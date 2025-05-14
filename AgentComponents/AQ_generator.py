from packages import *

import asyncio
import time
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict


class LLMClient(ABC):
    @abstractmethod
    async def batch_generate(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        pass


class OpenAIGPTClient(LLMClient):
    def __init__(self, api_key: str, gpt_version: str = "gpt-4o", temperature: float = 0, max_concurrency: int = 5):
        self.engine = openai.AsyncOpenAI(api_key=api_key)
        self.gpt_version = gpt_version
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def batch_generate(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        tasks = [self._generate(sys_prompt, usr_prompt) for sys_prompt, usr_prompt in zip(system_prompts, user_prompts)]
        return await asyncio.gather(*tasks)

    async def _generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:  # output str json
        start_time = time.time()

        request_params = {
            "model": self.gpt_version,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        response = await self.engine.chat.completions.create(**request_params)
        elapsed_time = time.time() - start_time
        # print(f"OpenAI API 请求耗时: {elapsed_time:.2f} 秒")

        tokens = {
            "total_token": response.usage.total_tokens,
            "input_token": response.usage.prompt_tokens,
            "output_token": response.usage.completion_tokens
        }

        return response.choices[0].message.content, tokens


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, gpt_version: str = "deepseek-chat", temperature: float = 0,
                 max_concurrency: int = 5):
        self.engine = openai.AsyncOpenAI(api_key=api_key, base_url='https://api.deepseek.com')
        self.gpt_version = gpt_version
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def batch_generate(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        tasks = [self._generate(sys_prompt, usr_prompt) for sys_prompt, usr_prompt in zip(system_prompts, user_prompts)]
        return await asyncio.gather(*tasks)

    async def _generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:  # output str json
        start_time = time.time()

        request_params = {
            "model": self.gpt_version,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
        response = await self.engine.chat.completions.create(**request_params)
        elapsed_time = time.time() - start_time
        # print(f"OpenAI API 请求耗时: {elapsed_time:.2f} 秒")

        tokens = {
            "total_token": response.usage.total_tokens,
            "input_token": response.usage.prompt_tokens,
            "output_token": response.usage.completion_tokens
        }

        return response.choices[0].message.content, tokens


class LLMQueue:
    def __init__(self, llm_client: LLMClient, batch_size=4, wait_time=0.5, num_workers=2, progress_bar=False):
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.queue = asyncio.Queue()
        self.task_results = {}
        self.stop_event = asyncio.Event()  # 用于标记是否停止 worker
        self.num_workers = num_workers
        self.total_tasks = 0  # 记录提交的任务总数
        self.completed_tasks = 0  # 记录完成的任务数

        self.total_token = 0
        self.input_token = 0
        self.output_token = 0

        if progress_bar:
            self.progress_bar = tqdm(total=1, desc="Processing Tasks", unit="task")
        else:
            self.progress_bar = None

    async def worker(self):
        """后台 worker，处理任务队列"""
        while not self.stop_event.is_set():
            batch_sys_prompts, batch_user_prompts, tasks = [], [], []

            try:
                start_time = time.time()
                while len(batch_sys_prompts) < self.batch_size:
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=self.wait_time)
                    except asyncio.TimeoutError:
                        break

                    if item is None:
                        print("终止信号收到，退出 worker")
                        self.stop_event.set()
                        return

                    try:
                        sys_prompt, user_prompt, task_id = item

                        batch_sys_prompts.append(sys_prompt)
                        batch_user_prompts.append(user_prompt)
                        tasks.append(task_id)
                    finally:
                        self.queue.task_done()

            except asyncio.TimeoutError:
                pass  # 超时后立即处理当前 batch

            if batch_sys_prompts and batch_user_prompts:
                results = await self.llm_client.batch_generate(batch_sys_prompts, batch_user_prompts)
                for (response, token_data), task_id, user_prompt in zip(results, tasks, batch_user_prompts):
                    self.task_results[task_id] = {
                        "user_prompt": user_prompt,
                        "response": response
                    }
                    self.total_token += token_data["total_token"]
                    self.input_token += token_data["input_token"]
                    self.output_token += token_data["output_token"]

                    self.completed_tasks += 1  # 任务完成计数

                    if self.progress_bar:
                        await asyncio.to_thread(self.progress_bar.update, 1)

    async def start_workers(self):
        """创建多个 worker 共享 queue"""
        self.workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]

    async def stop_workers(self):
        """停止所有 worker"""
        self.stop_event.set()
        await self.queue.join()

        for _ in range(self.num_workers):
            await self.queue.put(None)

        await asyncio.gather(*self.workers)

        if self.progress_bar:
            self.progress_bar.close()  # 关闭进度条

    async def generate(self, index: int, system_prompt: str, user_prompt: str) -> str:
        """提交任务到队列，并等待结果"""

        task_id = f"query_{index}"
        self.queue.put_nowait((system_prompt, user_prompt, task_id))
        # print(f"🔹 任务提交: {task_id}")

        self.total_tasks += 1  # 记录任务总数

        if self.progress_bar:
            self.progress_bar.total = self.total_tasks  # 更新进度条的总数
            self.progress_bar.refresh()

        while task_id not in self.task_results:
            await asyncio.sleep(0.1)

        return self.task_results[task_id]

