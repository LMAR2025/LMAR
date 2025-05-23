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
        self.stop_event = asyncio.Event()  
        self.num_workers = num_workers
        self.total_tasks = 0  
        self.completed_tasks = 0  

        self.total_token = 0
        self.input_token = 0
        self.output_token = 0

        if progress_bar:
            self.progress_bar = tqdm(total=1, desc="Processing Tasks", unit="task")
        else:
            self.progress_bar = None

    async def worker(self):
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
                        print("terminated，exit worker")
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
                pass 

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

                    self.completed_tasks += 1 

                    if self.progress_bar:
                        await asyncio.to_thread(self.progress_bar.update, 1)

    async def start_workers(self):
        self.workers = [asyncio.create_task(self.worker()) for _ in range(self.num_workers)]

    async def stop_workers(self):
        self.stop_event.set()
        await self.queue.join()

        for _ in range(self.num_workers):
            await self.queue.put(None)

        await asyncio.gather(*self.workers)

        if self.progress_bar:
            self.progress_bar.close() 

    async def generate(self, index: int, system_prompt: str, user_prompt: str) -> str:

        task_id = f"query_{index}"
        self.queue.put_nowait((system_prompt, user_prompt, task_id))

        self.total_tasks += 1  

        if self.progress_bar:
            self.progress_bar.total = self.total_tasks 
            self.progress_bar.refresh()

        while task_id not in self.task_results:
            await asyncio.sleep(0.1)

        return self.task_results[task_id]

