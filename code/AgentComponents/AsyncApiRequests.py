import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from packages import *

import asyncio
import time
import re
import json
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Tuple, Dict
import psutil
from pympler import asizeof
import multiprocessing

from utils import (access_token, openai_key, deepseek_api_key)
from utils import calculate_confidence_score

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn


def get_deep_size(obj, seen=None):
    """递归计算 obj 所占内存（单位：字节）"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items()])
    elif isinstance(obj, (list, tuple, set)):
        size += sum([get_deep_size(i, seen) for i in obj])
    return size


class JSONExtractionError(Exception):
    pass

class JSONDecodeFailure(Exception):
    pass

class DummyLLMQueue:
    pass

class LLMClient(ABC):
    @abstractmethod
    async def batch_generate(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        pass

    @staticmethod
    def extract_json(text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                raise JSONDecodeFailure(f"JSON decoding failed: {e}")
        else:
            raise JSONExtractionError("No JSON object found in text.")


class OpenAIGPTClient(LLMClient):
    def __init__(self, api_key: str, gpt_version: str = "gpt-4o", temperature: float = 0, max_concurrency: int = 5, return_log_prob = False, response_format=False, timeout=120):
        self.engine = openai.AsyncOpenAI(api_key=api_key)
        self.gpt_version = gpt_version
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.return_log_prob = return_log_prob
        self.response_format = response_format
        self.timeout = timeout

    async def batch_generate(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        tasks = [self._generate(sys_prompt, usr_prompt) for sys_prompt, usr_prompt in zip(system_prompts, user_prompts)]
        return await asyncio.gather(*tasks)

    async def _generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, Dict[str, int]]:  # output str json
        # await asyncio.sleep(10)
        start_time = time.time()

        request_params = {
            "model": self.gpt_version,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "logprobs": self.return_log_prob,
        }

        if self.response_format:
            request_params["response_format"] = self.response_format
        else:
            request_params["response_format"] = {'type': 'json_object'}


        # print(response)

        try:
            response = await asyncio.wait_for(self.engine.chat.completions.create(**request_params), timeout=self.timeout)
        except asyncio.TimeoutError:
            tokens = {
                "total_token": -1,
                "input_token": -1,
                "output_token": -1
            }
            return "", -1, tokens


        elapsed_time = time.time() - start_time
        # print(f"OpenAI API 请求耗时: {elapsed_time:.2f} 秒")

        tokens = {
            "total_token": response.usage.total_tokens,
            "input_token": response.usage.prompt_tokens,
            "output_token": response.usage.completion_tokens
        }

        data = ast.literal_eval(response.choices[0].message.content)

        if self.return_log_prob:
            confidence_score = calculate_confidence_score(response.choices[0].logprobs)
            return data, confidence_score, tokens

        return data, -1, tokens


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, gpt_version: str = "deepseek-chat", temperature: float = 0,
                 max_concurrency: int = 5, return_log_prob = False, response_format=False, timeout=120):
        self.engine = openai.AsyncOpenAI(api_key=api_key, base_url='https://api.deepseek.com')
        self.gpt_version = gpt_version
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.return_log_prob = return_log_prob
        self.response_format = response_format
        self.timeout = timeout

    async def batch_generate(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        tasks = [self._generate(sys_prompt, usr_prompt) for sys_prompt, usr_prompt in zip(system_prompts, user_prompts)]
        return await asyncio.gather(*tasks)

    async def _generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, Dict[str, int]]:  # output str json
        start_time = time.time()

        request_params = {
            "model": self.gpt_version,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "logprobs": self.return_log_prob,
        }

        if self.response_format:
            request_params["response_format"] = {'type': 'json_object'}

        generation_flag, attempt_count = False, 0
        data = ""
        total_token, input_token, output_token = -1, -1, -1
        log_prob = type('LogProb', (), {'content': []})
        while not generation_flag and attempt_count < 3:
            try:
                response = await asyncio.wait_for(self.engine.chat.completions.create(**request_params), timeout=self.timeout) # timeout=self.timeout

                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.strip("`")  # 去掉所有反引号
                    content_lines = content.split("\n")
                    if content_lines[0].strip().startswith("json"):
                        content = "\n".join(content_lines[1:])  # 去掉第一行的```json
                    else:
                        content = "\n".join(content_lines)

                data = ast.literal_eval(content)
                if data is not None:
                    generation_flag = True

                    total_token, input_token, output_token = response.usage.total_tokens, response.usage.prompt_tokens, response.usage.completion_tokens
                    log_prob = response.choices[0].logprobs

            except asyncio.TimeoutError:
                print("Run time Error")
                tokens = {
                    "total_token": total_token,
                    "input_token": input_token,
                    "output_token": output_token
                }
                return "", -1, tokens

            except Exception as e:
                if attempt_count == 2:
                    print(f"Error evaluating cluster: {e} with attempt time {attempt_count}") # with response {response}"

            finally:
                attempt_count += 1


        elapsed_time = time.time() - start_time
        # print(f"OpenAI API 请求耗时: {elapsed_time:.2f} 秒")

        tokens = {
            "total_token": total_token,
            "input_token": input_token,
            "output_token": output_token
        }

        if self.return_log_prob:
            # print(log_prob)
            confidence_score = calculate_confidence_score(log_prob)
            return data, confidence_score, tokens

        return data, -1, tokens


class LLamaHFClient(LLMClient):
    def __init__(self, huggingface_api_key:str, llm_version: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", temperature: float=0.7, max_concurrency: int = 5, return_log_prob=False, timeout=120, quantization=True, response_format=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_version, token=huggingface_api_key)
        # self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantization:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            self.quantization_config = None


        self.model = AutoModelForCausalLM.from_pretrained(
            llm_version,
            quantization_config=self.quantization_config,
            device_map="auto",
            token=huggingface_api_key
        )

        # self.model = copy.deepcopy(self.model_base)

        self.max_new_tokens = 512
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout = timeout
        self.model.eval()

        if response_format:
            self.response_format = response_format["json_schema"]["schema"]
            parser = JsonSchemaParser(self.response_format)
            self.prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)

        else:
            self.response_format = False
            self.prefix_allowed_tokens_fn = None



    async def batch_generate(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        tasks = [self._generate(system, user) for system, user in zip(system_prompts, user_prompts)]
        return await asyncio.gather(*tasks)

    def run_generation(self, inputs):

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]
        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs.to(self.device),
                max_new_tokens=self.max_new_tokens,
                temperature=1,
                top_p=None,
                do_sample=False,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_scores=False,
                repetition_penalty=1.2,
            )

            generated_ids = generated_ids.cpu()

            output_len = generated_ids.shape[1] - input_len
            token_info = {
                "input_token": input_len,
                "output_token": output_len,
                "total_token": input_len + output_len
            }

        return generated_ids, token_info

    async def _generate(self, system_prompt: str, user_prompt: str) -> Tuple[str, int, Dict[str, int]]:
        # print("-"*25+" start "+ "-"*25, flush=True)
        generation_flag, attempt_count = False, 0
        data = ""
        total_token, input_token, output_token = -1, -1, -1

        prompt = self._build_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_len = inputs["input_ids"].shape[1]

        max_allowed_input_len = self.model.config.max_position_embeddings - 200

        if input_len > max_allowed_input_len:
            print(f"[Truncate] Input too long ({input_len} tokens), truncating to {max_allowed_input_len}")
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False,
                                    truncation=True, max_length=max_allowed_input_len)

        while not generation_flag and attempt_count < 3:
            try:
                generated_ids, token_info = await asyncio.wait_for(
                    asyncio.to_thread(lambda: self.run_generation(inputs)),
                    timeout=self.timeout
                )

                decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                content = decoded[len(prompt):].strip()

                data = self.extract_json(content)

                if data is not None:
                    generation_flag = True
                    total_token, input_token, output_token = token_info["total_token"], token_info["input_token"], token_info["output_token"]

            except asyncio.TimeoutError:
                print("Run time Error", file=sys.stdout)
                tokens = {
                    "total_token": total_token,
                    "input_token": input_token,
                    "output_token": output_token
                }
                return "", -1, tokens

            except Exception as e:
                if attempt_count == 2:
                    print(f"Error evaluating cluster: {e} with attempt time {attempt_count}", file=sys.stderr) # , with content {content}

            finally:
                attempt_count += 1


        tokens = {
            "total_token": total_token,
            "input_token": input_token,
            "output_token": output_token
        }
        del inputs, generated_ids, decoded, content, prompt
        return data, -1, tokens

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"

    # def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
    #     system_prompt = system_prompt.strip()
    #     user_prompt = user_prompt.strip()
    #
    #     return (
    #         f"{system_prompt}\n\n{user_prompt}"
    #     )


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

        self.max_ram_gb = 6.5
        self.max_gpu_gb = 6
        self.sleep_time = 5


        if progress_bar:
            self.progress_bar = tqdm(total=1, desc="Processing Tasks", unit="task", file=sys.stderr, dynamic_ncols=True)
        else:
            self.progress_bar = None

    async def wait_if_oom_risk(self):
        while True:
            ram = psutil.Process().memory_info().rss / 1024 ** 3
            gpu_mem = torch.cuda.memory_allocated() / 1024 ** 3
            if ram > self.max_ram_gb or gpu_mem > self.max_gpu_gb:
                print(f"[Throttling] RAM={ram:.2f}GB, GPU={gpu_mem:.2f}GB. Sleeping {self.sleep_time}s...", file=sys.stdout)
                torch.cuda.empty_cache()
                gc.collect()

                await asyncio.sleep(self.sleep_time)
            else:
                break

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
                        print("Terminate signal received. Worker exit.", file=sys.stdout)
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
                await self.wait_if_oom_risk()

                results = await self.llm_client.batch_generate(batch_sys_prompts, batch_user_prompts)
                for (response, logprob ,token_data), task_id, user_prompt in zip(results, tasks, batch_user_prompts):
                    self.task_results[task_id] = {
                        "user_prompt": user_prompt,
                        "response": response,
                        "logprob": logprob
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
        try:
            await asyncio.wait_for(self.queue.join(), timeout=120)  # 设置队列任务完成的超时
        except asyncio.TimeoutError:
            print("Wait for queue task completion timeout to force all tasks to stop", file=sys.stdout)

        # 向队列里放入终止信号
        for _ in range(self.num_workers):
            await self.queue.put(None)

        # 强制取消所有未完成的 worker 任务
        for worker in self.workers:
            if not worker.done():
                print(f"Cancel unfinished workers: {worker.get_name()}" , file=sys.stdout)
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    print(f"Worker {worker.get_name()} is cancelled",file=sys.stdout)

        self.stop_event.set()

        if self.progress_bar:
            self.progress_bar.close()  # 关闭进度条

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

    async def _wait_for_result(self, task_id):
        while task_id not in self.task_results:
            await asyncio.sleep(0.1)

def create_llm_client(llm_type, return_log_prob, response_format, timeout):
    if llm_type == "DeepSeek":
        return DeepSeekClient(api_key=deepseek_api_key, return_log_prob=return_log_prob,
                              response_format=response_format, timeout=timeout)
    elif llm_type == "GPT":
        return OpenAIGPTClient(api_key=openai_key, gpt_version="gpt-4o", return_log_prob=return_log_prob,
                               response_format=response_format, timeout=timeout)
    elif llm_type == "Llama":
        return LLamaHFClient(huggingface_api_key=access_token, llm_version="meta-llama/Meta-Llama-3.1-8B-Instruct",
                             response_format=response_format, timeout=timeout)
    else:
        raise ValueError("Not applicable to specified LLM")

def run_chunk_in_process(chunk_start, chunk_end, user_prompt_lst, system_prompt,
                         batch_size, num_workers, shared_dict,
                         llm_type, return_log_prob, response_format, timeout):
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    llm_client = create_llm_client(llm_type, return_log_prob, response_format, timeout)
    llm_queue = LLMQueue(llm_client, batch_size=batch_size, num_workers=num_workers, progress_bar=True)

    async def process():
        await llm_queue.start_workers()
        await asyncio.gather(*[
            llm_queue.generate(i, system_prompt, user_prompt_lst[i])
            for i in range(chunk_start, chunk_end)
        ])
        await llm_queue.stop_workers()

    loop.run_until_complete(process())
    loop.close()

    shared_dict["task_results"] = dict(llm_queue.task_results)
    shared_dict["total_token"] = llm_queue.total_token
    shared_dict["input_token"] = llm_queue.input_token
    shared_dict["output_token"] = llm_queue.output_token


async def run_async_llm_tasks(system_prompt, user_prompt_lst, start, end, return_log_prob=False, response_format=None,
                                batch_size=4, num_workers=4, llm_type="DeepSeek", max_retries=2, timeout=120): # each question time

    chunk_size = 50 if llm_type == "Llama" else end - start
    task_results = {}
    total_token, input_token, output_token = 0, 0, 0
    print(f"Total Task Number: {end-start}")

    for chunk_start in range(start, end, chunk_size):
        chunk_end = min(chunk_start + chunk_size, end)

        print(f"\n Running chunk from {chunk_start} to {chunk_end}...")

        ctx = multiprocessing.get_context('spawn')
        manager = ctx.Manager()
        shared_dict = manager.dict()

        p = ctx.Process(
            target=run_chunk_in_process,
            args=(
                chunk_start, chunk_end,
                user_prompt_lst, system_prompt,
                batch_size, num_workers,
                shared_dict,
                llm_type, return_log_prob, response_format, timeout
            )
        )
        p.start()
        p.join()

        print(f"[RAM] {psutil.Process().memory_info().rss / 1024 ** 3:.2f} GB")
        print(f"[CUDA] {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

        task_results.update(shared_dict.get("task_results", {}))
        total_token += shared_dict.get("total_token", 0)
        input_token += shared_dict.get("input_token", 0)
        output_token += shared_dict.get("output_token", 0)

    final_llm_queue = DummyLLMQueue()
    final_llm_queue.task_results = task_results
    final_llm_queue.total_token = total_token
    final_llm_queue.input_token = input_token
    final_llm_queue.output_token = output_token

    return final_llm_queue


def execute_llm_tasks(system_prompt, user_prompt_lst, start, end,return_log_prob=False, response_format=None,
                                batch_size=1, num_workers=1, llm_type="DeepSeek"):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        llm_queue = loop.run_until_complete(
            run_async_llm_tasks(
                system_prompt=system_prompt,
                user_prompt_lst=user_prompt_lst,
                return_log_prob=return_log_prob,
                response_format=response_format,
                batch_size=batch_size,
                num_workers=num_workers,
                llm_type=llm_type,
                start = start,
                end= end
            )
        )
    finally:
        loop.close()

    return llm_queue

if __name__ == "__main__":

    system_prompt = '''
    You are a helpful assistant. Your task is to generate structured JSON outputs based on user instructions.
    
    Only return a valid JSON object. Do not include any explanation or extra text.
    Ensure the output can be parsed by Python's json.loads().
    '''

    user_prompt_lst = ["Please give me any data according to the output format." for i in range(1)]

    # # cluster_format_type
    # cluster_format_type = {
    #     "type": "json_schema",
    #     "json_schema": {
    #         "name": "paragraph_cluster_evaluation",
    #         "schema": {
    #             "type": "object",
    #             "properties": {
    #                 "cluster": {
    #                     "type": "array",
    #                     "items": {
    #                         "type": "object",
    #                         "properties": {
    #                             "indices": {
    #                                 "type": "array",
    #                                 "items": {"type": "integer"}
    #                             },
    #                             "description": {"type": "string"},
    #                             "label": {
    #                                 "type": "string",
    #                                 "enum": ["<STRONG>", "<Medium>", "<WEAK>"]
    #                             }
    #                         },
    #                         "required": ["indices", "description", "label"],
    #                         "additionalProperties": False
    #                     }
    #                 }
    #             },
    #             "required": ["cluster"],
    #             "additionalProperties": False
    #         }
    #     }
    # }
    #
    # result = execute_llm_tasks(system_prompt, user_prompt_lst, return_log_prob=True, response_format=cluster_format_type, llm_type="DeepSeek") # response_format=cluster_format_type

    result = execute_llm_tasks(system_prompt, user_prompt_lst, return_log_prob=False,
                               response_format=None,
                               llm_type="Llama", start=0, end=len(user_prompt_lst))  # response_format=cluster_format_type llm_type="DeepSeek"

    random_value = random.choice(list(result.task_results.values()))
    print(random_value)

    #
    # from openai import OpenAI
    #
    # # client = OpenAI(api_key= deepseek_api_key, base_url="https://api.deepseek.com")
    # client = OpenAI(api_key=openai_key)
    #
    # start_time = time.time()
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant"},
    #         {"role": "user", "content": "Hello"},
    #     ],
    #     # stream=False,
    #     response_format="json"
    # )
    # print(response)
    # print(response.choices[0].message.content)
    # print(f"Finished in {time.time()-start_time}")
    #

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        quantization_config = quantization_config,        # 可选：更省显存
        device_map="auto",                # 自动将模型分配到可用GPU
    )

    # 3. 构造输入文本
    prompt = "You are a helpful assistant. Hello!"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4. 推理
    start_time = time.time()
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95
    )

    # 5. 输出结果
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(f"Running time: {time.time() - start_time}s")

