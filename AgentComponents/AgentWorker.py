import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages import *
from utils import mean_pooling
from utils import access_token, openai_key, deepseek_api_key

# from AgentComponents.VectorStore import VectorStore
# from AgentComponents.QuestionClassifier import GPTQuestionClassifier
# from AgentComponents.LLMSetting.llama3Setting8B import llm_pipeline

class AgentWorker:
    def __init__(self, llm_pipeline = None, role = None):
        # Add asssertation check for type of llm and retriever
        self.llm = llm_pipeline
        self.role = role
        self.messages = [self.role]
        self.user_input = None # {"role":"user", "content":None}
        self.history = None
        
    def generate(self, text, max_new_tokens = 200):
        if self.history is None:
            # Combine input from users
            self.user_input = {"role":"user", "content": text}
            self.messages.append(self.user_input)

            # Get generation texts
            outputs = self.llm(self.messages, max_new_tokens = max_new_tokens)[0]["generated_text"]

            # Update History
            self.history = outputs
            
        else:
            # Combine input from users
            self.user_input = {"role":"user", "content": text}
            
            self.history.append(self.user_input)

            # Get generation texts
            outputs = self.llm(self.history, max_new_tokens = max_new_tokens)[0]["generated_text"]

            # Update History
            self.history = outputs
        
        return outputs

    def start_new_conversation(self):
        self.history = None
        self.messages = [self.role]

class GPTWorker:
    def __init__(self, openai_api_key=openai_key, model_name="gpt-4o-mini", temperature=None):
        self.MODEL = model_name
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", openai_api_key))
        self.temperature = temperature

    def generate(self, question, system_prompt, return_log_prob = False):
        if return_log_prob:
            request_params = {
                "model": self.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "logprobs": True
            }
        else:
            request_params = {
                "model": self.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "logprobs": False 
            }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        
        completion = self.client.chat.completions.create(**request_params)
        
        response_text = completion.choices[0].message.content
        if return_log_prob:
            token_logprobs = completion.choices[0].logprobs  # Extract log probabilities
            return response_text, token_logprobs
        else:
            return response_text
        

class DeepSeekWorker:
    def __init__(self, deepseek_api_key=deepseek_api_key, model_name="deepseek-chat", temperature=None):
        self.MODEL = model_name
        self.client = OpenAI(api_key=deepseek_api_key, base_url = 'https://api.deepseek.com')
        self.temperature = temperature

    def generate(self, question, system_prompt, return_log_prob = False):
        if return_log_prob:
            request_params = {
                "model": self.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "logprobs": True
            }
        else:
            request_params = {
                "model": self.MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                "logprobs": False 
            }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        
        completion = self.client.chat.completions.create(**request_params)
        
        response_text = completion.choices[0].message.content
        if return_log_prob:
            token_logprobs = completion.choices[0].logprobs  # Extract log probabilities
            return response_text, token_logprobs
        else:
            return response_text
