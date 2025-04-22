import os
import openai
import time

from .base_model import BaseModel
from utils import extract_answer_from_curly_brackets

class SambanovaModel(BaseModel):
    def __init__(self, api_key, model_name='Meta-Llama-3.1-70B-Instruct'):
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.sambanova.ai/v1"
        )

    def get_response(self, messages, temperature=1.0):
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"SambaNova API Error: {e}. Retrying in 60 seconds...")
                retries += 1
                time.sleep(60)
        
        print(f"Failed to get response from SambaNova after {max_retries} attempts. Skipping...")
        return None