# models/openai_model.py
import requests
import json
import time
from .base_model import BaseModel
from utils import extract_answer_from_curly_brackets

class OpenAIModel(BaseModel):
    def __init__(self, api_key, model_name='gpt-4o-mini'):
        self.api_key = api_key
        self.model_name = model_name
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def get_response(self, messages, temperature=1.0):
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        'model': self.model_name,
                        'messages': messages,
                        'temperature': temperature
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content']
                
            except Exception as e:
                print(f"OpenAI API Error: {e}. Retrying in 60 seconds...")
                retries += 1
                time.sleep(60)
        
        print(f"Failed to get response from OpenAI after {max_retries} attempts. Skipping...")
        return None

    def get_batch_responses(self, batch_messages, temperature=1.0, batch_size=300):
        responses = []
        for i in range(0, len(batch_messages), batch_size):
            batch = batch_messages[i:i + batch_size]
            payload = {
                'model': self.model_name,
                'messages': batch,
                'temperature': temperature
            }
            retries = 0
            max_retries = 3
            while retries < max_retries:
                try:
                    response = requests.post(self.api_url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    for choice in result['choices']:
                        content = choice['message']['content']
                        responses.append(content)
                    break
                except Exception as e:
                    print(f"OpenAI Batch API Error: {e}. Retrying in 60 seconds...")
                    retries += 1
                    time.sleep(60)
            else:
                print(f"Failed to process batch {i//batch_size +1}. Skipping.")
        return responses
