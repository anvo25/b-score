# models/cohere_model.py
import cohere
import json
import time
from .base_model import BaseModel
from utils import extract_answer_from_curly_brackets

class CohereModel(BaseModel):
    def __init__(self, api_key, model_name='command-r7b-12-2024'):
        self.api_key = api_key
        self.model_name = model_name
        self.co = cohere.ClientV2(api_key=self.api_key)

    def get_response(self, messages, temperature=1.0):
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                response = self.co.chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.message.content[0].text
                
            except Exception as e:
                print(f"Cohere API Error: {e}. Retrying in 60 seconds...")
                retries += 1
                time.sleep(60)
        
        print(f"Failed to get response from Cohere after {max_retries} attempts. Skipping...")
        return None
    
    def get_batch_responses(self, batch_messages, temperature=1.0, batch_size=300):
        responses = []
        for i in range(0, len(batch_messages), batch_size):
            batch = batch_messages[i:i + batch_size]
            try:
                batch_response = self.client.chat(
                    model=self.model_name,
                    messages=batch,
                    temperature=temperature
                )
                for msg in batch_response.message.content:
                    responses.append(msg.text)
            except Exception as e:
                print(f"Cohere Batch API Error: {e}. Retrying in 10 seconds...")
                time.sleep(10)
                # Retry once
                try:
                    batch_response = self.client.chat(
                        model=self.model_name,
                        messages=batch,
                        temperature=temperature
                    )
                    for msg in batch_response.message.content:
                        responses.append(msg.text)
                except Exception as e:
                    print(f"Failed to process batch {i//batch_size +1}: {e}. Skipping.")
        return responses
