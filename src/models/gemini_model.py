import google.generativeai as genai
from .base_model import BaseModel
import time


class GeminiModel(BaseModel):
    def __init__(self, api_key, model_name='gemini-1.5-flash'):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
    def _convert_messages_to_history(self, messages):
        """Convert OpenAI-style messages to Gemini chat history format."""
        history = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else msg["role"]
            history.append({
                "role": role,
                "parts": msg["content"]
            })
        return history
        
    def get_response(self, messages, temperature=1.0):
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                # Convert messages to Gemini format
                history = self._convert_messages_to_history(messages[:-1])
                current_prompt = messages[-1]["content"]
                
                # Start chat with history if it exists
                if history:
                    chat = self.model.start_chat(history=history)
                    response = chat.send_message(
                        current_prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=temperature
                        )
                    )
                else:
                    response = self.model.generate_content(
                        current_prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=temperature
                        )
                    )
                
                return response.text
                
            except Exception as e:
                print(f"Gemini API Error: {e}. Retrying in 60 seconds...")
                retries += 1
                time.sleep(60)
        
        print(f"Failed to get response from Gemini after {max_retries} attempts. Skipping...")
        return None