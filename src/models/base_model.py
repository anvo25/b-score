# models/base_model.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def get_response(self, messages, temperature=1.0):
        """
        Sends a single-turn request to the model and returns the response.

        Args:
            messages (list): List of message dictionaries.
            temperature (float): Sampling temperature.

        Returns:
            str: Model's response content.
        """
        pass

