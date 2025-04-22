# models/__init__.py
from .openai_model import OpenAIModel
from .cohere_model import CohereModel
from .sambanova_model import SambanovaModel
from .gemini_model import GeminiModel

def get_model(model_name, api_key):
    """
    Factory function to instantiate the appropriate model class based on model_name.
    
    Args:
        model_name (str): Name of the model.
        api_key (str): API key for the model provider.
    
    Returns:
        BaseModel: An instance of a model class inheriting from BaseModel.
    """
    if model_name.startswith('gpt') or model_name.startswith('o1'):  # Adjust condition based on your naming convention
        return OpenAIModel(api_key=api_key, model_name=model_name)
    elif model_name.startswith('command'):
        return CohereModel(api_key=api_key, model_name=model_name)
    elif model_name.startswith('Meta-Llama'):
        return SambanovaModel(api_key=api_key, model_name=model_name)
    elif model_name.startswith('gemini'):
        return GeminiModel(api_key=api_key, model_name=model_name)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")
