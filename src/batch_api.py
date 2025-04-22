# batch_api.py
import time
import os
import json
import requests

def process_batch_jobs(model, batch_file_path, output_dir, batch_file_name):
    """
    Process batch jobs using the appropriate model's batch API.
    
    Args:
        model (BaseModel): An instance of a model class.
        batch_file_path (str): Path to the batch request file.
        output_dir (str): Directory to save responses.
        batch_file_name (str): Name of the batch file.
    
    Returns:
        list: List of batch responses.
    """
    if hasattr(model, 'get_batch_responses'):
        with open(batch_file_path, 'r') as f:
            batch_requests = [json.loads(line) for line in f]
        messages = [req['body']['messages'] for req in batch_requests]
        responses = model.get_batch_responses(messages, batch_size=300)
        # Save responses
        response_file_name = batch_file_name.replace('requests', 'responses')
        response_file_path = os.path.join(output_dir, response_file_name)
        with open(response_file_path, 'w') as f:
            json.dump(responses, f, indent=4)
        print(f"Batch results saved to {response_file_path}")
        return responses
    else:
        raise NotImplementedError("Batch processing not implemented for this model.")
