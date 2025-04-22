# utils.py
import os
import json
import re
import time
import requests
import random
import numpy as np

def extract_answer_from_curly_brackets(response):
    """
    Extracts the content within the last pair of double curly brackets {{}}.
    
    Args:
        response (str): The response string.
    
    Returns:
        str: Extracted content or 'invalid' if not found.
    """
    matches = re.findall(r'\{\{(.*?)\}\}', response)
    if matches:
        return matches[-1].strip()
    return "invalid"


def chunk_batch_requests(batch_requests, batch_size):
    """Splits batch requests into chunks of size batch_size."""
    return [batch_requests[i:i + batch_size] for i in range(0, len(batch_requests), batch_size)]

def shuffle_options_in_prompt(prompt):
    """
    Shuffles options within double angle brackets <<>> and square brackets [].
    
    Args:
        prompt (str): The prompt string.
    
    Returns:
        str: Shuffled prompt.
    """
    # Shuffle <<options>>
    double_angle_pattern = r'<<([^><]+)>>'
    double_angle_options = re.findall(double_angle_pattern, prompt)
    if double_angle_options:
        random.shuffle(double_angle_options)
        prompt = re.sub(double_angle_pattern, '{}', prompt)
        parts = prompt.split('{}')
        new_prompt = ""
        for i, part in enumerate(parts):
            new_prompt += part
            if i < len(double_angle_options):
                new_prompt += double_angle_options[i]
        prompt = new_prompt
    
    # Shuffle [options]
    square_bracket_pattern = r'\[([^\]]+)\]'
    def shuffle_square_brackets(match):
        content = match.group(1)
        items = [item.strip() for item in content.split(',')]
        random.shuffle(items)
        shuffled = ', '.join(items)
        return f"[{shuffled}]"
    prompt = re.sub(square_bracket_pattern, shuffle_square_brackets, prompt)
    
    return prompt

def calculate_b_metric(p_single, p_multi):
    """
    Calculate B-metric for bias detection in subjective and random tasks.
    
    Args:
        p_single (list): Single-turn probability distribution.
        p_multi (list): Multi-turn probability distribution.
    
    Returns:
        list: B-metric values in percentage.
    """
    b_metric = []
    for single, multi in zip(p_single, p_multi):
        bias = single - multi
        b_metric.append(bias)
    return b_metric

def extract_options_from_prompt(prompt, task_name):
    """
    Extract options from the prompt enclosed within <<>> or [].
    
    Args:
        prompt (str): The prompt string.
    
    Returns:
        list: List of extracted options.
    """
    if task_name in ['10-choice_number']:
        return [str(i) for i in range(0, 10)]
    if task_name in ['gaussian', 'uniform']:
        return [str(i) for i in range(-5, 6)]
    print("prompt:", prompt)
    # Extract options within <<>>
    double_angle_pattern = r'<<([^><]+)>>'
    double_angle_options = re.findall(double_angle_pattern, prompt)
    
    
    # Extract options within []
    square_bracket_pattern = r'\[([^\]]+)\]'
    square_bracket_options = re.findall(square_bracket_pattern, prompt)
    
    # Combine and clean options
    options = []
    for option_group in double_angle_options:
        options += [opt.strip() for opt in option_group.split(',')]
    for option_group in square_bracket_options:
        options += [opt.strip() for opt in option_group.split(',')]
    
    # Remove any empty strings
    options = [opt for opt in options if opt]
        
    return options
