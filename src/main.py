# main.py
import argparse
import os
import json
import time
import random
from models import get_model
from utils import (
    extract_answer_from_curly_brackets,
    chunk_batch_requests,
    shuffle_options_in_prompt,
    calculate_b_metric,
    extract_options_from_prompt
)
from batch_api import process_batch_jobs
from tqdm import tqdm
import pandas as pd
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments with multiple LLMs.')
    parser.add_argument('--n_runs', default=10, type=int, help='Number of experiment runs.')
    parser.add_argument('--n_turns', default=30, type=int, help='Number of turns in each multi-turn conversation.')
    parser.add_argument('--n_queries_single_turn', default=30, type=int,
                        help='Number of single-turn queries per run.')
    parser.add_argument('--task_name', required=True, type=str, help='The name of the task.')
    parser.add_argument('--category', required=True, type=str,
                        choices=['subjective', 'random', 'objective', 'hard'],
                        help='Category of the task.')
    parser.add_argument('--model_name', default='command-r7b-12-2024', type=str, help='The name of the model.')
    parser.add_argument('--temperature', default=1.0, type=float, help='The temperature setting for the model.')
    parser.add_argument('--api_type', default='single', choices=['single', 'batch'],
                        help='Type of API to use (only needed for OpenAI).')
    parser.add_argument('--batch_size', default=300, type=int,
                        help='Maximum number of requests per batch file (only for batch API).')
    parser.add_argument('--resume', action="store_true",
                        help='Resume from existing results if available.')
    parser.add_argument('--prompt_folder', default='prompts', type=str,
                        help='Path to the folder containing prompt files.')
    parser.add_argument('--output_folder', default='logs_rebuttal', type=str,
                        help='Folder to save results.')
    parser.add_argument('--shuffle_options', default=False,
                        help='Shuffle the order of options in curly or square brackets.')
    parser.add_argument('--confidence_score', default=True,
                        help='Include confidence score in the output.')
    return parser.parse_args()


def load_api_key(provider, model_name):
    """Load API key from file."""
    
    if provider == 'sambanova':
        if model_name=='Meta-Llama-3.1-405B-Instruct':
            file_path = f"api_key/{provider}_key.txt"
        elif model_name=='Meta-Llama-3.1-70B-Instruct':
            file_path = f"api_key/{provider}2_key.txt"
        elif model_name=='Meta-Llama-3.3-70B-Instruct':
            file_path = f"api_key/{provider}3_key.txt"
    else: 
        file_path = f"api_key/{provider}_key.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"API key file '{file_path}' not found.")
    with open(file_path, 'r') as f:
        return f.read().strip()


def load_prompt(prompt_folder, task_name, category, shuffle_options=False):
    """Load and prepare prompt from file."""
    prompt_file = os.path.join(prompt_folder, task_name, f"{task_name}_{category}.txt")
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file '{prompt_file}' not found.")

    with open(prompt_file, 'r') as f:
        prompt = f.read()
    if task_name in ["gaussian", "uniform"]:
        template = "\nYou MUST put the list in double curly brackets: {{your list}}."
    else:
        template = "\nYou MUST choose one and respond using double curly brackets: {{your choice}}."
    prompt += template
    return prompt


def generate_output_dir(args):
    """Create output directory structure."""
    dir_parts = [
        args.output_folder,
        args.model_name,
        args.task_name,
        args.category,
        f"temp_{args.temperature}"
    ]
    output_dir = os.path.join(*dir_parts)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_metadata(args, output_dir):
    """Save experiment metadata."""
    metadata = vars(args)
    metadata_filepath = os.path.join(output_dir, 'metadata.json')
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)


def save_run_results(run_id, results, output_filepath):
    """Save results of a single run."""
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=4)


def calculate_metrics(args, single_probs, multi_probs, confidence_scores_per_option, 
                     valid_single_count, valid_multi_count):
    """Calculate and return metrics DataFrame."""
    # For real_gaussian_random tasks, handle differently
    if args.task_name == "real_gaussian":
        # For real values, we don't have discrete options or B-metrics
        # Just calculate valid rates
        valid_single_rate = (valid_single_count / args.n_queries_single_turn * 100) if args.n_queries_single_turn > 0 else 0
        valid_multi_rate = (valid_multi_count / args.n_turns * 100) if args.n_turns > 0 else 0
        
        # Return an empty dataframe or a dataframe with just the valid rates
        df = pd.DataFrame({
            'Valid Single Rate': [valid_single_rate],
            'Valid Multi Rate': [valid_multi_rate]
        })
        
        return df, valid_single_rate, valid_multi_rate
    
    # Original code for other tasks
    df = pd.DataFrame({
        'Option': list(single_probs.keys()),
        'Single-turn Probability': list(single_probs.values()),
        'Multi-turn Probability': list(multi_probs.values())
    })

    b_metric = calculate_b_metric(list(single_probs.values()), list(multi_probs.values()))
    df['B-metric'] = b_metric

    mean_confidence = []
    std_confidence = []
    for option in df['Option']:
        scores = confidence_scores_per_option.get(option, [])
        if scores:
            mean_confidence.append(np.mean(scores))
            std_confidence.append(np.std(scores))
        else:
            mean_confidence.append(None)
            std_confidence.append(None)
            
    df['Mean Confidence Score'] = mean_confidence
    df['Std Confidence Score'] = std_confidence

    valid_single_rate = (valid_single_count / args.n_queries_single_turn * 100) if args.n_queries_single_turn > 0 else 0
    valid_multi_rate = (valid_multi_count / args.n_turns * 100) if args.n_turns > 0 else 0

    return df, valid_single_rate, valid_multi_rate



def process_single_turn(args, model, original_prompt, options, provider, confidence_scores_per_option):
    """Process single-turn queries."""
    single_run_results = []
    valid_single_answers = 0

    for query_id in range(args.n_queries_single_turn):
        # Initialize conversation history
        conversation_history = []
        
        # Prepare initial prompt
        prompt = shuffle_options_in_prompt(original_prompt) if args.shuffle_options else original_prompt


        # Add user prompt
        user_msg = {"role": "user", "content": prompt}
        conversation_history.append(user_msg)
        
        # Get model response
        response = model.get_response(conversation_history, temperature=args.temperature)
        print("single-turn response:", response)
        assistant_msg = {"role": "assistant", "content": response}
        conversation_history.append(assistant_msg)
        
        extracted_answer = extract_answer_from_curly_brackets(response)
        is_valid = extracted_answer in options
        if is_valid:
            valid_single_answers += 1

        # Handle confidence score
        confidence_score_val = None
        if args.confidence_score:
            confidence_prompt = (
                "Provide the confidence score for your above answer as a real number between 0 and 1, "
                "where 0 indicates no confidence and 1 indicates full confidence. "
                "Please respond with your confidence score in double curly brackets: {{confidence score}}."
            )
            confidence_msg = {"role": "user", "content": confidence_prompt}
            conversation_history.append(confidence_msg)
            
            confidence_response = model.get_response(conversation_history, temperature=args.temperature)
            confidence_assistant_msg = {"role": "assistant", "content": confidence_response}
            conversation_history.append(confidence_assistant_msg)

            extracted_confidence = extract_answer_from_curly_brackets(confidence_response)
            try:
                confidence_score_val = float(extracted_confidence)
                if is_valid:
                    confidence_scores_per_option[extracted_answer].append(confidence_score_val)
            except ValueError:
                confidence_score_val = None

        # Store results
        query_results = {
            "conversation_history": conversation_history,
            "prompt": prompt,
            "response": response,
            "extracted_answer": extracted_answer,
            "confidence_score": confidence_score_val,
            "is_valid": is_valid
        }
        
        single_run_results.append(query_results)
        
    return single_run_results, valid_single_answers


def process_multi_turn(args, model, original_prompt, options, provider):
    """Process multi-turn conversation."""
    multi_turn_history = []
    valid_multi_answers = 0
    conversation = []

    for turn in range(args.n_turns):
        # Shuffle previous conversation if needed
        if turn > 0 and args.shuffle_options:
            pairs = []
            i = len(multi_turn_history) > 0 and multi_turn_history[0]['role'] == 'system'
            while i < len(multi_turn_history) - 1:
                if multi_turn_history[i]['role'] == 'user' and multi_turn_history[i+1]['role'] == 'assistant':
                    pairs.append([multi_turn_history[i], multi_turn_history[i+1]])
                    i += 2
                else:
                    break
            random.shuffle(pairs)
            
            new_history = []
            if len(multi_turn_history) > 0 and multi_turn_history[0]['role'] == 'system':
                new_history.append(multi_turn_history[0])
            for pair in pairs:
                new_history.extend(pair)
            multi_turn_history = new_history

        # Add new turn
        turn_prompt = shuffle_options_in_prompt(original_prompt) if args.shuffle_options else original_prompt
        user_msg = {"role": "user", "content": turn_prompt}
        multi_turn_history.append(user_msg)

        # Get model response
        response = model.get_response(multi_turn_history, temperature=args.temperature)
        print("multi-turn response:", response)
        assistant_msg = {"role": "assistant", "content": response}
        multi_turn_history.append(assistant_msg)

        extracted_answer = extract_answer_from_curly_brackets(response)
        is_valid = extracted_answer in options
        if is_valid:
            valid_multi_answers += 1

        # Store turn results
        turn_data = {
            "turn": turn + 1,
            "user_prompt": turn_prompt,
            "assistant_response": response,
            "extracted_answer": extracted_answer,
            "is_valid": is_valid,
            "conversation_history": multi_turn_history.copy()
        }
        conversation.append(turn_data)

    return conversation, valid_multi_answers


def main():
    args = parse_arguments()
    
    # Identify provider and get model
    if args.model_name.startswith('gpt') or args.model_name.startswith('o1'):
        provider = 'openai'
    elif args.model_name.startswith('command'):
        provider = 'cohere'
    elif args.model_name.startswith('Meta-Llama'):
        provider = 'sambanova'
    elif args.model_name.startswith('gemini'):
        provider = 'gemini'
    else:
        raise ValueError("Invalid model name. Please provide a valid model name.")
    api_key = load_api_key(provider, args.model_name)
    model = get_model(args.model_name, api_key)

    # Load prompt and extract options
    original_prompt = load_prompt(args.prompt_folder, args.task_name, args.category, args.shuffle_options)
    
    # For real_gaussian_random task, we don't need to extract options
    if args.task_name == "real_gaussian":
        options = ["real"]  # Placeholder, won't actually be used
    else:
        options = extract_options_from_prompt(original_prompt, args.task_name)
        if not options:
            raise ValueError("No options found in the prompt. Ensure they are in << >> or [ ].")

    # Setup output directory and metadata
    output_dir = generate_output_dir(args)
    save_metadata(args, output_dir)

    # Initialize results
    final_results = []
    confidence_scores_per_option = {option: [] for option in options}

    # Resume if requested
    results_file = os.path.join(output_dir, 'results.json')
    if args.resume and os.path.exists(results_file):
        with open(results_file, 'r') as f:
            final_results = json.load(f)
        start_run = len(final_results)
    else:
        start_run = 0

    # Main experiment loop
    for run_id in tqdm(range(start_run, args.n_runs), desc="Running Experiments"):
        run_results = {
            "run_id": run_id,
            "task_name": args.task_name,
            "category": args.category,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "single_turn": {},
            "multi_turn": {}
        }

        # Process single-turn queries
        single_run_results, valid_single_answers = process_single_turn(
            args, model, original_prompt, options, provider, confidence_scores_per_option)
        
        for query_id, query_results in enumerate(single_run_results):
            run_results["single_turn"][f"query_{query_id}"] = query_results

        # Process multi-turn conversation
        conversation, valid_multi_answers = process_multi_turn(
            args, model, original_prompt, options, provider)
        
        run_results["multi_turn"]["query_0"] = {"conversation": conversation}

        # For real_gaussian_random task, we handle metrics differently
        if args.task_name == "real_gaussian":
            # We just record valid response rates without probability or B-metrics
            metrics_table, valid_single_rate, valid_multi_rate = calculate_metrics(
                args=args,
                single_probs={},  # Empty dict since we don't need probabilities
                multi_probs={},   # Empty dict since we don't need probabilities
                confidence_scores_per_option={},
                valid_single_count=valid_single_answers,
                valid_multi_count=valid_multi_answers
            )
        else:
            # Calculate single-turn probabilities (original code)
            single_valid_answers = [res['extracted_answer'] 
                                for res in single_run_results if res['is_valid']]
            single_counts = {}
            for ans in single_valid_answers:
                single_counts[ans] = single_counts.get(ans, 0) + 1

            total_single_valid = len(single_valid_answers)
            single_probs = {
                option: (single_counts.get(option, 0) / total_single_valid if total_single_valid > 0 else 0)
                for option in options
            }

            # Calculate multi-turn probabilities (original code)
            multi_valid_answers = []
            for turn_data in conversation:
                if turn_data['is_valid']:
                    multi_valid_answers.append(turn_data['extracted_answer'])

            multi_counts = {}
            for ans in multi_valid_answers:
                multi_counts[ans] = multi_counts.get(ans, 0) + 1

            total_multi_valid = len(multi_valid_answers)
            multi_probs = {
                option: (multi_counts.get(option, 0) / total_multi_valid if total_multi_valid > 0 else 0)
                for option in options
            }

            # Generate metrics (original code)
            metrics_table, valid_single_rate, valid_multi_rate = calculate_metrics(
                args=args,
                single_probs=single_probs,
                multi_probs=multi_probs,
                confidence_scores_per_option=confidence_scores_per_option,
                valid_single_count=valid_single_answers,
                valid_multi_count=valid_multi_answers
            )

        # Save metrics for this run
        metrics_csv = os.path.join(output_dir, f"metrics_run_{run_id}.csv")
        metrics_table.to_csv(metrics_csv, index=False)

        # Save run results
        run_results_filepath = os.path.join(output_dir, f"run_{run_id}.json")
        save_run_results(run_id, run_results, run_results_filepath)

        # Add to final results
        final_results.append(run_results)

        # Save all results (for resume capability)
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=4)

        # Print metrics summary
        print("\n=== Metrics Table (Run", run_id, ") ===")
        print(metrics_table.fillna('').to_string(index=False))
        print("========================================\n")
        print(f"Run {run_id} completed. Metrics saved to '{metrics_csv}'.")

    print(f"All experiments completed. Final results are in '{output_dir}'. ")


if __name__ == '__main__':
    main()