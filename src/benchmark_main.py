# benchmark_main.py
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

from benchmark_utils import (
    load_api_key, download_benchmark_data, get_fixed_sample,
    format_arc_prompt, format_mmlu_prompt, calculate_metrics,
    calculate_overall_metrics, format_commonsense_prompt, format_hle_prompt, format_bbq_prompt
)
from benchmark_processing import process_single_turn, process_multi_turn

# Constants
SAMPLE_SIZE = 400
SEED = 42

def parse_benchmark_arguments():
    """Parse command line arguments for the benchmark script."""
    parser = argparse.ArgumentParser(description='Run benchmark experiments with LLMs.')
    parser.add_argument('--benchmark', type=str, required=True,
                      choices=['arc-challenge', 'mmlu', 'commonsense', 'hle', 'bbq'],
                      help='Benchmark to run')
    parser.add_argument('--model_name', type=str, required=True,
                      help='The name of the model')
    parser.add_argument('--n_runs', default=10, type=int,
                      help='Number of experiment runs per data point')
    parser.add_argument('--n_turns', default=30, type=int,
                      help='Number of turns in each multi-turn conversation')
    parser.add_argument('--n_queries_single_turn', default=30, type=int,
                      help='Number of single-turn queries per run')
    parser.add_argument('--output_folder', default='benchmark_logs', type=str,
                      help='Folder to save results')
    parser.add_argument('--temperature', default=1.0, type=float,
                      help='Temperature setting for the model')
    parser.add_argument('--resume', action="store_true",
                      help='Resume from existing results')
    parser.add_argument('--confidence_score', default=True, type=bool,
                      help='Include confidence score in the output')
    parser.add_argument('--shuffle_options', default=True, type=bool,
                      help='Shuffle the order of options')
    return parser.parse_args()

def setup_model(args):
    """Setup provider and model based on model name."""
    if args.model_name.startswith('gpt'):
        provider = 'openai'
    elif args.model_name.startswith('command'):
        provider = 'cohere'
    elif args.model_name.startswith('Meta-Llama'):
        provider = 'sambanova'
    elif args.model_name.startswith('gemini'):
        provider = 'gemini'
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    api_key = load_api_key(provider, args.model_name)
    from models import get_model
    return get_model(args.model_name, api_key)

def setup_output_directory(args):
    """Create and setup output directory structure."""
    output_dir = Path(args.output_folder) / args.benchmark / args.model_name / f"temp_{args.temperature}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = vars(args)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    return output_dir

def process_example(args, model, data_idx, row, output_dir, progress_file, completed_examples):
    """Updated process_example function with better CommonsenseQA and BBQ handling."""
    example_dir = output_dir / f"example_{data_idx}"
    example_dir.mkdir(exist_ok=True)

    # Format prompt and get correct answer with info based on benchmark type
    try:
        if args.benchmark == 'hle':
            prompt, options, answer_info = format_hle_prompt(row)
            if prompt is None:
                print(f"\nSkipping example {data_idx} due to formatting error")
                return
            # For HLE, always use correct_letter since we've already filtered for multiple choice
            correct_answer = answer_info['correct_letter']
                
        elif args.benchmark == 'arc-challenge':
            prompt, options, answer_info = format_arc_prompt(row)
            correct_answer = row['answerKey'].upper()
            
        elif args.benchmark == 'mmlu':
            prompt, options, answer_info = format_mmlu_prompt(row)
            correct_answer = answer_info['correct_letter']
            
        elif args.benchmark == 'commonsense':
            prompt, options, answer_info = format_commonsense_prompt(row)
            correct_answer = answer_info['correct_letter']
            
            # Additional validation for CommonsenseQA
            if not correct_answer:
                print(f"\nWarning: No correct answer found for example {data_idx}")
                print(f"Row data: {row}")
                return
                
            if correct_answer not in options:
                print(f"\nWarning: Correct answer {correct_answer} not in options {options}")
                print(f"Row data: {row}")
                return
        elif args.benchmark == 'bbq':
            prompt, options, answer_info = format_bbq_prompt(row)
            if prompt is None:
                print(f"\nSkipping example {data_idx} due to formatting error")
                return
            # For BBQ we use expected_letter instead of correct_answer
            expected_answer = answer_info['expected_letter']
            # For debugging purposes, assign it to correct_answer too
            correct_answer = expected_answer  
        
        else:
            raise ValueError(f"Unknown benchmark type: {args.benchmark}")
            
    except Exception as e:
        print(f"\nError formatting example {data_idx}: {str(e)}")
        print(f"Row data: {row}")
        return

    # Print debug information
    print(f"\nDebug - Processing example {data_idx}:")
    if args.benchmark == 'bbq':
        print(f"Expected answer: {correct_answer}")
    else:
        print(f"Correct answer: {correct_answer}")
    print(f"Answer info: {answer_info}")
    print(f"Available options: {options}")

    # Initialize results for this example
    example_results = []
    
    # Resume handling for individual runs
    results_file = example_dir / 'results.json'
    if args.resume and results_file.exists():
        with open(results_file) as f:
            example_results = json.load(f)
        start_run = len(example_results)
    else:
        start_run = 0

    # Run experiments for this example
    for run_id in range(start_run, args.n_runs):
        confidence_scores_per_option = {option: [] for option in options}
        
        run_results = {
            "run_id": run_id,
            "benchmark": args.benchmark,
            "data_id": data_idx,
            "model_name": args.model_name,
            "temperature": args.temperature,
            "original_answer_info": answer_info,
            "single_turn": {},
            "multi_turn": {}
        }

        # Process single-turn queries
        single_run_results, valid_single_answers = process_single_turn(
            args, model, prompt, options, confidence_scores_per_option)
        
        for query_id, query_results in enumerate(single_run_results):
            run_results["single_turn"][f"query_{query_id}"] = query_results

        # Process multi-turn conversation
        conversation, valid_multi_answers = process_multi_turn(
            args, model, prompt, options)
        
        run_results["multi_turn"]["query_0"] = {"conversation": conversation}

        # Calculate probabilities with error handling
        try:
            # Different handling for BBQ vs other benchmarks
            if args.benchmark == 'bbq':
                # For BBQ, use extracted_answer directly
                single_valid_answers = [res['extracted_answer'] 
                                      for res in single_run_results if res['is_valid']]
                
                multi_valid_answers = [turn['extracted_answer'] 
                                     for turn in conversation if turn['is_valid']]
            else:
                # For other benchmarks, use original_answer (mapped letter)
                single_valid_answers = [res['original_answer'] 
                                      for res in single_run_results if res['is_valid']]
                
                multi_valid_answers = [turn['original_answer'] 
                                     for turn in conversation if turn['is_valid']]
            
            # Count answer frequencies
            single_counts = {}
            for ans in single_valid_answers:
                single_counts[ans] = single_counts.get(ans, 0) + 1

            total_single_valid = len(single_valid_answers)
            single_probs = {
                option: (single_counts.get(option, 0) / total_single_valid 
                        if total_single_valid > 0 else 0)
                for option in options
            }

            multi_counts = {}
            for ans in multi_valid_answers:
                multi_counts[ans] = multi_counts.get(ans, 0) + 1

            total_multi_valid = len(multi_valid_answers)
            multi_probs = {
                option: (multi_counts.get(option, 0) / total_multi_valid 
                        if total_multi_valid > 0 else 0)
                for option in options
            }
        except Exception as e:
            print(f"Warning: Error calculating probabilities: {str(e)}")
            single_probs = {option: 0 for option in options}
            multi_probs = {option: 0 for option in options}

        # Calculate metrics - Note the proper indentation here
        if args.benchmark == 'bbq':
            metrics_table, valid_single_rate, valid_multi_rate = calculate_metrics(
                args=args,
                single_probs=single_probs,
                multi_probs=multi_probs,
                confidence_scores_per_option=confidence_scores_per_option,
                valid_single_count=valid_single_answers,
                valid_multi_count=valid_multi_answers,
                correct_answer=expected_answer,  # For BBQ this is not really "correct" but "expected"
                answer_info=answer_info
            )
        else:
            metrics_table, valid_single_rate, valid_multi_rate = calculate_metrics(
                args=args,
                single_probs=single_probs,
                multi_probs=multi_probs,
                confidence_scores_per_option=confidence_scores_per_option,
                valid_single_count=valid_single_answers,
                valid_multi_count=valid_multi_answers,
                correct_answer=correct_answer,
                answer_info=answer_info
            )

        # Save metrics for this run
        metrics_csv = example_dir / f"metrics_run_{run_id}.csv"
        metrics_table.to_csv(metrics_csv, index=False)

        # Add run results and save
        example_results.append(run_results)
        with open(results_file, 'w') as f:
            json.dump(example_results, f, indent=4)

        print(f"\nExample {data_idx}, Run {run_id} completed")
        print(metrics_table.to_string(index=False))

    # Mark example as completed
    completed_examples.add(data_idx)
    with open(progress_file, 'w') as f:
        json.dump(list(completed_examples), f)

def run_benchmark(args):
    """Run benchmark experiments."""
    model = setup_model(args)
    
    # Download/load benchmark data
    full_df = download_benchmark_data(args.benchmark)
    data = get_fixed_sample(args.benchmark, full_df, SAMPLE_SIZE, SEED)
    
    # Setup output directory
    output_dir = setup_output_directory(args)
    
    # Save sample info
    sample_info = {
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "data_indices": data.index.tolist()
    }
    with open(output_dir / 'sample_info.json', 'w') as f:
        json.dump(sample_info, f, indent=4)

    # Handle resume functionality
    progress_file = output_dir / 'progress.json'
    if args.resume and progress_file.exists():
        with open(progress_file) as f:
            completed_examples = set(json.load(f))
        print(f"\n=== Resuming previous run ===")
        print(f"Found {len(completed_examples)} completed examples")
        print(f"Will skip examples: {sorted(list(completed_examples))}")
        print(f"Continuing with remaining {len(data) - len(completed_examples)} examples")
        print("=" * 30)
    else:
        completed_examples = set()
        if args.resume:
            print("\nNo previous progress file found. Starting fresh run.")

    try:
        skipped = 0
        for data_idx, row in tqdm(data.iterrows(), total=len(data)):
            if data_idx in completed_examples:
                skipped += 1
                continue

            if skipped > 0 and data_idx == next(iter(set(range(len(data))) - completed_examples)):
                print(f"\nSkipped {skipped} completed examples. Starting with example {data_idx}")
                skipped = 0

            process_example(args, model, data_idx, row, output_dir, 
                          progress_file, completed_examples)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        # Save progress even if there's an error
        with open(progress_file, 'w') as f:
            json.dump(list(completed_examples), f)
        raise e

    print(f"\nAll benchmark experiments completed. Results are in '{output_dir}'")

    # Calculate and save overall metrics
    return calculate_overall_metrics(output_dir)

def main():
    """Main entry point for the benchmark script."""
    args = parse_benchmark_arguments()
    
    try:
        overall_metrics = run_benchmark(args)
        print("\nOverall Metrics:")
        print(json.dumps(overall_metrics, indent=2))
    except Exception as e:
        print(f"Error occurred during benchmark execution: {str(e)}")
        raise e

if __name__ == '__main__':
    main()
