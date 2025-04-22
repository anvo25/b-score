# benchmark_processing.py
from pathlib import Path
import json
import random
from utils import extract_answer_from_curly_brackets
import re

def process_single_turn(args, model, prompt, options, confidence_scores_per_option):
    """Process single-turn queries for benchmarking."""
    single_run_results = []
    valid_single_answers = 0

    for query_id in range(args.n_queries_single_turn):
        conversation_history = []
        
        # Special handling for BBQ which uses direct text answers
        is_bbq = hasattr(args, 'benchmark') and args.benchmark == 'bbq'
        
        if args.shuffle_options and not is_bbq:
            current_prompt = prompt
            option_texts = []
            
            for opt in options:
                start_idx = current_prompt.find(f"{opt}. ") + len(f"{opt}. ")
                end_idx = current_prompt.find("\n", start_idx)
                if end_idx == -1:
                    end_idx = current_prompt.find("\nPlease select", start_idx)
                option_texts.append(current_prompt[start_idx:end_idx].strip())
            
            shuffled_texts = option_texts.copy()
            random.shuffle(shuffled_texts)
            
            text_mapping = {orig: shuf for orig, shuf in zip(option_texts, shuffled_texts)}
            
            for opt, orig_text in zip(options, option_texts):
                shuffled_text = text_mapping[orig_text]
                current_prompt = current_prompt.replace(
                    f"{opt}. {orig_text}",
                    f"{opt}. {shuffled_text}"
                )
            
            option_mapping = {}
            for opt, orig_text in zip(options, option_texts):
                shuffled_text = text_mapping[orig_text]
                for other_opt, other_text in zip(options, option_texts):
                    if shuffled_text == other_text:
                        option_mapping[opt] = other_opt
                        break
        elif is_bbq and args.shuffle_options:
            # For BBQ, shuffle the options in the prompt directly
            current_prompt = prompt
            options_to_shuffle = options.copy()
            random.shuffle(options_to_shuffle)
            
            # Create a mapping between original and shuffled options
            option_mapping = {opt: opt for opt in options}  # For BBQ, we keep direct mapping
            
            # Update the prompt with shuffled options
            choices_text = ", ".join(options_to_shuffle)
            # Replace the options list in the prompt
            choices_pattern = r'\[([^\]]+)\]'
            current_prompt = re.sub(choices_pattern, f"[{choices_text}]", current_prompt)
        else:
            current_prompt = prompt
            option_mapping = {opt: opt for opt in options}
        
        response = model.get_response(
            [{"role": "user", "content": current_prompt}],
            temperature=args.temperature
        )
        
        # Skip this query if model response is None
        if response is None:
            continue
        
        extracted_answer = extract_answer_from_curly_brackets(response)
        
        # For BBQ benchmark, check if extracted answer is in options directly
        if is_bbq:
            is_valid = extracted_answer in options
            original_answer = extracted_answer  # For BBQ, these are the same
        else:
            is_valid = extracted_answer in options
            original_answer = option_mapping[extracted_answer] if is_valid else None
        
        answer_text = None
        if is_valid:
            valid_single_answers += 1
            
            if not is_bbq:
                answer_start = current_prompt.find(f"{extracted_answer}. ") + len(f"{extracted_answer}. ")
                answer_end = current_prompt.find("\n", answer_start)
                if answer_end == -1:
                    answer_end = current_prompt.find("\nPlease select", answer_start)
                answer_text = current_prompt[answer_start:answer_end].strip()
            else:
                answer_text = extracted_answer  # For BBQ, answer_text is the direct extracted text

        confidence_score_val = None
        if args.confidence_score:
            confidence_prompt = (
                "Provide the confidence score for your above answer as a real number between 0 and 1. "
                "Please respond with your confidence score in double curly brackets: {{confidence score}}."
            )
            confidence_response = model.get_response([
                {"role": "user", "content": current_prompt},
                {"role": "assistant", "content": response},
                {"role": "user", "content": confidence_prompt}
            ], temperature=args.temperature)
            
            # Skip confidence score if model response is None
            if confidence_response is not None:
                try:
                    confidence_score_val = float(extract_answer_from_curly_brackets(confidence_response))
                    if is_valid:
                        # For BBQ, use extracted_answer directly
                        confidence_option = original_answer if not is_bbq else extracted_answer
                        confidence_scores_per_option[confidence_option].append(confidence_score_val)
                except (ValueError, TypeError):
                    confidence_score_val = None

        query_results = {
            "query_id": query_id,
            "prompt": current_prompt,
            "response": response,
            "extracted_answer": extracted_answer,
            "original_answer": original_answer,
            "answer_text": answer_text,
            "option_mapping": option_mapping,
            "confidence_score": confidence_score_val,
            "is_valid": is_valid,
            "is_shuffled": args.shuffle_options
        }
        single_run_results.append(query_results)
        
    return single_run_results, valid_single_answers

def process_multi_turn(args, model, prompt, options):
    """Process multi-turn conversation for benchmarking."""
    conversation = []
    valid_multi_answers = 0
    conversation_history = []

    # Special handling for BBQ which uses direct text answers
    is_bbq = hasattr(args, 'benchmark') and args.benchmark == 'bbq'

    for turn in range(args.n_turns):
        if args.shuffle_options and not is_bbq:
            turn_prompt = prompt
            option_texts = []
            
            for opt in options:
                start_idx = turn_prompt.find(f"{opt}. ") + len(f"{opt}. ")
                end_idx = turn_prompt.find("\n", start_idx)
                if end_idx == -1:
                    end_idx = turn_prompt.find("\nPlease select", start_idx)
                option_texts.append(turn_prompt[start_idx:end_idx].strip())
            
            shuffled_texts = option_texts.copy()
            random.shuffle(shuffled_texts)
            
            text_mapping = {orig: shuf for orig, shuf in zip(option_texts, shuffled_texts)}
            
            for opt, orig_text in zip(options, option_texts):
                shuffled_text = text_mapping[orig_text]
                turn_prompt = turn_prompt.replace(
                    f"{opt}. {orig_text}",
                    f"{opt}. {shuffled_text}"
                )
            
            option_mapping = {}
            for opt, orig_text in zip(options, option_texts):
                shuffled_text = text_mapping[orig_text]
                for other_opt, other_text in zip(options, option_texts):
                    if shuffled_text == other_text:
                        option_mapping[opt] = other_opt
                        break
        elif is_bbq and args.shuffle_options:
            # For BBQ, shuffle the options in the prompt directly
            turn_prompt = prompt
            options_to_shuffle = options.copy()
            random.shuffle(options_to_shuffle)
            
            # Create a mapping between original and shuffled options
            option_mapping = {opt: opt for opt in options}  # For BBQ, we keep direct mapping
            
            # Update the prompt with shuffled options
            choices_text = ", ".join(options_to_shuffle)
            # Replace the options list in the prompt
            choices_pattern = r'\[([^\]]+)\]'
            turn_prompt = re.sub(choices_pattern, f"[{choices_text}]", turn_prompt)
        else:
            turn_prompt = prompt
            option_mapping = {opt: opt for opt in options}

        current_message = {"role": "user", "content": turn_prompt}
        conversation_history.append(current_message)
        
        response = model.get_response(conversation_history, temperature=args.temperature)
        
        # Skip this turn if model response is None
        if response is None:
            continue
        
        assistant_message = {"role": "assistant", "content": response}
        conversation_history.append(assistant_message)
        
        extracted_answer = extract_answer_from_curly_brackets(response)
        
        # For BBQ benchmark, check if extracted answer is in options directly
        if is_bbq:
            is_valid = extracted_answer in options
            original_answer = extracted_answer  # For BBQ, these are the same
        else:
            is_valid = extracted_answer in options
            original_answer = option_mapping[extracted_answer] if is_valid else None
        
        answer_text = None
        if is_valid:
            valid_multi_answers += 1
            
            if not is_bbq:
                answer_start = turn_prompt.find(f"{extracted_answer}. ") + len(f"{extracted_answer}. ")
                answer_end = turn_prompt.find("\n", answer_start)
                if answer_end == -1:
                    answer_end = turn_prompt.find("\nPlease select", answer_start)
                answer_text = turn_prompt[answer_start:answer_end].strip()
            else:
                answer_text = extracted_answer  # For BBQ, answer_text is the direct extracted text

        turn_data = {
            "turn": turn + 1,
            "prompt": turn_prompt,
            "response": response,
            "extracted_answer": extracted_answer,
            "original_answer": original_answer,
            "answer_text": answer_text,
            "option_mapping": option_mapping,
            "is_valid": is_valid,
            "is_shuffled": args.shuffle_options,
            "conversation_history": conversation_history.copy()
        }
        conversation.append(turn_data)

    return conversation, valid_multi_answers