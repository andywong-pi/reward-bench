import argparse
import json
import re
import pandas as pd
from pathlib import Path
from datasets import Dataset, concatenate_datasets, load_dataset
from vllm import LLM, SamplingParams
from typing import List, Dict, Union, Optional, Tuple
import random
from tqdm import tqdm
import numpy as np
# Set a fixed random seed for reproducibility
random.seed(42)  # You can use any number (42 is a common choice)
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from itertools import product
from rewardbench import process_single_model
from datetime import datetime

INF2_SETS = [
    "/mnt/vast/home/sanjana/dpo_data/dpojpi_chatml_no_names_llama33i_resample.jsonl",
    "/mnt/vast/home/jimmy/data/inf2/rl/validation.parquet",
    "/mnt/vast/home/andy/data/inf1_eclairselfharm+core+support+justpi/annotations_pm_test.jsonl"
]
REWARDBENCH_v1_SET = "allenai/reward-bench"
JUDGEBENCH_SET = "ScalerLab/JudgeBench"
RM_BENCH_SET = "THU-KEG/RM-Bench"
REWARDBENCH_v2_SET = "allenai/reward-bench-2"

HELPSTEER3_SYSTEM_PROMPT = (
    "You are a skilled little expert at scoring responses. "
    "You should evaluate given responses based on the given judging criteria.\n"
    "Given the context of the conversation (the last turn is the User's query) and two responses from the Assistant, "
    "you need to refer to the [General Scoring Guidelines] to score each individual response. "
    "Based on the general scoring guidelines, state potential other specific criteria to the query, "
    "the weights of different criteria, and then you need to also give a ranking score based on the [Ranking Scoring Guidelines].\n"
    "Before scoring, please analyze step by step. "
    "Your scoring needs to be as strict as possible.\n"
    "[General Scoring Guideline]\n"
    "When evaluating individual responses, consider the following criteria and other criteria that are specific to the query and the context:\n"
    "- Correctness/Completeness: Is the response accurate and complete?\n"
    "- Coherence/Clarity: Is the response clear, coherent, and easy to understand?\n"
    "- Instruction following: Does the response follow the instructions and fulfill the user's request?\n"
    "- Relevance: Is the response relevant to the user's query/input, with information closely aligned with the topic.?\n"
    "- Level of Detail and Creativity: Does the response provide enough detail without being too verbose? "
    "Does it show creativity but not hallucinations?\n"
    "**Score 5: Excellent Response**\n"
    "- The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.\n"
    "- It accurately acts on the user's request, without unnecessary information.\n"
    "- If a user request is not possible/in line with desired model behavior, a helpful response provides useful context and rationale.\n"
    "**Score 4: Good Response**\n"
    "- The response is mostly helpful and mainly aligned with what the user was looking for.\n"
    "- There is still some room for improvement, but the response is generally useful.\n"
    "**Score 3: Fair Response**\n"
    "- The response is partially helpful but misses the overall goal of the user's query/input in some way.\n"
    "- The response did not fully satisfy what the user was looking for.\n"
    "**Score 2: Poor Response**\n"
    "- The response is borderline unhelpful and mostly does not capture what the user was looking for.\n"
    "- However, it is still usable and helpful in a small way.\n"
    "**Score 1: Bad Response**\n"
    "- The response is not useful or helpful at all.\n"
    "- The response completely missed the essence of what the user wanted.\n"
    "[Ranking Scoring Guidelines]\n"
    "Ranking score is used to rank the two responses based on their quality. "
    "Even if you give the same individual score for both responses, you need to differentiate them strictly. "
    "The ranking score is a number between 1 and 6, where:\n"
    "1 = Response 1 is much better than Response 2\n"
    "2 = Response 1 is better than Response 2\n"
    "3 = Response 1 is slightly better than Response 2\n"
    "4 = Response 2 is slightly better than Response 1\n"
    "5 = Response 2 is better than Response 1\n"
    "6 = Response 2 is much better than Response 1\n"
)
HELPSTEER3_USER_PROMPT = (
    "#### Conversation Context ####\n"
    "{Query}\n"
    "#### Responses to be Scored ####\n"
    "Response 1:\n{Response_1}\nResponse 2:\n{Response_2}\n\n"
    "#### Output Format Requirements ####\n"
    "First state other potential criteria specific to the query and the context, and the weights of each criteria in the format of:\n"
    "[The Begin of other criteria and weights]\n"
    "State the criteria and the weights of each criteria\n"
    "[The End of other criteria and weights]\n"
    "Second give your analysis on each responses in the format of:\n"
    "[The Begin of Analysis on Response i]\n"
    "Analysis on the i-th response\n"
    "[The End of Analysis on Response i]\n"
    "Then give the scores of each response in order, separate by comma in the boxed, adhering this format:\n"
    "[The Begin of Individual Scores]\n"
    "\\boxed{{x, y}} if there exists 2 responses\n"
    "[The End of Individual Scores]\n"
    "Finally, give the relative ranking score in the format of:\n"
    "[The Begin of Ranking Score]\n"
    "\\boxed{{z}}\n"
    "[The End of Ranking Score]\n"
)

GENERIC_CONVERSATIONAL_INTELLIGENCE_SYSTEM_PROMPT = (
    "You are a skilled little expert at scoring responses. "
    "You should evaluate given responses based on the given judging criteria.\n"
    "Given the context of the conversation (the last turn is the User's query) and two responses from the Assistant, "
    "you need to refer to the [General Scoring Guidelines] to score each individual response. "
    "Based on the general scoring guidelines, state potential other specific criteria to the query, "
    "the weights of different criteria, and then you need to also give a ranking score based on the [Ranking Scoring Guidelines].\n"
    "Before scoring, please analyze step by step. "
    "Your scoring needs to be as strict as possible.\n"
    "[General Scoring Guideline]\n"
    "When evaluating individual responses, consider the following criteria and other criteria that are specific to the query and the context:\n"
    "- Correctness/Completeness: Is the response accurate and complete?\n"
    "- Coherence/Clarity: Is the response clear, coherent, and easy to understand?\n"
    "- Instruction following: Does the response follow the instructions and fulfill the user's request?\n"
    "- Relevance: Is the response relevant to the user's query/input, with information closely aligned with the topic.?\n"
    "- Level of Detail and Creativity: Does the response provide enough detail without being too verbose? "
    "Does it show creativity but not hallucinations?\n"
    "**Score 5: Excellent Response**\n"
    "- The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.\n"
    "- It accurately acts on the user's request, without unnecessary information.\n"
    "- If a user request is not possible/in line with desired model behavior, a helpful response provides useful context and rationale.\n"
    "**Score 4: Good Response**\n"
    "- The response is mostly helpful and mainly aligned with what the user was looking for.\n"
    "- There is still some room for improvement, but the response is generally useful.\n"
    "**Score 3: Fair Response**\n"
    "- The response is partially helpful but misses the overall goal of the user's query/input in some way.\n"
    "- The response did not fully satisfy what the user was looking for.\n"
    "**Score 2: Poor Response**\n"
    "- The response is borderline unhelpful and mostly does not capture what the user was looking for.\n"
    "- However, it is still usable and helpful in a small way.\n"
    "**Score 1: Bad Response**\n"
    "- The response is not useful or helpful at all.\n"
    "- The response completely missed the essence of what the user wanted.\n"
    "[Ranking Scoring Guidelines]\n"
    "Ranking score is used to rank the two responses based on their quality. "
    "Even if you give the same individual score for both responses, you need to differentiate them strictly. "
    "The ranking score is a number between 1 and 6, where:\n"
    "1 = Response 1 is much better than Response 2\n"
    "2 = Response 1 is better than Response 2\n"
    "3 = Response 1 is slightly better than Response 2\n"
    "4 = Response 2 is slightly better than Response 1\n"
    "5 = Response 2 is better than Response 1\n"
    "6 = Response 2 is much better than Response 1\n"
)
GENERIC_CONVERSATIONAL_INTELLIGENCE_USER_PROMPT = (
    "#### Conversation Context ####\n"
    "{Query}\n"
    "#### Responses to be Scored ####\n"
    "Response 1:\n{Response_1}\nResponse 2:\n{Response_2}\n\n"
    "#### Output Format Requirements ####\n"
    "First state other potential criteria specific to the query and the context, and the weights of each criteria in the format of:\n"
    "[The Begin of other criteria and weights]\n"
    "State the criteria and the weights of each criteria\n"
    "[The End of other criteria and weights]\n"
    "Second give your analysis on each responses in the format of:\n"
    "[The Begin of Analysis on Response i]\n"
    "Analysis on the i-th response\n"
    "[The End of Analysis on Response i]\n"
    "Then give the scores of each response in order, separate by comma in the boxed, adhering this format:\n"
    "[The Begin of Individual Scores]\n"
    "\\boxed{{x, y}} if there exists 2 responses\n"
    "[The End of Individual Scores]\n"
    "Finally, give the relative ranking score in the format of:\n"
    "[The Begin of Ranking Score]\n"
    "\\boxed{{z}}\n"
    "[The End of Ranking Score]\n"
)

TIES_SYSTEM_PROMPT = ""

TIES_USER_PROMPT = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.

Notes:
1- Your evaluation should consider factors such as the helpfulness, relevance, and accuracy of the response, but need not consider depth or level of detail of the response.
2- Begin your evaluation by providing a short explanation.
3- Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10. For your rating, only give a number between 1 and 10 (inclusive), do not use any markdown, and do not put any text after your final rating.

[Query]
{prompt}

[Response]
{completion}

[Your judgement]"""


def find_first_difference(str1, str2):
    """
    Find the index of the first character that differs between two strings.
    Returns the index where the strings start to differ.
    """
    assert len(str1) == len(str2)
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            return i
    return len(str1)

def generate_prompt_response(dataset, set_name, swap=False):
    """Process dataset to split into prompts and responses, optionally doubling size with swapped versions."""
    valid_sets = {"inf2_sets", "rewardbench_v1_set", "judgebench_gpt_set", "judgebench_claude_set", "rm_bench_set"}
    if set_name not in valid_sets:
        raise ValueError(f"Invalid set_name: {set_name}. Must be one of {valid_sets}")

    def create_entry(row, prompt, chosen_resp, rejected_resp, response1, response2, is_shuffled):
        """Helper function to create a standardized entry while preserving original fields."""
        new_entry = row.copy()
        new_entry.update({
            "prompt": prompt,
            "text_chosen": chosen_resp,
            "text_rejected": rejected_resp,
            "response1": response1,
            "response2": response2,
            "is_shuffled": is_shuffled
        })
        return new_entry

    def process_row(row):
        chosen, rejected = row["text_chosen"], row["text_rejected"]
        split_idx = find_first_difference(chosen, rejected)
        prompt, chosen_resp, rejected_resp = chosen[:split_idx], chosen[split_idx:], rejected[split_idx:]
        
        entry = create_entry(row, prompt, chosen_resp, rejected_resp, 
                           chosen_resp, rejected_resp, False)
        swapped_entry = create_entry(row, prompt, chosen_resp, rejected_resp, 
                                   rejected_resp, chosen_resp, True)
        if swap:
            return [entry, swapped_entry]
        return [entry] if random.random() < 0.5 else [swapped_entry]

    # Single iteration that processes rows and collects keys
    all_keys = set()
    processed_rows = []
    
    for row in dataset:
        processed = process_row(row)  # Always returns a list now
        processed_rows.extend(processed)
        # Update keys with the first processed item's keys
        if processed and not all_keys:  # Only need to do this once
            all_keys.update(processed[0].keys())
    
    return Dataset.from_dict({k: [row[k] for row in processed_rows] for k in all_keys})

def generate_prompt_response_rewardbench_v2(dataset, set_name, swap=False):
    """Process dataset to split into prompts and responses, optionally doubling size with swapped versions."""
    valid_sets = {"rewardbench_v2_set"}
    if set_name not in valid_sets:
        raise ValueError(f"Invalid set_name: {set_name}. Must be one of {valid_sets}")

    def process_row(row):
        chosen, rejected = row["text_chosen"], row["text_rejected"]
        split_idx = find_first_difference(chosen[0], rejected[0])
        prompt = chosen[0][:split_idx]
        
        # Create list of all responses
        responses = [
            chosen[0][split_idx:],    # chosen response
            rejected[0][split_idx:],  # rejected0
            rejected[1][split_idx:],  # rejected1
            rejected[2][split_idx:]   # rejected2
        ]
        
        # Remember which one is chosen (index 0)
        chosen_index = 0
        
        # Get shuffled indices
        indices = list(range(len(responses)))
        random.shuffle(indices)
        
        # Find where the chosen response moved to
        new_chosen_index = indices.index(chosen_index)
        
        # Shuffle the responses according to the shuffled indices
        shuffled_responses = [responses[i] for i in indices]

        row["prompt"] = prompt
        row["response1"] = shuffled_responses[0]
        row["response2"] = shuffled_responses[1]
        row["response3"] = shuffled_responses[2]
        row["response4"] = shuffled_responses[3]
        row["is_shuffled"] = str(new_chosen_index+1) # 1-4
        return row

    dataset = dataset.map(process_row)
    return dataset

def filter_long_turns(batch, max_turns):
    return len(batch["text_chosen"]) // 2 <= max_turns

def apply_prompt_templates(example, args) -> dict:
    """
    Generate system_prompt and user_prompt for given prompt type
    Returns a dictionary with these new columns
    """
    # Determine the base prompts based on prompt_type
    if args.prompt_type == "helpsteer3":
        system_prompt = HELPSTEER3_SYSTEM_PROMPT
        user_prompt_template = HELPSTEER3_USER_PROMPT
    elif args.prompt_type == "generic_conversational_intellegence":
        system_prompt = GENERIC_CONVERSATIONAL_INTELLIGENCE_SYSTEM_PROMPT
        user_prompt_template = GENERIC_CONVERSATIONAL_INTELLIGENCE_USER_PROMPT
    else:
        raise ValueError(f"Unknown prompt type: {args.prompt_type}")

    # Handle non-rewardbench_v2_set case
    if args.dataset != "rewardbench_v2_set":
        user_prompt = user_prompt_template.format(
            Query=example['prompt'],
            Response_1=example['response1'],
            Response_2=example['response2'],
        )
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "messages": messages,
        }
    else:
        # Handle rewardbench_v2_set case
        response_pairs = [
            ('12', 'response1', 'response2'),
            ('34', 'response3', 'response4'),
            ('13', 'response1', 'response3'),
            ('14', 'response1', 'response4'),
            ('23', 'response2', 'response3'),
            ('24', 'response2', 'response4')
        ]

        result = {"system_prompt": system_prompt}
        
        for suffix, resp1, resp2 in response_pairs:
            user_prompt = user_prompt_template.format(
                Query=example['prompt'],
                Response_1=example[resp1],
                Response_2=example[resp2],
            )
            result[f"user_prompt_{suffix}"] = user_prompt
            result[f"messages_{suffix}"] = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]

        return result

def apply_prompt_templates_rewardbench_v2_ties(example, args) -> dict:
    """
    Generate system_prompt and user_prompt for given prompt type
    Returns a dictionary with these new columns
    """
    num_answers = len(example['answers'])
    example['messages'] = []

    for i in range(num_answers):
        system_prompt = TIES_SYSTEM_PROMPT
        user_prompt = TIES_USER_PROMPT.format(
            prompt=example['prompt'],
            completion=example['answers'][i]
        )

        message = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
            ]
        example['messages'].append(message)
    return example

def load_inf2_dataset(data_paths):
    datasets = []
    
    for fname in data_paths:
        subset = fname.split("/")[-1].split(".")[0]
        if fname.endswith('.jsonl'):
            ds = load_dataset("json", data_files=fname, split="train")
        elif fname.endswith('.parquet'):
            ds = load_dataset("parquet", data_files=fname, split="train")
        else:
            continue  # skip unknown file types
        ds = ds.add_column("subset", [subset] * len(ds))
        datasets.append(ds)
    raw_dataset = concatenate_datasets(datasets)
    raw_dataset = raw_dataset.add_column("text_chosen", raw_dataset["chosen"])
    raw_dataset = raw_dataset.add_column("text_rejected", raw_dataset["rejected"])
    return raw_dataset

def load_rewardbench_v1_dataset(data_paths):
    raw_dataset = load_dataset(data_paths, split="filtered")

    def format_conversation(example):
        """Convert prompt/chosen/rejected into structured conversations."""
        return {
            "text_chosen": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]}
            ],
            "text_rejected": [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]}
            ]
        }

    # Apply the transformation
    raw_dataset = raw_dataset.map(format_conversation)    
    return raw_dataset

def load_judgebench_dataset(args):
    if "gpt" in args.dataset:
        raw_dataset = load_dataset(JUDGEBENCH_SET, split="gpt")
    elif "claude" in args.dataset:
        raw_dataset = load_dataset(JUDGEBENCH_SET, split="claude")
    else:
        raise ValueError(f"the dataset {args.dataset} cannot be found.")

    def format_conversation(example):
        if example['label'] == "A>B":
            chosen_response = example['response_A']
            rejected_response = example['response_B']
        elif example['label'] == "B>A":
            chosen_response = example['response_B']
            rejected_response = example['response_A']
        else:
            raise ValueError(f"Invalid label {example['label']}")
        """Convert prompt/chosen/rejected into structured conversations."""
        return {
            "text_chosen": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": chosen_response}
            ],
            "text_rejected": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": rejected_response}
            ],
            "subset": example['source']
        }

    # Apply the transformation
    raw_dataset = raw_dataset.map(format_conversation)    
    return raw_dataset

def load_rm_bench_dataset(args):
    raw_dataset = load_dataset(RM_BENCH_SET, split="train")

    # Assuming raw_dataset is already loaded
    expanded_data = []

    for example in raw_dataset:
        chosen_items = example['chosen']  # List of 3 chosen responses
        rejected_items = example['rejected']  # List of 3 rejected responses
        
        # Generate all 3x3=9 combinations
        for chosen, rejected in product(chosen_items, rejected_items):
            # chosen_items = ['A', 'B', 'C']
            # rejected_items = ['X', 'Y', 'Z']
            # A X / A Y / A Z
            # B X / B Y / B Z
            # C X / C Y / C Z
            text_chosen = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": chosen}
            ]
            text_rejected = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": rejected}
            ]
            expanded_data.append({
                'id': example['id'],
                'prompt': example['prompt'],
                'chosen': chosen,  # Note: Your original has 'chosen' (corrected spelling)
                'rejected': rejected,
                'text_chosen': text_chosen,
                'text_rejected': text_rejected,
                'subset': example['domain']
            })
    # Convert the list of dictionaries to a Hugging Face Dataset
    expanded_dataset = Dataset.from_list(expanded_data)
    return expanded_dataset

def load_rewardbench_v2_dataset(data_paths):
    raw_dataset = load_dataset(data_paths, split="test")

    def format_conversation(example):
        """Convert prompt/chosen/rejected into structured conversations."""
        return {
            "text_chosen": [
                [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["chosen"][0]}
                ]
            ],
            "text_rejected": [
                [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"][0]}
                ],
                [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"][1]}
                ],
                [   
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"][2]}
                ]
            ]
        }

    def format_ratings(batch, is_ties=True):
        """Format batch for ratings-based evaluation"""
        num_chosen = len(batch["chosen"])
        num_rejected = len(batch["rejected"])

        batch["text_chosen"] = []
        for i in range(num_chosen):
            batch["text_chosen"].append(
                [
                    {"role": "user", "content": batch["prompt"]},
                    {"role": "assistant", "content": batch["chosen"][i]}
                ]
            )
        batch["text_rejected"] = []
        for i in range(num_rejected):
            batch["text_rejected"].append(
                [
                    {"role": "user", "content": batch["prompt"]},
                    {"role": "assistant", "content": batch["rejected"][i]}
                ]
            )

        prompt = batch["text_chosen"][0][0]  # Get the user question

        # Combine chosen and rejected answers
        # texts_chosen is [[messages]], texts_rejected is [messages, messages, messages]
        all_answers = batch["text_chosen"] + batch["text_rejected"]  # Remove the extra [0] indexing

        # Format each answer for rating
        formatted_answers = []
        for answer in all_answers:
            answer_text = answer[1]  # Get the assistant's response
            formatted_answers.append(answer_text)

        batch["prompt"] = prompt
        batch["answers"] = formatted_answers
        return batch
    
    # Filter the dataset to exclude examples where 'subset' is 'Ties'
    main_dataset = raw_dataset.filter(lambda example: example['subset'] != 'Ties')
    ties_dataset = raw_dataset.filter(lambda example: example['subset'] == 'Ties')
    # Apply the transformation
    main_dataset = main_dataset.map(format_conversation)  
    ties_dataset = ties_dataset.map(format_ratings)

    return main_dataset, ties_dataset

def load_datasets(args) -> Tuple[Dataset, List[str]]:
    """Load datasets with subset tracking"""
    # Extract dataset path from set_name
    if args.dataset == "inf2_sets":
        raw_dataset = load_inf2_dataset(INF2_SETS)
    elif args.dataset == "rewardbench_v1_set":
        raw_dataset = load_rewardbench_v1_dataset(REWARDBENCH_v1_SET)
    elif args.dataset == "judgebench_gpt_set" or args.dataset == "judgebench_claude_set":
        raw_dataset = load_judgebench_dataset(args)
    elif args.dataset == "rm_bench_set":
        raw_dataset = load_rm_bench_dataset(args)
    elif args.dataset == "rewardbench_v2_set":
        raw_dataset, ties_dataset = load_rewardbench_v2_dataset(REWARDBENCH_v2_SET)
    else:
        raise ValueError(f"set_name {args.dataset} cannot be found")

    if args.dataset == "rewardbench_v2_set":
        # prompt is different and ties_dataset
        formatted_dataset = generate_prompt_response_rewardbench_v2(raw_dataset, set_name=args.dataset, swap=False)
    else:
        # swap is only utilized for judgebench
        formatted_dataset = generate_prompt_response(raw_dataset, set_name=args.dataset, swap=args.swap)
    filtered_dataset = formatted_dataset.filter(lambda x: filter_long_turns(x, args.max_turns))
    # Apply to your dataset
    dataset = filtered_dataset.map(lambda x: apply_prompt_templates(x, args), batched=False)
    subsets = dataset["subset"]
    if args.dataset == "rewardbench_v2_set":
        ties_dataset = ties_dataset.map(lambda x: apply_prompt_templates_rewardbench_v2_ties(x, args), batched=False)
        return dataset, subsets, ties_dataset
    return dataset, subsets

import torch
from vllm import LLM, SamplingParams
from typing import List
from datasets import Dataset

class vLLMInferenceEngine:
    def __init__(self, args):
        """Initialize vLLM with explicit GPU configuration"""
        self.args = args
        
        # Verify GPU availability and set device
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")
        
        # Get number of available GPUs if not specified
        if not hasattr(args, 'num_gpus') or args.num_gpus is None:
            args.num_gpus = torch.cuda.device_count()
            print(f"Automatically detected {args.num_gpus} GPUs")
        
        # Initialize LLM with explicit GPU configuration
        self.llm = LLM(
            model=args.model,
            tensor_parallel_size=args.num_gpus,
            tokenizer=args.tokenizer if hasattr(args, 'tokenizer') else None,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            dtype="auto"
        )
        
        self.sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens
        )
        
        self.tokenizer = self.llm.get_tokenizer() if hasattr(self.llm, 'get_tokenizer') else None
        self._print_gpu_info()
    
    def _print_gpu_info(self):
        """Print GPU information for debugging"""
        print(f"\nGPU Configuration:")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Using {self.args.num_gpus} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print()
    
    def format_chat_prompt(self, messages: List[dict]) -> Optional[str]:
        """Apply chat template to format messages into a prompt string.
        Returns None if the resulting prompt exceeds max_prompt_length."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available for applying chat template")
        
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            raise ValueError("Tokenizer does not support chat templates")
            
        # Format the prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Check length if max_prompt_length is specified
        if hasattr(self.args, 'max_prompt_length') and self.args.max_prompt_length is not None:
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            if len(tokenized.input_ids[0]) > self.args.max_prompt_length:
                return " "
        
        return prompt
            
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses with batch processing"""
        # Process in batches with a single progress bar
        batch_size = self.args.batch_size
        all_outputs = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
                
            if self.args.debug:
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")
                
            batch_outputs = self.llm.generate(batch_prompts, self.sampling_params)
            all_outputs.extend([output.outputs[0].text for output in batch_outputs])
        
        return all_outputs

    def batch_predict(self, dataset: Dataset, args) -> Dataset:
        """Run batch prediction with either chat template or traditional prompt system"""
        if args.dataset != "rewardbench_v2_set":
            prompts = []
            # Optional: Add a progress bar for prompt preparation if dataset is large
            for example in tqdm(dataset, desc="Preparing prompts"):
                prompt = self.format_chat_prompt(example['messages']) if args.use_chat_template else example['messages']
                prompts.append(prompt)
            responses = self.generate(prompts)

            import pdb;pdb.set_trace()
            
            # responses = self.llm.generate(prompts, sampling_params=self.sampling_params)
            return responses, [None] * len(responses)
        else:
            prompts_12, prompts_34 = [], []
            # Optional: Add a progress bar for prompt preparation if dataset is large
            for example in tqdm(dataset, desc="Preparing prompts"):
                prompt_12 = self.format_chat_prompt(example['messages_12']) if args.use_chat_template else example['messages_12']
                prompts_12.append(prompt_12)
                prompt_34 = self.format_chat_prompt(example['messages_34']) if args.use_chat_template else example['messages_34']
                prompts_34.append(prompt_34)
            responses_12 = self.generate(prompts_12)
            responses_34 = self.generate(prompts_34)
            dataset = dataset.add_column('evaluation', responses_12)
            answers_12 = [output_parser(example, args) for example in dataset]
            dataset = dataset.remove_columns('evaluation')
            dataset = dataset.add_column('evaluation', responses_34)
            answers_34 = [output_parser(example, args) for example in dataset]
            dataset = dataset.remove_columns('evaluation')

            prompts, candidates = [], []
            for example, ans_12, ans_34 in tqdm(zip(dataset, answers_12, answers_34), 
                                     total=len(dataset), 
                                     desc="Preparing prompts"):
                if ans_12 == "A" and ans_34 == "A":
                    message = example['messages_13']
                    candidate = ['1', '3']
                elif ans_12 == "A" and ans_34 == "B":
                    message = example['messages_14']
                    candidate = ['1', '4']
                elif ans_12 == "B" and ans_34 == "A":
                    message = example['messages_23']
                    candidate = ['2', '3']
                elif ans_12 == "B" and ans_34 == "B":
                    message = example['messages_24']
                    candidate = ['2', '4']
                else:
                    # random
                    message = "error"
                    candidate = ['1', '2']

                prompt = self.format_chat_prompt(message) if args.use_chat_template else message
                prompts.append(prompt)
                candidates.append(candidate)
            responses = self.generate(prompts)

            return responses, candidates
    
    def batch_predict_ties(self, dataset: Dataset, args) -> Dataset:
        """Run batch prediction with either chat template or traditional prompt system"""
        all_responses = []
        # Optional: Add a progress bar for prompt preparation if dataset is large
        for example in tqdm(dataset, desc="Preparing prompts"):
            prompts = []
            for message in example['messages']:
                prompt = self.format_chat_prompt(message) if args.use_chat_template else message
                prompts.append(prompt)
            responses = self.generate(prompts)
            all_responses.append(responses)

        return all_responses

import re
from typing import Optional

def output_parser(example, args):
    judgment = example['evaluation']
    subset = example['subset']
    prompt_type = args.prompt_type
    
    if subset != 'Ties' and prompt_type in ["helpsteer3", "generic_conversational_intellegence"]:
        # Helper functions for boxed string processing
        def find_boxed_string(s: str, pattern: str, first: bool = True) -> Optional[str]:
            matches = list(re.finditer(pattern, s))
            return matches[0].group(0) if matches else None

        def last_boxed_only_string(s: str) -> Optional[str]:
            return find_boxed_string(s, r"\\boxed\{\{?[^,{}]+\}?\}", first=False)

        def remove_boxed(s: str) -> Optional[str]:
            if not s or not s.startswith("\\boxed{") or not s.endswith("}"):
                return None
            inner = s[7:-1]  # Remove \boxed{}
            return inner[1:-1] if inner.startswith("{") and inner.endswith("}") else inner

        # Extract ranking score section
        pattern = re.compile(r"\\?\[The Begin of Ranking Score\\?\](.*?)\\?\[The End of Ranking Score\\?\]", re.DOTALL)
        match = re.search(pattern, judgment)
        if not match:
            return "error"
            
        ranking_section = match.group(1)
        boxed_pred = last_boxed_only_string(ranking_section) if ranking_section else None
        extracted_score = remove_boxed(boxed_pred) if boxed_pred else None
        
        if not extracted_score:
            return "error"
            
        try:
            score = int(extracted_score)
            if args.dataset == "rewardbench_v2_set" and 'candidates' in example:
                return example['candidates'][0] if score <= 3 else example['candidates'][1]
            else:
                return "A" if score <= 3 else "B"
        except ValueError:
            return "error"

    elif subset == 'Ties' and prompt_type in ["helpsteer3", "generic_conversational_intellegence"]:
        # Process TIES dataset
        scores = []
        for j in judgment:
            match = re.search(r"\b([1-9]|10)\b\s*$", j.strip())
            if match:
                try:
                    rating = int(match.group(1))
                    scores.append(rating if 1 <= rating <= 10 else "error")
                except ValueError:
                    scores.append("error")
        return scores

    raise ValueError("The model parser is not defined")

# Iterate through the dataset and apply the logic
def process_example(example, args):
    answer = example['answers']  # replace with your answer column name
    is_shuffled = example['is_shuffled']  # replace with your is_shuffled column name
    
    if args.prompt_type in ['helpsteer3', 'generic_conversational_intellegence'] and args.dataset != 'rewardbench_v2_set':
        if (answer == 'A' and not is_shuffled) or (answer == 'B' and is_shuffled):
            return {'score': 1}
        elif (answer == 'A' and is_shuffled) or (answer == 'B' and not is_shuffled):
            return {'score': 0}
        else:
            # return {'score': 0.5} remove this impact
            return {'score': 0}
    elif args.prompt_type in ['helpsteer3', 'generic_conversational_intellegence'] and args.dataset == 'rewardbench_v2_set':
        if answer == is_shuffled:
            return {'score': 1}
        else:
            return {'score': 0}
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

def calculate_judgebench_accuracy_swap(dataset, subsets):
    print("###\nThe start of swap analysis\n###\n")
    # print per subset and log into results_grouped file
    sources = ["mmlu-pro", "livebench-reasoning", "livebench-math", "livecodebench", ""]
    results_grouped = {}
    for subset in sources:
        subset_dataset = dataset.filter(lambda example: example["subset"].startswith(subset))
        scores = subset_dataset["score"]
        num_total = len(scores)

        # Swap case - analyze pairs
        both_correct = 0
        both_wrong = 0
        one_correct_one_wrong = 0
            
        for i in range(0, num_total, 2):
            if i+1 >= num_total:
                break  # skip last item if odd number
                
            score1 = scores[i]
            score2 = scores[i+1]
                
            if score1 and score2:
                both_correct += 1
            elif not score1 and not score2:
                both_wrong += 1
            else:
                one_correct_one_wrong += 1
            
        total_pairs = both_correct + both_wrong + one_correct_one_wrong
        results_grouped[subset] = {
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'one_correct_one_wrong': one_correct_one_wrong,
            'total_pairs': total_pairs,
            'both_correct_ratio': both_correct / total_pairs if total_pairs > 0 else 0,
            'both_wrong_ratio': both_wrong / total_pairs if total_pairs > 0 else 0,
            'mixed_ratio': one_correct_one_wrong / total_pairs if total_pairs > 0 else 0
        }
            
        print(f"\n{subset} (swap analysis):")
        print(f"Total pairs: {total_pairs}")
        print(f"Both correct: {both_correct}/{total_pairs} ({both_correct/total_pairs if total_pairs > 0 else 0})")
        print(f"Both wrong: {both_wrong}/{total_pairs} ({both_wrong/total_pairs if total_pairs > 0 else 0})")
        print(f"One correct one wrong: {one_correct_one_wrong}/{total_pairs} ({one_correct_one_wrong/total_pairs if total_pairs > 0 else 0})")
    return results_grouped

def calculate_rm_bench_accuracy(dataset, subsets):
    hard_indices = [1, 2, 5]
    normal_indices = [0, 4, 8]
    easy_indices = [3, 6, 7]

    print("###\nThe start of hard/normal/easy analysis\n###\n")
    
    # Initialize counters for overall dataset
    overall_counts = [0, 0, 0]  # hard, normal, easy
    overall_totals = [0, 0, 0]  # total hard, normal, easy examples in entire dataset
    
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    results_grouped = {}
    for subset in present_subsets:
        subset_dataset = dataset.filter(lambda example: example["subset"] == subset)
        scores = subset_dataset["score"]

        counts = [0, 0, 0]  # hard, normal, easy
        num_examples = [0, 0, 0]  # hard, normal, easy totals
        
        for i, example in enumerate(subset_dataset):
            if i % 9 in hard_indices:
                counts[0] += example['score']
                num_examples[0] += 1
                overall_counts[0] += example['score']
                overall_totals[0] += 1
            elif i % 9 in normal_indices:
                counts[1] += example['score']
                num_examples[1] += 1
                overall_counts[1] += example['score']
                overall_totals[1] += 1
            elif i % 9 in easy_indices:
                counts[2] += example['score']
                num_examples[2] += 1
                overall_counts[2] += example['score']
                overall_totals[2] += 1

        # Store results for this subset
        results_grouped[subset] = {
            'hard': {
                'correct': counts[0],
                'total': num_examples[0],
                'accuracy': counts[0]/num_examples[0] if num_examples[0] > 0 else 0
            },
            'normal': {
                'correct': counts[1],
                'total': num_examples[1],
                'accuracy': counts[1]/num_examples[1] if num_examples[1] > 0 else 0
            },
            'easy': {
                'correct': counts[2],
                'total': num_examples[2],
                'accuracy': counts[2]/num_examples[2] if num_examples[2] > 0 else 0
            }
        }

        print(f"Subset {subset}:")
        print(f"  Hard accuracy: {counts[0]}/{num_examples[0]} ({counts[0]/num_examples[0] if num_examples[0] > 0 else 0})")
        print(f"  Normal accuracy: {counts[1]}/{num_examples[1]} ({counts[1]/num_examples[1] if num_examples[1] > 0 else 0})")
        print(f"  Easy accuracy: {counts[2]}/{num_examples[2]} ({counts[2]/num_examples[2] if num_examples[2] > 0 else 0})")
        print()
    
    # Store overall results
    results_grouped['overall'] = {
        'hard': {
            'correct': overall_counts[0],
            'total': overall_totals[0],
            'accuracy': overall_counts[0]/overall_totals[0] if overall_totals[0] > 0 else 0
        },
        'normal': {
            'correct': overall_counts[1],
            'total': overall_totals[1],
            'accuracy': overall_counts[1]/overall_totals[1] if overall_totals[1] > 0 else 0
        },
        'easy': {
            'correct': overall_counts[2],
            'total': overall_totals[2],
            'accuracy': overall_counts[2]/overall_totals[2] if overall_totals[2] > 0 else 0
        }
    }
    
    # Print overall dataset statistics
    print("Overall dataset:")
    print(f"  Hard accuracy: {overall_counts[0]}/{overall_totals[0]} ({overall_counts[0]/overall_totals[0] if overall_totals[0] > 0 else 0})")
    print(f"  Normal accuracy: {overall_counts[1]}/{overall_totals[1]} ({overall_counts[1]/overall_totals[1] if overall_totals[1] > 0 else 0})")
    print(f"  Easy accuracy: {overall_counts[2]}/{overall_totals[2]} ({overall_counts[2]/overall_totals[2] if overall_totals[2] > 0 else 0})")    
    return results_grouped
    
def evaluation(dataset, subsets, args):
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    results_grouped = {}
    for subset in present_subsets:
        subset_dataset = dataset.filter(lambda example: example["subset"] == subset)
        scores = subset_dataset["score"]
        num_total = len(scores)

        num_correct = sum(scores)
        results_grouped[subset] = num_correct / num_total
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")

    if args.dataset == "rewardbench_v1_set":
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(f"rewardbench_v1_leadboard: {results_leaderboard}")
        results_grouped.update(results_leaderboard)
    elif (args.dataset == "judgebench_gpt_set" or args.dataset == "judgebench_claude_set") and args.swap:
        results_leaderboard = calculate_judgebench_accuracy_swap(dataset, subsets)
        results_grouped.update(results_leaderboard)
    elif args.dataset == "rm_bench_set":
        results_leaderboard = calculate_rm_bench_accuracy(dataset, subsets)
        results_grouped.update(results_leaderboard)
    elif args.dataset == "rewardbench_v2_set":
        print(f"TIES: {args.ties_score}")
        results_grouped["TIES"] = args.ties_score
    return results_grouped

def setup_argparse() -> argparse.Namespace:
    """Set up argument parser"""
    parser = argparse.ArgumentParser(description='LLM Evaluation Pipeline')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path for vLLM')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['inf2_sets', 'rewardbench_v1_set', 'judgebench_gpt_set', 'judgebench_claude_set', 'rm_bench_set', 'rewardbench_v2_set'],
                       help='Different dataset(s)')
    parser.add_argument('--max_turns', type=int, default=4,
                       help='Maximum turns to be maintained, otherwise will be filtered out')
    parser.add_argument('--swap', action='store_true', default=False,
                       help="Whether the sequences of responses will be swapped")
    
    # New prompt selection argument
    parser.add_argument('--prompt_type', type=str, required=True,
                       choices=['helpsteer3', 'generic_conversational_intellegence'],
                       help='Type of prompt template to use')
    
    # New GPU control argument
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    
    # Existing inference parameters
    parser.add_argument('--use_chat_template', type=str, default=True,
                       help='Use the default chat template within the tokenizer')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Number of prompts to process in each generation batch')
    parser.add_argument('--max_prompt_length', type=int, default=8192,
                       help='Maximum prompt length')
    parser.add_argument('--max_tokens', type=int, default=8192,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p sampling value')
    
    # Evaluation parameters
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (limit samples)')
    parser.add_argument('--max_debug_examples', type=int, default=10240,
                       help='Maximum debug examples')
    
    # New argument for specifying complete output file path
    parser.add_argument('--output_file', type=str, default=None,
                       help='Complete path for output JSON file (including filename). '
                            'If not specified, will generate automatically in results/ directory')
    
    return parser.parse_args()

def save_results(results, model_name, prompt_type, output_file=None):
    """Save evaluation results to a JSON file.
    If output_file is specified, uses that path exactly.
    Otherwise generates a filename automatically in results/ directory."""
    import os
    from datetime import datetime
    
    if output_file is None:
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/results_{model_name.replace('/', '_')}_{prompt_type}_{timestamp}.json"
    else:
        # Use the specified path exactly
        filename = output_file
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    
    # Save results
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    return filename

def main():
    args = setup_argparse()
    
    print("Loading dataset...")
    if args.dataset == "rewardbench_v2_set":
        dataset, subsets, ties_dataset = load_datasets(args)
        print(f"Loaded {len(ties_dataset)} samples from {args.dataset} TIES subset")
    else:
        dataset, subsets = load_datasets(args)
    print(f"Loaded {len(dataset)} samples from {args.dataset}")
    
    if args.debug and not args.swap:
        dataset = dataset.shuffle(seed=42).select(range(min(args.max_debug_examples, len(dataset))))
        if args.dataset == "rewardbench_v2_set":
            ties_dataset = ties_dataset.shuffle(seed=42).select(range(min(args.max_debug_examples, len(ties_dataset))))
            print(f"Debug mode: Limited to {len(ties_dataset)} samples from TIES subset")
        print(f"Debug mode: Limited to {len(dataset)} samples")

    print("Initializing LLM...")
    engine = vLLMInferenceEngine(args)

    print("Running inference...")
    results, candidates = engine.batch_predict(dataset, args)
    dataset = dataset.add_column('evaluation', results)
    if args.dataset == "rewardbench_v2_set":
        dataset = dataset.add_column('candidates', candidates)

    if args.dataset == "rewardbench_v2_set":
        ties_results = engine.batch_predict_ties(ties_dataset, args)
        ties_dataset = ties_dataset.add_column('evaluation', ties_results)

    print("Parsing results...")
    answers = [output_parser(example, args) for example in dataset]
    dataset = dataset.add_column('answers', answers)
    if args.dataset == "rewardbench_v2_set":
        ties_answers = [output_parser(example, args) for example in ties_dataset]
        ties_dataset = ties_dataset.add_column('scores', ties_answers)

    # Apply the function to the dataset
    dataset = dataset.map(process_example, fn_kwargs={"args": args})
    if args.dataset == "rewardbench_v2_set":
        ties_dataset, ties_score = process_single_model(ties_dataset)
        args.ties_score = ties_score

    print("Evaluation...")
    return_values = evaluation(dataset, subsets, args)
    
    # Add metadata to results
    return_values['metadata'] = {
        'model': args.model,
        'dataset': args.dataset,
        'prompt_type': args.prompt_type,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to file
    results_file = save_results(return_values, args.model, args.prompt_type, args.output_file)
    print(f"Results saved to {results_file}")
    
    return return_values

if __name__ == "__main__":
    main()
