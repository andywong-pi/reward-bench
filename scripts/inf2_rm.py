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
    #"/mnt/vast/home/sanjana/dpo_data/dpojpi_chatml_no_names_llama33i_resample.jsonl",
    # "/mnt/vast/home/jimmy/data/inf2/rl/validation.parquet",
    # "/mnt/vast/home/andy/data/inf1_eclairselfharm+core+support+justpi/annotations_pm_test.jsonl",
    "/mnt/vast/home/jimmy/data/july_2025_justpi_multiturn_test_prolific/annotations_pm_test_july_2025_prolific.jsonl"
]
REWARDBENCH_v1_SET = "allenai/reward-bench"
JUDGEBENCH_SET = "ScalerLab/JudgeBench"
RM_BENCH_SET = "THU-KEG/RM-Bench"
REWARDBENCH_v2_SET = "allenai/reward-bench-2"


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

    def create_entry(
        row: dict,
        prompt: list[dict],
        chosen_resp: list[dict],
        rejected_resp: list[dict],
        response1: list[dict],
        response2: list[dict],
        is_shuffled: bool
    ) -> dict:
        """
        Creates a standardized entry while preserving original fields from the input row.
        
        Args:
            row: Original dictionary containing all fields to be preserved
            prompt: List of dictionaries representing the prompt messages
            chosen_resp: List of dictionaries representing the chosen response
            rejected_resp: List of dictionaries representing the rejected response
            response1: First response option
            response2: Second response option
            is_shuffled: Boolean indicating if responses were shuffled
            
        Returns:
            A new dictionary combining original fields with the standardized structure
        """
        # Create a deep copy to avoid modifying the original row
        new_entry = row.copy()
        
        # Create message chains by combining prompt with responses (without modifying originals)
        messages_1 = prompt + response1
        messages_2 = prompt + response2
        
        # Update with new fields
        new_entry.update({
            "prompt": prompt.copy(),
            "text_chosen": chosen_resp.copy(),
            "text_rejected": rejected_resp.copy(),
            "response1": response1.copy(),
            "response2": response2.copy(),
            "messages_1": messages_1,
            "messages_2": messages_2,
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
        """
        Process a row of data by:
        1. Finding the common prefix between chosen and rejected responses
        2. Extracting the prompt (common prefix) and individual responses
        3. Shuffling the responses while tracking the original chosen response
        4. Adding shuffled responses and other metadata to the row
        
        Args:
            row: Dictionary containing 'text_chosen' and 'text_rejected' keys
            
        Returns:
            Processed row with additional keys for prompt, responses, and shuffle info
        """
        # Extract texts and verify structure
        chosen_texts = row["text_chosen"]
        rejected_texts = row["text_rejected"]
        
        # Find the split point between prompt and responses
        split_idx = find_first_difference(chosen_texts[0], rejected_texts[0])
        prompt = chosen_texts[0][:split_idx]
        
        # Prepare all responses (chosen + 3 rejected)
        responses = [
            chosen_texts[0][split_idx:],    # chosen response (index 0)
            rejected_texts[0][split_idx:],  # rejected0
            rejected_texts[1][split_idx:],  # rejected1
            rejected_texts[2][split_idx:]   # rejected2
        ]
        
        # Shuffle responses while tracking original chosen (index 0)
        shuffled_indices = random.sample(range(len(responses)), k=len(responses))
        shuffled_responses = [responses[i] for i in shuffled_indices]
        new_chosen_position = shuffled_indices.index(0)  # 0-based index of original chosen
        
        # Build the output row
        processed_row = {
            **row,
            "prompt": prompt,
            **{f"response{i+1}": resp for i, resp in enumerate(shuffled_responses)},
            **{f"messages_{i+1}": prompt + resp for i, resp in enumerate(shuffled_responses)},
            "is_shuffled": str(new_chosen_position + 1),  # 1-based position
            "original_chosen_index": 0,  # For debugging
            "shuffled_indices": shuffled_indices  # For debugging/verification
        }
        
        return processed_row

    dataset = dataset.map(process_row)
    return dataset

def generate_prompt_response_rewardbench_v2_ties(example) -> dict:  # Remove unused `args`
    """
    Generate system_prompt and user_prompt for each answer.
    Returns the original example with a new 'messages' field.
    """
    messages = [
        [
            example["prompt"],
            answer
        ]
        for answer in example["answers"]  # List comprehension
    ]
    return {**example, "messages": messages}  # Merge with original data

def filter_long_turns(batch, max_turns):
    return len(batch["text_chosen"]) // 2 <= max_turns

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
        ties_dataset = ties_dataset.map(generate_prompt_response_rewardbench_v2_ties)
    else:
        # swap is only utilized for judgebench
        formatted_dataset = generate_prompt_response(raw_dataset, set_name=args.dataset, swap=args.swap)
        
    dataset = formatted_dataset.filter(lambda x: filter_long_turns(x, args.max_turns))
    subsets = dataset["subset"]

    if args.dataset != "rewardbench_v2_set":
        return dataset, subsets
    else:
        return dataset, subsets, ties_dataset

import torch
from vllm import LLM, SamplingParams
from datasets import Dataset

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional, Tuple
from datasets import Dataset
from tqdm import tqdm
import gc

class RewardModelInferenceEngine:
    def __init__(self, args):
        """Initialize reward model with optimized multi-GPU support for large models."""
        self.args = args
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        
        print("\nInitializing reward model with optimized configuration...")
        
        # Initialize with lower memory footprint
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        self.device = torch.device("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        # Handle padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "[PAD]"
            if self.tokenizer.pad_token == "[PAD]":
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Model loading with memory optimization
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
            device_map="auto"  # Let Hugging Face handle device placement
        )
        
        # Configure model for inference
        if self.tokenizer.pad_token is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Configure for inference
        self.model.eval()
        
        # Print GPU info
        self._print_gpu_info()
        
        # Warm up the model
        self._warmup_model()

    def _print_gpu_info(self):
        """Print detailed GPU memory information."""
        print("\nGPU Configuration:")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: Allocated = {torch.cuda.memory_allocated(i)/1024**3:.2f}GB, "
                  f"Reserved = {torch.cuda.memory_reserved(i)/1024**3:.2f}GB")
        print()

    def _warmup_model(self):
        """Run a small warmup inference to initialize memory."""
        print("Warming up model...")
        dummy_input = self.tokenizer("Warmup", return_tensors="pt").to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            _ = self.model(**dummy_input)
        torch.cuda.empty_cache()

    def format_chat_prompt(self, messages: List[dict]) -> Optional[str]:
        """Optimized chat prompt formatting with memory awareness."""
        try:
            if not hasattr(self.tokenizer, 'apply_chat_template'):
                raise ValueError("Tokenizer does not support chat templates")
                
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            if hasattr(self.args, 'max_prompt_length'):
                tokenized = self.tokenizer(prompt, return_tensors="pt")
                if len(tokenized.input_ids[0]) > self.args.max_prompt_length:
                    return " "
            
            return prompt
        except Exception as e:
            print(f"Prompt formatting error: {str(e)}")
            return " "

    def _cleanup_memory(self):
        """Clean up memory between batches."""
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, text_batch: List[str]) -> List[float]:
        """
        Optimized batch processing with memory management.
        """
        if not text_batch:
            return []
            
        batch_size = min(
            getattr(self.args, 'batch_size', 4),  # Default to smaller batches for large models
            len(text_batch)
        )
        
        scores = []
        try:
            # Process in smaller sub-batches if needed
            for i in range(0, len(text_batch), batch_size):
                batch = text_batch[i:i + batch_size]
                
                # Tokenize with memory efficiency
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.args, 'max_length', 2048),
                    return_tensors="pt"
                )
                
                # Move inputs to device in a memory-efficient way
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                
                # Inference with memory optimization
                with torch.no_grad(), torch.cuda.amp.autocast():
                    outputs = self.model(**inputs)
                    batch_scores = outputs.logits[:, -1].squeeze()
                    
                    # Handle single-item batches
                    if batch_scores.dim() == 0:
                        batch_scores = batch_scores.unsqueeze(0)
                    
                    scores.extend(batch_scores.cpu().tolist())  # Move to CPU immediately
                
                # Clean up
                del inputs, outputs, batch_scores
                self._cleanup_memory()
            
            return scores
            
        except torch.cuda.OutOfMemoryError:
            print("Out of memory, retrying with smaller batch size...")
            self._cleanup_memory()
            return self.generate(text_batch)  # Recursively try with smaller batches

    def batch_predict(self, dataset: Dataset, args) -> List[Tuple[int, Optional[int]]]:
        """Optimized batch prediction with memory management."""
        is_rewardbench_v2 = args.dataset == "rewardbench_v2_set"
        num_responses = 4 if is_rewardbench_v2 else 2
        message_keys = [f'messages_{i+1}' for i in range(num_responses)]
        
        # Pre-process all prompts first to manage memory better
        all_prompts = []
        for example in tqdm(dataset, desc="Preparing prompts"):
            example_prompts = []
            for key in message_keys:
                message = example[key]
                prompt = self.format_chat_prompt(message) if args.use_chat_template else message
                example_prompts.append(prompt)
            all_prompts.append(example_prompts)
        
        # Process responses in optimized batches
        winning_response_indices = []
        for i in tqdm(range(0, len(all_prompts), args.batch_size), desc="Processing batches"):
            batch_prompts = all_prompts[i:i + args.batch_size]
            
            # Transpose to get responses together
            response_batches = list(zip(*batch_prompts))
            batch_scores = []
            
            for response_batch in response_batches:
                scores = self.generate(list(response_batch))
                batch_scores.append(scores)
            
            # Determine winners for this batch
            for scores in zip(*batch_scores):
                try:
                    winning_idx = scores.index(max(scores))
                    winning_response_indices.append(winning_idx)
                except:
                    winning_response_indices.append(-1)
            
            # Clean up between batches
            self._cleanup_memory()
        
        return winning_response_indices

    def batch_predict_ties(self, dataset: Dataset, args) -> Dataset:
        """Optimized version for tie detection."""
        all_responses = []
        
        # Pre-process all prompts first
        all_prompts = []
        for example in tqdm(dataset, desc="Preparing prompts"):
            example_prompts = []
            for message in example['messages']:
                prompt = self.format_chat_prompt(message) if args.use_chat_template else message
                example_prompts.append(prompt)
            all_prompts.append(example_prompts)
        
        # Process in optimized batches
        for i in tqdm(range(0, len(all_prompts), args.batch_size), desc="Processing batches"):
            batch_prompts = all_prompts[i:i + args.batch_size]
            
            for example_prompts in batch_prompts:
                responses = self.generate(example_prompts)
                all_responses.append(responses)
            
            # Clean up between batches
            self._cleanup_memory()
        
        return all_responses

import re
from typing import Optional

def output_parser(example, args):
    judgment = example['evaluation']
    subset = example['subset']
    
    if subset != 'Ties':
        # Ensure judgment is between 0 and 3
        if judgment not in {0, 1, 2, 3}:
            raise ValueError(f"Invalid judgment value: {judgment}. Expected 0, 1, 2, or 3.")
        if args.dataset == "rewardbench_v2_set":
            # Map judgment to letters (1-4)
            judgment_map = {0: "1", 1: "2", 2: "3", 3: "4"}
            letter_judgment = judgment_map[judgment]  # Safe since we already validated
        else:
            judgment_map = {0: "A", 1: "B"}
            letter_judgment = judgment_map[judgment]
        return letter_judgment
    elif subset == 'Ties':
        judgment_array = np.array(judgment)

        # Normalize between 1 and 10
        min_val = np.min(judgment_array)
        max_val = np.max(judgment_array)

        normalized = 1 + (judgment_array - min_val) * (9) / (max_val - min_val)
        normalized_integers = np.round(normalized).astype(int)

        return normalized_integers.tolist()

# Iterate through the dataset and apply the logic
def process_example(example, args):
    answer = example['answers']  # replace with your answer column name
    is_shuffled = example['is_shuffled']  # replace with your is_shuffled column name
    
    if args.dataset != 'rewardbench_v2_set':
        if (answer == 'A' and not is_shuffled) or (answer == 'B' and is_shuffled):
            return {'score': 1}
        elif (answer == 'A' and is_shuffled) or (answer == 'B' and not is_shuffled):
            return {'score': 0}
        else:
            # return {'score': 0.5} remove this impact
            return {'score': 0}
    elif args.dataset == 'rewardbench_v2_set':
        if answer == is_shuffled:
            return {'score': 1}
        else:
            return {'score': 0}
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

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
    
    # New GPU control argument
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    
    # Existing inference parameters
    parser.add_argument('--use_chat_template', type=str, default=True,
                       help='Use the default chat template within the tokenizer')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Number of prompts to process in each generation batch')
    parser.add_argument('--max_prompt_length', type=int, default=8192,
                       help='Maximum prompt length')
    parser.add_argument('--max_tokens', type=int, default=8192,
                       help='Maximum tokens to generate')
    
    # Evaluation parameters
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (limit samples)')
    parser.add_argument('--max_debug_examples', type=int, default=10240,
                       help='Maximum debug examples')
    
    # New argument to limit maximum examples per dataset
    parser.add_argument('--max_examples', type=int, default=20000,
                       help='Maximum number of examples to load from each dataset (default: 200000)')
    # New argument for specifying complete output file path
    parser.add_argument('--output_file', type=str, default=None,
                       help='Complete path for output JSON file (including filename). '
                            'If not specified, will generate automatically in results/ directory')
    
    return parser.parse_args()

def save_results(results, dataset, model_name, output_file=None):
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
        filename = f"results/results_{model_name.replace('/', '_')}_{timestamp}.json"
    else:
        # Use the specified path exactly
        filename = output_file
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    
    # Convert dataset to a JSON-serializable format (e.g., dictionary)
    try:
        # Assuming dataset is a Hugging Face Dataset object
        dataset_serializable = dataset.to_dict()  # Convert to dictionary
    except AttributeError:
        # If dataset is already a list or dict, use it directly
        dataset_serializable = dataset if isinstance(dataset, (dict, list)) else list(dataset)
    
    # Prepare the data to save
    data_to_save = {
        'results': results,
        'dataset': dataset_serializable
    }
    
    # Save results
    with open(filename, "w") as f:
        json.dump(data_to_save, f, indent=2)
    
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
    
    # Apply max_examples limit to both datasets and corresponding subsets
    if len(dataset) > args.max_examples:
        # Shuffle once and get indices
        shuffled_indices = list(range(len(dataset)))
        random.Random(42).shuffle(shuffled_indices)
        selected_indices = shuffled_indices[:args.max_examples]
        
        # Apply same selection to both
        dataset = dataset.select(selected_indices)
        subsets = [subsets[i] for i in selected_indices]

    if args.debug and not args.swap:
        dataset = dataset.shuffle(seed=42).select(range(min(args.max_debug_examples, len(dataset))))
        if args.dataset == "rewardbench_v2_set":
            ties_dataset = ties_dataset.shuffle(seed=42).select(range(min(args.max_debug_examples, len(ties_dataset))))
            print(f"Debug mode: Limited to {len(ties_dataset)} samples from TIES subset")
        print(f"Debug mode: Limited to {len(dataset)} samples")

    print("Initializing LLM...")
    engine = RewardModelInferenceEngine(args)

    print("Running inference...")
    results = engine.batch_predict(dataset, args)

    dataset = dataset.add_column('evaluation', results)

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
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to file
    results_file = save_results(return_values, dataset, args.model, args.output_file)
    print(f"Results saved to {results_file}")
    
    return return_values

if __name__ == "__main__":
    main()
