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

INF2_SETS = [
    "/mnt/vast/home/sanjana/dpo_data/dpojpi_chatml_no_names_llama33i_resample.jsonl",
    "/mnt/vast/home/jimmy/data/inf2/rl/validation.parquet"
]
REWARDBENCH_v1_SET = "allenai/reward-bench"
JUDGEBENCH_SET = "ScalerLab/JudgeBench"

HELPSTEER3_PRINCIPLES_SYSTEM_PROMPT = (
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
HELPSTEER3_PRINCIPLES_USER_PROMPT = (
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

def generate_prompt_response(dataset, set_name):
    if set_name in ["inf2_sets", "rewardbench_v1_set", "judgebench_gpt_set", "judgebench_claude_set"]:        
        # Process each row to split into prompt and responses
        def split_prompt_response(row):
            chosen = row["text_chosen"]
            rejected = row["text_rejected"]
            
            # Find the first differing character
            split_idx = find_first_difference(chosen, rejected)
            
            # Split into prompt and responses
            prompt = chosen[:split_idx]
            new_chosen = chosen[split_idx:]
            new_rejected = rejected[split_idx:]

            if random.random() < 0.5:
                response1 = new_chosen
                response2 = new_rejected
                is_shuffled = False
            else:
                response1 = new_rejected
                response2 = new_chosen
                is_shuffled = True
            
            return {
                "prompt": prompt,
                "text_chosen": new_chosen,
                "text_rejected": new_rejected,
                "response1": response1,
                "response2": response2,
                "is_shuffled": is_shuffled,
            }
        
        # Apply the transformation to each row
        dataset = dataset.map(split_prompt_response)
        
    else:
        raise ValueError(f"set_name {set_name} formatting is not defined")
    
    return dataset

def filter_long_turns(batch, max_turns):
    return len(batch["text_chosen"]) // 2 <= max_turns

def apply_prompt_templates(example, prompt_type: str) -> dict:
    """
    Generate system_prompt and user_prompt for given prompt type
    Returns a dictionary with these new columns
    """
    if prompt_type == "helpsteer3":
        system_prompt = HELPSTEER3_PRINCIPLES_SYSTEM_PROMPT
        user_prompt = HELPSTEER3_PRINCIPLES_USER_PROMPT.format(
            Query=example['prompt'],
            Response_1=example['response1'],
            Response_2=example['response2'],
        )
    elif prompt_type == "generic_conversational_intellegence":
        system_prompt = GENERIC_CONVERSATIONAL_INTELLIGENCE_SYSTEM_PROMPT
        user_prompt = GENERIC_CONVERSATIONAL_INTELLIGENCE_USER_PROMPT.format(
            Query=example['prompt'],
            Response_1=example['response1'],
            Response_2=example['response2'],
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": messages,
    }

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

def load_datasets(args) -> Tuple[Dataset, List[str]]:
    """Load datasets with subset tracking"""
    # Extract dataset path from set_name
    if args.dataset == "inf2_sets":
        raw_dataset = load_inf2_dataset(INF2_SETS)
    elif args.dataset == "rewardbench_v1_set":
        raw_dataset = load_rewardbench_v1_dataset(REWARDBENCH_v1_SET)
    elif args.dataset == "judgebench_gpt_set" or args.dataset == "judgebench_claude_set":
        raw_dataset = load_judgebench_dataset(args)
    else:
        raise ValueError(f"set_name {args.dataset} cannot be found")

    formatted_dataset = generate_prompt_response(raw_dataset, set_name=args.dataset)
    filtered_dataset = formatted_dataset.filter(lambda x: filter_long_turns(x, args.max_turns))
    # Apply to your dataset
    dataset = filtered_dataset.map(lambda x: apply_prompt_templates(x, args.prompt_type), batched=False)
    # Debug: select certain samples
    # take column subset from dataset
    subsets = dataset["subset"]

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

    def batch_predict(self, dataset: Dataset, use_chat_template: bool = False) -> Dataset:
        """Run batch prediction with either chat template or traditional prompt system"""
        prompts = []
        # Optional: Add a progress bar for prompt preparation if dataset is large
        for example in tqdm(dataset, desc="Preparing prompts"):
            prompt = self.format_chat_prompt(example['messages']) if use_chat_template else example['prompt']
            prompts.append(prompt)
        responses = self.generate(prompts)
        # responses = self.llm.generate(prompts, sampling_params=self.sampling_params)

        return responses

def output_parser(example, prompt_type):
    judgment = example['evaluation']

    if prompt_type in ["helpsteer3", "generic_conversational_intellegence", "helpsteer3_principles"]:
        # Standardized extraction with verl-private/verl/utils/reward_score/inf1_pie_platform.py

        def find_boxed_string(string: str, pattern: str, first: bool = True) -> Optional[str]:
            """Extract the first or last LaTeX boxed expression matching a regex pattern from a string.

            Args:
                string: Input string containing LaTeX code
                pattern: Regex pattern to match boxed expressions
                first: If True, return the first match; if False, return the last match

            Returns:
                The matched boxed expression or None if not found
            """
            matches = list(re.finditer(pattern, string))
            if not matches:
                return None
            return matches[0].group(0) if first else matches[-1].group(0)

        def last_boxed_only_string(string: str) -> Optional[str]:
            """Extract the last LaTeX boxed expression (single or double curly braces) from a string.

            Args:
                string: Input string containing LaTeX code

            Returns:
                The last boxed expression or None if not found
            """
            # Accepts \boxed{...} or \boxed{{...}}
            return find_boxed_string(string, r"\\boxed\{\{?[^,{}]+\}?\}", first=False)
            # return find_boxed_string(string, r'boxed\{(-?\d+)\}', first=False)

        def remove_boxed(s: str) -> str:
            """Remove the LaTeX boxed command with single or double curly braces from a string.

            Supports both single and pair boxed expressions (e.g., '\boxed{x}', '\boxed{x, y}', '\boxed{{x}}', or '\boxed{{x, y}}').

            Args:
                s: String with format "\\boxed{content}" or "\\boxed{{content}}"

            Returns:
                The content inside the boxed command
            """
            if s is None:
                return None
            left = "\\boxed{"
            right = "}"
            if not (s.startswith(left) and s.endswith(right)):
                return None
            inner = s[len(left):-len(right)]
            # If wrapped in extra curly braces, remove them
            if inner.startswith("{") and inner.endswith("}"):
                inner = inner[1:-1]
            return inner
        
        pattern = re.compile(
            r"\\?\[The Begin of Ranking Score\\?\](.*?)\s*"
            r"\\?\[The End of Ranking Score\\?\]",
            re.DOTALL
        )
        match_result = re.search(pattern, judgment)
        if match_result is not None:
            ranking_score_section = match_result.group(1)
        else:
            #print(f"DEBUG: judgment: {judgment}")
            #print(f"DEBUG: match_result: {match_result}")
            return "error"  # no ranking score section found
        preference_ranking_boxed_pred = last_boxed_only_string(ranking_score_section) if ranking_score_section is not None else None
        preference_ranking_extracted_pred = remove_boxed(preference_ranking_boxed_pred) if preference_ranking_boxed_pred is not None else None
        if preference_ranking_extracted_pred:
            score = int(preference_ranking_extracted_pred)
            if score <= 3: # 3 is the threshold for helpsteer3
                return "A"
            elif score > 3: # 3 is the threshold for helpsteer3
                return "B"
            else:
                return "error"
        else:
            return "error" # no boxed score found in the Ranking Score section
    else:
        raise ValueError(f"The model parser is not defined")

# Iterate through the dataset and apply the logic
def process_example(example):
    answer = example['answers']  # replace with your answer column name
    is_shuffled = example['is_shuffled']  # replace with your is_shuffled column name
    
    if (answer == 'A' and not is_shuffled) or (answer == 'B' and is_shuffled):
        return {'score': 1}
    elif (answer == 'A' and is_shuffled) or (answer == 'B' and not is_shuffled):
        return {'score': 0}
    else:
        # return {'score': 0.5} remove this impact
        return {'score': 0.5}

def evaluation(dataset, subsets, args):
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    results_grouped = {}
    for subset in present_subsets:
        subset_dataset = dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["score"])
        num_total = len(subset_dataset["score"])
        results_grouped[subset] = num_correct / num_total
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")

    if args.dataset == "rewardbench_v1_set":
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(f"rewardbench_v1_leadboard: {results_leaderboard}")

def setup_argparse() -> argparse.Namespace:
    """Set up argument parser"""
    parser = argparse.ArgumentParser(description='LLM Evaluation Pipeline')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path for vLLM')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['inf2_sets', 'rewardbench_v1_set', 'judgebench_gpt_set', 'judgebench_claude_set'],
                       help='Different dataset(s)')
    parser.add_argument('--max_turns', type=int, default=4,
                       help='Maximum turns to be maintained, otherwise will be filtered out')
    
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
    parser.add_argument('--batch_size', type=int, default=5120,
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
                       
    
    return parser.parse_args()

def main():
    args = setup_argparse()
    
    print("Loading dataset...")
    dataset, subsets = load_datasets(args)
    print(f"Loaded {len(dataset)} samples from {args.dataset}")
    
    if args.debug:
        dataset = dataset.shuffle(seed=42).select(range(min(args.max_debug_examples, len(dataset))))
        #dataset = dataset.select(range(min(args.max_debug_examples, len(dataset))))
        print(f"Debug mode: Limited to {len(dataset)} samples")

    print("Initializing LLM...")
    engine = vLLMInferenceEngine(args)
        
    print("Running inference...")
    results = engine.batch_predict(dataset, args.use_chat_template)
    dataset = dataset.add_column('evaluation', results)

    print("Parsing results...")
    answers = [output_parser(example, args.prompt_type) for example in dataset]
    dataset = dataset.add_column('answers', answers)
    # Apply the function to the dataset
    dataset = dataset.map(process_example)

    print("Evaluation...")
    evaluation(dataset, subsets, args)

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()