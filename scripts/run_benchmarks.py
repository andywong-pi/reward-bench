import argparse
import os
import subprocess
import time
import json
import glob
from datetime import datetime
import csv 

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run all benchmark experiments')
    parser.add_argument('--model', type=str, required=True,
                      help='Model name or path (e.g., Qwen/Qwen3-14B)')
    parser.add_argument('--prompt_type', type=str, required=True,
                      choices=['helpsteer3', 'generic_conversational_intellegence'],
                      help='Prompt type to use for all experiments')
    parser.add_argument('--max_turns', type=int, default=4,
                      help='Maximum turns to be maintained (default: 4)')
    return parser.parse_args()

def run_experiments(model_name, prompt_type, max_turns):
    """Run all benchmark experiments sequentially"""
    base_command = "python inf2_vllm.py"
    
    # Define all experiments with their specific parameters
    experiments = [
        {
            "name": "inf2",
            "dataset": "inf2_sets",
            "max_turns": f"{max_turns}",
            "extra_args": ""
        },
        {
            "name": "rewardbench_v1",
            "dataset": "rewardbench_v1_set",
            "max_turns": f"{max_turns}",
            "extra_args": ""
        },
        {
            "name": "judgebench_gpt",
            "dataset": "judgebench_gpt_set",
            "max_turns": f"{max_turns}",
            "extra_args": "--swap"
        },
        {
            "name": "judgebench_claude",
            "dataset": "judgebench_claude_set",
            "max_turns": f"{max_turns}",
            "extra_args": "--swap"
        },
        {
            "name": "rm_bench",
            "dataset": "rm_bench_set",
            "max_turns": f"{max_turns}",
            "extra_args": ""
        },
        {
            "name": "rewardbench_v2",
            "dataset": "rewardbench_v2_set",
            "max_turns": f"{max_turns}",
            "extra_args": ""
        }
    ]

    # Create timestamp for all experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_name.replace("/", "_")
    
    # Run each experiment
    for exp in experiments:
        # Create base directory with dataset name included
        output_dir = f"results/{model_safe}_{prompt_type}_{timestamp}/{exp['dataset']}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Build command with output file path
        output_file = f"{output_dir}/results.json"
        command = (
            f"{base_command} --model={model_name} "
            f"--dataset={exp['dataset']} "
            f"--prompt_type={prompt_type} "
            f"--max_turns={max_turns} "
            f"--output_file={output_file} "
            f"{exp['extra_args']}"
        )
        
        print(f"Running command: {command}")
        start_time = time.time()
        
        # Execute the command
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Print execution time
        elapsed = time.time() - start_time
        print(f"Completed {exp['name']} in {elapsed:.2f} seconds")
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error running {exp['name']}:")
            print(result.stderr)
        else:
            print(f"Results saved to {output_file}")
    
    print(f"\nAll experiments completed! Results saved in various directories")
    return f"results/{model_safe}_{prompt_type}_{timestamp}/*"

def load_all_results(base_path_pattern="results/*"):
    """Load all result JSON files into a dictionary, keyed by their immediate parent directory"""
    all_results = {}
    for base_path in glob.glob(base_path_pattern):
        for json_file in glob.glob(f"{base_path}/**/results.json", recursive=True):
            with open(json_file, "r") as f:
                data = json.load(f)
                
                # Get the immediate parent directory name (the key we want)
                key = os.path.basename(os.path.dirname(json_file))
                
                # Add metadata
                if "metadata" not in data:
                    data["metadata"] = {}
                data["metadata"]["experiment"] = os.path.basename(base_path)
                data["metadata"]["dataset"] = key  # Using the same key as dataset name
                
                all_results[key] = data
    return all_results

def transform_results(input_data, model_name, prompt_type):
    all_results = [
        {"model_name": model_name},
        {"prompt_type": prompt_type}
    ]

    def add_average(results_dict, prefix, keys):
        values = [results_dict[k] for k in keys]
        all_results.append({f"{prefix}-Average": sum(values) / len(values)})

    for key, val in input_data.items():
        if key == "inf2_sets":
            all_results.extend([
                {"inf2-dpojpi": val['dpojpi_chatml_no_names_llama33i_resample']},
                {"inf2-validation": val['validation']}
            ])
        elif key == "rewardbench_v1_set":
            keys = ['Chat', 'Chat Hard', 'Safety', 'Reasoning']
            all_results.extend([{"rewardbench_v1-" + k: val[k]} for k in keys])
            add_average(val, "rewardbench_v1", keys)
        elif key == "rewardbench_v2_set":
            keys = ['Factuality', 'Focus', 'Math', 'Precise IF', 'Safety', 'TIES']
            all_results.extend([{"rewardbench_v2-" + k: val[k]} for k in keys])
            add_average(val, "rewardbench_v2", keys)
        elif key in ("judgebench_gpt_set", "judgebench_claude_set"):
            prefix = key.replace("_set", "")
            benchmarks = [
                ('mmlu-pro', 'mmlu_pro'),
                ('livebench-reasoning', 'livebench_reasoning'),
                ('livebench-math', 'livebench_math'),
                ('livecodebench', 'livecodebench'),
                ('', 'overall')
            ]
            for bench_key, bench_name in benchmarks:
                metrics = {
                    f"{prefix}-{bench_name}-{m}": val[bench_key][f"{m}_ratio"]
                    for m in ['both_correct', 'both_wrong', 'mixed']
                }
                all_results.append(metrics)
        elif key == "rm_bench_set":
            categories = [
                'chat', 'code', 'math', 
                'safety-refuse', 'safety-response', 'overall'
            ]
            for cat in categories:
                metrics = {
                    f"rm_bench_set-{cat}-{level}": val[cat][level]['accuracy']
                    for level in ['hard', 'normal', 'easy']
                }
                all_results.append(metrics)

    return all_results

def save_csv(combined_file, merged_results):
    # Create the results directory if it doesn't exist
    os.makedirs(os.path.dirname(combined_file), exist_ok=True)

    # Flatten the dictionary list into a single dictionary
    flat_data = {}
    for item in merged_results:
        flat_data.update(item)

    # Prepare the header and row data
    header = flat_data.keys()
    row = flat_data.values()

    # Check if file exists to determine write mode
    file_exists = os.path.isfile(combined_file)

    with open(combined_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is being created
        if not file_exists:
            writer.writerow(header)
        
        # Write the data row
        writer.writerow(row)

def main():
    args = parse_args()
    
    # Run all experiments
    results_dir_pattern = run_experiments(
        model_name=args.model,
        prompt_type=args.prompt_type,
        max_turns=args.max_turns
    )
    print(f"results_dir_pattern: {results_dir_pattern}")

    # Load and concatenate all results
    combined_results = load_all_results(results_dir_pattern)

    # put all of these results into a list of dictionary
    merged_results = transform_results(combined_results, args.model, args.prompt_type)

    # Save combined results
    combined_file = f"results/combined_results.csv"
    save_csv(combined_file, merged_results)

if __name__ == "__main__":
    main()