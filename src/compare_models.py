import json
import time
import pandas as pd
import os
import multiprocessing
import sys
import argparse

# Specify to use only GPU 0 and 1 for vLLM
gpu_devices_list = [0, 1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_devices_list)

# Define paths to the models
original_model_path = "model/Mistral-7B-Instruct-v0.2"
dpo_model_path = "model/Mistral-7B-Instruct-v0.2-dpo"

# Sample instructions that were not seen in training
instructions = [
    "Explain the process of photosynthesis in trees using only words that start with the letter 'P'.",
    "Write a short poem about quantum physics and love.",
    "Describe the taste of water to someone who has never tasted it before.",
    "If colors had personalities, what would be the personality of the color purple?",
    "Design a sustainable city of the future that addresses climate change concerns.",
    "Create a dialogue between a smartphone and an old rotary phone discussing modern communication.",
    "Explain blockchain technology to a 10-year-old child.",
    "What would be different if humans had evolved with four arms instead of two?",
    "Write a recipe for happiness that includes exact measurements.",
    "Describe the internet to someone from the 18th century."
]

def process_model(model_path, output_file):
    """
    Function to run in a separate process - loads model and processes all instructions
    """
    # Import required libraries here, so each process has its own copy
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    
    # Set vLLM to use spawn method
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = 'spawn'
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )
    
    print(f"Process for {model_path} starting...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        print(f"Loading model from {model_path}...")
        model = LLM(
            model=model_path, 
            trust_remote_code=True, 
            tensor_parallel_size=len(gpu_devices_list)
        )
        
        results = []
        for i, instruction in enumerate(instructions):
            print(f"[{model_path}] Processing instruction {i+1}/{len(instructions)}: {instruction[:50]}...")
            
            # Apply chat template
            messages = [{"role": "user", "content": instruction}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Generate completion
            outputs = model.generate([prompt], sampling_params)
            completion = outputs[0].outputs[0].text.strip()
            
            # Store result
            results.append({
                "instruction": instruction,
                "completion": completion
            })
        
        # Save results to the output file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results for {model_path} saved to {output_file}")
        
    except Exception as e:
        print(f"Error in process for {model_path}: {str(e)}")
        # Create an empty file to indicate failure
        with open(output_file + '.error', 'w') as f:
            f.write(str(e))

def run_original_model():
    """Function to run in a separate process for the original model"""
    process_model(original_model_path, "original_model_results.json")

def run_dpo_model():
    """Function to run in a separate process for the DPO model"""
    process_model(dpo_model_path, "dpo_model_results.json")

def run_comparison(model_name):
    """Run only one model based on command line argument"""
    if model_name == "original":
        run_original_model()
    elif model_name == "dpo":
        run_dpo_model()
    else:
        print(f"Unknown model name: {model_name}")
        sys.exit(1)

def combine_results():
    """Combine results from both models into a final dataframe"""
    try:
        # Load results from both model runs
        with open("original_model_results.json", 'r') as f:
            original_results = json.load(f)
        
        with open("dpo_model_results.json", 'r') as f:
            dpo_results = json.load(f)
        
        # Combine the results
        combined_results = []
        for orig, dpo in zip(original_results, dpo_results):
            # Verify instructions match
            assert orig["instruction"] == dpo["instruction"], "Instructions don't match"
            
            combined_results.append({
                "instruction": orig["instruction"],
                "Mistral-7B-Instruct-v0.2": orig["completion"],
                "Mistral-7B-Instruct-v0.2-dpo": dpo["completion"]
            })
        
        # Save to final JSON file
        with open("model_comparison_results.json", 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        # Create pandas DataFrame
        df = pd.DataFrame(combined_results)
        
        # Display all rows in the DataFrame instead of just the first 5
        print("\nFull comparison DataFrame:")
        # Set display options to show all rows
        with pd.option_context('display.max_rows', None, 'display.max_colwidth', None):
            print(df)
        
        return True
    except Exception as e:
        print(f"Error combining results: {str(e)}")
        return False

def main():
    # Create argument parser to determine which part to run
    parser = argparse.ArgumentParser(description="Compare original and DPO fine-tuned models")
    parser.add_argument('mode', choices=['original', 'dpo', 'combine'], 
                        help='Which mode to run: original, dpo, or combine')
    
    args = parser.parse_args()
    
    if args.mode == 'original':
        run_original_model()
    elif args.mode == 'dpo':
        run_dpo_model()
    elif args.mode == 'combine':
        combine_results()
    else:
        print("Invalid mode specified")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Create wrapper script if no arguments provided
        with open("run_comparison.sh", "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("echo 'Running original model...'\n")
            f.write(f"python {sys.argv[0]} original\n\n")
            f.write("echo 'Running DPO model...'\n")
            f.write(f"python {sys.argv[0]} dpo\n\n")
            f.write("echo 'Combining results...'\n")
            f.write(f"python {sys.argv[0]} combine\n")
        
        # Make the script executable
        os.chmod("run_comparison.sh", 0o755)
        print("Created run_comparison.sh script. Run this script to process both models and combine results.")
    else:
        main()