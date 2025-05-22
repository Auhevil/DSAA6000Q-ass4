import json
import random
from typing import List, Dict
import torch
from vllm import LLM, SamplingParams
import pandas as pd
from transformers import AutoTokenizer
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
import os
import jsonlines


# Specify to use only GPU 0 and 1 for vLLM
gpu_devices_list = [0, 1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_devices_list)

def load_instructions(n_samples: int = 50) -> List[str]:
    """Load and sample instructions from local LIMA dataset."""
    instructions = []
    lima_path = os.path.join("dataset", "lima", "train.jsonl")
    
    # Read instructions from local jsonl file
    with jsonlines.open(lima_path) as reader:
        for item in reader:
            instructions.append(item['conversations'][0])
    
    return random.sample(instructions, n_samples)

def generate_responses(instructions: List[str], model_name: str = "Mistral-7B-Instruct-v0.2") -> Dict[str, List[str]]:
    """Generate multiple responses for each instruction using Mistral-7B."""
    model_path = os.path.join("model", model_name)
    llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=len(gpu_devices_list))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Use the model's chat template
    def create_prompt(instruction: str) -> str:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048,
        n=5  # Generate 5 responses per instruction
    )
    
    responses_dict = {}
    for instruction in instructions:
        prompt = create_prompt(instruction)
        outputs = llm.generate([prompt], sampling_params)
        
        # Extract all generated texts from the outputs
        # outputs[0] contains all 5 responses for the single prompt
        responses = []
        for output in outputs[0].outputs:
            responses.append(output.text.strip())
        responses_dict[instruction] = responses
        
        # Print progress
        print(f"Generated {len(responses)} responses for instruction: {instruction[:50]}...")
    
    return responses_dict

def create_preference_dataset(responses_dict: Dict[str, List[str]], output_path: str):
    """Create preference dataset using PairRM."""
    # Initialize PairRM from local path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairrm_path = os.path.join("model", "PairRM-hf")
    pairrm = DebertaV2PairRM.from_pretrained(pairrm_path, device_map=device).eval()
    tokenizer = AutoTokenizer.from_pretrained(pairrm_path)
    
    # Define prefixes as per PairRM documentation
    source_prefix = "<|source|>"
    cand1_prefix = "<|candidate1|>"
    cand2_prefix = "<|candidate2|>"
    
    preference_data = []
    
    def tokenize_pair(sources: List[str], candidate1s: List[str], candidate2s: List[str], 
                     source_max_length=1024, candidate_max_length=512):
        """Tokenize pairs of responses for comparison."""
        ids = []
        assert len(sources) == len(candidate1s) == len(candidate2s)
        max_length = source_max_length + 2 * candidate_max_length
        for i in range(len(sources)):
            source_ids = tokenizer.encode(source_prefix + sources[i], max_length=source_max_length, truncation=True)
            candidate_max_length = (max_length - len(source_ids)) // 2
            candidate1_ids = tokenizer.encode(cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True)
            candidate2_ids = tokenizer.encode(cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True)
            ids.append(source_ids + candidate1_ids + candidate2_ids)
        encodings = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
        return encodings
    
    # Process each instruction and its responses
    for instruction, responses in responses_dict.items():
        # For each response, compare it with a dummy baseline to get individual scores
        response_scores = []
        dummy_response = "This is a baseline response."  # Used as a constant reference point
        
        for response in responses:
            # Compare each response with the dummy response
            sources = [instruction]
            candidates_A = [response]
            candidates_B = [dummy_response]
            
            encodings = tokenize_pair(sources, candidates_A, candidates_B)
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = pairrm(**encodings)
            score = outputs.logits.item()
            response_scores.append((response, score))
        
        # Sort responses by their scores
        response_scores.sort(key=lambda x: x[1], reverse=True)
        best_response = response_scores[0][0]
        
        # Create preference pairs between the best response and others
        for response, score in response_scores[1:]:
            # Compare best response with current response to get the actual score difference
            sources = [instruction]
            candidates_A = [best_response]
            candidates_B = [response]
            
            encodings = tokenize_pair(sources, candidates_A, candidates_B)
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            with torch.no_grad():
                outputs = pairrm(**encodings)
            score_diff = outputs.logits.item()
            
            preference_data.append({
                'instruction': instruction,
                'response_better': best_response,
                'response_worse': response,
                'score_diff': abs(score_diff)
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(preference_data)
    df.to_json(output_path, orient='records', lines=True)
    print(f"Saved preference dataset to {output_path}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs("dataset", exist_ok=True)
    output_path = "dataset/curated-preference-dataset.jsonl"
    
    # 1. Sample instructions
    print("Sampling instructions...")
    instructions = load_instructions(50)
    
    # 2. Generate responses
    print("Generating responses using Mistral-7B...")
    responses_dict = generate_responses(instructions)
    
    # Save raw responses
    with open("dataset/curated-raw-responses.json", 'w') as f:
        json.dump(responses_dict, f, indent=2)
    
    # 3. Create preference dataset
    print("Creating preference dataset using PairRM...")
    create_preference_dataset(responses_dict, output_path)

if __name__ == "__main__":
    main() 