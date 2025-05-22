import json
import os

def process_preference_data():
    """
    Processes a JSONL file to extract unique instructions and their last seen data.
    - Reads from 'dataset/curated-preference-dataset.jsonl'.
    - Creates a dictionary where keys are unique instruction strings and
      values are the last JSON object encountered for that instruction.
    - Saves this dictionary to 'dataset/preference-dataset.json'.
    - Also collects a list of all instruction strings encountered.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(script_dir, "../dataset")

        input_file_path = os.path.join(dataset_folder, "curated-preference-dataset.jsonl")
        output_file_path = os.path.join(dataset_folder, "preference-dataset.json")

        all_instructions_list = []
        preference_data_map = {}  # To store {"instruction_string": last_seen_json_object}
        lines_processed_count = 0
        valid_json_objects_count = 0

        print(f"Attempting to read from: {input_file_path}")

        if not os.path.exists(input_file_path):
            print(f"Error: Input file not found at {input_file_path}")
            return

        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line_number, line_content in enumerate(infile, 1):
                lines_processed_count += 1
                stripped_line = line_content.strip()
                if not stripped_line:
                    print(f"Info: Skipping empty line at line number {line_number}.")
                    continue
                
                try:
                    data_item = json.loads(stripped_line)
                    valid_json_objects_count += 1
                    
                    if "instruction" in data_item:
                        instruction_text = data_item["instruction"]
                        all_instructions_list.append(instruction_text)
                        # Update the map with the latest data_item for this instruction
                        preference_data_map[instruction_text] = data_item
                    else:
                        print(f"Warning: 'instruction' key missing in JSON object at line {line_number}. Object: {str(data_item)[:100]}...")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line {line_number}. Content: {stripped_line[:100]}...")

        print(f"Total lines read from input file: {lines_processed_count}")
        print(f"Valid JSON objects processed: {valid_json_objects_count}")
        
        # For information: a structure containing all instructions encountered (including duplicates)
        instructions_summary_structure = {
            "all_instructions_encountered": all_instructions_list
        }
        print(f"Collected {len(all_instructions_list)} instruction instances (includes duplicates).")
        print(f"Found {len(preference_data_map)} unique instructions.")

        if preference_data_map: # If the map is not empty
            final_dpo_list = []
            for key in preference_data_map:
                data_item = preference_data_map[key]
                dpo_item = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": data_item["instruction"]
                        },
                    ],
                    "chosen": {
                        "from": "assistant",
                        "value": data_item["response_better"]
                    },
                    "rejected": {
                        "from": "assistant",
                        "value": data_item["response_worse"]
                    }
                }
                final_dpo_list.append(dpo_item)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            try:
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    json.dump(final_dpo_list, outfile, indent=4)
                print(f"Successfully saved data for {len(final_dpo_list)} unique instructions to: {output_file_path}")
            except IOError as e:
                print(f"Error writing output file {output_file_path}: {e}")
        else:
            if valid_json_objects_count == 0 and lines_processed_count > 0 :
                 print(f"Warning: Input file {input_file_path} was read but contained no valid JSON objects with 'instruction' keys.")
            elif not os.path.exists(input_file_path):
                 print(f"Error: Input file {input_file_path} seems to have disappeared or was not found initially.")
            else:
                 print(f"Warning: No valid JSON data with 'instruction' keys processed from {input_file_path}.")
            print("No data to write to preference-dataset.json.")

    except FileNotFoundError:
        print(f"Critical Error: Could not determine script path or input file path correctly.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_preference_data() 