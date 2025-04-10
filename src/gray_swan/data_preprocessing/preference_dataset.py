import json
import os

def load_jsonl(file_path: str):
    """
    Load a JSONL file and return a list of JSON objects.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def combine_preference_data(good_file: str, bad_file: str):
    """
    Combines the good (preferred) and bad (rejected) datasets into a single list of examples.
    
    Each entry is a dictionary with keys:
    - prompt: the user prompt from the first message.
    - chosen: the assistant response from the good (preferred) dataset.
    - rejected: the assistant response from the bad (unpreferred) dataset.
    
    Assumes that the good and bad files are aligned (i.e. the Nth entry in both corresponds to the same prompt).
    """
    good_data = load_jsonl(good_file)
    bad_data  = load_jsonl(bad_file)
    
    if len(good_data) != len(bad_data):
        raise ValueError("Good and Bad datasets are not of the same length.")
    
    combined = []
    for good_entry, bad_entry in zip(good_data, bad_data):
        prompt = good_entry["messages"][0]["content"]
        chosen = good_entry["messages"][1]["content"]
        rejected = bad_entry["messages"][1]["content"]
        combined.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    return combined

if __name__ == "__main__":
    # expects there to be a data/raw directory with the preference data
    base_dir = os.path.join("data", "raw", "preference")
    good_file = os.path.join(base_dir, "Llama-3.1-8B-Instruct-good.jsonl")
    bad_file = os.path.join(base_dir, "Llama-3.1-8B-Instruct-bad.jsonl")
    
    preference_data = combine_preference_data(good_file, bad_file)
    
    print("First example in the combined preference dataset:")
    print("Prompt:", preference_data[0]["prompt"])
    print("Chosen (Preferred):", preference_data[0]["chosen"])
    print("Rejected (Harmful):", preference_data[0]["rejected"])

    output_file = os.path.join(base_dir, "combined_preference_data.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in preference_data:
            f.write(json.dumps(entry) + "\n")
