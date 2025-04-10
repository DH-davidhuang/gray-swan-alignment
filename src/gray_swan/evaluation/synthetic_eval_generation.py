import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List
import math
import os
import json
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List

class SyntheticEvaluator:
    """
    A class-based approach to run an evaluation on 
    a synthetic dataset with 'benign' and 'harmful' prompts.
    It loads two models (original + post-trained),
    generates responses for each category in batches,
    then saves them to a JSON file with a 'label' field.
    """

    def __init__(
        self,
        orig_checkpoint: str,
        tuned_checkpoint: str,
        eval_path: str,
        device: str = "cuda",
        output_file: str = "model_comparison.json",
        batch_size: int = 8
    ):
        """
        Initialize and load the models + tokenizer, plus store paths/params.
        """
        self.orig_checkpoint = orig_checkpoint
        self.tuned_checkpoint = tuned_checkpoint
        self.eval_path = eval_path
        self.device = device
        self.output_file = output_file
        self.batch_size = batch_size

        self.orig_model, self.tuned_model, self.tokenizer = self.load_two_models(
            self.orig_checkpoint, self.tuned_checkpoint, device=self.device
        )


    def load_two_models(
        self,
        orig_checkpoint: str,
        tuned_checkpoint: str,
        device: str = "cuda"
    ):
        """
        Loads original and tuned models + a shared tokenizer (assuming they are compatible).
        Returns (orig_model, tuned_model, tokenizer).
        """
        print(f"[SyntheticEvaluator] Loading original model from {orig_checkpoint} ...")
        orig_model = AutoModelForCausalLM.from_pretrained(orig_checkpoint).to(device)
        orig_model.eval()

        print(f"[SyntheticEvaluator] Loading post-trained model from {tuned_checkpoint} ...")
        tuned_model = AutoModelForCausalLM.from_pretrained(tuned_checkpoint).to(device)
        tuned_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(orig_checkpoint)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        tokenizer.padding_side = "left"

        return orig_model, tuned_model, tokenizer

    def generate_responses_batch(
        self,
        model,
        prompts: list[str],
        max_new_tokens=200,
    ) -> list[str]:
        """
        Generates responses from 'model' for each prompt in 'prompts', 
        processing up to self.batch_size prompts at once.
        
        Returns list of strings (the model-generated responses).
        """
        results = []
        num_prompts = len(prompts)
        num_batches = math.ceil(num_prompts / self.batch_size)

        model.eval()
        with torch.no_grad():
            for b_idx in tqdm(range(num_batches), desc="Generating responses in batches"):
                start = b_idx * self.batch_size
                end = min(start + self.batch_size, num_prompts)
                batch_prompts = prompts[start:end]

                # 1. Build strings for each prompt (per your chat template).
                batch_texts = []
                for prompt in batch_prompts:
                    messages = [{"role": "user", "content": prompt}]
                    input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    batch_texts.append(input_text)

                # 2. Tokenize as a single batch. The tokenizer will pad automatically.
                batch_enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                # 3. Generate for the batch.
                gen_out = model.generate(
                    **batch_enc,
                    max_new_tokens=max_new_tokens,
                )

                # 4. Decode each generation in the batch.
                for gen_seq in gen_out:
                    text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
                    results.append(text)

        return results

    def run_evaluation(self):
        """
        Main method that:
          1) Reads the synthetic eval dataset with 'benign'/'harmful' prompts
          2) Generates responses in batches for each model
          3) Saves results to JSON with 'label' field
          4) Prints a small sample
        """
        with open(self.eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        benign_prompts = data["benign"]
        harmful_prompts = data["harmful"]

        # 3) Generate responses for BENIGN
        print(f"Generating for {len(benign_prompts)} benign prompts...")
        orig_benign_resps = self.generate_responses_batch(
            self.orig_model,
            benign_prompts,
            max_new_tokens=200
        )
        tuned_benign_resps = self.generate_responses_batch(
            self.tuned_model,
            benign_prompts,
            max_new_tokens=200
        )

        # 4) Generate responses for HARMFUL
        print(f"Generating for {len(harmful_prompts)} harmful prompts...")
        orig_harmful_resps = self.generate_responses_batch(
            self.orig_model,
            harmful_prompts,
            max_new_tokens=200
        )
        tuned_harmful_resps = self.generate_responses_batch(
            self.tuned_model,
            harmful_prompts,
            max_new_tokens=200
        )

        results = []

        for i, prompt in enumerate(benign_prompts):
            results.append({
                "label": "benign",
                "prompt": prompt,
                "original_model_response": orig_benign_resps[i],
                "tuned_model_response": tuned_benign_resps[i]
            })

        for i, prompt in enumerate(harmful_prompts):
            results.append({
                "label": "harmful",
                "prompt": prompt,
                "original_model_response": orig_harmful_resps[i],
                "tuned_model_response": tuned_harmful_resps[i]
            })

        out_dir = os.path.dirname(self.output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n[SyntheticEvaluator] Saved synthetic eval comparison to {self.output_file}. Here's a sample:")

        for item in results[:3]:
            print(f"\nLabel: {item['label']}")
            print(f"Prompt: {item['prompt']}")
            print(f"  Original => {item['original_model_response']}")
            print(f"  Tuned    => {item['tuned_model_response']}")


def main():

    evaluator = SyntheticEvaluator(
        orig_checkpoint="/data1/shared_models/SmolLM2-135M-Instruct", # change this to wherever your model is
        tuned_checkpoint="models/dpo_utility_finetuned",
        eval_path="data/synthetic_eval/synthetic_dataset.json",
        device="cuda:2",
        output_file="full_dpo_utility_model_comparison.json",
        batch_size=8
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()