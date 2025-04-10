import os
import json
import torch
import torch.nn.functional as F
from colorama import Fore, Style, init
from transformers import AutoModelForCausalLM, AutoTokenizer

init(autoreset=True)


def extract_assistant_text(full_response: str):
    """
    Finds the last occurrence of "assistant\n" and returns everything after that.
    If not found, return the entire string unmodified.
    """
    marker = "assistant\n"
    idx = full_response.rfind(marker)
    if idx == -1:
        return full_response
    return full_response[idx + len(marker):]


def compute_completion_logprob(model, ids: torch.LongTensor, prompt_len: int):
    """
    Given a single concatenation of prompt+completion in 'ids' (shape [seq_len]),
    we mask out the first 'prompt_len' tokens from the loss and compute the log-prob
    of the 'completion' portion only.
    """
    ids = ids.unsqueeze(0)  # batch dimension
    labels = ids.clone()
    labels[0, :prompt_len] = -100  # mask prompt

    with torch.no_grad():
        outputs = model(ids, labels=labels)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        mask = (shift_labels.view(-1) != -100).float()
        sum_ce = (per_token_ce * mask).sum()
        logprob = -sum_ce
        return logprob


def token_attribution_occlusion(model, tokenizer, prompt: str, completion: str, device="cuda"):
    """
    For each token in the prompt, replace it with [UNK], measure the difference
    in log-prob of 'completion' => importance score.
    Returns [(token_str, float_attribution)] in prompt token order.
    """
    model.eval()
    model.to(device)

    full_text = prompt + completion
    enc_full = tokenizer(full_text, return_tensors='pt')
    full_ids = enc_full["input_ids"][0].to(device)

    enc_prompt = tokenizer(prompt, return_tensors='pt')
    prompt_ids = enc_prompt["input_ids"][0]
    prompt_len = prompt_ids.size(0)

    
    baseline_lp = compute_completion_logprob(model, full_ids, prompt_len)
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)

    unk_id = tokenizer.unk_token_id
    if unk_id is None:
        unk_id = tokenizer.pad_token_id or 0

    attributions = []
    for i in range(prompt_len):
        occluded = full_ids.clone()
        occluded[i] = unk_id
        occl_lp = compute_completion_logprob(model, occluded, prompt_len)
        diff = baseline_lp - occl_lp
        attributions.append((prompt_tokens[i], diff.item()))
    return attributions


def colorize_prompt_tokens(tokens_with_scores):
    """
    Color tokens from green->yellow->red based on the score, 
    skipping certain special tokens, etc.
    """
    if not tokens_with_scores:
        return ""
    max_score = max(s for _, s in tokens_with_scores if s is not None)
    if max_score <= 0:
        max_score = 1.0  # fallback to avoid zero dividing

    colored_text = ""
    for tok, score in tokens_with_scores:
        norm = max(0.0, min(1.0, score / max_score))
        if norm > 0.7:
            color = Fore.RED
        elif norm > 0.4:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN

        display_tok = tok
        if display_tok.startswith("Ġ"):
            display_tok = " " + display_tok[1:]

        if display_tok.startswith("<") and display_tok.endswith(">"):
            continue

        colored_text += f"{color}{display_tok}{Style.RESET_ALL}"
    return colored_text

def colorize_tokens_html(tokens_with_scores):
    """
    Returns an HTML string that colorizes tokens 
    from green -> yellow -> red based on the score.
    """
    if not tokens_with_scores:
        return ""

    max_score = max(s for _, s in tokens_with_scores if s is not None)
    if max_score <= 0:
        max_score = 1.0

    html_fragments = []
    for tok, score in tokens_with_scores:
        norm = max(0.0, min(1.0, score / max_score))  # 0..1 scale

        r = int(255 * norm)
        g = int(255 * (1 - norm))
        b = 0
        color_hex = f"#{r:02x}{g:02x}{b:02x}"

        display_tok = tok
        if display_tok.startswith("Ġ"):
            display_tok = " " + display_tok[1:]
        if display_tok.startswith("<") and display_tok.endswith(">"):
            continue

        html_fragments.append(
            f'<span style="color: {color_hex};">'
            f"{display_tok}"
            f"</span>"
        )

    return "".join(html_fragments)


class AttributionAnalyzer:
    """
    Class-based approach encapsulating:
      1) Loading original & tuned model + tokenizer
      2) Reading from the model_comparison.json
      3) Doing token-level occlusion for each entry
      4) Printing colorized results
    """

    def __init__(
        self,
        orig_checkpoint: str,
        tuned_checkpoint: str,
        comparison_path: str,
        device: str = "cuda"
    ):
        """
        :param orig_checkpoint: path to original model
        :param tuned_checkpoint: path to post-trained model
        :param comparison_path: path to JSON with entries
        :param device: e.g. 'cuda' or 'cpu'
        """
        self.orig_checkpoint = orig_checkpoint
        self.tuned_checkpoint = tuned_checkpoint
        self.comparison_path = comparison_path
        self.device = device

        # Load models
        print(f"[AttributionAnalyzer] Loading original model: {orig_checkpoint}")
        self.orig_model = AutoModelForCausalLM.from_pretrained(orig_checkpoint).to(device)
        self.orig_model.eval()

        print(f"[AttributionAnalyzer] Loading tuned model: {tuned_checkpoint}")
        self.tuned_model = AutoModelForCausalLM.from_pretrained(tuned_checkpoint).to(device)
        self.tuned_model.eval()

        # Load tokenizer from original checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(orig_checkpoint)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Read data
        print(f"[AttributionAnalyzer] Reading from {comparison_path} ...")
        with open(comparison_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        print(f"[AttributionAnalyzer] Found {len(self.entries)} entries.")

    def run_analysis(self, max_count: int = 10):
        """
        Loops over first 'max_count' entries, does token occlusion 
        for both original & tuned model, prints results.
        """
        for idx, entry in enumerate(self.entries[:max_count], start=1):
            prompt = entry["prompt"]
            orig_full = entry["original_model_response"]
            tuned_full = entry["tuned_model_response"]

            # Parse out the assistant portion
            orig_completion = extract_assistant_text(orig_full)
            tuned_completion = extract_assistant_text(tuned_full)

            print("=" * 80)
            print(f"Entry #{idx}")
            print(f"PROMPT:\n{prompt}")

            # 1) Original model attribution
            orig_attribs = token_attribution_occlusion(
                self.orig_model,
                self.tokenizer,
                prompt,
                orig_completion,
                device=self.device
            )
            colored_orig = colorize_prompt_tokens(orig_attribs)
            print("\n--- Original Model's Completion (truncated) ---")
            print(orig_completion[:200], "...\n")
            print("Attribution =>", colored_orig)

            # 2) Tuned model attribution
            tuned_attribs = token_attribution_occlusion(
                self.tuned_model,
                self.tokenizer,
                prompt,
                tuned_completion,
                device=self.device
            )
            colored_tuned = colorize_prompt_tokens(tuned_attribs)
            print("\n--- Tuned Model's Completion (truncated) ---")
            print(tuned_completion[:200], "...\n")
            print("Attribution =>", colored_tuned)
            print()

        print("[AttributionAnalyzer] Done.\n")
            
    def run_analysis_html(self, max_count: int = 5):
            """
            Like run_analysis(), but produce inline HTML for color-coded tokens, 
            and display them in the notebook via IPython.display.
            """
            from IPython.display import display, HTML

            for idx, entry in enumerate(self.entries[:max_count], start=1):
                prompt = entry["prompt"]
                orig_full = entry["original_model_response"]
                tuned_full = entry["tuned_model_response"]

                orig_completion = extract_assistant_text(orig_full)
                tuned_completion = extract_assistant_text(tuned_full)

                orig_attribs = token_attribution_occlusion(
                    self.orig_model,
                    self.tokenizer,
                    prompt,
                    orig_completion,
                    device=self.device
                )
                tuned_attribs = token_attribution_occlusion(
                    self.tuned_model,
                    self.tokenizer,
                    prompt,
                    tuned_completion,
                    device=self.device
                )

                html_orig = colorize_tokens_html(orig_attribs)
                html_tuned = colorize_tokens_html(tuned_attribs)

                block_html = f"""
                <div style="border:1px solid #ccc; padding:1em; margin-bottom:1em;">
                <h3>Entry #{idx}</h3>
                <p><strong>Prompt:</strong><br>{prompt}</p>

                <p><strong>Original Model's Completion (truncated)</strong><br>
                {orig_completion[:200]}...</p>
                <p><strong>Original Model Attribution:</strong><br>
                <code>{html_orig}</code></p>

                <p><strong>Tuned Model's Completion (truncated)</strong><br>
                {tuned_completion[:200]}...</p>
                <p><strong>Tuned Model Attribution:</strong><br>
                <code>{html_tuned}</code></p>
                </div>
                """
                display(HTML(block_html))

def main():
    comparison_path = "/home/davidh/gray-swan-alignment/5_example_model_comparison.json"
    orig_checkpoint = "/data1/shared_models/SmolLM2-135M-Instruct"
    tuned_checkpoint = "/home/davidh/gray-swan-alignment/src/gray_swan/dpo_training/models/dpo_finetuned_grad_accum"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    analyzer = AttributionAnalyzer(
        orig_checkpoint=orig_checkpoint,
        tuned_checkpoint=tuned_checkpoint,
        comparison_path=comparison_path,
        device=device
    )
    analyzer.run_analysis(max_count=10)

if __name__ == "__main__":
    main()
