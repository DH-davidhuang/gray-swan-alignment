import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

def build_conversation_batch(
    system_texts, user_texts, assistant_texts, tokenizer, device
):
    """
    Builds a BATCH of conversation inputs:
      System: system_text[i]
      User:   user_text[i]
      Assistant: assistant_text[i]
    for i in [0..batch_size-1].

    Returns:
      input_ids: shape (B, T) (padded)
      prompt_lens: list of length B
        prompt_lens[i] is how many tokens in example i
        belong to (system+user) so that assistant tokens
        start at prompt_lens[i].
    """
    input_ids_list = []
    prompt_lens = []

    for sys_txt, usr_txt, asst_txt in zip(system_texts, user_texts, assistant_texts):
        # 1) system+user prompt
        prompt_messages = []
        if sys_txt:
            prompt_messages.append({"role": "system", "content": sys_txt})
        prompt_messages.append({"role": "user", "content": usr_txt})

        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        prompt_enc = tokenizer(prompt_text, return_tensors="pt")
        prompt_ids = prompt_enc["input_ids"][0]
        prompt_len = prompt_ids.size(0)

        # 2) full conversation = system+user+assistant
        full_messages = list(prompt_messages)
        full_messages.append({"role": "assistant", "content": asst_txt})

        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        full_enc = tokenizer(full_text, return_tensors="pt")
        full_ids = full_enc["input_ids"][0]

        input_ids_list.append(full_ids)
        prompt_lens.append(prompt_len)

    # Pad to the same length
    # Find max length in this batch
    max_len = max(seq.size(0) for seq in input_ids_list)
    padded_input_ids = []
    for seq in input_ids_list:
        pad_size = max_len - seq.size(0)
        # pad on the right with pad_token_id
        pad_ids = torch.full((pad_size,), tokenizer.pad_token_id, dtype=torch.long)
        padded_seq = torch.cat([seq, pad_ids], dim=0)
        padded_input_ids.append(padded_seq)

    # stack => shape (B, T)
    padded_input_ids = torch.stack(padded_input_ids, dim=0).to(device)

    return padded_input_ids, prompt_lens

def compute_response_logprob_batch(model, input_ids, prompt_lens, tokenizer):
    """
    input_ids: (B, T) int64
    prompt_lens: list of length B with #tokens for system+user
                 so assistant tokens start at prompt_lens[i].
    Returns:
      log_probs: shape (B,) => sum log-prob of each sample's assistant content
    """

    B, T = input_ids.shape

    labels = input_ids.clone()

    for i in range(B):
        labels[i, : prompt_lens[i]] = -100

    special_ids = []
    for special_token in [
        "<|im_start|>assistant",
        "<|im_end|>",
        # Add others if needed
    ]:
        tid = tokenizer.convert_tokens_to_ids(special_token)
        if isinstance(tid, int) and tid not in special_ids and tid != tokenizer.unk_token_id:
            special_ids.append(tid)

    for i in range(B):
        row_len = input_ids[i].size(0)
        for idx in range(prompt_lens[i], row_len):
            if labels[i, idx].item() in special_ids:
                labels[i, idx] = -100

    outputs = model(input_ids, labels=labels)

  
    shift_logits = outputs.logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()              # (B, T-1)


    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )

    valid_mask = (shift_labels.view(-1) != -100).float()
    loss_per_token = loss_per_token * valid_mask

    # reshape back to (B, T-1) and sum along dim=1
    loss_per_token = loss_per_token.view(B, -1)
    sum_ce = loss_per_token.sum(dim=1) 

    # convert from cross-entropy to log-prob
    log_probs = -sum_ce
    return log_probs  


def dpo_loss_batch(
    model,
    ref_model,
    tokenizer,
    system_texts,
    user_texts,
    chosen_texts,
    rejected_texts,
    beta=1.0
):
    """
    system_texts, user_texts, chosen_texts, rejected_texts => lists of length B
    Returns a shape (B,) tensor of losses for each sample in the batch.
    """
    # 1) Build input_ids for chosen and rejected
    chosen_input_ids, chosen_prompt_lens = build_conversation_batch(
        system_texts, user_texts, chosen_texts, tokenizer, model.device
    )
    rejected_input_ids, rejected_prompt_lens = build_conversation_batch(
        system_texts, user_texts, rejected_texts, tokenizer, model.device
    )

    logp_chosen = compute_response_logprob_batch(
        model, chosen_input_ids, chosen_prompt_lens, tokenizer
    )
    logp_rejected = compute_response_logprob_batch(
        model, rejected_input_ids, rejected_prompt_lens, tokenizer
    )
    # 2) Compute log-prob for reference model
    # We don't need to compute the log-prob for the reference model
    # for the rejected input_ids, since we only need the difference
    # between the two log-probs.
    # We can compute the log-prob for the chosen input_ids and
    # the rejected input_ids in a single batch.
    with torch.no_grad():
        ref_logp_chosen = compute_response_logprob_batch(
            ref_model, chosen_input_ids, chosen_prompt_lens, tokenizer
        )
        ref_logp_rejected = compute_response_logprob_batch(
            ref_model, rejected_input_ids, rejected_prompt_lens, tokenizer
        )

    diff = beta * ((logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected))
    
    loss_batch = -F.logsigmoid(diff)
    return loss_batch  


def train_dpo_grad_accum(
    checkpoint: str,
    data_path: str,
    beta: float = 1.0,
    lr: float = 1e-4,
    epochs: int = 1,
    desired_batch_size: int = 16,    
    micro_batch_size: int = 4,  
    device: str = "cuda",
    log_file: str = "logs/dpo_training.log"
):
    """
    Train a model using DPO with a two-level approach:
      - micro_batch_size at a time on GPU
      - accumulate gradients
      - step the optimizer once per 'desired_batch_size' examples.

    checkpoint: path to instruct-tuned base model
    data_path: path to JSONL with {system, prompt, chosen, rejected}
    beta: preference hyperparam
    lr: learning rate
    epochs: number of epochs
    desired_batch_size: the total batch size you want for each optimizer step
    micro_batch_size: how many we can realistically fit in GPU at once
    device: "cuda" or "cpu"
    log_file: path to store logs
    """

    # 1) Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"Loading target model from {checkpoint} ...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.train()

    print(f"Loading reference model from {checkpoint} ...")
    ref_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 2) Load dataset
    dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            dataset.append(example)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3) Setup logging
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logf = open(log_file, "w", encoding="utf-8")

    gradient_accumulation_steps = desired_batch_size // micro_batch_size
    assert desired_batch_size % micro_batch_size == 0, (
        "desired_batch_size should be divisible by micro_batch_size!"
    )

    print(
        f"Using micro_batch_size={micro_batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, "
        f"for an effective batch size of {desired_batch_size}."
    )

    # main training loop
    num_samples = len(dataset)
    steps_per_epoch = (num_samples + desired_batch_size - 1) // desired_batch_size
    global_step = 0
    total_loss = 0.0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        logf.write(f"Epoch {epoch+1}/{epochs}\n")
        random.shuffle(dataset)

        step_count_this_epoch = 0
        total_loss_this_epoch = 0.0

        for start_idx in range(0, num_samples, desired_batch_size):
            end_idx = start_idx + desired_batch_size
            big_batch = dataset[start_idx:end_idx]

            micro_steps = (len(big_batch) + micro_batch_size - 1) // micro_batch_size

            # zero the gradients once per 'big batch'
            model.zero_grad()

            for micro_step in range(micro_steps):
                mb_start = micro_step * micro_batch_size
                mb_end = mb_start + micro_batch_size
                micro_batch = big_batch[mb_start:mb_end]
                if len(micro_batch) == 0:
                    break

                # build lists for dpo_loss_batch
                system_texts = [ex.get("system", "") for ex in micro_batch]
                user_texts   = [ex["prompt"] for ex in micro_batch]
                chosen_texts = [ex["chosen"] for ex in micro_batch]
                rejected_texts = [ex["rejected"] for ex in micro_batch]

                loss_vec = dpo_loss_batch(
                    model, ref_model, tokenizer,
                    system_texts, user_texts, chosen_texts, rejected_texts,
                    beta=beta
                )
                # shape (mb_size,)
                loss = loss_vec.mean()

                # scale the loss by 1/gradient_accumulation_steps
                # so that the effective gradient is the same as if we fed the entire big batch at once
                loss = loss / gradient_accumulation_steps

                loss.backward()

                # we accumulate gradients, so we don't step optimizer until we finish all micro-batches
                total_loss += loss.item() * gradient_accumulation_steps 
                total_loss_this_epoch += loss.item() * gradient_accumulation_steps

            optimizer.step()

            step_count_this_epoch += 1
            global_step += 1

            if step_count_this_epoch % 10 == 0:
                avg_loss = total_loss_this_epoch / step_count_this_epoch
                print(f"  Epoch {epoch+1}, step {step_count_this_epoch}/{steps_per_epoch}, avg_loss={avg_loss:.4f}")
                logf.write(f"Epoch {epoch+1}, step {step_count_this_epoch}, avg_loss={avg_loss:.4f}\n")

        final_epoch_loss = total_loss_this_epoch / max(1, step_count_this_epoch)
        print(f"End of epoch {epoch+1}, average loss={final_epoch_loss:.4f}")
        logf.write(f"Epoch {epoch+1} average loss={final_epoch_loss:.4f}\n")

    logf.close()

    output_dir = "models/dpo_finetuned_grad_accum"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete. Model saved to", output_dir)

if __name__ == "__main__":
    train_dpo_grad_accum(
        checkpoint="/data1/shared_models/SmolLM2-135M-Instruct",
        data_path="data/raw/preference/combined_preference_data.jsonl", 
        beta=0.5,
        lr=1e-4,
        epochs=10,
        desired_batch_size=64,
        micro_batch_size=1,
        device="cuda:3",
        log_file="logs/dpo_training_grad_accum.log"
    )
 