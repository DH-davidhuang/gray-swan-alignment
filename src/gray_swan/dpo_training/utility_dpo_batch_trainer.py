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
    max_len = max(seq.size(0) for seq in input_ids_list)
    padded_input_ids = []
    for seq in input_ids_list:
        pad_size = max_len - seq.size(0)
        pad_ids = torch.full((pad_size,), tokenizer.pad_token_id, dtype=torch.long)
        padded_seq = torch.cat([seq, pad_ids], dim=0)
        padded_input_ids.append(padded_seq)

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

    # 1) Mask out the prompt region
    for i in range(B):
        labels[i, : prompt_lens[i]] = -100

    # 2) Also mask out special tokens in the assistant portion
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
    # set last token to -100
    labels[:, -1] = -100
    # find the starting mask of the -100 tokens 
    # and set them to -100 basically setting assistant to 0 
    B, L = labels.size()
    for i in range(B):
        row = labels[i]
        seq_len = row.size(0)
        end_of_run = 0
        while end_of_run < seq_len and row[end_of_run].item() == -100:
            end_of_run += 1
        if end_of_run < seq_len:
            row[end_of_run] = -100
        if end_of_run + 1 < seq_len:
            row[end_of_run + 1] = -100
        if end_of_run + 2 < seq_len:
            row[end_of_run + 2] = -100
    

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

    chosen_input_ids, chosen_prompt_lens = build_conversation_batch(
        system_texts, user_texts, chosen_texts, tokenizer, model.device
    )
    rejected_input_ids, rejected_prompt_lens = build_conversation_batch(
        system_texts, user_texts, rejected_texts, tokenizer, model.device
    )

    # model log-probs
    logp_chosen = compute_response_logprob_batch(
        model, chosen_input_ids, chosen_prompt_lens, tokenizer
    )
    logp_rejected = compute_response_logprob_batch(
        model, rejected_input_ids, rejected_prompt_lens, tokenizer
    )

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


def utility_loss_batch(
    model,
    tokenizer,
    user_texts,
    assistant_texts,
    device
):
    """
    For a batch of user_texts, assistant_texts => compute a standard next-token
    cross-entropy. (We can treat it like a conversation with just user->assistant.)

    Returns shape (B,) of cross-entropies. We'll do a simple approach:
      - build user+assistant conversation
      - mask out the user portion for the loss
    """
    # build batch
    # we assume no system message, or empty
    system_texts = ["" for _ in user_texts]
    input_ids, prompt_lens = build_conversation_batch(
        system_texts, user_texts, assistant_texts, tokenizer, device
    )

    B, T = input_ids.shape
    labels = input_ids.clone()
    special_ids = []
    for special_token in [
        "<|im_start|>assistant\n",
        "<|im_end|>",
        "<|im_start|>",
        "assistant\n",
        # Add others as needed
    ]:
        tid = tokenizer.convert_tokens_to_ids(special_token)
        if isinstance(tid, int) and tid not in special_ids and tid != tokenizer.unk_token_id:
            special_ids.append(tid)

    for i in range(B):
        row_len = input_ids[i].size(0)
        for idx in range(prompt_lens[i], row_len):
            if labels[i, idx].item() in special_ids:
                labels[i, idx] = -100

    for i in range(B):
       labels[i, : prompt_lens[i]] = -100
    
    # set last token to -100
    labels[:, -1] = -100
    # find the starting mask of the -100 tokens 
    # and set them to -100 basically setting assistant to 0 
    B, L = labels.size()
    for i in range(B):
        row = labels[i]
        seq_len = row.size(0)
        end_of_run = 0
        while end_of_run < seq_len and row[end_of_run].item() == -100:
            end_of_run += 1
        if end_of_run < seq_len:
            row[end_of_run] = -100
        if end_of_run + 1 < seq_len:
            row[end_of_run + 1] = -100
        if end_of_run + 2 < seq_len:
            row[end_of_run + 2] = -100

    outputs = model(input_ids, labels=labels)

    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )

    loss_per_token = loss_per_token.view(B, -1).mean(dim=1)
    return loss_per_token



def train_dpo_with_utility_grad_accum(
    checkpoint: str,
    data_path: str,
    utility_data_path: str,
    alpha: float = 0.2,
    beta: float = 1.0,
    lr: float = 1e-4,
    epochs: int = 1,
    desired_batch_size: int = 16,
    micro_batch_size: int = 4,
    device: str = "cuda",
    log_file: str = "logs/dpo_utility_training.log"
):
    """
    Train a model using DPO + Utility next-token loss, 
    with gradient accumulation. Final loss = (1-alpha)*DPO_loss + alpha*Utility_loss.

    :param checkpoint: path to instruct-tuned base model
    :param data_path: path to JSONL with {system, prompt, chosen, rejected} (for DPO)
    :param utility_data_path: path to JSONL with "messages" => user+assistant, e.g. alpaca_cleaned
    :param alpha: weighting for utility portion [0..1]
    :param beta: DPO preference hyperparam
    :param lr: learning rate
    :param epochs: number of epochs
    :param desired_batch_size: total batch size for each optimizer step
    :param micro_batch_size: micro-batch size
    :param device: "cuda" or "cpu"
    :param log_file: path to store logs
    """

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[train_dpo_with_utility] Loading target model from {checkpoint} ...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    model.train()

    print(f"[train_dpo_with_utility] Loading reference model from {checkpoint} ...")
    ref_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 1) Load preference (DPO) dataset
    pref_dataset = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            pref_dataset.append(ex)

    # 2) Load utility dataset (Alpaca-like)
    utility_dataset = []
    with open(utility_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            utility_dataset.append(ex)

    # We'll assume each ex in utility_dataset has a structure like:
    # {
    #   "messages": [
    #       {"role": "user", "content": "..."},
    #       {"role": "assistant", "content": "..."}
    #   ]
    # }
    # We'll just parse them

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3) Logging
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logf = open(log_file, "w", encoding="utf-8")

    gradient_accumulation_steps = desired_batch_size // micro_batch_size
    assert desired_batch_size % micro_batch_size == 0, (
        "desired_batch_size should be divisible by micro_batch_size!"
    )

    print(f"[train_dpo_with_utility] alpha={alpha}, beta={beta}, micro_batch_size={micro_batch_size}, accum={gradient_accumulation_steps}")
    logf.write(f"alpha={alpha}, beta={beta}\n")

    # We'll just do a simple approach: shuffle both datasets, 
    # and sample pairs of micro-batches: one from pref, one from utility
    # Then do final_loss = (1-alpha)*dpo_loss + alpha*utility_loss

    num_samples_pref = len(pref_dataset)
    num_samples_util = len(utility_dataset)
    print(f"Loaded pref dataset size={num_samples_pref}, util size={num_samples_util}")

    steps_per_epoch = (num_samples_pref + desired_batch_size - 1) // desired_batch_size
    steps_per_epoch = min(steps_per_epoch, (num_samples_util + desired_batch_size - 1)//desired_batch_size)
    print(f"[train_dpo_with_utility] steps_per_epoch={steps_per_epoch}")

    global_step = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        logf.write(f"Epoch {epoch+1}/{epochs}\n")

        random.shuffle(pref_dataset)
        random.shuffle(utility_dataset)

        # We'll slice them to the same length for demonstration
        max_len = min(num_samples_pref, num_samples_util)
        pref_dataset_this = pref_dataset[:max_len]
        util_dataset_this = utility_dataset[:max_len]

        step_count_epoch = 0
        total_loss_this_epoch = 0.0

        for start_idx in range(0, max_len, desired_batch_size):
            end_idx = start_idx + desired_batch_size
            # big batch from preference
            pref_big_batch = pref_dataset_this[start_idx:end_idx]
            # big batch from utility
            util_big_batch = util_dataset_this[start_idx:end_idx]

            # micro steps
            micro_steps = (len(pref_big_batch) + micro_batch_size - 1)//micro_batch_size

            model.zero_grad()
            for micro_step in range(micro_steps):
                mb_start = micro_step * micro_batch_size
                mb_end = mb_start + micro_batch_size
                p_batch = pref_big_batch[mb_start:mb_end]
                u_batch = util_big_batch[mb_start:mb_end]

                if len(p_batch) == 0 or len(u_batch) == 0:
                    break

                # 4) compute DPO loss for pref
                # build lists
                system_texts = [ex.get("system","") for ex in p_batch]
                user_texts = [ex["prompt"] for ex in p_batch]
                chosen_texts = [ex["chosen"] for ex in p_batch]
                rejected_texts = [ex["rejected"] for ex in p_batch]

                pref_loss_vec = dpo_loss_batch(
                    model, ref_model, tokenizer,
                    system_texts, user_texts, chosen_texts, rejected_texts,
                    beta=beta
                ) 

        
                user_texts_u = []
                assistant_texts_u = []
                for ex_util in u_batch:
                    msgs = ex_util["messages"]
                    user_msg = next((m for m in msgs if m["role"]=="user"), {"content":""})
                    assistant_msg = next((m for m in msgs if m["role"]=="assistant"), {"content":""})
                    user_texts_u.append(user_msg["content"])
                    assistant_texts_u.append(assistant_msg["content"])

                util_loss_vec = utility_loss_batch(
                    model, tokenizer,
                    user_texts_u, assistant_texts_u, device
                )  

                dpo_loss_avg = pref_loss_vec.mean()
                util_loss_avg = util_loss_vec.mean()
                combined_loss = (1 - alpha)*dpo_loss_avg + alpha*util_loss_avg

                # accum gradients here
                combined_loss = combined_loss / gradient_accumulation_steps
                combined_loss.backward()

                total_loss_this_epoch += combined_loss.item() * gradient_accumulation_steps

            # step after finishing mini batching microsteps
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_count_epoch += 1
            global_step += 1

            if step_count_epoch % 10 == 0:
                avg_loss = total_loss_this_epoch / step_count_epoch
                print(f"  Step {step_count_epoch}/{steps_per_epoch} in epoch {epoch+1}, avg_loss={avg_loss:.4f}")
                logf.write(f"Epoch {epoch+1}, step {step_count_epoch}, avg_loss={avg_loss:.4f}\n")

        # end of epoch
        final_epoch_loss = total_loss_this_epoch / max(1, step_count_epoch)
        print(f"End of epoch {epoch+1}, average loss={final_epoch_loss:.4f}")
        logf.write(f"Epoch {epoch+1} average loss={final_epoch_loss:.4f}\n")

    logf.close()
    # import pdb; pdb.set_trace()

    # Save final
    output_dir = "models/dpo_utility_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[train_dpo_with_utility] Training complete. Model saved to", output_dir)


if __name__ == "__main__":
    train_dpo_with_utility_grad_accum(
        checkpoint="/data1/shared_models/SmolLM2-135M-Instruct",
        data_path="/home/davidh/gray-swan-alignment/data/raw/preference/combined_preference_data.jsonl",
        utility_data_path="/home/davidh/gray-swan-alignment/data/raw/preference/alpaca_cleaned.jsonl",
        alpha=0.2, # sets the balance between utility and dpo alignment
        beta=0.5, # sets the beta term in dpo 
        lr=1e-4,
        epochs=20,
        desired_batch_size=64,
        micro_batch_size=1,
        device="cuda:3",
        log_file="logs/dpo_utility_training.log"
    )
