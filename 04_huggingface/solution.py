"""Module 04 reference solution."""
from __future__ import annotations

import time

import torch


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    model.eval()
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = ids.shape[1]
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if tokenizer.eos_token_id is not None and nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0, prompt_len:], skip_special_tokens=True)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


@torch.no_grad()
def tokens_per_second(model, tokenizer, prompts: list[str], max_new_tokens: int, device: torch.device) -> dict:
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    model.generate(**batch, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    if device.type == "mps":
        torch.mps.synchronize()
    seconds = time.perf_counter() - t0
    new_tokens = max_new_tokens * len(prompts)
    return {"new_tokens": new_tokens, "seconds": seconds, "tokens_per_second": new_tokens / seconds}
