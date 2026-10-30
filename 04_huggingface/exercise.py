"""Module 04: tokenizers, a hand-written generation loop, pooling, a throughput measurement."""
from __future__ import annotations

import time

import torch


def count_tokens(tokenizer, text: str) -> int:
    """Number of tokens for `text`, without special tokens."""
    raise NotImplementedError("TODO")


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    """Manual greedy decoding. Encode the prompt; loop: logits of the last position -> argmax -> append;
    stop early on tokenizer.eos_token_id. Return the decoded NEW tokens only (skip_special_tokens=True)."""
    raise NotImplementedError("TODO")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(B, T, D), (B, T) -> (B, D): mean over positions where attention_mask == 1."""
    raise NotImplementedError("TODO")


@torch.no_grad()
def tokens_per_second(model, tokenizer, prompts: list[str], max_new_tokens: int, device: torch.device) -> dict:
    """Left-pad the batch (tokenizer.padding_side = "left"; pad_token = eos if missing), generate greedily,
    and return {"new_tokens": int, "seconds": float, "tokens_per_second": float}.
    new_tokens = max_new_tokens * len(prompts) (count the full budget; early EOS is rare with tiny budgets)."""
    raise NotImplementedError("TODO")
