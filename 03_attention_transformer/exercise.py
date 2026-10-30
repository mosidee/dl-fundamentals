"""Module 03: attention, multi-head attention, a transformer block, a tiny GPT."""
from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """softmax(q k^T / sqrt(d)) v. mask: bool, True = may attend; masked scores become -inf."""
    raise NotImplementedError("TODO")


def causal_mask(T: int, device: torch.device | None = None) -> torch.Tensor:
    """(T, T) bool, True on and below the diagonal."""
    raise NotImplementedError("TODO")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # TODO: self.qkv = nn.Linear(d_model, 3 * d_model); self.out = nn.Linear(d_model, d_model)
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model).
        Steps: qkv = self.qkv(x); split into q, k, v along the last dim; reshape each to (B, n_heads, T, head_dim);
        attention per head (mask broadcasts over batch and heads); merge heads back to (B, T, d_model); self.out."""
        raise NotImplementedError("TODO")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # TODO: ln1, attn, ln2, mlp = Linear(d, 4d) -> GELU -> Linear(4d, d)
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x = x + attn(ln1(x), mask); x = x + mlp(ln2(x))."""
        raise NotImplementedError("TODO")


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 4, block_size: int = 128):
        super().__init__()
        self.block_size = block_size
        # TODO: tok_emb, pos_emb (nn.Embedding(block_size, d_model)), blocks (nn.ModuleList), ln_f, head
        raise NotImplementedError("TODO")

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) int64 with T <= block_size -> logits (B, T, vocab_size). Use a causal mask."""
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Append max_new_tokens sampled tokens to idx (B, T). Crop the context to the last block_size tokens
        before each forward. Sample from softmax(logits[:, -1] / temperature)."""
        raise NotImplementedError("TODO")
