"""Module 02: a two-layer MLP from raw tensors, then as nn.Module."""
from __future__ import annotations

import math

import torch
from torch import nn


def init_params(in_dim: int, hidden: int, out_dim: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """W1 (in_dim, hidden) scaled by sqrt(2/in_dim); b1 zeros; W2 (hidden, out_dim) scaled by sqrt(1/hidden); b2 zeros.
    Use torch.randn(..., generator=generator). All tensors require grad."""
    raise NotImplementedError("TODO")


def mlp_forward(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """relu(x @ W1 + b1) @ W2 + b2 -> logits of shape (N, out_dim)."""
    raise NotImplementedError("TODO")


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    """log(softmax(logits)) along the last dim, computed stably with the max subtracted.
    Do not call torch.log_softmax or torch.logsumexp."""
    raise NotImplementedError("TODO")


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean of -log p(target) over the batch, using your log_softmax. targets: (N,) int64."""
    raise NotImplementedError("TODO")


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    raise NotImplementedError("TODO")


def sgd_step(params: dict[str, torch.Tensor], lr: float) -> None:
    """In place: p -= lr * p.grad for every parameter, under no_grad; then set p.grad = None."""
    raise NotImplementedError("TODO")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        raise NotImplementedError("TODO: two nn.Linear layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO")


def copy_params_into_module(params: dict[str, torch.Tensor], module: MLP) -> None:
    """Copy W1/b1/W2/b2 into module.fc1 and module.fc2. nn.Linear.weight is (out, in): transpose."""
    raise NotImplementedError("TODO")
