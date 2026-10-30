"""Module 02 reference solution."""
from __future__ import annotations

import math

import torch
from torch import nn


def init_params(in_dim: int, hidden: int, out_dim: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    w1 = torch.randn(in_dim, hidden, generator=generator) * math.sqrt(2.0 / in_dim)
    w2 = torch.randn(hidden, out_dim, generator=generator) * math.sqrt(1.0 / hidden)
    params = {
        "W1": w1,
        "b1": torch.zeros(hidden),
        "W2": w2,
        "b2": torch.zeros(out_dim),
    }
    for p in params.values():
        p.requires_grad_(True)
    return params


def mlp_forward(params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    h = torch.relu(x @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=-1, keepdim=True).values
    shifted = logits - m
    lse = shifted.exp().sum(dim=-1, keepdim=True).log()
    return shifted - lse


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logp = log_softmax(logits)
    picked = logp[torch.arange(logits.shape[0]), targets]
    return -picked.mean()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def sgd_step(params: dict[str, torch.Tensor], lr: float) -> None:
    with torch.no_grad():
        for p in params.values():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad = None


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def copy_params_into_module(params: dict[str, torch.Tensor], module: MLP) -> None:
    with torch.no_grad():
        module.fc1.weight.copy_(params["W1"].T)
        module.fc1.bias.copy_(params["b1"])
        module.fc2.weight.copy_(params["W2"].T)
        module.fc2.bias.copy_(params["b2"])
