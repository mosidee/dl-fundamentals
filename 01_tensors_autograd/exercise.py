"""Module 01: tensors and autograd. Fill in every TODO; run `pytest` in this folder to check."""
from __future__ import annotations

from typing import Callable

import torch


def linear_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """x: (N, D); w: (D,); b: scalar tensor. Returns (N,)."""
    raise NotImplementedError("TODO")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error as a 0-dim tensor."""
    raise NotImplementedError("TODO")


def manual_gradients(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dL/dw, dL/db) for L = mse_loss(linear_forward(x, w, b), y), derived by hand. No autograd.

    Derivation (write it out):
      pred = x @ w + b
      r = pred - y
      L = (1/N) * sum(r^2)
      dL/dpred = ...
      dL/dw = ...
      dL/db = ...
    """
    raise NotImplementedError("TODO")


def finite_difference_grad(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Numerical gradient of scalar f at x using central differences, one coordinate at a time.
    Must not use autograd. Works for any shape of x; returns the same shape."""
    raise NotImplementedError("TODO")


def train_linear_regression(x: torch.Tensor, y: torch.Tensor, lr: float = 0.1, steps: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit y ~ x @ w + b with plain gradient descent using autograd.
    Start from zeros. Each step: loss -> backward -> update under no_grad -> zero grads.
    Return (w, b) detached from the graph."""
    raise NotImplementedError("TODO")
