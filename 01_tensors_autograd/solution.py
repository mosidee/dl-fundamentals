"""Module 01 reference solution. Read after your own tests pass."""
from __future__ import annotations

from typing import Callable

import torch


def linear_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return x @ w + b


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def manual_gradients(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # pred = x @ w + b ; r = pred - y ; L = (1/N) sum r^2
    # dL/dpred = 2 r / N                      (elementwise)
    # dL/dw    = x^T @ dL/dpred               (pred_i depends on w through x_i)
    # dL/db    = sum(dL/dpred)                (b is added to every pred_i)
    with torch.no_grad():
        n = x.shape[0]
        r = x @ w + b - y
        dpred = 2.0 * r / n
        dw = x.T @ dpred
        db = dpred.sum()
    return dw, db


def finite_difference_grad(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.detach().clone()
    grad = torch.zeros_like(x)
    flat_x = x.view(-1)
    flat_g = grad.view(-1)
    for i in range(flat_x.numel()):
        old = flat_x[i].item()
        flat_x[i] = old + eps
        f_plus = f(x).item()
        flat_x[i] = old - eps
        f_minus = f(x).item()
        flat_x[i] = old
        flat_g[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def train_linear_regression(x: torch.Tensor, y: torch.Tensor, lr: float = 0.1, steps: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, requires_grad=True)
    for _ in range(steps):
        loss = mse_loss(linear_forward(x, w, b), y)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None
        b.grad = None
    return w.detach(), b.detach()
