"""Module 05: fine-tuning a sequence classifier with a manual loop, plus LoRA."""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def tokenize_batch(tokenizer, texts: list[str], max_length: int) -> dict:
    """tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt") -> dict of tensors."""
    raise NotImplementedError("TODO")


def build_lr_schedule(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Multiplier: step/warmup during warm-up, then linear from 1 at warmup to 0 at total_steps."""
    raise NotImplementedError("TODO")


def training_step(model, batch: dict, labels: torch.Tensor, optimizer, scheduler) -> float:
    """model.train(); out = model(**batch, labels=labels); out.loss.backward(); clip grad norm 1.0;
    optimizer.step(); scheduler.step(); optimizer.zero_grad(); return float loss."""
    raise NotImplementedError("TODO")


@torch.no_grad()
def evaluate(model, tokenizer, texts: list[str], labels: list[int], device: torch.device, batch_size: int = 32, max_length: int = 256) -> float:
    """Accuracy over texts in batches. model.eval(); logits.argmax(-1) vs labels."""
    raise NotImplementedError("TODO")


def count_trainable(model) -> int:
    raise NotImplementedError("TODO")


def add_lora(model, r: int = 8, target_modules: list[str] | None = None):
    """from peft import LoraConfig, TaskType, get_peft_model. Return the wrapped model."""
    raise NotImplementedError("TODO")
