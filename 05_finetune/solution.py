"""Module 05 reference solution."""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import LambdaLR


def tokenize_batch(tokenizer, texts: list[str], max_length: int) -> dict:
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def build_lr_schedule(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        remaining = max(total_steps - warmup_steps, 1)
        return max(0.0, (total_steps - step) / remaining)

    return LambdaLR(optimizer, factor)


def training_step(model, batch: dict, labels: torch.Tensor, optimizer, scheduler) -> float:
    model.train()
    out = model(**batch, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return out.loss.item()


@torch.no_grad()
def evaluate(model, tokenizer, texts: list[str], labels: list[int], device: torch.device, batch_size: int = 32, max_length: int = 256) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(texts), batch_size):
        batch = tokenize_batch(tokenizer, texts[i : i + batch_size], max_length).to(device)
        preds = model(**batch).logits.argmax(dim=-1).cpu()
        correct += (preds == torch.tensor(labels[i : i + batch_size])).sum().item()
    return correct / len(texts)


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_lora(model, r: int = 8, target_modules: list[str] | None = None):
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=2 * r, lora_dropout=0.1, target_modules=target_modules)
    return get_peft_model(model, cfg)
