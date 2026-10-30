"""Fine-tune DistilBERT on an IMDB subset with a manual loop; optional LoRA. Writes results/05_finetune.json.

    ../.venv/bin/python finetune.py --impl exercise [--lora] [--train-size 2000 --test-size 1000]
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.device import describe_device, get_device, seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="exercise", choices=["exercise", "solution"])
    ap.add_argument("--model", default="distilbert/distilbert-base-uncased")
    ap.add_argument("--train-size", type=int, default=2000)
    ap.add_argument("--test-size", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora", action="store_true")
    args = ap.parse_args()
    impl = importlib.import_module(args.impl)

    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    seed_everything(0)
    device = get_device()
    print(f"device: {describe_device(device)}  model: {args.model}  lora: {args.lora}")

    ds = load_dataset("stanfordnlp/imdb")
    train = ds["train"].shuffle(seed=0).select(range(args.train_size))
    test = ds["test"].shuffle(seed=0).select(range(args.test_size))
    tr_x, tr_y = train["text"], train["label"]
    te_x, te_y = test["text"], test["label"]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    full_params = sum(p.numel() for p in model.parameters())
    if args.lora:
        model = impl.add_lora(model, r=8, target_modules=["q_lin", "v_lin"])
        lr = args.lr * 10  # adapters train with a higher rate
    else:
        lr = args.lr
    model.to(device)
    trainable = impl.count_trainable(model)
    print(f"parameters: {full_params:,} total, {trainable:,} trainable ({100 * trainable / full_params:.2f}%)")

    before = impl.evaluate(model, tok, te_x, te_y, device, batch_size=32, max_length=args.max_length)
    print(f"accuracy before: {before:.4f}")

    steps_per_epoch = math.ceil(len(tr_x) / args.batch)
    total = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = impl.build_lr_schedule(opt, warmup_steps=max(1, total // 10), total_steps=total)

    t0 = time.perf_counter()
    losses = []
    step = 0
    for epoch in range(args.epochs):
        order = torch.randperm(len(tr_x)).tolist()
        for i in range(0, len(order), args.batch):
            idx = order[i : i + args.batch]
            batch = impl.tokenize_batch(tok, [tr_x[j] for j in idx], args.max_length).to(device)
            labels = torch.tensor([tr_y[j] for j in idx], device=device)
            loss = impl.training_step(model, batch, labels, opt, sched)
            losses.append(loss)
            step += 1
            if step % 25 == 0 or step == total:
                print(f"step {step}/{total}  loss {sum(losses[-25:]) / len(losses[-25:]):.4f}  ({time.perf_counter() - t0:.0f}s)")
    seconds = time.perf_counter() - t0

    after = impl.evaluate(model, tok, te_x, te_y, device, batch_size=32, max_length=args.max_length)
    print(f"accuracy after:  {after:.4f}  (trained {seconds:.0f}s)")

    tag = "lora" if args.lora else "full"
    save_dir = ROOT / "results" / f"finetuned-distilbert-imdb-{tag}"
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)

    # reload and use it like any Hub model
    if args.lora:
        from peft import AutoPeftModelForSequenceClassification

        reloaded = AutoPeftModelForSequenceClassification.from_pretrained(save_dir).to(device).eval()
    else:
        reloaded = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device).eval()
    demo = ["A warm, funny, beautifully shot film that I did not want to end.", "Two hours of my life I will never get back."]
    with torch.no_grad():
        preds = reloaded(**impl.tokenize_batch(tok, demo, args.max_length).to(device)).logits.argmax(-1).tolist()
    for s, p in zip(demo, preds):
        print(f"  {'positive' if p == 1 else 'negative'}  {s}")

    out = {
        "module": "05_finetune",
        "impl": args.impl,
        "device": describe_device(device),
        "model": args.model,
        "lora": args.lora,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "epochs": args.epochs,
        "batch": args.batch,
        "max_length": args.max_length,
        "lr": lr,
        "params_total": full_params,
        "params_trainable": trainable,
        "accuracy_before": before,
        "accuracy_after": after,
        "seconds": round(seconds, 1),
        "first_loss": losses[0],
        "last_loss": sum(losses[-25:]) / len(losses[-25:]),
        "saved_to": str(save_dir.relative_to(ROOT)),
    }
    (ROOT / "results" / f"05_finetune_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/05_finetune_{tag}.json")


if __name__ == "__main__":
    main()
