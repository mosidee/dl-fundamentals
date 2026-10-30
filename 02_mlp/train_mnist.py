"""Train the module-02 MLP on MNIST (via the Hugging Face `datasets` library) and record the result.

    ../.venv/bin/python train_mnist.py --impl exercise   # your code
    ../.venv/bin/python train_mnist.py --impl solution   # reference
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.device import describe_device, get_device, seed_everything  # noqa: E402


def load_mnist():
    from datasets import load_dataset

    ds = load_dataset("ylecun/mnist")

    def to_tensors(split):
        imgs = np.stack([np.asarray(im, dtype=np.uint8) for im in split["image"]])
        x = torch.from_numpy(imgs).float().div_(255.0).view(len(imgs), -1)
        y = torch.tensor(split["label"], dtype=torch.long)
        return x, y

    return to_tensors(ds["train"]), to_tensors(ds["test"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="exercise", choices=["exercise", "solution"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    impl = importlib.import_module(args.impl)
    seed_everything(0)
    device = get_device()
    print(f"device: {describe_device(device)}")

    (xtr, ytr), (xte, yte) = load_mnist()
    print(f"train {tuple(xtr.shape)}  test {tuple(xte.shape)}")
    xtr, ytr, xte, yte = xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)

    model = impl.MLP(784, args.hidden, 10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = xtr.shape[0]
    t0 = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad()
            loss = impl.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        model.eval()
        with torch.no_grad():
            acc = impl.accuracy(model(xte), yte)
        history.append({"epoch": epoch + 1, "train_loss": total / n, "test_accuracy": acc})
        print(f"epoch {epoch + 1}: train loss {total / n:.4f}  test accuracy {acc:.4f}")
    seconds = time.perf_counter() - t0

    out = {
        "module": "02_mlp",
        "impl": args.impl,
        "device": describe_device(device),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "hidden": args.hidden,
        "seconds": round(seconds, 1),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "02_mnist.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/02_mnist.json  ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
