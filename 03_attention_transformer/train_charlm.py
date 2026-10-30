"""Train the module-03 TinyGPT as a character-level language model.

    ../.venv/bin/python train_charlm.py --impl exercise            # tiny shakespeare (downloaded to data/)
    ../.venv/bin/python train_charlm.py --impl exercise --text f.txt
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.device import describe_device, get_device, seed_everything  # noqa: E402

TINY_SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    target = ROOT / "data" / "tinyshakespeare.txt"
    if not target.exists():
        target.parent.mkdir(exist_ok=True)
        print(f"downloading {TINY_SHAKESPEARE}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE, target)
    return target.read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="exercise", choices=["exercise", "solution"])
    ap.add_argument("--text", default=None)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sample-chars", type=int, default=400)
    args = ap.parse_args()

    impl = importlib.import_module(args.impl)
    seed_everything(0)
    device = get_device()
    print(f"device: {describe_device(device)}")

    text = load_text(args.text)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_train = int(0.9 * len(data))
    train, val = data[:n_train], data[n_train:]
    print(f"{len(text):,} characters, vocab {len(chars)}, train {len(train):,} / val {len(val):,}")

    def batch(split: torch.Tensor):
        ix = torch.randint(len(split) - args.block - 1, (args.batch,))
        x = torch.stack([split[i : i + args.block] for i in ix])
        y = torch.stack([split[i + 1 : i + args.block + 1] for i in ix])
        return x.to(device), y.to(device)

    model = impl.TinyGPT(len(chars), args.d_model, args.heads, args.layers, args.block).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        losses = []
        for _ in range(10):
            x, y = batch(val)
            losses.append(F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1)).item())
        model.train()
        return sum(losses) / len(losses)

    history = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = batch(train)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, len(chars)), y.view(-1))
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            vl = val_loss()
            history.append({"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(vl, 4)})
            print(f"step {step:5d}  train {loss.item():.4f}  val {vl:.4f}  ({time.perf_counter() - t0:.0f}s)")
    seconds = time.perf_counter() - t0

    model.eval()
    start = torch.tensor([[stoi[text[0]]]], device=device)
    sample = "".join(itos[i] for i in model.generate(start, args.sample_chars, temperature=0.8)[0].tolist())
    print("\n--- sample ---\n" + sample + "\n--- end ---")

    out = {
        "module": "03_attention_transformer",
        "impl": args.impl,
        "device": describe_device(device),
        "text": args.text or "tinyshakespeare",
        "vocab": len(chars),
        "parameters": n_params,
        "steps": args.steps,
        "batch": args.batch,
        "block": args.block,
        "seconds": round(seconds, 1),
        "first": history[0],
        "last": history[-1],
        "history": history,
        "sample": sample,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "03_charlm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote results/03_charlm.json  ({seconds:.0f}s)")


if __name__ == "__main__":
    main()
