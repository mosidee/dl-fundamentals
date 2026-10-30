"""Tokens per second for a small chat model at batch sizes 1, 4 and 8. Writes results/04_bench.json.

    ../.venv/bin/python bench_generate.py --impl exercise
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.device import describe_device, get_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="exercise", choices=["exercise", "solution"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--batches", default="1,4,8")
    args = ap.parse_args()
    impl = importlib.import_module(args.impl)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_device()
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    print(f"device: {describe_device(device)}  dtype: {dtype}  model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()

    base_prompts = [
        "Write one sentence about the sea.",
        "Explain what a GPU is in one sentence.",
        "Name three fruits.",
        "What is 12 times 12?",
        "Describe Bangkok in one sentence.",
        "Give one tip for learning Python.",
        "What does an API do?",
        "Say hello in Thai.",
    ]
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    # warm-up (first MPS run compiles kernels)
    impl.tokens_per_second(model, tok, [chat(base_prompts[0])], 8, device)

    rows = []
    for b in [int(x) for x in args.batches.split(",")]:
        prompts = [chat(p) for p in base_prompts[:b]]
        r = impl.tokens_per_second(model, tok, prompts, args.new_tokens, device)
        mem = torch.mps.current_allocated_memory() / 2**20 if device.type == "mps" else None
        rows.append({"batch": b, **r, "mps_allocated_mb": round(mem) if mem else None})
        print(f"batch {b}: {r['tokens_per_second']:7.1f} tokens/s  ({r['new_tokens']} tokens in {r['seconds']:.2f}s)" + (f"  mem {mem:.0f} MB" if mem else ""))

    out = {"model": args.model, "device": describe_device(device), "dtype": str(dtype), "new_tokens_per_sequence": args.new_tokens, "impl": args.impl, "rows": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "04_bench.json").write_text(json.dumps(out, indent=2))
    print("wrote results/04_bench.json")


if __name__ == "__main__":
    main()
