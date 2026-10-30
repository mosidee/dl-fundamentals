"""Tour of the Hugging Face ecosystem. Downloads a few models on first run; writes results/04_explore.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.device import describe_device, get_device  # noqa: E402

CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SENTIMENT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def main() -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    device = get_device()
    print(f"device: {describe_device(device)}\n")
    out: dict = {"device": describe_device(device)}

    # 1. Tokenizers: the same sentence in English and Thai
    print("== 1. tokenizers ==")
    tok = AutoTokenizer.from_pretrained(CHAT_MODEL)
    pairs = [("The weather in Bangkok is hot today.", "วันนี้อากาศที่กรุงเทพร้อนมาก"), ("I am learning deep learning.", "ฉันกำลังเรียนการเรียนรู้เชิงลึก")]
    out["tokens"] = []
    for en, th in pairs:
        n_en = len(tok(en, add_special_tokens=False)["input_ids"])
        n_th = len(tok(th, add_special_tokens=False)["input_ids"])
        pieces_th = tok.convert_ids_to_tokens(tok(th, add_special_tokens=False)["input_ids"])
        print(f"  EN {n_en:2d} tokens: {en}")
        print(f"  TH {n_th:2d} tokens: {th}")
        print(f"     Thai pieces: {pieces_th[:12]}{' ...' if len(pieces_th) > 12 else ''}")
        out["tokens"].append({"en": en, "en_tokens": n_en, "th": th, "th_tokens": n_th})

    # 2. pipeline: the one-line way to run a task
    print("\n== 2. pipeline (sentiment) ==")
    clf = pipeline("text-classification", model=SENTIMENT, device=device)
    samples = ["This library is a joy to use.", "The documentation is confusing and the install broke twice."]
    preds = clf(samples)
    for s, p in zip(samples, preds):
        print(f"  {p['label']:9s} {p['score']:.3f}  {s}")
    out["sentiment"] = [{"text": s, **p} for s, p in zip(samples, preds)]

    # 3. AutoModelForCausalLM + chat template
    print("\n== 3. chat generation ==")
    model = AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=torch.float16 if device.type != "cpu" else torch.float32).to(device).eval()
    messages = [{"role": "user", "content": "In two sentences, what does a tokenizer do?"}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(device)
    gen = model.generate(**ids, max_new_tokens=80, do_sample=False)
    answer = tok.decode(gen[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
    print("  prompt (templated, first 120 chars):", repr(prompt[:120]))
    print("  answer:", answer)
    out["chat"] = {"model": CHAT_MODEL, "question": messages[0]["content"], "answer": answer}

    # 4. datasets: load a subset and tokenise it with map
    print("\n== 4. datasets ==")
    ds = load_dataset("stanfordnlp/imdb", split="train").shuffle(seed=0).select(range(200))
    print("  columns:", ds.column_names, "| rows:", len(ds))
    tokenised = ds.map(lambda b: tok(b["text"], truncation=True, max_length=64), batched=True)
    print("  after map:", tokenised.column_names)
    print("  first review (truncated):", ds[0]["text"][:100].replace("\n", " "), "| label:", ds[0]["label"])
    out["datasets"] = {"name": "stanfordnlp/imdb", "rows": len(ds), "columns_after_map": tokenised.column_names}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "04_explore.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nwrote results/04_explore.json")


if __name__ == "__main__":
    main()
