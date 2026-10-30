"""Minimal inference server: a classifier route and an OpenAI-style chat route. Not covered by tests."""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.device import describe_device, get_device  # noqa: E402

CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED = ROOT / "results" / "finetuned-distilbert-imdb-full"
FALLBACK_CLF = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

app = FastAPI(title="dl-fundamentals serve")
device = get_device()
_models: dict = {}


def classifier():
    if "clf" not in _models:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        src = str(FINETUNED) if FINETUNED.exists() else FALLBACK_CLF
        _models["clf"] = (AutoTokenizer.from_pretrained(src), AutoModelForSequenceClassification.from_pretrained(src).to(device).eval(), src)
    return _models["clf"]


def chat_model():
    if "chat" not in _models:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if device.type != "cpu" else torch.float32
        _models["chat"] = (AutoTokenizer.from_pretrained(CHAT_MODEL), AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=dtype).to(device).eval())
    return _models["chat"]


class ClassifyIn(BaseModel):
    text: str


class Message(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[Message]
    max_tokens: int = 64
    model: str = CHAT_MODEL


@app.get("/health")
def health():
    return {"ok": True, "device": describe_device(device)}


@app.post("/classify")
@torch.no_grad()
def classify(body: ClassifyIn):
    tok, model, src = classifier()
    t0 = time.perf_counter()
    enc = tok(body.text, return_tensors="pt", truncation=True, max_length=256).to(device)
    probs = model(**enc).logits.softmax(-1)[0]
    label = int(probs.argmax())
    return {"label": "positive" if label == 1 else "negative", "score": float(probs[label]), "model": src, "ms": round(1000 * (time.perf_counter() - t0), 1)}


@app.post("/v1/chat/completions")
@torch.no_grad()
def chat(body: ChatIn):
    tok, model = chat_model()
    t0 = time.perf_counter()
    prompt = tok.apply_chat_template([m.model_dump() for m in body.messages], tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=body.max_tokens, do_sample=False)
    new = out[0, enc["input_ids"].shape[1] :]
    text = tok.decode(new, skip_special_tokens=True)
    seconds = time.perf_counter() - t0
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": CHAT_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": int(enc["input_ids"].shape[1]), "completion_tokens": int(new.numel()), "total_tokens": int(enc["input_ids"].shape[1] + new.numel())},
        "timing": {"seconds": round(seconds, 3), "tokens_per_second": round(new.numel() / seconds, 1)},
    }
