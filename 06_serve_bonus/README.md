# 06 Serve it (bonus)

A model becomes a product when it sits behind an HTTP endpoint. This is a minimal FastAPI server with two routes:

- `POST /classify` with `{"text": "..."}` runs the sentiment model you fine-tuned in module 05 (falls back to the public SST-2 DistilBERT if you have not run it).
- `POST /v1/chat/completions` with an OpenAI-style body (`{"messages": [...], "max_tokens": 64}`) runs `Qwen/Qwen2.5-0.5B-Instruct`. The response shape matches the OpenAI API closely enough that an OpenAI SDK client can point at it by changing `base_url`, which is exactly how an inference API is consumed.

```bash
../.venv/bin/uvicorn serve:app --app-dir 06_serve_bonus --port 8000
curl -s localhost:8000/classify -H 'content-type: application/json' -d '{"text": "What a great film."}'
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{"messages": [{"role": "user", "content": "Say hi in Thai."}], "max_tokens": 32}'
```

Each response reports the time spent, so you can see where it goes: tokenisation is microseconds, the forward pass is everything, and generation cost scales with the number of new tokens.
