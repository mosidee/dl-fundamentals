# CLAUDE.md

A learning project: PyTorch and Hugging Face fundamentals on an Apple M2 (MPS). The owner is learning; the rules below keep the project useful for that.

## Commands

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt   # one-time setup (torch 2.15 has cp314 arm64 wheels)
cd 0N_module && ../.venv/bin/python -m pytest -q                        # test the owner's exercise.py
cd 0N_module && IMPL=solution ../.venv/bin/python -m pytest -q          # test the reference solution
.venv/bin/python -m pytest -q 0*/                                       # everything, exercise versions
IMPL=solution .venv/bin/python -m pytest -q 0*/                         # everything, reference versions
```

Training scripts are listed in README.md. They write to `results/` (JSON summaries are committed; weights are ignored by git).

## Rules

- **Do not fill in `exercise.py` for the owner.** The TODOs are theirs. Explain, point to the README, or show a smaller example; do not paste the solution. If asked directly to reveal a solution, point at `solution.py` instead.
- **Every concept has a test.** A new function in `solution.py` needs a test in `test_0N_*.py` that compares against PyTorch's own implementation or a known value.
- **Tests must run in under a minute per module on CPU**, with no network after the first download. Use tiny models (`hf-internal-testing/tiny-random-*`, `prajjwal1/bert-tiny`) and synthetic data in tests; real models and datasets belong in the training scripts.
- **Device selection goes through `common/device.py`.** MPS when available, else CPU. Never hard-code a device.
- **Results are measured, not typed.** `results/*.json` files are written by the scripts. Do not edit them by hand.
- Dependencies: `transformers` is v5 (use `dtype=`, not `torch_dtype=`); `datasets` is v5 (no dataset scripts; all datasets used here are parquet-backed on the Hub).

## Layout

```
common/device.py            device + seed helpers
01_tensors_autograd/        README, exercise, solution, tests
02_mlp/                     + train_mnist.py
03_attention_transformer/   + train_charlm.py (downloads tiny shakespeare into data/ on first run; --text for your own file)
04_huggingface/             exercise/solution/tests + explore.py + bench_generate.py
05_finetune/                exercise/solution/tests + finetune.py (+ --lora)
06_serve_bonus/             serve.py (FastAPI), not covered by tests
learning-log.md             the owner's answers, in their own words
results/                    JSON summaries from real runs
```
