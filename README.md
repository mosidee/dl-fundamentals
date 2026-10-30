# Deep learning fundamentals, hands-on

A learning project with one goal: a basic, working understanding of neural networks and transformers, plus real hands-on use of PyTorch and the Hugging Face ecosystem. Built and run on an Apple M2 (MPS backend). Every claim here is backed by a script you can run and a test that passes.

## How it works

Each module has four files:

- `README.md` explains the idea in plain words and says what to implement.
- `exercise.py` has the functions and classes with `TODO` bodies. You write these.
- `solution.py` is a reference implementation. Read it only after your tests pass, or when stuck.
- `test_0N_*.py` checks the implementation against PyTorch's own functions. By default it tests `exercise.py`; set `IMPL=solution` to run it against the reference.

Workflow for a module: read the README, fill in `exercise.py`, run `pytest` in that folder until green, then run the module's training script to see it work on real data, then answer the questions for that module in `learning-log.md` in your own words.

## Curriculum

| Module | What you build | What you learn |
|---|---|---|
| 01 Tensors and autograd | Linear regression by hand and with autograd; finite-difference gradient check | Tensors, computational graph, `backward()`, why gradients must be zeroed |
| 02 MLP | A two-layer network from raw tensors, then as `nn.Module`; cross-entropy from log-sum-exp; MNIST to about 97% | What a layer is, activations, loss, SGD, batching, train/eval modes |
| 03 Attention and transformer | Scaled dot-product attention, multi-head attention that matches `nn.MultiheadAttention` weight for weight, a transformer block, a tiny GPT that writes text | Attention, heads, causal masking, residuals and LayerNorm, positional embeddings |
| 04 Hugging Face ecosystem | Tokenizer exploration (English and Thai), `pipeline`, `AutoModel`, a greedy decoder written by hand, `datasets`, a tokens-per-second benchmark on MPS | How pretrained models are packaged, loaded and run; where throughput comes from |
| 05 Fine-tuning | Fine-tune a small BERT on sentiment with a manual training loop; before/after evaluation; save and reload; LoRA with `peft` | Transfer learning, learning-rate schedules, evaluation discipline, parameter-efficient fine-tuning |
| 06 Serve (bonus) | A FastAPI endpoint that serves the fine-tuned model and a small chat model | How a model becomes an API |

## Setup

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

## Run

```bash
cd 01_tensors_autograd && ../.venv/bin/python -m pytest -q            # tests your exercise.py
cd 01_tensors_autograd && IMPL=solution ../.venv/bin/python -m pytest -q  # tests the reference
.venv/bin/python 02_mlp/train_mnist.py                                   # trains on MNIST, writes results/02_mnist.json
.venv/bin/python 03_attention_transformer/train_charlm.py                # trains a tiny GPT, writes a sample
.venv/bin/python 04_huggingface/explore.py                               # tokenizers, pipeline, generation
.venv/bin/python 04_huggingface/bench_generate.py                        # tokens/s at batch 1, 4, 8
.venv/bin/python 05_finetune/finetune.py                                 # fine-tunes, evaluates, saves
```

## Results

Filled in from actual runs; see `results/`. Numbers are for this machine (Apple M2, 24 GB, MPS) and are not claims about any other hardware.

## Honesty note

This is a fundamentals project, not production ML. The models are small, the datasets are standard, and the point is understanding. Anything stated here can be reproduced with the commands above.
