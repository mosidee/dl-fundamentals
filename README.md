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

Reference runs of the solution code on this machine (Apple M2, 24 GB, MPS, torch 2.15, 2026-10-30). The JSON files in `results/` are written by the scripts, not by hand. Re-run with `--impl exercise` once your own code passes the tests; your numbers replace these.

| Module | Run | Result |
|---|---|---|
| 02 MLP | MNIST, 784-256-10, Adam 1e-3, 2 epochs, batch 128 | test accuracy 96.1% in 3 s |
| 03 Transformer | character-level GPT, 826,433 parameters, 1500 steps, block 128 | validation loss 4.17 to 1.93 in 98 s; sample in `results/03_charlm.json` |
| 04 Tokenizers | Qwen2.5 tokenizer, two sentence pairs | English vs Thai tokens: 8 vs 13; 6 vs 16 |
| 05 Fine-tune, full | DistilBERT on IMDB, 2000 train / 1000 test, 1 epoch | accuracy 53.3% before, 86.3% after; 66,955,010 trainable parameters; 120 s |
| 05 Fine-tune, LoRA r=8 | same data | accuracy 53.3% before, 86.7% after; 739,586 trainable parameters (1.1%); 105 s |

Generation throughput, `Qwen/Qwen2.5-0.5B-Instruct`, float16 on MPS, 64 new tokens per sequence, greedy:

| Batch size | Tokens per second | MPS memory (MB) |
|---|---|---|
| 1 | 74 | 960 |
| 4 | 99 | 960 |
| 8 | 234 | 960 |

Batching eight prompts gives about three times the throughput of one, from the same weights read once per step. That is the first fact of inference optimisation, and you can see it on a laptop.

## Honesty note

This is a fundamentals project, not production ML. The models are small, the datasets are standard, and the point is understanding. Anything stated here can be reproduced with the commands above.
