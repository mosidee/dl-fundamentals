# 04 The Hugging Face ecosystem

## The idea

Hugging Face is three things you will use constantly:

- **The Hub**: a git-like store of model weights, tokenizers and datasets, addressed by name (`Qwen/Qwen2.5-0.5B-Instruct`, `stanfordnlp/imdb`). The first `from_pretrained` downloads to `~/.cache/huggingface`; later calls are offline.
- **`transformers`**: the library that turns a name into a `tokenizer` and a `model`. `AutoTokenizer` and `AutoModel*` read the repo's `config.json` and pick the right class. `AutoModel` gives you hidden states; `AutoModelForCausalLM` adds the language-model head (next-token logits); `AutoModelForSequenceClassification` adds a classification head.
- **`datasets`**: loads datasets as Arrow tables, with `.map()` for tokenisation and `.select()` for subsets.

A **tokenizer** turns text into integer ids. Modern ones use byte-pair encoding: frequent strings become one token; rare ones split into several. English words are usually one token; Thai text, which has no spaces and is underrepresented in most training data, splits into many more. You will measure that.

**Generation** is a loop: run the model on the ids so far, take the logits of the last position, pick a token (argmax for greedy, sample for variety), append it, repeat. `model.generate()` does this with caching and batching, but you will write the loop yourself first so there is nothing hidden. Then you will measure tokens per second at different batch sizes, which is where the "inference optimisation" conversation starts: one sequence at a time leaves the GPU mostly idle; a batch of eight shares every weight read.

## Implement, in `exercise.py`

- `count_tokens(tokenizer, text)`: number of tokens without special tokens.
- `greedy_generate(model, tokenizer, prompt, max_new_tokens, device)`: the manual loop above, stopping at the EOS token. Return only the generated text. The test checks that it matches `model.generate(do_sample=False)` exactly.
- `mean_pool(last_hidden_state, attention_mask)`: average the hidden states over real tokens only (what sentence-embedding models do). Shape `(B, T, D)` and `(B, T)` to `(B, D)`.
- `tokens_per_second(model, tokenizer, prompts, max_new_tokens, device)`: batch the prompts with left padding, call `model.generate(do_sample=False)`, time it, return `{"new_tokens", "seconds", "tokens_per_second"}`.

## Check

```bash
cd 04_huggingface && ../.venv/bin/python -m pytest -q    # uses a tiny GPT-2 from the Hub (first run downloads it)
```

## Explore and measure

```bash
../.venv/bin/python explore.py          # tokenizers on English and Thai, pipeline, chat generation, datasets
../.venv/bin/python bench_generate.py   # Qwen2.5-0.5B-Instruct on MPS: tokens/s at batch 1, 4, 8
```

Both write to `results/`. The first run downloads about 1 GB of weights.

## Then answer in `learning-log.md`

The questions under "04".
