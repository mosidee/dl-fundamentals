# 03 Attention and the transformer

## The idea

A transformer processes a sequence of vectors, one per token. The only operation that lets tokens see each other is **attention**. For each token you compute a query `q`; for every token you compute a key `k` and a value `v`. The score between token i and token j is `q_i . k_j`; softmax over j turns the scores into weights; the output for token i is the weighted sum of the values. In matrix form:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

The division by `sqrt(d_k)` keeps the dot products from growing with the dimension, which would push softmax into a regime where it outputs almost one-hot weights and gradients vanish.

**Multi-head** attention runs several attentions in parallel on slices of the vector (heads), then concatenates and projects. Different heads learn different relations (previous token, matching bracket, subject of the verb). In code it is one big linear layer producing Q, K and V for all heads, a reshape to `(batch, heads, time, head_dim)`, attention per head, a reshape back, and an output projection. You will build it and prove it is identical to `torch.nn.MultiheadAttention` by copying the weights across and comparing outputs.

**Causal masking:** a language model predicts the next token, so token i may only attend to tokens 0..i. You set the scores for j > i to minus infinity before the softmax, so their weights become exactly zero. Without it, the model would see the answer during training and learn nothing useful.

A **transformer block** is: `x = x + Attention(LayerNorm(x))`, then `x = x + MLP(LayerNorm(x))`. The additions are residual connections: each block only has to learn a correction to its input, which is why stacks of dozens of blocks still train. LayerNorm keeps the scale of activations steady.

A **GPT** is: token embedding plus position embedding, N blocks, a final LayerNorm, and a linear head to vocabulary logits. The position embedding is needed because attention itself has no notion of order.

Hugging Face models like GPT-2, Llama and Qwen are this architecture with more layers, a few refinements (rotary positions, RMSNorm, grouped-query attention) and a lot more data.

## Implement, in `exercise.py`

- `scaled_dot_product_attention(q, k, v, mask=None)`: `q, k, v` of shape `(..., T, d)`; `mask` is a boolean tensor broadcastable to `(..., T, T)` where `True` means "may attend". Masked positions get `-inf` before softmax. Return `(..., T, d)`.
- `causal_mask(T, device=None)`: boolean `(T, T)`, lower triangular including the diagonal.
- `MultiHeadAttention(d_model, n_heads)`: one packed `nn.Linear(d_model, 3 * d_model)` for Q, K, V (same layout as `nn.MultiheadAttention.in_proj_weight`: rows `[0:d]` are Q, `[d:2d]` K, `[2d:3d]` V), and `nn.Linear(d_model, d_model)` for the output. `forward(x, mask=None)` for `x` of shape `(B, T, d_model)`.
- `TransformerBlock(d_model, n_heads)`: pre-LayerNorm block as above; MLP is `Linear(d, 4d) -> GELU -> Linear(4d, d)`.
- `TinyGPT(vocab_size, d_model, n_heads, n_layers, block_size)`: embeddings, blocks, final LayerNorm, head. `forward(idx)` returns logits `(B, T, vocab)`. `generate(idx, max_new_tokens, temperature=1.0)` appends tokens one at a time, cropping the context to `block_size`, sampling from the softmax of the last position's logits.

## Check

```bash
cd 03_attention_transformer && ../.venv/bin/python -m pytest -q
```

## Run on real text

```bash
../.venv/bin/python train_charlm.py --impl exercise              # tiny shakespeare, downloaded on first run
../.venv/bin/python train_charlm.py --impl exercise --text my.txt  # any UTF-8 text; Thai works, it is character-level
```

Trains a 4-layer model on MPS for 1,500 steps (a few minutes), prints the loss every 100 steps, writes a 400-character sample and `results/03_charlm.json`. The sample will be imperfect; that is the point. Watch it go from random characters to word-shaped nonsense to almost-English.

## Then answer in `learning-log.md`

The questions under "03".
