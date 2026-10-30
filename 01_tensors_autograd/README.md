# 01 Tensors and autograd

## The idea

A tensor is an n-dimensional array that remembers how it was computed. When you set `requires_grad=True` on a tensor and build other tensors from it, PyTorch records every operation in a graph. Calling `.backward()` on a scalar at the end of that graph walks it backwards, applying the chain rule, and leaves `dL/dx` in `x.grad` for every leaf tensor that asked for it.

That is all a neural network needs: a function from parameters to a loss, and the gradient of the loss with respect to the parameters. Training is "move the parameters a little against the gradient, repeat".

Three things trip people up:

1. **Gradients accumulate.** `backward()` adds into `.grad`; it does not overwrite. Zero them before each step or you are summing gradients across steps.
2. **Updates must not be recorded.** `w -= lr * w.grad` is itself an operation; do it under `torch.no_grad()` or the graph grows forever and the leaf stops being a leaf.
3. **Autograd is exact, finite differences are approximate.** `(f(x + eps) - f(x - eps)) / (2 eps)` is the definition of a derivative with a small error of order eps squared, plus floating-point noise. It is the standard way to check that a hand-written gradient is right.

## Implement, in `exercise.py`

- `linear_forward(x, w, b)`: `x @ w + b` for `x` of shape `(N, D)`, `w` of shape `(D,)`, scalar `b`. Returns shape `(N,)`.
- `mse_loss(pred, target)`: mean of squared differences.
- `manual_gradients(x, w, b, y)`: the gradients of the MSE loss with respect to `w` and `b`, derived by hand, without autograd. Write the derivation in a comment. Hint: with `r = pred - y` and `N` rows, `dL/dpred = 2 r / N`, then apply the chain rule through `pred = x @ w + b`.
- `finite_difference_grad(f, x, eps)`: numerical gradient of a scalar function `f` at `x`, one coordinate at a time. Use float64 inputs for precision.
- `train_linear_regression(x, y, lr, steps)`: start from `w = zeros(D)`, `b = 0`, both with `requires_grad=True`; for each step compute the loss, call `backward()`, update under `no_grad`, zero the gradients. Return `(w, b)` detached.

## Check

```bash
cd 01_tensors_autograd && ../.venv/bin/python -m pytest -q
```

The tests compare your `manual_gradients` with autograd, your `finite_difference_grad` with autograd, and your trainer with the known line the data came from.

## Then answer in `learning-log.md`

The three questions under "01". The last one is answered by printing both gradients and looking at the difference.
