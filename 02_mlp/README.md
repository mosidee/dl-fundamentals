# 02 A multilayer perceptron

## The idea

A layer is `x @ W + b`: every output is a weighted sum of the inputs. Stack two of those with nothing between them and you still have a single linear map (`(xW1 + b1)W2 + b2` is just `xW + b`). The nonlinearity between layers, here ReLU, `max(0, z)`, is what lets the stack represent curved decision boundaries. That is the whole reason "deep" networks exist.

For classification the output of the last layer is a vector of **logits**, one per class. Softmax turns logits into probabilities; cross-entropy is minus the log probability of the correct class, averaged over the batch. Computing `log(softmax(z))` naively overflows for large logits, so everyone uses the log-sum-exp trick: subtract the maximum before exponentiating. You will implement that.

Training: take a mini-batch, forward, loss, `backward()`, update, repeat. First you do it with raw tensors and a hand-written SGD step, so there is no magic. Then you rebuild the same network as an `nn.Module` with `nn.Linear` and train it with `torch.optim`, which is how everyone writes it in practice. Both must give the same forward output when the weights are the same; the test checks that.

## Implement, in `exercise.py`

- `init_params(in_dim, hidden, out_dim, generator)`: dict with `W1 (in_dim, hidden)`, `b1 (hidden,)`, `W2 (hidden, out_dim)`, `b2 (out_dim,)`. Scale `W1` by `sqrt(2 / in_dim)` and `W2` by `sqrt(1 / hidden)` (He-style init keeps activations from exploding or vanishing). All with `requires_grad=True`.
- `mlp_forward(params, x)`: `relu(x @ W1 + b1) @ W2 + b2`, returning logits.
- `log_softmax(logits)`: along the last dimension, via log-sum-exp with the max subtracted. No `torch.log_softmax`, no `torch.logsumexp`.
- `cross_entropy(logits, targets)`: mean negative log-likelihood of the target class, using your `log_softmax`.
- `accuracy(logits, targets)`: fraction correct, as a Python float.
- `sgd_step(params, lr)`: for every tensor, `p -= lr * p.grad` under `no_grad`, then clear the grad.
- `class MLP(nn.Module)`: same architecture with `nn.Linear`; `forward` returns logits.
- `copy_params_into_module(params, module)`: copy your raw tensors into the module's weights. Note `nn.Linear` stores `weight` as `(out, in)`, so transpose.

## Run on real data

```bash
../.venv/bin/python train_mnist.py --impl exercise     # after your tests pass
```

Downloads MNIST through `datasets`, trains 784 -> 256 -> 10 for two epochs on MPS, prints test accuracy, writes `results/02_mnist.json`. Expect about 96% after two epochs (the reference run got 96.1%); a third epoch or a lower learning rate gets a little more. Then try `--lr 0.01` and `--lr 1.0` and watch what happens to the loss curve.

## Then answer in `learning-log.md`

The questions under "02".
