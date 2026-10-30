import torch
import torch.nn.functional as F

from conftest import load_impl

impl = load_impl("01_tensors_autograd")
torch.manual_seed(0)


def _data(n=64, d=3, dtype=torch.float64):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, d, generator=g, dtype=dtype)
    w_true = torch.tensor([2.0, -3.0, 0.5], dtype=dtype)
    b_true = torch.tensor(0.7, dtype=dtype)
    y = x @ w_true + b_true + 0.01 * torch.randn(n, generator=g, dtype=dtype)
    return x, y, w_true, b_true


def test_linear_forward_matches_matmul():
    x, _, w, b = _data()
    out = impl.linear_forward(x, w, b)
    assert out.shape == (64,)
    assert torch.allclose(out, x @ w + b)


def test_mse_matches_torch():
    pred = torch.randn(50)
    target = torch.randn(50)
    assert torch.allclose(impl.mse_loss(pred, target), F.mse_loss(pred, target))


def test_manual_gradients_match_autograd():
    x, y, _, _ = _data()
    w = torch.randn(3, dtype=torch.float64, requires_grad=True)
    b = torch.randn((), dtype=torch.float64, requires_grad=True)
    loss = F.mse_loss(x @ w + b, y)
    loss.backward()
    dw, db = impl.manual_gradients(x, w.detach(), b.detach(), y)
    assert torch.allclose(dw, w.grad, atol=1e-10)
    assert torch.allclose(db, b.grad, atol=1e-10)


def test_finite_difference_matches_autograd():
    def f(t):
        return (torch.sin(t) * t**2).sum()

    x = torch.randn(2, 3, dtype=torch.float64)
    num = impl.finite_difference_grad(f, x, eps=1e-5)
    xa = x.clone().requires_grad_(True)
    f(xa).backward()
    assert num.shape == x.shape
    assert torch.allclose(num, xa.grad, atol=1e-6)


def test_train_recovers_true_line():
    x, y, w_true, b_true = _data(n=256)
    w, b = impl.train_linear_regression(x, y, lr=0.1, steps=300)
    assert not w.requires_grad and not b.requires_grad
    assert torch.allclose(w, w_true, atol=0.05)
    assert torch.allclose(b, b_true, atol=0.05)
