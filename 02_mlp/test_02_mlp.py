import torch
import torch.nn.functional as F

from conftest import load_impl

impl = load_impl("02_mlp")


def _blobs(n_per_class=200, classes=3, dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [0.0, -2.5]])
    xs, ys = [], []
    for c in range(classes):
        xs.append(centers[c] + 0.6 * torch.randn(n_per_class, dim, generator=g))
        ys.append(torch.full((n_per_class,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def test_init_shapes_and_grad():
    g = torch.Generator().manual_seed(0)
    p = impl.init_params(5, 7, 3, generator=g)
    assert p["W1"].shape == (5, 7) and p["b1"].shape == (7,)
    assert p["W2"].shape == (7, 3) and p["b2"].shape == (3,)
    assert all(t.requires_grad for t in p.values())
    assert torch.all(p["b1"] == 0) and torch.all(p["b2"] == 0)


def test_log_softmax_matches_torch_and_is_stable():
    logits = torch.randn(4, 6) * 50  # large values: naive log(softmax) would overflow
    assert torch.allclose(impl.log_softmax(logits), F.log_softmax(logits, dim=-1), atol=1e-5)
    assert torch.isfinite(impl.log_softmax(logits)).all()


def test_cross_entropy_matches_torch():
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    assert torch.allclose(impl.cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)


def test_accuracy():
    logits = torch.tensor([[2.0, 0.0], [0.0, 1.0], [3.0, 1.0], [0.0, 5.0]])
    targets = torch.tensor([0, 1, 1, 1])
    assert abs(impl.accuracy(logits, targets) - 0.75) < 1e-9


def test_module_matches_raw_forward_when_weights_copied():
    g = torch.Generator().manual_seed(1)
    p = impl.init_params(4, 9, 3, generator=g)
    m = impl.MLP(4, 9, 3)
    impl.copy_params_into_module(p, m)
    x = torch.randn(10, 4)
    assert torch.allclose(impl.mlp_forward(p, x), m(x), atol=1e-6)


def test_raw_training_learns_blobs():
    x, y = _blobs()
    g = torch.Generator().manual_seed(2)
    p = impl.init_params(2, 32, 3, generator=g)
    first = None
    for step in range(300):
        loss = impl.cross_entropy(impl.mlp_forward(p, x), y)
        if first is None:
            first = loss.item()
        loss.backward()
        impl.sgd_step(p, lr=0.1)
    assert loss.item() < first * 0.2
    assert impl.accuracy(impl.mlp_forward(p, x), y) > 0.95
    assert all(t.grad is None for t in p.values())


def test_module_training_learns_blobs():
    x, y = _blobs(seed=3)
    torch.manual_seed(0)
    m = impl.MLP(2, 32, 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        loss = impl.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    assert impl.accuracy(m(x), y) > 0.95
