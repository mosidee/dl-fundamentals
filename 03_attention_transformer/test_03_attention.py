import torch
import torch.nn.functional as F
from torch import nn

from conftest import load_impl

impl = load_impl("03_attention_transformer")


def test_sdpa_matches_torch_without_mask():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))
    assert torch.allclose(impl.scaled_dot_product_attention(q, k, v), F.scaled_dot_product_attention(q, k, v), atol=1e-6)


def test_sdpa_matches_torch_with_causal_mask():
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 3, 6, 8) for _ in range(3))
    mask = impl.causal_mask(6)
    ours = impl.scaled_dot_product_attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_causal_mask_shape_and_values():
    m = impl.causal_mask(4)
    assert m.dtype == torch.bool and m.shape == (4, 4)
    assert m.tolist() == [[True, False, False, False], [True, True, False, False], [True, True, True, False], [True, True, True, True]]


def test_mha_matches_nn_multihead_attention():
    torch.manual_seed(2)
    d, h, B, T = 16, 4, 2, 7
    ours = impl.MultiHeadAttention(d, h)
    ref = nn.MultiheadAttention(d, h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(ours.qkv.weight)
        ref.in_proj_bias.copy_(ours.qkv.bias)
        ref.out_proj.weight.copy_(ours.out.weight)
        ref.out_proj.bias.copy_(ours.out.bias)
    x = torch.randn(B, T, d)
    y_ours = ours(x)
    y_ref, _ = ref(x, x, x, need_weights=False)
    assert torch.allclose(y_ours, y_ref, atol=1e-5)
    # with a causal mask (nn.MultiheadAttention's bool mask means "blocked", so invert ours)
    causal = impl.causal_mask(T)
    y_ours_c = ours(x, causal)
    y_ref_c, _ = ref(x, x, x, attn_mask=~causal, need_weights=False)
    assert torch.allclose(y_ours_c, y_ref_c, atol=1e-5)


def test_transformer_block_shape_and_gradient():
    torch.manual_seed(3)
    blk = impl.TransformerBlock(16, 4)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = blk(x, impl.causal_mask(5))
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_tiny_gpt_forward_shape_and_overfits_a_pattern():
    torch.manual_seed(4)
    vocab = 10
    model = impl.TinyGPT(vocab, d_model=32, n_heads=4, n_layers=2, block_size=16)
    seq = torch.tensor([list(range(10)) + list(range(6))])  # 0..9,0..5 repeated pattern, length 16
    x, y = seq[:, :-1], seq[:, 1:]
    logits = model(x)
    assert logits.shape == (1, 15, vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, vocab), y.view(-1))
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 < first
    out = model.generate(torch.tensor([[0, 1, 2]]), max_new_tokens=5, temperature=0.1)
    assert out.shape == (1, 8)
    assert out[0, 3:].tolist() == [3, 4, 5, 6, 7]


def test_generate_respects_block_size():
    model = impl.TinyGPT(5, d_model=16, n_heads=2, n_layers=1, block_size=4)
    out = model.generate(torch.zeros(1, 2, dtype=torch.long), max_new_tokens=6)
    assert out.shape == (1, 8)
