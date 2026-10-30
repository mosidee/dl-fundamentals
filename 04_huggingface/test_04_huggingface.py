import pytest
import torch

from conftest import load_impl

impl = load_impl("04_huggingface")
TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY).eval()
    return tok, model


def test_count_tokens_english_vs_thai(tiny):
    tok, _ = tiny
    en = impl.count_tokens(tok, "hello world")
    th = impl.count_tokens(tok, "สวัสดีชาวโลก")
    assert en == 2
    assert th > en  # byte-level BPE trained on English splits Thai into many pieces


def test_greedy_generate_matches_model_generate(tiny):
    tok, model = tiny
    device = torch.device("cpu")
    prompt = "The quick brown"
    ours = impl.greedy_generate(model, tok, prompt, max_new_tokens=8, device=device)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    ref_ids = model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    ref = tok.decode(ref_ids[0, ids.shape[1] :], skip_special_tokens=True)
    assert ours == ref


def test_mean_pool_ignores_padding():
    h = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.allclose(impl.mean_pool(h, mask), torch.tensor([[2.0, 3.0]]))


def test_tokens_per_second_shape(tiny):
    tok, model = tiny
    out = impl.tokens_per_second(model, tok, ["a b", "c d e f"], max_new_tokens=4, device=torch.device("cpu"))
    assert set(out) == {"new_tokens", "seconds", "tokens_per_second"}
    assert out["new_tokens"] == 8 and out["seconds"] > 0 and out["tokens_per_second"] > 0
