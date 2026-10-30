import pytest
import torch

from conftest import load_impl

impl = load_impl("05_finetune")
TINY = "prajjwal1/bert-tiny"

POS = ["great film", "wonderful acting", "loved every minute", "a joy to watch", "brilliant and moving", "superb cast", "truly excellent", "delightful story"]
NEG = ["terrible film", "awful acting", "hated every minute", "a chore to watch", "dull and boring", "weak cast", "truly dreadful", "tedious story"]


def _toy():
    texts = (POS + NEG) * 4
    labels = ([1] * len(POS) + [0] * len(NEG)) * 4
    return texts, labels


@pytest.fixture()
def tiny():
    # BertTokenizerFast explicitly: this old test model ships only vocab.txt, which AutoTokenizer
    # cannot convert under transformers v5, and a config.json without model_type, so the Bert* classes are named directly. Real models (distilbert, Qwen) ship tokenizer.json and load via Auto*.
    from transformers import BertForSequenceClassification, BertTokenizerFast

    torch.manual_seed(0)
    tok = BertTokenizerFast.from_pretrained(TINY)
    model = BertForSequenceClassification.from_pretrained(TINY, num_labels=2)
    return tok, model


def test_tokenize_batch_shapes(tiny):
    tok, _ = tiny
    b = impl.tokenize_batch(tok, ["short", "a somewhat longer sentence here"], max_length=8)
    assert b["input_ids"].shape[0] == 2 and b["input_ids"].shape[1] <= 8
    assert b["attention_mask"].shape == b["input_ids"].shape


def test_lr_schedule_shape():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = impl.build_lr_schedule(opt, warmup_steps=4, total_steps=12)
    lrs = []
    for _ in range(13):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == 0.0
    assert abs(lrs[2] - 0.5) < 1e-9
    assert abs(lrs[4] - 1.0) < 1e-9
    assert abs(lrs[8] - 0.5) < 1e-9
    assert lrs[12] == 0.0


def test_training_improves_toy_accuracy(tiny):
    tok, model = tiny
    device = torch.device("cpu")
    texts, labels = _toy()
    before = impl.evaluate(model, tok, texts, labels, device, batch_size=16, max_length=8)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    steps = 40
    sched = impl.build_lr_schedule(opt, warmup_steps=4, total_steps=steps)
    losses = []
    for s in range(steps):
        i = (s * 16) % len(texts)
        batch = impl.tokenize_batch(tok, texts[i : i + 16], max_length=8)
        losses.append(impl.training_step(model, batch, torch.tensor(labels[i : i + 16]), opt, sched))
    after = impl.evaluate(model, tok, texts, labels, device, batch_size=16, max_length=8)
    assert losses[-1] < losses[0]
    assert after >= 0.9 and after > before - 1e-9


def test_count_trainable_and_lora(tiny):
    _, model = tiny
    full = impl.count_trainable(model)
    assert full == sum(p.numel() for p in model.parameters())
    lora = impl.add_lora(model, r=4, target_modules=["query", "value"])
    trainable = impl.count_trainable(lora)
    assert 0 < trainable < 0.1 * full
