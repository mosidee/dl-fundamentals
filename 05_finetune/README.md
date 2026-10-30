# 05 Fine-tuning a pretrained model

## The idea

A pretrained model already knows the language; fine-tuning teaches it your task with a small amount of labelled data. For classification you load `AutoModelForSequenceClassification`, which puts a fresh, randomly initialised head on top of the pretrained body, and train the whole thing with a small learning rate for an epoch or two.

Discipline that matters more than the model:

- **Evaluate before you train**, so you know what the head does at random (about 50% on a binary task). An "after" number means nothing without a "before".
- **Learning-rate schedule**: warm up from zero for the first few percent of steps (the fresh head would otherwise send huge gradients through the pretrained body), then decay linearly to zero.
- **Gradient clipping** at 1.0 is standard and cheap insurance.
- **Keep a held-out test set** the model never sees during training.

**LoRA** (Low-Rank Adaptation) freezes the pretrained weights and trains small rank-`r` matrices added to a few projection layers. It trains under 1% of the parameters, uses far less memory, and the adapter is a few megabytes you can swap. `peft` adds it in three lines. You will measure the trainable-parameter difference.

## Implement, in `exercise.py`

- `tokenize_batch(tokenizer, texts, max_length)`: padding and truncation, return tensors.
- `build_lr_schedule(optimizer, warmup_steps, total_steps)`: a `torch.optim.lr_scheduler.LambdaLR` with linear warm-up then linear decay to zero.
- `training_step(model, batch, labels, optimizer, scheduler)`: forward with `labels=` (the model computes cross-entropy), backward, clip to 1.0, optimizer step, scheduler step, zero grads. Return the loss as a float.
- `evaluate(model, tokenizer, texts, labels, device, batch_size)`: accuracy in eval mode under `no_grad`.
- `count_trainable(model)`: number of parameters with `requires_grad`.
- `add_lora(model, r, target_modules)`: wrap with `peft` (`LoraConfig(task_type=TaskType.SEQ_CLS, ...)`, `get_peft_model`).

## Check

```bash
cd 05_finetune && ../.venv/bin/python -m pytest -q    # uses prajjwal1/bert-tiny and 64 synthetic sentences
```

## Fine-tune for real

```bash
../.venv/bin/python finetune.py --impl exercise            # DistilBERT on IMDB (2,000 train / 1,000 test), full fine-tune
../.venv/bin/python finetune.py --impl exercise --lora     # same with LoRA; compare trainable parameters and accuracy
```

Each run prints accuracy before and after, saves the model under `results/`, reloads it and classifies two sentences, and writes `results/05_finetune.json`. Expect roughly 50% before and 85 to 90% after on this subset; it takes a few minutes on MPS.

## Then answer in `learning-log.md`

The questions under "05".
