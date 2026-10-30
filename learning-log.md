# Learning log

Answer each question in your own words after the module's tests pass and its script has run. Short is fine. These are the questions an interviewer asks; writing the answers is the preparation.

## 01 Tensors and autograd
- What does `requires_grad=True` change about a tensor?
- Why must gradients be zeroed between steps?
- Your finite-difference check: how close was the numerical gradient to autograd's, and why not exactly equal?

## 02 MLP
- Why does a network with no activation function collapse to a single linear layer?
- Why compute cross-entropy through log-sum-exp instead of `log(softmax(x))`?
- MNIST result (accuracy, epochs, time on MPS):
- What happened when you changed the learning rate by 10x in each direction?

## 03 Attention and transformer
- In one sentence each: what are Q, K and V?
- Why divide by sqrt(d_k)?
- What does the causal mask prevent, and what would go wrong without it?
- Why do residual connections and LayerNorm make deep stacks trainable?
- Your tiny GPT: loss at start and end, and one sample of generated text:

## 04 Hugging Face ecosystem
- How many tokens did an English sentence and a Thai sentence of similar meaning use, and why do they differ?
- What does a chat template do?
- Tokens/s at batch 1, 4 and 8 on MPS, and why batching helps:
- What is the difference between `AutoModel` and `AutoModelForCausalLM`?

## 05 Fine-tuning
- Accuracy before and after fine-tuning, and on how many examples:
- Why use a learning-rate warm-up and decay?
- What does LoRA change, and how many parameters did it train compared with full fine-tuning?
- One mistake you made during this module and how you found it:

## 06 Serve (bonus)
- What does the server do per request, and where does the time go?
