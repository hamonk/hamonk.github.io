# Loss Functions for Classification

1. TOC
{:toc}

## Introduction

Loss functions are fundamental to machine learning and deep learning, yet their specifics can be easy to forget. This page serves as a comprehensive reference for understanding and implementing classification loss functions.

These notes are based on [Module 3 - Loss functions for classification](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/), enhanced with hand-crafted examples and practical PyTorch implementations to illustrate key concepts and edge cases.

## Gradient Descent Variants

There are three main variants of gradient descent:

- **Batch Gradient Descent**: Uses the full training set for each update
- **Stochastic Gradient Descent (SGD)**: Updates parameters using one example at a time
- **Mini-batch Gradient Descent**: Uses a subset (batch) of the training set for each update

## Optimization Problem

The **likelihood** is the probability of observing $y$ given $x$ (training data). Under the assumption of Gaussian-distributed errors, we can estimate this likelihood.

During training, we typically **maximize the likelihood** to make the observed data as probable as possible under our model. Since likelihood is a product, it's complex to maximize directly. Taking the logarithm simplifies this: the log-likelihood becomes a sum of squared errors (the Ordinary Least Squares problem).

## Logistic Regression

### Binary Classification

The **sigmoid function** maps any real-valued number to the range $(0, 1)$:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

For binary classification with labels $y_i \in \{0, 1\}$, the likelihood is:

$$
L(w, b) = \prod_{i=1}^n 
\big[ \sigma(z_i) \big]^{y_i} 
\big[ 1 - \sigma(z_i) \big]^{1 - y_i}
$$

where $z_i = w^T x_i + b$ is the logit for sample $i$.

The log-likelihood simplifies to:

$$
\ell(w, b) = \sum_{i=1}^n \left[ y_i \log \sigma(z_i) + (1 - y_i) \log \big(1 - \sigma(z_i)\big) \right]
$$

**PyTorch Implementation:**

- **`BCELoss`**: Binary Cross-Entropy Loss. Requires that you apply sigmoid to your model outputs first.
- **`BCEWithLogitsLoss`**: Combines sigmoid and BCE loss for better numerical stability. This is the **recommended approach** as it uses the log-sum-exp trick internally to avoid numerical issues with very large or very small values in the exponential function.

### Softmax Regression (Multi-class Classification)

For multi-class classification with $K > 2$ classes:

- **Model output**: Logits (real-valued scores)
- **Probability model**: Categorical distribution via softmax
- **Loss function**: Negative log-likelihood (cross-entropy)

The **softmax function** converts logits into a probability distribution:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}
$$

**PyTorch Implementation:**

- Apply `LogSoftmax` at the end of your network, then use `NLLLoss` (Negative Log Likelihood)
- Alternatively, use `CrossEntropyLoss` directly on logits (it combines `LogSoftmax` + `NLLLoss`)

## Code examples

### Sigmoid + BCELoss = BCEWithLogitsLoss

```python
import torch
import torch.nn as nn

m = nn.Sigmoid()
loss = nn.BCELoss()
loss_with_logits = nn.BCEWithLogitsLoss()

input = torch.randn(3, 4, 5)
target = torch.rand(3, 4, 5)

# Using Sigmoid + BCELoss is equivalent to BCEWithLogitsLoss
result1 = loss(m(input), target)
result2 = loss_with_logits(input, target)

print(f"Sigmoid + BCELoss: {result1.item():.6f}")
print(f"BCEWithLogitsLoss: {result2.item():.6f}")
print(f"Are they equal? {torch.allclose(result1, result2)}")
```

### NLLLoss + LogSoftmax = CrossEntropyLoss

```python
import torch
import torch.nn as nn

m = nn.LogSoftmax(dim=1)
loss_nll = nn.NLLLoss()
loss_ce = nn.CrossEntropyLoss()

C = 8  # number of classes
input = torch.randn(3, C, 4, 5)
target = torch.empty(3, 4, 5, dtype=torch.long).random_(0, C)

# NLLLoss + LogSoftmax is equivalent to CrossEntropyLoss
result1 = loss_nll(m(input), target)
result2 = loss_ce(input, target)

print(f"NLLLoss + LogSoftmax: {result1.item():.6f}")
print(f"CrossEntropyLoss: {result2.item():.6f}")
print(f"Are they equal? {torch.allclose(result1, result2)}")
```

**Important**: 
- If your model outputs **raw logits**, use `CrossEntropyLoss` directly.
- If you already apply `LogSoftmax` in your model, use `NLLLoss` instead. Using `CrossEntropyLoss` would apply log-softmax twice and break training.

### NLLLoss (Negative Log-Likelihood Loss)

NLLLoss expects log-probabilities as input and computes the negative log-likelihood by selecting the log-probability of the true class.

```python
import torch
import torch.nn as nn

# Log probabilities for a batch of 3 samples and 4 classes
log_probs = torch.tensor([
    [-0.5, -1.2, -2.0, -3.0],   # sample 1
    [-1.0, -0.2, -3.0, -2.0],   # sample 2
    [-2.0, -1.0, -0.1, -3.0]    # sample 3
])

targets = torch.tensor([3, 1, 2])  # ground truth class indices

# Pick the log-probability of the correct class for each sample
picked = log_probs[torch.arange(len(targets)), targets]

print("Picked log-probs:", picked)
# Picked log-probs: tensor([-3.0000, -0.2000, -0.1000])

# NLL = -mean(correct log-probabilities)
nll_manual = -picked.mean()
nll_builtin = nn.NLLLoss()(log_probs, targets)

print(f"Manual NLLLoss: {nll_manual.item():.6f}")
print(f"Built-in NLLLoss: {nll_builtin.item():.6f}")
# Manual NLLLoss: 1.100000
```

### Softmax Implementation

The softmax function converts logits into probabilities that sum to 1.

```python
import torch
import torch.nn as nn

m = nn.Softmax(dim=1)
input = torch.tensor([[-0.2869,  0.8709, -1.0575],
                       [-0.6224,  0.5318, -2.3918]])

print("Input logits:")
print(input)

# Built-in softmax
output_builtin = m(input)

print("\nBuilt-in Softmax:")
print(output_builtin)
# tensor([[0.2153, 0.6851, 0.0996],
#         [0.2303, 0.7304, 0.0393]])

# Manual implementation
exp_vals = torch.exp(input)
sum_exp = torch.sum(exp_vals, dim=1, keepdim=True)
output_manual = exp_vals / sum_exp

print("\nManual Softmax:")
print(output_manual)

print(f"\nAre they equal? {torch.allclose(output_builtin, output_manual)}")
print(f"Row sums (should be ~1.0): {output_manual.sum(dim=1)}")
```

### Sigmoid Implementation

The sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$ squashes input values to the range $(0, 1)$.

```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# Built-in sigmoid
sig_builtin = torch.sigmoid(x)

# Manual implementation
sig_manual = 1 / (1 + torch.exp(-x))

print("Input:", x)
print("Built-in Sigmoid:", sig_builtin)
print("Manual Sigmoid:", sig_manual)
print(f"Max difference: {(sig_builtin - sig_manual).abs().max().item():.10f}")

# Output:
# Input: tensor([-2., -1.,  0.,  1.,  2.])
# Built-in Sigmoid: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
# Manual Sigmoid: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])
# Max difference: 0.0000000000
```

### BCELoss (Binary Cross-Entropy Loss)

BCE Loss formula: $\text{BCE} = -\frac{1}{n}\sum_{i=1}^n [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$

Intuition:
- When **target is 1**: We want $p$ (prediction) close to 1, so $\log(p) \approx \log(1) = 0$ (minimal loss)
- When **target is 0**: We want $p$ close to 0, so $\log(1-p) \approx \log(1) = 0$ (minimal loss)

```python
import torch
import torch.nn as nn

# Logits (raw model outputs)
logits = torch.tensor([[0.2], [-1.0], [2.0]])

# Targets (binary labels in [0,1])
targets = torch.tensor([[1.0], [0.0], [1.0]])

# ---- Built-in way ----
bce = nn.BCELoss()
preds = torch.sigmoid(logits)  # must apply sigmoid first
loss_builtin = bce(preds, targets)

# ---- Manual BCE ----
preds_manual = 1 / (1 + torch.exp(-logits))  # sigmoid
eps = 1e-12  # to avoid log(0)
loss_manual = - (targets * torch.log(preds_manual + eps) +
                 (1 - targets) * torch.log(1 - preds_manual + eps)).mean()

print(f"Built-in BCE: {loss_builtin.item():.6f}")
print(f"Manual BCE: {loss_manual.item():.6f}")

# Output:
# Built-in BCE: 0.346110
# Manual BCE: 0.346110
```


### CrossEntropyLoss Implementation

Cross-entropy loss combines log-softmax and negative log-likelihood. The formula is:

$$
\text{CE} = -\frac{1}{n}\sum_{i=1}^n \log\left(\frac{e^{z_{i,y_i}}}{\sum_{j=1}^K e^{z_{i,j}}}\right)
$$

where $z_{i,y_i}$ is the logit for the true class of sample $i$.

```python
import torch
import torch.nn as nn

# Example logits (batch=3, classes=4)
logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, -1.0],
    [0.5, 0.5, 0.5, 0.5]
])

targets = torch.tensor([3, 0, 2])  # true class indices

# ---- Step 1: Numerical stability trick (subtract max per row) ----
max_logits, _ = torch.max(logits, dim=1, keepdim=True)
logits_shifted = logits - max_logits

# ---- Step 2: Compute log-sum-exp ----
sum_exp = torch.sum(torch.exp(logits_shifted), dim=1, keepdim=True)
log_sum_exp = torch.log(sum_exp)

# ---- Step 3: Pick the logits of the true classes ----
true_class_logits = logits_shifted[torch.arange(len(targets)), targets]

# ---- Step 4: Compute CrossEntropyLoss ----
ce_manual = -(true_class_logits - log_sum_exp.squeeze()).mean()

# ---- Compare with built-in ----
ce_builtin = nn.CrossEntropyLoss()(logits, targets)

print(f"Manual CrossEntropyLoss: {ce_manual.item():.6f}")
print(f"Built-in CrossEntropyLoss: {ce_builtin.item():.6f}")
print(f"Are they equal? {torch.allclose(ce_manual, ce_builtin)}")
```

### Exercise: Implement CrossEntropyLoss

Try implementing cross-entropy loss yourself:

```python
import torch
import torch.nn as nn

# Example logits (batch=3, classes=4)
logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, -1.0],
    [0.5, 0.5, 0.5, 0.5]
])

targets = torch.tensor([3, 0, 2])  # true class indices

# TODO: Implement ce_manual
# ce_manual = ?

# TODO: Compare with built-in
# ce_builtin = nn.CrossEntropyLoss()(logits, targets)
```

**My first attempt:**

```python
import torch
import torch.nn as nn

# Example logits (batch=3, classes=4)
logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, -1.0],
    [0.5, 0.5, 0.5, 0.5]
])

targets = torch.tensor([3, 0, 2])  # true class indices

# Compute softmax manually
logits_exp = torch.exp(logits)
logits_exp_sum = logits_exp.sum(dim=1, keepdim=True)
probs = logits_exp / logits_exp_sum

# Compute cross-entropy
ce_manual = -torch.log(probs[torch.arange(len(targets)), targets]).mean()
ce_builtin = nn.CrossEntropyLoss()(logits, targets)

print(f"Manual implementation: {ce_manual.item():.6f}")
print(f"Built-in implementation: {ce_builtin.item():.6f}")
```

**Note on optimization**: This implementation doesn't use the numerical stability trick from the previous example. When computing $\log(\frac{\exp(x)}{\sum \exp(x)})$, we can simplify using logarithm properties:

$$
\log\left(\frac{a}{b}\right) = \log(a) - \log(b)
$$

Since $a = e^x$, we have $\log(e^x) = x$, which avoids computing the exponential and logarithm. This is more efficient and numerically stable.


### Log-Sum-Exp Trick

The **log-sum-exp trick** is crucial for numerical stability when computing:

$$
\log \sum_i e^{x_i}
$$

Direct computation can overflow when $x_i$ values are large. The trick subtracts the maximum value:

$$
\log \sum_i e^{x_i} = \log \sum_i e^{x_i - m} \cdot e^m = m + \log \sum_i e^{x_i - m}
$$

where $m = \max_i x_i$. This keeps all exponentials in a reasonable range.

```python
import torch

x = torch.tensor([1000.0, 1001.0, 1002.0])

# Naive approach - causes overflow!
naive = torch.log(torch.exp(x).sum())
print(f"Naive computation: {naive}")
# Output: tensor(inf)

# Stable approach using log-sum-exp trick
m = x.max()
safe = m + torch.log(torch.exp(x - m).sum())
print(f"Stable computation: {safe:.4f}")
# Output: tensor(1002.4076)

# PyTorch has a built-in function for this
builtin = torch.logsumexp(x, dim=0)
print(f"Built-in logsumexp: {builtin:.4f}")
# Output: tensor(1002.4076)
```

## Summary

| Loss Function | Use Case | PyTorch Class | Input Requirements |
|---------------|----------|---------------|--------------------| 
| BCELoss | Binary classification | `nn.BCELoss()` | Probabilities (after sigmoid) |
| BCEWithLogitsLoss | Binary classification | `nn.BCEWithLogitsLoss()` | Raw logits (more stable) |
| NLLLoss | Multi-class classification | `nn.NLLLoss()` | Log-probabilities (after LogSoftmax) |
| CrossEntropyLoss | Multi-class classification | `nn.CrossEntropyLoss()` | Raw logits (recommended) |

**Key Takeaways:**

1. Always use the numerically stable versions: `BCEWithLogitsLoss` and `CrossEntropyLoss`
2. Don't apply activation functions before loss functions that expect raw logits
3. The log-sum-exp trick prevents overflow in softmax computations
4. Cross-entropy loss is equivalent to negative log-likelihood with softmax probabilities

