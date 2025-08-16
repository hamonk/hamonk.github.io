# Loss function for classification

These are my notes from [Module 3 - Loss functions for classification](https://dataflowr.github.io/website/modules/3-loss-functions-for-classification/)

I also added some hand crafted examples to better understand some of the concepts.

## gradient descent
- batch gradient descent -> full training set
- stochastic gradient descent -> 1 example at a time
- mini batch gradient descent -> subset (=batch) of training set

## optimization problem
The likelihood is the probability of y knowing x (training data).
Under the assumption we have a gaussian distributed error, we can estimate the likelihood.
When training, we often maximize the likelihood (make the observed data as probable as possible under the model).
Likelihood is a product - complex to maximize. Taking the log helps and we just have the cost function is the sum of  square errors (ordinary least square problem).

## logistic regression
### binary classification
sigmoid function:
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

We have the likelihood as:
The likelihood is:

$$
L(w, b) = \prod_{i=1}^n 
\big[ \sigma(z_i) \big]^{y_i} 
\big[ 1 - \sigma(z_i) \big]^{1 - y_i}
$$

The log-likelihood simplifies to:

$$
\ell(w, b) = \sum_{i=1}^n \left[ y_i \log \sigma(z_i) + (1 - y_i) \log \big(1 - \sigma(z_i)\big) \right]
$$

In pytorch, we can use **BCELoss**. You need to apply the **sigmoid** before using BCELoss

BCEWithLogitLoss is more stable numerically than BCELoss + sigmoid.
They use the log-sum trick.
In sigmoid, there's an exponential. With big numbers or small numbers, it can lead to issues.

### softmax regression

Now we have more than 2 classes

- Model output: Logits (real-valued scores)
- Probability model: Categorical distribution via softmax
- Loss function Negative: log likelihood (cross-entropy)

Softmax function: 
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}
$$

At the of network: LogSoftmax
Then use NLLLoss (negative log likelihood)

## Code examples

### Sigmoid + BCELoss = BCEWithLogitsLoss
```python
import torch
import torch.nn as nn
m = nn.Sigmoid()
loss = nn.BCELoss()
loss_WithLogitsLoss = nn.BCEWithLogitsLoss()

input = torch.randn(3,4,5)
target = torch.rand(3,4,5)

# using sigmoid + BCELoss is the same as BCEWithLogitsLoss
loss(m(input), target) == loss_WithLogitsLoss(input, target)
```

### NLLLoss + LogSoftMax = CrossEntropyLoss
```python
import torch
import torch.nn as nn
m = nn.LogSoftmax(dim=1)
loss1 = nn.NLLLoss()
loss2 = nn.CrossEntropyLoss()
C = 8
input = torch.randn(3,C,4,5)
target = torch.empty(3,4,5, dtype=torch.long).random_(0,C) 

# using NLLL + logsoftmax is the same as CrossEntropyLoss
assert loss1(m(input),target) == loss2(input,target)
```

If your model outputs raw logits, you should use CrossEntropyLoss directly.

If you already apply a LogSoftmax in your model, then you must use NLLLoss instead, or you’ll “double log-softmax” and break training.

### NLLLoss

```python
import torch

# Fake "log probabilities" for a batch of 3 samples and 4 classes
log_probs = torch.tensor([
    [-0.5, -1.2, -2.0, -3.0],   # sample 1
    [-1.0, -0.2, -3.0, -2.0],   # sample 2
    [-2.0, -1.0, -0.1, -3.0]    # sample 3
])

targets = torch.tensor([3, 1, 2])  # ground truth class indices

# Pick the log-probability of the correct class for each sample
picked = log_probs[torch.arange(len(targets)), targets]

print("Picked log-probs:", picked)
# >> Picked log-probs: tensor([-3.0000, -0.2000, -0.1000])

# NLL = -mean(correct log-probabilities)
nll_loss = -picked.mean()

print("Manual NLLLoss:", nll_loss.item())
# >> Manual NLLLoss: 1.100000023841858
```

### Softmax

```python
m = nn.Softmax(dim=1)
input = torch.randn(2, 3)

# >>> input
# tensor([[-0.2869,  0.8709, -1.0575],
#         [-0.6224,  0.5318, -2.3918]])

output = m(input)

# >> output
# tensor([[0.2153, 0.6851, 0.0996],
#         [0.2303, 0.7304, 0.0393]])

```

```python
exp_vals = torch.exp(input)

# >>> exp_vals
# tensor([[0.7506, 2.3891, 0.3473],
#         [0.5366, 1.7020, 0.0915]])

sum_exp = torch.sum(exp_vals, dim=1, keepdim=True)

# >>> sum_exp
# tensor([[3.4870],
#         [2.3301]])

out_manual = exp_vals / sum_exp 

# >> out_manual
# tensor([[0.2153, 0.6851, 0.0996],
#         [0.2303, 0.7304, 0.0393]])
```

### Sigmoid

```python
import torch

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])  # some test values

# Built-in
sig_builtin = torch.sigmoid(x)

# Manual
sig_manual = 1 / (1 + torch.exp(-x))

print("Input:", x)
# Input: tensor([-2., -1.,  0.,  1.,  2.])

print("Built-in Sigmoid:", sig_builtin)
# Built-in Sigmoid: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

print("Manual Sigmoid:", sig_manual)
# Manual Sigmoid: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808])

print("Difference:", (sig_builtin - sig_manual).abs().max().item())
# Difference: 0.0

```

### BCELoss

In the formula, when:
- **target is 1**: we want the x in log(x) to be close to 1 (`log(1) = 0`).
- **target is 0**: we want x in log(1-x) to be close to 0 (`log(1-0) = 0`)


```python-repl
>>> import torch
>>> import torch.nn as nn
>>> 
>>> # Fake logits (raw model outputs)
>>> logits = torch.tensor([[0.2], [-1.0], [2.0]])
>>> 
>>> # Targets (binary labels in [0,1])
>>> targets = torch.tensor([[1.0], [0.0], [1.0]])
>>> 
>>> # ---- Built-in way ----
>>> bce = nn.BCELoss()
>>> preds = torch.sigmoid(logits)  # must apply sigmoid first
>>> loss_builtin = bce(preds, targets)
>>> 
>>> # ---- Manual BCE ----
>>> preds_manual = 1 / (1 + torch.exp(-logits))  # sigmoid
>>> eps = 1e-12  # to avoid log(0)
>>> loss_manual = - (targets * torch.log(preds_manual + eps) +
...                  (1 - targets) * torch.log(1 - preds_manual + eps)).mean()
>>> 
>>> print("Built-in BCE:", loss_builtin.item())
Built-in BCE: 0.346109539270401
>>> print("Manual BCE:", loss_manual.item())
Manual BCE: 0.346109539270401
```


### CrossEntropyLoss

```python
import torch

# Example logits (batch=3, classes=4)
logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, -1.0],
    [0.5, 0.5, 0.5, 0.5]
])

targets = torch.tensor([3, 0, 2])  # true class indices

# ---- Step 1: stability trick: subtract max per row ----
max_logits, _ = torch.max(logits, dim=1, keepdim=True)
logits_shifted = logits - max_logits

# ---- Step 2: compute log-sum-exp ----
sum_exp = torch.sum(torch.exp(logits_shifted), dim=1, keepdim=True)
log_sum_exp = torch.log(sum_exp)

# ---- Step 3: pick the logits of the true classes ----
true_class_logits = logits_shifted[torch.arange(len(targets)), targets]

# ---- Step 4: compute CrossEntropyLoss ----
ce_manual = - (true_class_logits - log_sum_exp.squeeze()).mean()

print("Manual CrossEntropyLoss:", ce_manual.item())

# ---- Optional: check against built-in ----
import torch.nn as nn
ce_builtin = nn.CrossEntropyLoss()(logits, targets)
print("Built-in CrossEntropyLoss:", ce_builtin.item())

```

Exercise - implement it!

```python
import torch

# Example logits (batch=3, classes=4)
logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, -1.0],
    [0.5, 0.5, 0.5, 0.5]
])

targets = torch.tensor([3, 0, 2])  # true class indices

# ce_manual = ?

# ce_built_in = ?
```

My first try:

```python
import torch

# Example logits (batch=3, classes=4)
logits = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, -1.0],
    [0.5, 0.5, 0.5, 0.5]
])

targets = torch.tensor([3, 0, 2])  # true class indices

logits_exp = torch.exp(logits)

logits_exp_sum = logits_exp.sum(dim=1, keepdim=True)

my_logits = logits_exp / logits_exp_sum

loss = - (torch.log(my_logits[torch.arange(len(targets)), targets]).mean()).item()

# ce_manual = ?
print(loss)

# ce_built_in = ?
print(nn.CrossEntropyLoss()(logits, targets))

```

With this implementation, I am not using the trick used in the 1st implementation. When I do `log(logits_exp / logits_exp_sum)` I could avoid computing some computations.
Indeed, `log(a/b) = log(a) - log(b)` and here, `a=exp(x)` so when using `log(exp(a))`, we should not do any computation.


### log-sum-exp trick

The **log-sum-exp trick** is used to compute:


$$
\log \sum_i e^{x_i} = \log \sum_i e^{x_i - m} \cdot e^m = m + \log \sum_i e^{x_i - m}
$$


This avoids overflow when some `x_i` are very large.


```python
import torch

x = torch.tensor([1000.0, 1001.0, 1002.0])

torch.log(torch.exp(x).sum())
# >>> tensor(inf)

safe = m + torch.log(torch.exp(x - m).sum())
# >>> tensor(1002.4076)

