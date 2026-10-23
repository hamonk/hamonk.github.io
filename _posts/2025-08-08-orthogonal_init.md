# The value of initialization 📐🧠

I started reading [The Little Book of Deep Learning by François Fleuret](https://fleuret.org/francois/lbdl.html).

One of the first examples poses a simple-looking classification problem: given a set of 2D points, predict whether each point is red or blue. The two classes curve around each other like a yin-yang symbol:

![task](/images/original.png)

Because the classes are interleaved, a straight line cannot separate them in the original input space. A neural network can instead transform the points into a new representation where a final linear layer can distinguish the two classes.

The interesting question is whether a very narrow network—with just two units in each hidden layer—can learn that transformation, and how its weight initialization affects the result.

## First results

I set up the network described in the book:
- 8 hidden layers, each with 2 units
- `tanh` activation applied component-wise
- a final linear layer that produces scores for the 2 classes

```python
# Define the custom MLP model with 8 layers and final linear classifier
class DeepTanhMLP(nn.Module):
    def __init__(self, depth=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(2, 2))  # each layer is 2x2
        self.final = nn.Linear(2, 2)  # classifier output (2 classes)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.final(x)

    def transform(self, x, layer_index):
        """Transform data up to a specific layer (for visualization)"""
        for i in range(layer_index):
            x = torch.tanh(self.layers[i](x))
        return x.detach().numpy()
```

Although the network is deep, it is very narrow. I expected it to solve this small problem and reach 100% accuracy quickly.

That didn’t happen. The loss decreased, but accuracy plateaued well below 100%:

![loss](/images/loss_no_init.png)

## What could be wrong?

I started troubleshooting:

- **Vanishing gradients?** — Yes. Gradients became tiny after only a few iterations.
- **Data normalization?** — I initially skipped it. That caused other issues, but did not seem to be the main culprit here.
- **Training loop, optimizer, learning rate, or batch size?** — These seemed less likely: a larger, shallower model (one hidden layer with 8 units) trained successfully.
- **Random chance?** — Sometimes the narrow model did reach 100% accuracy, but not reliably.

One thing stood out: after a few layers, the transformed data collapsed onto nearly a single line:

![bad learning](/images/layers_output_no_init.png)

The grey crosses show the input space being compressed along one direction. The network is losing much of the data’s original 2D structure, making it harder for later layers to separate the classes.

## Orthogonal Initialization

I experimented with different initialization methods. Xavier initialization did not help much (see [Glorot Initialization](https://en.wikipedia.org/wiki/Weight_initialization#Glorot_initialization)).

The breakthrough came with orthogonal initialization. With it, training became much more reliable, and the network reached 100% accuracy consistently in my runs.

![good loss](/images/loss_with_init.png)

The layers now transform the space in a way that separates the blue and red points, making them easier for the final layer to distinguish.

Here is the effect of the layers on the data:

![good learning](/images/layers_output_orthogonal_init.png)

Orthogonal initialization sets a weight matrix so its rows or columns are orthogonal.

This can help preserve the scale of signals and gradients as they pass through a network. It is not a guarantee against vanishing or exploding gradients—nonlinearities such as `tanh`, as well as training updates, still affect the values.

In this experiment, the initialized layers kept more of the data’s spread, giving later layers useful information to work with. That made this narrow network much better at learning the transformation needed for the yin-yang classification problem.

## SVD = Singular Value Decomposition

Singular value decomposition (SVD) gives another way to inspect how a weight matrix scales different directions. In PyTorch, we can compute it with `torch.linalg.svd`.

Here are the values from the model weights I inspected:
```python
# RANDOM INITIALIZATION

for x in range(8):
    u, s, v = torch.linalg.svd(model.layers[x].weight.detach())
    print(f"Layer {x} singular values: {s}")

Layer 0 singular values: tensor([0.9221, 0.4884])
Layer 1 singular values: tensor([0.9710, 0.0936]) <- problematic
Layer 2 singular values: tensor([0.6273, 0.1235])
Layer 3 singular values: tensor([0.4757, 0.1770])
Layer 4 singular values: tensor([0.7705, 0.2222])
Layer 5 singular values: tensor([0.9553, 0.4274])
Layer 6 singular values: tensor([0.6217, 0.1350])
Layer 7 singular values: tensor([0.7096, 0.5015])

# ORTHOGONAL INITIALIZATION

for x in range(8):
    u, s, v = torch.linalg.svd(model_with_initialization.layers[x].weight.detach())
    print(f"Layer {x} singular values: {s}")

Layer 0 singular values: tensor([1.7645, 0.2109])
Layer 1 singular values: tensor([1.5999, 1.0709])
Layer 2 singular values: tensor([1.4989, 0.8290])
Layer 3 singular values: tensor([1.5814, 0.5803])
Layer 4 singular values: tensor([1.6148, 0.7659])
Layer 5 singular values: tensor([1.6656, 0.8657])
Layer 6 singular values: tensor([1.8517, 0.8446])
Layer 7 singular values: tensor([2.1327, 0.8531])
```

These are values from the weights in my runs, not the untouched initialization. In particular, orthogonally initialized square matrices start with singular values of 1, but training can change them.

If `s[1]` is very small compared to `s[0]`, then `W` strongly compresses one direction and is close to rank 1. In the randomly initialized model, Layer 1 has singular values 0.9710 and 0.0936—a large difference that is consistent with the collapse seen in the visualization. This is a useful clue, though the behavior of the full network also depends on its other layers and nonlinearities.

## Code

The full code can be found [here](https://github.com/hamonk/deep-learning/blob/main/the_value_of_initialization/the_value_of_initialization.ipynb)