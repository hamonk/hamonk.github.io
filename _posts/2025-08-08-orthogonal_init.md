# The value of initialization

I started reading [The Little Book of Deep Learning by Francois Fleuret](https://fleuret.org/francois/lbdl.html)

One of the first examples is about classifying data that looks like the yin and yang.

![task](/images/original.png)

## First results

I set up my deep neural network as explained in the book: 
- 8 layers of size 2×2
- tanh activation applied component-wise
- a final classification layer

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

I expected this small problem to be solved with 100% accuracy pretty quickly.

It didn’t happen.
The loss decreased, but accuracy plateaued far below perfection:

![loss](/images/loss_no_init.png)

## What could be wrong?

I started troubleshooting:

- **Vanishing gradients?** — Yes, gradients became tiny after just a few iterations.
- **Data normalization?** — Initially missing, and while it caused other issues, it wasn’t the main culprit here.
- **Training loop?** / optimizer / learning rate / batch size? — Unlikely. When I switched to a larger model (e.g., 1 hidden layer of size 8), training worked fine.
- **Random chance?** — Sometimes the model did reach 100% accuracy, but inconsistently.

What really stood out: after a few layers, the data collapsed onto a single line:

![bad learning](/images/layers_output_no_init.png)

Thanks to the grey crosses on the plot, we can see that the entire input space is compressed onto a single line, losing its original 2D structure.

## Orthogonal Initialization

I experimented with initialization methods. Xavier didn’t help much.

The breakthrough came with orthogonal initialization. As soon as I applied it, the network became much more reliable and reached 100% accuracy consistently.

![good loss](/images/loss_with_init.png)

We can see the different transformation of the space are clearly separating the blue / red data making it easy for the final layer to discriminate between the two.

Visualization showing the effect of the layers on the data:

![good learning](/images/layers_output_orthogonal_init.png)

Orthogonal initialization sets the weight matrix so its rows (or columns) are orthogonal vectors.

**Key property:** Orthogonal matrices preserve variance. This helps keep activations and gradients from exploding or vanishing, even across many layers.

By preserving the spread of the data in early layers, the network keeps more useful information for the later ones — which is exactly what fixed my yin–yang classification problem.

## SVD = Singular Value Decomposition

One way to check if matrices are problematic is by using `torch.linalg.svd`.

Here are the values I get:
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

If `s[1]` is very small compared to `s[0]`, then `W` is nearly rank 1.

We can see that for Layer 1 with random initialization, `s[1]` is 0.0936 << 0.9710 (`s[0]`).

## Code

The full code can be found [here](https://github.com/hamonk/deep-learning/blob/main/the_value_of_initialization/the_value_of_initialization.ipynb)