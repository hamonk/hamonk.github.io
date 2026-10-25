# The value of initialization 📐🧠

In [The Little Book of Deep Learning by François Fleuret](https://fleuret.org/francois/lbdl.html), one of the first examples poses a simple-looking classification problem: given a set of 2D points, predict whether each point is red or blue. The two classes curve around each other like a yin-yang symbol:

![task](/images/original.png)

Because the classes are interleaved, a straight line cannot separate them in the original input space. A neural network can instead transform the points into a new representation where a final linear layer can distinguish the two classes.

I was intrigued by this example. Training a small neural network to solve such a simple problem seemed straightforward, so I set up an experiment. However, my initial results were not what I expected.

In this post, I walk through the experiments that led me to identify weight initialization as the culprit. Because the problem is two-dimensional, we can visualize the network's behavior at each stage, making it easier to see what went wrong and how the initialization affected the learning process.

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

Orthogonal initialization sets a weight matrix so its rows or columns are orthogonal. It helps preserve the scale of signals as they pass through the network, which can make training more stable and reliable.

In this experiment, the initialized layers kept more of the data’s spread, giving later layers useful information to work with. That made this narrow network much better at learning the transformation needed for the yin-yang classification problem.

## SVD = Singular Value Decomposition

Singular value decomposition (SVD) is one way to investigate how a layer’s weight matrix stretches or compresses different directions. Singular values measure the amount of stretching or shrinking along those directions. For example, in one of my randomly initialized models, Layer 1 had singular values 0.9710 and 0.0936. The smaller value indicates that this layer strongly compresses one direction, which is consistent with the collapse in the visualization.

I’ve left the per-layer SVD outputs and code in the [notebook](https://github.com/hamonk/deep-learning/blob/main/the_value_of_initialization/the_value_of_initialization.ipynb) for anyone who wants to dig into the details.

## Conclusion

I find setting up machine learning experiments and figuring out how to tell whether learning is actually happening fascinating. Being able to visualize the 2D network and how it transforms the data space makes it even more interesting!

## Code

The full code can be found [here](https://github.com/hamonk/deep-learning/blob/main/the_value_of_initialization/the_value_of_initialization.ipynb)