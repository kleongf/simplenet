# simplenet

A small NumPy-based neural-network framework built to understand the mathematics behind tensors, backpropagation, layers, losses, optimizers, data loading, and training loops.

## Included components

- reverse-mode autodifferentiation with `Tensor`
- linear, convolution, pooling, flatten, dropout, batch-normalisation, and activation modules
- MSE, MAE, cross-entropy, and binary cross-entropy losses
- SGD, momentum, Adam, and AdamW optimizers
- dataset helpers, train/test splitting, batching, metrics, and a `Trainer`
- unit tests covering the tensor engine and higher-level components

## Setup

```bash
python -m pip install numpy pytest
```

There is no packaging configuration yet. The example scripts add `src/` to `sys.path` automatically; for an interactive session, use `PYTHONPATH=src`.

## Examples

```bash
python main.py   # linear regression and a small MLP that learns sin(x)
python mnist.py  # convolutional MNIST classifier using the bundled IDX files
```

`visualize_mnist.py` and `mnist_inspector_template.html` provide an additional way to inspect the bundled dataset.

## Tests

```bash
pytest
```

This project is educational and prioritizes readable implementations over production performance.
