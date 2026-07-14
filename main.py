import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from simplenet.tensor import Tensor
from simplenet.modules import Linear
from simplenet.optimizers import SGD
from simplenet.losses import MSELoss
import numpy as np

X = Tensor(np.random.randn(100, 1))
y_true = Tensor(2 * X.data + 1 + np.random.randn(100, 1) * 0.1)  # noisy target

model = Linear(1, 1)
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(1000):
    y_pred = model.forward(X)
    loss = MSELoss()(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss {loss.data:.4f}")

print(model.weight.data, model.bias.data)   # should approach [[2.0]], [1.0]