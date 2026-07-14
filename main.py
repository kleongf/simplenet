import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from simplenet.tensor import Tensor
from simplenet.modules import Linear, ReLU, Sequential
from simplenet.optimizers import Adam
from simplenet.losses import MSELoss
import numpy as np

X = Tensor(np.random.randn(100, 1))
y_true = Tensor(2 * X.data + 1 + np.random.randn(100, 1) * 0.1)  # noisy target

model = Linear(1, 1)
optimizer = Adam(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model.forward(X)
    loss = MSELoss()(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss {loss.data:.4f}")

print(model.weight.data, model.bias.data)   # should approach [[2.0]], [1.0]

# Example 2: y = sin(x). A single Linear layer (or even a quadratic feature)
# can't fit this over multiple periods, so this needs actual hidden layers
# with a nonlinear activation between them.
X2 = Tensor(np.random.uniform(-2 * np.pi, 2 * np.pi, size=(300, 1)))
y2_true = Tensor(np.sin(X2.data))

model2 = Sequential(Linear(1, 64), ReLU(), Linear(64, 64), ReLU(), Linear(64, 1))
optimizer2 = Adam(model2.parameters(), lr=0.01)

for epoch in range(2000):
    y2_pred = model2.forward(X2)
    loss2 = MSELoss()(y2_pred, y2_true)

    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    if epoch % 200 == 0:
        print(f"epoch {epoch}, loss {loss2.data:.4f}")

test_x = np.array([[-np.pi / 2], [0.0], [np.pi / 2], [np.pi]])
test_pred = model2.forward(Tensor(test_x))
for x_val, pred_val in zip(test_x.flatten(), test_pred.data.flatten()):
    print(f"sin({x_val:.3f}) actual={np.sin(x_val):.4f} pred={pred_val:.4f}")