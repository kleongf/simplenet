import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from simplenet.tensor import Tensor
from simplenet.modules import Conv2d, ReLU, Flatten, Linear, Sequential
from simplenet.optimizers import Adam
from simplenet.losses import MSELoss

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "mnist")


def load_idx_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def load_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def accuracy(model, X, y_labels, batch_size=500):
    correct = 0
    for i in range(0, len(X), batch_size):
        pred = model.forward(Tensor(X[i:i + batch_size]))
        correct += np.sum(np.argmax(pred.data, axis=1) == y_labels[i:i + batch_size])
    return correct / len(X)


np.random.seed(0)

X_train = load_idx_images(os.path.join(DATA_DIR, "train-images.idx3-ubyte")).astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
y_train = load_idx_labels(os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
X_test = load_idx_images(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte")).astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
y_test = load_idx_labels(os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))
y_train_onehot = np.eye(10)[y_train]

# CrossEntropyLoss needs Tensor.log_softmax/__getitem__, which don't exist
# yet, and Tensor.softmax's backward only supports a single 1D vector (not
# a batch), so it can't be used for a batched loss either. MSELoss against
# one-hot targets sidesteps both and is a standard from-scratch approach.
model = Sequential(
    Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # (N, 8, 14, 14)
    ReLU(),
    Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # (N, 16, 7, 7)
    ReLU(),
    Flatten(),
    Linear(16 * 7 * 7, 64),
    ReLU(),
    Linear(64, 10),
)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = MSELoss()

num_train = X_train.shape[0]
batch_size = 128
epochs = 5

start = time.time()
for epoch in range(epochs):
    perm = np.random.permutation(num_train)
    epoch_loss = 0.0
    for i in range(0, num_train, batch_size):
        idx = perm[i:i + batch_size]
        x_batch = Tensor(X_train[idx])
        y_batch = Tensor(y_train_onehot[idx])

        pred = model.forward(x_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss.data) * len(idx)

    train_acc = accuracy(model, X_train[:2000], y_train[:2000])
    print(f"epoch {epoch}, loss {epoch_loss / num_train:.4f}, train acc (2000 subset) {train_acc:.4f}, "
          f"elapsed {time.time() - start:.1f}s")

test_acc = accuracy(model, X_test, y_test)
print(f"test accuracy: {test_acc:.4f}")
