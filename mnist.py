import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from simplenet.dataset import ArrayDataset, train_test_split
from simplenet.modules import Conv2d, ReLU, Flatten, Linear, Sequential
from simplenet.optimizers import Adam
from simplenet.losses import CrossEntropyLoss
from simplenet.trainer import Trainer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "mnist")


def load_idx_images(path):
    with open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def load_idx_labels(path):
    with open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def accuracy(pred, y_true):
    return np.mean(np.argmax(pred, axis=1) == y_true)


np.random.seed(0)

X_train_all = load_idx_images(os.path.join(DATA_DIR, "train-images.idx3-ubyte")).astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
y_train_all = load_idx_labels(os.path.join(DATA_DIR, "train-labels.idx1-ubyte"))
X_test = load_idx_images(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte")).astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
y_test = load_idx_labels(os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))

train_ds, val_ds = train_test_split(ArrayDataset(X_train_all, y_train_all), test_size=0.05, random_state=0)
test_ds = ArrayDataset(X_test, y_test)

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

trainer = Trainer(
    model=model,
    optimizer=Adam(model.parameters(), lr=0.01),
    loss_fn=CrossEntropyLoss(),
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=128,
    metrics={"accuracy": accuracy},
)
trainer.fit(epochs=5)

test_result = trainer.evaluate(test_ds)
print(f"test loss: {test_result['loss']:.4f}, test accuracy: {test_result['accuracy']:.4f}")
