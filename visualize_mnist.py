import base64
import json
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
OUTPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "mnist_inspector.html"


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
X_test_raw = load_idx_images(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"))
X_test = X_test_raw.astype(np.float64).reshape(-1, 1, 28, 28) / 255.0
y_test = load_idx_labels(os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))
y_train_onehot = np.eye(10)[y_train]

model = Sequential(
    Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
    ReLU(),
    Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
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
    for i in range(0, num_train, batch_size):
        idx = perm[i:i + batch_size]
        pred = model.forward(Tensor(X_train[idx]))
        loss = loss_fn(pred, Tensor(y_train_onehot[idx]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch}, elapsed {time.time() - start:.1f}s")

test_pred = model.forward(Tensor(X_test)).data
test_pred_labels = np.argmax(test_pred, axis=1)
test_acc = float(np.mean(test_pred_labels == y_test))
print(f"test accuracy: {test_acc:.4f}")

# Sample a mix that's actually interesting to inspect: every misclassified
# example (capped) plus enough correct ones to fill out a gallery.
rng = np.random.default_rng(0)
wrong_idx = np.where(test_pred_labels != y_test)[0]
right_idx = np.where(test_pred_labels == y_test)[0]
wrong_sample = rng.choice(wrong_idx, size=min(60, len(wrong_idx)), replace=False)
right_sample = rng.choice(right_idx, size=min(200, len(right_idx)), replace=False)
sample_idx = np.concatenate([wrong_sample, right_sample])
rng.shuffle(sample_idx)

records = []
for i in sample_idx:
    img_b64 = base64.b64encode(X_test_raw[i].tobytes()).decode("ascii")
    records.append({
        "img": img_b64,
        "true": int(y_test[i]),
        "pred": int(test_pred_labels[i]),
        "scores": [round(float(s), 4) for s in test_pred[i]],
        "correct": bool(test_pred_labels[i] == y_test[i]),
    })

payload = {
    "test_accuracy": test_acc,
    "test_count": int(len(y_test)),
    "sample_wrong_count": int(len(wrong_sample)),
    "sample_total_count": int(len(records)),
    "images": records,
}

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mnist_inspector_template.html")
with open(TEMPLATE_PATH, "r") as f:
    template = f.read()

html = template.replace("__MNIST_DATA__", json.dumps(payload))
with open(OUTPUT_PATH, "w") as f:
    f.write(html)

print(f"wrote {OUTPUT_PATH} ({len(html) / 1024:.0f} KB)")
