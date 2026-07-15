from simplenet.dataset import Dataset
import numpy as np

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_dataset, val_dataset=None, collate_fn=None, batch_size=32, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.metrics = metrics if metrics is not None else {}

    def fit(self, epochs, verbose=True):
        n = len(self.train_dataset)
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            total_loss = 0.0
            for batch_start in range(0, n, self.batch_size):
                idx = perm[batch_start:batch_start + self.batch_size]
                xb, yb = self.train_dataset[idx]
                # collate function: takes in two arguments X, y, and returns X, y, which are modified
                # i know it's different than most collate functions, but this is a simple implementation
                # will be fixed later
                if self.collate_fn:
                    xb, yb = self.collate_fn(xb, yb)

                y_pred = self.model.forward(xb)
                loss = self.loss_fn(y_pred, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.data) * len(idx)

            log = f"epoch {epoch}, loss {total_loss / n:.4f}"
            if self.val_dataset is not None:
                val_result = self.evaluate(self.val_dataset)
                log += ", " + ", ".join(f"val_{k} {v:.4f}" for k, v in val_result.items())
            if verbose:
                print(log)

    def evaluate(self, dataset: Dataset):
        total_loss = 0.0
        metric_totals = {name: 0.0 for name in self.metrics}
        n = len(dataset)

        for i in range(0, n, self.batch_size):
            idx = np.arange(i, min(i + self.batch_size, n))
            xb, yb = dataset[idx]
            if self.collate_fn:
                xb, yb = self.collate_fn(xb, yb)
            pred = self.model.forward(xb)
            loss = self.loss_fn(pred, yb)

            total_loss += float(loss.data) * len(idx)
            for name, metric_fn in self.metrics.items():
                metric_totals[name] += metric_fn(pred.data, yb.data) * len(idx)

        result = {"loss": total_loss / n}
        result.update({name: total / n for name, total in metric_totals.items()})
        return result
