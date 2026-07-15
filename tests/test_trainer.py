"""
Unit tests for src/simplenet/trainer.py.

Trainer previously had `batch_size=...` (the literal Ellipsis object) as
its default, which crashed the instant fit()/evaluate() hit
`range(0, n, self.batch_size)` unless a caller passed batch_size
explicitly. `metrics` defaulted to `[]`, but evaluate() calls
`self.metrics.items()`, which only a dict supports -- so the default
crashed too, just one step later (inside evaluate(), e.g. via the
val_dataset path). fit() also never shuffled between epochs and never
used the val_dataset it stored. Several tests below guard those
regressions directly.
"""

import numpy as np
import pytest

from simplenet.tensor import Tensor
from simplenet.dataset import ArrayDataset
from simplenet.modules import Linear
from simplenet.optimizers import SGD, Adam
from simplenet.losses import MSELoss
from simplenet.trainer import Trainer


def make_linear_problem(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    true_w = np.array([[2.0], [-1.0]])
    y = X @ true_w + 0.5
    return ArrayDataset(X, y)


class TestTrainerDefaults:
    def test_default_batch_size_is_usable(self):
        # regression test: batch_size used to default to Ellipsis, which
        # crashed range(0, n, self.batch_size) immediately.
        ds = make_linear_problem(n=10)
        model = Linear(2, 1)
        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds)
        assert isinstance(trainer.batch_size, int) and trainer.batch_size > 0
        trainer.fit(epochs=1)

    def test_default_metrics_is_a_dict_not_a_list(self):
        # regression test: metrics used to default to [], but evaluate()
        # calls self.metrics.items(), which only dict supports.
        ds = make_linear_problem(n=10)
        model = Linear(2, 1)
        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds)
        assert trainer.metrics == {}
        result = trainer.evaluate(ds)
        assert set(result.keys()) == {"loss"}


class TestFit:
    def test_runs_without_error(self):
        ds = make_linear_problem(n=32)
        model = Linear(2, 1)
        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=8)
        trainer.fit(epochs=1)

    def test_loss_decreases_over_epochs(self):
        ds = make_linear_problem(n=200)
        model = Linear(2, 1)
        trainer = Trainer(model, Adam(model.parameters(), lr=0.05), MSELoss(), ds, batch_size=32)
        before = trainer.evaluate(ds)["loss"]
        trainer.fit(epochs=20)
        after = trainer.evaluate(ds)["loss"]
        assert after < before * 0.1

    def test_shuffles_batches_every_epoch(self, monkeypatch):
        ds = make_linear_problem(n=16)
        model = Linear(2, 1)
        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=4)

        calls = []
        real_permutation = np.random.permutation

        def spy(n):
            calls.append(n)
            return real_permutation(n)

        monkeypatch.setattr(np.random, "permutation", spy)
        trainer.fit(epochs=3)
        assert calls == [16, 16, 16]

    def test_uses_val_dataset_when_provided(self, capsys):
        # regression test: val_dataset used to be stored but never used.
        train_ds = make_linear_problem(n=32, seed=0)
        val_ds = make_linear_problem(n=16, seed=1)
        model = Linear(2, 1)
        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), train_ds, val_dataset=val_ds, batch_size=8)
        trainer.fit(epochs=1)
        out = capsys.readouterr().out
        assert "val_loss" in out

    def test_no_val_output_when_val_dataset_omitted(self, capsys):
        ds = make_linear_problem(n=16)
        model = Linear(2, 1)
        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=8)
        trainer.fit(epochs=1)
        out = capsys.readouterr().out
        assert "val_loss" not in out

    def test_collate_fn_applied_during_training(self):
        ds = make_linear_problem(n=16)
        model = Linear(2, 1)
        seen_shapes = []

        def collate(xb, yb):
            seen_shapes.append(yb.data.shape)
            return xb, yb

        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, collate_fn=collate, batch_size=4)
        trainer.fit(epochs=1)
        assert len(seen_shapes) == 4  # 16 samples / batch_size 4


class TestEvaluate:
    def test_loss_matches_manual_computation(self):
        ds = make_linear_problem(n=20)
        model = Linear(2, 1)
        result = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=32).evaluate(ds)

        xb, yb = ds[np.arange(len(ds))]
        expected = float(((model.forward(xb) - yb) ** 2).mean().data)
        assert result["loss"] == pytest.approx(expected, rel=1e-5)

    def test_custom_metric_is_computed(self):
        ds = make_linear_problem(n=20)
        model = Linear(2, 1)

        def within_one(pred, y_true):
            return float(np.mean(np.abs(pred - y_true) < 1.0))

        trainer = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=8, metrics={"within_one": within_one})
        result = trainer.evaluate(ds)
        assert set(result.keys()) == {"loss", "within_one"}
        assert 0.0 <= result["within_one"] <= 1.0

    def test_does_not_mutate_model_gradients(self):
        # evaluate() is forward-only: no zero_grad/backward/step should run.
        ds = make_linear_problem(n=16)
        model = Linear(2, 1)
        model.weight.grad = np.full_like(model.weight.data, 7.0)
        Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=8).evaluate(ds)
        np.testing.assert_allclose(model.weight.grad, np.full_like(model.weight.data, 7.0))

    def test_collate_fn_applied_during_evaluation(self):
        ds = make_linear_problem(n=16)
        model = Linear(2, 1)

        def double_target(xb, yb):
            return xb, Tensor(yb.data * 2)

        plain_loss = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=16).evaluate(ds)["loss"]
        collated_loss = Trainer(model, SGD(model.parameters(), lr=0.01), MSELoss(), ds, batch_size=16, collate_fn=double_target).evaluate(ds)["loss"]
        assert collated_loss != pytest.approx(plain_loss)
