"""
Unit tests for src/simplenet/losses.py.

CrossEntropyLoss and BCELoss were both unusable until Tensor grew the
primitives they're built from: `log()` (missing entirely -- both losses
call it), a batch-correct `softmax()` backward (previously only valid for
a single 1-D vector), `__getitem__` (needed for CrossEntropyLoss's
per-row class selection), and `__rsub__` (needed for BCELoss's `1 - pred`).
"""

import numpy as np
import pytest

from simplenet.tensor import Tensor
from simplenet.losses import MSELoss, MAELoss, CrossEntropyLoss, BCELoss


def t(data):
    return Tensor(np.array(data, dtype=np.float64))


class TestMSELoss:
    def test_forward_mean_reduction(self):
        pred, target = t([1.0, 2.0, 3.0]), t([0.0, 0.0, 0.0])
        loss = MSELoss()(pred, target)
        assert loss.data == pytest.approx((1.0 + 4.0 + 9.0) / 3)

    def test_forward_sum_reduction(self):
        pred, target = t([1.0, 2.0]), t([0.0, 0.0])
        loss = MSELoss(reduction="sum")(pred, target)
        assert loss.data == pytest.approx(5.0)

    def test_backward_matches_finite_differences(self):
        pred_data = np.array([1.5, -2.0, 0.3])
        target_data = np.array([1.0, 0.0, 0.0])
        eps = 1e-6

        def loss_fn(p):
            return np.mean((p - target_data) ** 2)

        pred = Tensor(pred_data.copy())
        loss = MSELoss()(pred, Tensor(target_data))
        loss.backward()

        num_grad = np.zeros_like(pred_data)
        for i in range(len(pred_data)):
            perturbed = pred_data.copy()
            perturbed[i] += eps
            num_grad[i] = (loss_fn(perturbed) - loss_fn(pred_data)) / eps
        np.testing.assert_allclose(pred.grad, num_grad, atol=1e-4)


class TestMAELoss:
    def test_forward(self):
        pred, target = t([1.0, -2.0, 3.0]), t([0.0, 0.0, 0.0])
        loss = MAELoss()(pred, target)
        assert loss.data == pytest.approx((1.0 + 2.0 + 3.0) / 3)

    def test_backward_matches_sign(self):
        pred, target = t([2.0, -3.0]), t([0.0, 0.0])
        loss = MAELoss()(pred, target)
        loss.backward()
        np.testing.assert_allclose(pred.grad, [0.5, -0.5])


class TestCrossEntropyLoss:
    def _reference(self, pred, target):
        shifted = pred - pred.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        log_probs = shifted - logsumexp
        return -log_probs[np.arange(len(target)), target]

    def test_forward_matches_reference(self):
        np.random.seed(0)
        pred_data = np.random.randn(5, 4)
        target_data = np.array([0, 3, 1, 2, 0])
        loss = CrossEntropyLoss()(Tensor(pred_data), Tensor(target_data))
        expected = self._reference(pred_data, target_data).mean()
        assert loss.data == pytest.approx(expected)

    def test_backward_matches_closed_form_softmax_minus_onehot(self):
        # d(mean CE)/d(logits) = (softmax(pred) - one_hot(target)) / batch_size
        np.random.seed(1)
        pred_data = np.random.randn(6, 5)
        target_data = np.array([0, 4, 2, 1, 3, 0])

        pred = Tensor(pred_data.copy())
        loss = CrossEntropyLoss()(pred, Tensor(target_data))
        loss.backward()

        def softmax_np(x):
            e = np.exp(x - x.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)

        one_hot = np.eye(5)[target_data]
        expected_grad = (softmax_np(pred_data) - one_hot) / len(target_data)
        np.testing.assert_allclose(pred.grad, expected_grad, atol=1e-8)

    def test_backward_matches_finite_differences(self):
        np.random.seed(2)
        pred_data = np.random.randn(4, 3)
        target_data = np.array([1, 0, 2, 1])
        eps = 1e-6

        def loss_fn(p):
            return self._reference(p, target_data).mean()

        pred = Tensor(pred_data.copy())
        loss = CrossEntropyLoss()(pred, Tensor(target_data))
        loss.backward()

        num_grad = np.zeros_like(pred_data)
        it = np.nditer(pred_data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            perturbed = pred_data.copy()
            perturbed[idx] += eps
            num_grad[idx] = (loss_fn(perturbed) - loss_fn(pred_data)) / eps
        np.testing.assert_allclose(pred.grad, num_grad, atol=1e-4)

    def test_correct_confident_prediction_has_low_loss(self):
        pred_data = np.array([[10.0, -10.0, -10.0]])
        loss = CrossEntropyLoss()(Tensor(pred_data), Tensor(np.array([0])))
        assert loss.data < 0.01


class TestBCELoss:
    def _reference(self, pred, target):
        return np.mean(-(target * np.log(pred) + (1 - target) * np.log(1 - pred)))

    def test_forward_matches_reference(self):
        np.random.seed(0)
        pred_data = np.random.uniform(0.05, 0.95, size=(6, 1))
        target_data = np.random.randint(0, 2, size=(6, 1)).astype(np.float64)
        loss = BCELoss()(Tensor(pred_data), Tensor(target_data))
        assert loss.data == pytest.approx(self._reference(pred_data, target_data))

    def test_backward_matches_finite_differences(self):
        np.random.seed(3)
        pred_data = np.random.uniform(0.05, 0.95, size=(6, 1))
        target_data = np.random.randint(0, 2, size=(6, 1)).astype(np.float64)
        eps = 1e-6

        def loss_fn(p):
            return self._reference(p, target_data)

        pred = Tensor(pred_data.copy())
        loss = BCELoss()(pred, Tensor(target_data))
        loss.backward()

        num_grad = np.zeros_like(pred_data)
        it = np.nditer(pred_data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            perturbed = pred_data.copy()
            perturbed[idx] += eps
            num_grad[idx] = (loss_fn(perturbed) - loss_fn(pred_data)) / eps
        np.testing.assert_allclose(pred.grad, num_grad, atol=1e-4)
