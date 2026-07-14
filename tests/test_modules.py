"""
Unit tests for src/simplenet/modules.py.

Sequential.parameters() previously returned a list of per-layer parameter
lists (e.g. [[w1, b1], [], [w2, b2]]) instead of a flat list of Tensors,
which crashed as soon as an Optimizer tried to iterate over it. Several
tests below specifically guard against that regression, since a plain
forward-pass test wouldn't have caught it.
"""

import numpy as np
import pytest

from simplenet.tensor import Tensor
from simplenet.modules import (
    Module, Linear, Sequential, ReLU, Sigmoid, Tanh, Softmax,
    Flatten, Dropout, BatchNorm1D,
)
from simplenet.optimizers import SGD


def t(data):
    return Tensor(np.array(data, dtype=np.float64))


class TestModuleBase:
    def test_parameters_empty_by_default(self):
        assert Module().parameters() == []

    def test_zero_grad_with_no_parameters_does_not_raise(self):
        Module().zero_grad()


class TestLinear:
    def test_forward_shape(self):
        layer = Linear(3, 5)
        x = t(np.random.randn(10, 3))
        out = layer.forward(x)
        assert out.data.shape == (10, 5)

    def test_forward_shape_with_different_in_out_features(self):
        # regression check for the weight-transpose bug: in_features != out_features
        layer = Linear(4, 2)
        x = t(np.random.randn(7, 4))
        out = layer.forward(x)
        assert out.data.shape == (7, 2)

    def test_forward_matches_manual_computation(self):
        layer = Linear(3, 2)
        x = t(np.random.randn(5, 3))
        out = layer.forward(x)
        expected = x.data @ layer.weight.data.T + layer.bias.data
        np.testing.assert_allclose(out.data, expected)

    def test_parameters_returns_weight_and_bias(self):
        layer = Linear(3, 2)
        params = layer.parameters()
        assert params == [layer.weight, layer.bias]

    def test_zero_grad_resets_weight_and_bias(self):
        layer = Linear(2, 2)
        layer.weight.grad = np.ones_like(layer.weight.data)
        layer.bias.grad = np.ones_like(layer.bias.data)
        layer.zero_grad()
        np.testing.assert_allclose(layer.weight.grad, np.zeros_like(layer.weight.data))
        np.testing.assert_allclose(layer.bias.grad, np.zeros_like(layer.bias.data))

    def test_backward_matches_finite_differences(self):
        layer = Linear(3, 2)
        x_data = np.random.randn(4, 3)
        x = t(x_data)
        out = layer.forward(x).sum()
        out.backward()

        eps = 1e-6

        def loss(weight):
            return (x_data @ weight.T + layer.bias.data).sum()

        num_grad = np.zeros_like(layer.weight.data)
        for i in range(layer.weight.data.shape[0]):
            for j in range(layer.weight.data.shape[1]):
                perturbed = layer.weight.data.copy()
                perturbed[i, j] += eps
                num_grad[i, j] = (loss(perturbed) - loss(layer.weight.data)) / eps

        np.testing.assert_allclose(layer.weight.grad, num_grad, atol=1e-4)


class TestSequential:
    def test_forward_chains_layers_in_order(self):
        l1, l2 = Linear(3, 4), Linear(4, 2)
        model = Sequential(l1, ReLU(), l2)
        x = t(np.random.randn(5, 3))

        expected = l2.forward(l1.forward(x).relu())
        actual = model.forward(x)
        np.testing.assert_allclose(actual.data, expected.data)

    def test_forward_with_no_layers_returns_input_unchanged(self):
        model = Sequential()
        x = t([1.0, 2.0, 3.0])
        out = model.forward(x)
        np.testing.assert_allclose(out.data, x.data)

    def test_parameters_is_flat_not_nested(self):
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        params = model.parameters()
        assert all(isinstance(p, Tensor) for p in params)

    def test_parameters_returns_all_layer_parameters_in_order(self):
        l1, l2 = Linear(2, 4), Linear(4, 1)
        model = Sequential(l1, ReLU(), l2)
        assert model.parameters() == [l1.weight, l1.bias, l2.weight, l2.bias]

    def test_optimizer_integration_does_not_raise(self):
        # regression test: this crashed with AttributeError('list' object
        # has no attribute 'data') before Sequential.parameters() was fixed.
        model = Sequential(Linear(2, 4), ReLU(), Linear(4, 1))
        opt = SGD(model.parameters(), lr=0.01)
        x = t(np.random.randn(8, 2))
        out = model.forward(x).sum()
        opt.zero_grad()
        out.backward()
        opt.step()

    def test_trains_a_nonlinear_function(self):
        # a single Linear can't fit x0*x1; this only converges if Sequential
        # correctly composes Linear -> ReLU -> Linear and passes gradients
        # through the whole chain.
        np.random.seed(0)
        model = Sequential(Linear(2, 16), ReLU(), Linear(16, 1))
        opt = SGD(model.parameters(), lr=0.05)

        X = t(np.random.uniform(-2, 2, size=(200, 2)))
        y_true = t(X.data[:, 0:1] * X.data[:, 1:2])

        losses = []
        for _ in range(300):
            y_pred = model.forward(X)
            loss = ((y_pred - y_true) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.data))

        assert losses[-1] < losses[0]
        assert losses[-1] < 0.5


@pytest.mark.parametrize("module_cls,tensor_method", [
    (ReLU, "relu"),
    (Sigmoid, "sigmoid"),
    (Tanh, "tanh"),
    (Softmax, "softmax"),
])
class TestActivationModules:
    def test_forward_matches_underlying_tensor_method(self, module_cls, tensor_method):
        a = t([-1.0, 0.0, 2.0])
        b = t([-1.0, 0.0, 2.0])
        out_module = module_cls().forward(a)
        out_direct = getattr(b, tensor_method)()
        np.testing.assert_allclose(out_module.data, out_direct.data)

    def test_has_no_learnable_parameters(self, module_cls, tensor_method):
        assert module_cls().parameters() == []

    def test_backward_flows_through_module(self, module_cls, tensor_method):
        a = t([-1.0, 0.5, 2.0])
        b = t([-1.0, 0.5, 2.0])
        out_module = module_cls().forward(a).sum()
        out_direct = getattr(b, tensor_method)().sum()
        out_module.backward()
        out_direct.backward()
        np.testing.assert_allclose(a.grad, b.grad)


class TestFlatten:
    def test_forward_shape(self):
        x = t(np.random.randn(4, 3, 2))
        out = Flatten().forward(x)
        assert out.data.shape == (4, 6)

    def test_forward_preserves_row_major_values(self):
        x = t([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # (2, 2, 2)
        out = Flatten().forward(x)
        np.testing.assert_allclose(out.data, [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    def test_backward_reshapes_grad_back_to_input_shape(self):
        x = t(np.arange(24).reshape(4, 3, 2).astype(np.float64))
        out = Flatten().forward(x).sum()
        out.backward()
        np.testing.assert_allclose(x.grad, np.ones((4, 3, 2)))

    def test_gradient_flows_to_upstream_linear(self):
        lin = Linear(6, 2)
        x = t(np.random.randn(4, 3, 2))
        out = lin.forward(Flatten().forward(x)).sum()
        out.backward()
        assert np.any(lin.weight.grad != 0)


class TestDropout:
    def test_eval_mode_is_identity(self):
        drop = Dropout(p=0.5)
        drop.training = False
        x = t([1.0, 2.0, 3.0])
        out = drop.forward(x)
        np.testing.assert_allclose(out.data, x.data)

    def test_training_mode_zeros_out_roughly_p_fraction(self):
        np.random.seed(0)
        drop = Dropout(p=0.3)
        x = t(np.ones(5000))
        out = drop.forward(x)
        zero_fraction = np.mean(out.data == 0.0)
        assert abs(zero_fraction - 0.3) < 0.02

    def test_training_mode_scales_kept_units_by_inverse_keep_prob(self):
        np.random.seed(0)
        drop = Dropout(p=0.5)
        x = t(np.ones(2000))
        out = drop.forward(x)
        kept = out.data[out.data != 0.0]
        np.testing.assert_allclose(kept, 2.0)  # 1 / (1 - 0.5)

    def test_p_zero_keeps_everything_unscaled(self):
        drop = Dropout(p=0.0)
        x = t([1.0, 2.0, 3.0])
        out = drop.forward(x)
        np.testing.assert_allclose(out.data, [1.0, 2.0, 3.0])

    def test_gradient_flows_to_upstream_parameters(self):
        # regression test: forward() used to return a disconnected Tensor
        # (no children, no _backward_fn), which silently zeroed gradients
        # for every layer before the Dropout.
        lin = Linear(3, 5)
        x = t(np.random.randn(6, 3))
        h = lin.forward(x)
        out = Dropout(p=0.5).forward(h).sum()
        out.backward()
        assert np.any(lin.weight.grad != 0)

    def test_gradient_is_zero_at_dropped_positions(self):
        np.random.seed(0)
        drop = Dropout(p=0.5)
        x = t(np.random.randn(200))
        out = drop.forward(x)
        out.backward()
        dropped = out.data == 0.0
        assert np.all(x.grad[dropped] == 0.0)
        assert np.all(x.grad[~dropped] != 0.0)


class TestBatchNorm1D:
    def test_forward_output_shape(self):
        bn = BatchNorm1D(4)
        x = t(np.random.randn(10, 4))
        out = bn.forward(x)
        assert out.data.shape == (10, 4)

    def test_gamma_ones_beta_zeros_by_default(self):
        bn = BatchNorm1D(3)
        np.testing.assert_allclose(bn.gamma.data, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(bn.beta.data, [0.0, 0.0, 0.0])

    def test_parameters_returns_gamma_and_beta_only(self):
        bn = BatchNorm1D(3)
        assert bn.parameters() == [bn.gamma, bn.beta]

    def test_training_mode_normalizes_to_zero_mean_unit_variance(self):
        np.random.seed(0)
        bn = BatchNorm1D(4)
        x = t(np.random.randn(500, 4) * 7.0 + 3.0)
        out = bn.forward(x)
        np.testing.assert_allclose(out.data.mean(axis=0), np.zeros(4), atol=1e-6)
        np.testing.assert_allclose(out.data.std(axis=0), np.ones(4), atol=1e-6)

    def test_running_stats_update_towards_batch_statistics(self):
        bn = BatchNorm1D(2, momentum=0.1)
        x_data = np.array([[10.0, -10.0], [20.0, -20.0]])
        x = t(x_data)
        bn.forward(x)
        expected_mean = 0.1 * x_data.mean(axis=0)  # running_mean started at 0
        expected_var = 0.9 * 1.0 + 0.1 * x_data.var(axis=0)  # running_var started at 1
        np.testing.assert_allclose(bn.running_mean.data, expected_mean)
        np.testing.assert_allclose(bn.running_var.data, expected_var)

    def test_running_stats_stay_out_of_the_autograd_graph(self):
        # regression test: running_mean/running_var used to be rebuilt via
        # Tensor arithmetic each forward call, chaining every past batch's
        # computation into their _children forever.
        bn = BatchNorm1D(3)
        for _ in range(5):
            bn.forward(t(np.random.randn(8, 3)))
        assert bn.running_mean._children == set()
        assert bn.running_var._children == set()

    def test_eval_mode_uses_running_stats_not_batch_stats(self):
        bn = BatchNorm1D(2)
        bn.running_mean.data = np.array([5.0, -5.0])
        bn.running_var.data = np.array([4.0, 4.0])
        bn.training = False
        x = t([[5.0, -5.0]])
        out = bn.forward(x)
        # (x - running_mean) / sqrt(running_var + eps) == 0 regardless of x's own stats
        np.testing.assert_allclose(out.data, [[0.0, 0.0]], atol=1e-4)

    def test_backward_matches_finite_differences(self):
        np.random.seed(1)
        bn = BatchNorm1D(3)
        x_data = np.random.randn(6, 3)

        def loss_fn(xd, gamma, beta):
            mean = xd.mean(axis=0)
            var = ((xd - mean) ** 2).mean(axis=0)
            xn = (xd - mean) / np.sqrt(var + bn.eps)
            return ((gamma * xn + beta) ** 2).sum()

        x = t(x_data.copy())
        out = (bn.forward(x) ** 2).sum()
        out.backward()

        eps = 1e-6
        num_grad_x = np.zeros_like(x_data)
        for i in range(x_data.shape[0]):
            for j in range(x_data.shape[1]):
                perturbed = x_data.copy()
                perturbed[i, j] += eps
                num_grad_x[i, j] = (
                    loss_fn(perturbed, bn.gamma.data, bn.beta.data)
                    - loss_fn(x_data, bn.gamma.data, bn.beta.data)
                ) / eps

        np.testing.assert_allclose(x.grad, num_grad_x, atol=1e-3)

    def test_trains_inside_a_sequential_with_optimizer(self):
        np.random.seed(0)
        model = Sequential(Linear(3, 8), BatchNorm1D(8), ReLU(), Linear(8, 1))
        opt = SGD(model.parameters(), lr=0.01)
        x = t(np.random.randn(16, 3))
        y_true = t(np.random.randn(16, 1))

        losses = []
        for _ in range(50):
            y_pred = model.forward(x)
            loss = ((y_pred - y_true) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.data))

        assert losses[-1] < losses[0]
