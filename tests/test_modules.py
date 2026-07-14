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
from simplenet.modules import Module, Linear, Sequential, ReLU, Sigmoid, Tanh, Softmax
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
