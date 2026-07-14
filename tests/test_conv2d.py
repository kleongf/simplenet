"""
Unit tests for the Conv2d module in src/simplenet/modules.py.

Conv2d was broken in several ways that a pure shape/smoke test wouldn't
catch: forward() crashed on the bias broadcast and (for non-"same" padding)
on the final reshape; im2col returned a Tensor disconnected from the
autograd graph so gradients never reached the input; weight_reshaped was
also disconnected from self.weight so the kernel itself never received a
gradient (it would silently never train, with no crash); and
Tensor.__matmul__'s backward didn't support the 2D-weight-broadcast-over-
3D-batch shape Conv2d's im2col approach relies on. Forward correctness is
checked against a naive reference conv2d; gradients are checked against
finite differences for weight, bias, and input.
"""

import numpy as np
import pytest

from simplenet.tensor import Tensor
from simplenet.modules import Conv2d, ReLU, Flatten, Linear, Sequential
from simplenet.optimizers import Adam


def naive_conv2d(x, weight, bias, stride, padding):
    batch, in_c, h, w = x.shape
    out_c, _, kh, kw = weight.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    out = np.zeros((batch, out_c, out_h, out_w))
    for b in range(batch):
        for oc in range(out_c):
            for oy in range(out_h):
                for ox in range(out_w):
                    patch = x_padded[b, :, oy * stride:oy * stride + kh, ox * stride:ox * stride + kw]
                    out[b, oc, oy, ox] = np.sum(patch * weight[oc]) + bias[oc]
    return out


CONFIGS = [
    dict(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=0, xshape=(2, 2, 7, 7)),
    dict(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, xshape=(2, 2, 7, 7)),
    dict(in_channels=3, out_channels=4, kernel_size=2, stride=2, padding=0, xshape=(2, 3, 8, 8)),
    dict(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1, xshape=(3, 1, 9, 9)),
]


@pytest.mark.parametrize("cfg", CONFIGS)
class TestConv2dForward:
    def test_output_shape_matches_reference(self, cfg):
        cfg = dict(cfg)
        xshape = cfg.pop("xshape")
        conv = Conv2d(**cfg)
        x = Tensor(np.random.randn(*xshape))
        out = conv.forward(x)
        ref = naive_conv2d(x.data, conv.weight.data, conv.bias.data, conv.stride, conv.padding)
        assert out.data.shape == ref.shape

    def test_values_match_naive_reference(self, cfg):
        cfg = dict(cfg)
        xshape = cfg.pop("xshape")
        conv = Conv2d(**cfg)
        x = Tensor(np.random.randn(*xshape))
        out = conv.forward(x)
        ref = naive_conv2d(x.data, conv.weight.data, conv.bias.data, conv.stride, conv.padding)
        np.testing.assert_allclose(out.data, ref, atol=1e-8)


class TestConv2dParameters:
    def test_parameters_returns_weight_and_bias(self):
        conv = Conv2d(2, 3, kernel_size=3)
        assert conv.parameters() == [conv.weight, conv.bias]

    def test_weight_and_bias_shapes(self):
        conv = Conv2d(in_channels=3, out_channels=5, kernel_size=4)
        assert conv.weight.data.shape == (5, 3, 4, 4)
        assert conv.bias.data.shape == (5,)


class TestConv2dGradients:
    def test_weight_gradient_matches_finite_differences(self):
        np.random.seed(2)
        conv = Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1)
        x_data = np.random.randn(2, 2, 7, 7)
        x = Tensor(x_data.copy())

        def loss_fn(w, b):
            out = naive_conv2d(x_data, w, b, conv.stride, conv.padding)
            return (out ** 2).sum()

        loss = (conv.forward(x) ** 2).sum()
        loss.backward()

        eps = 1e-6
        base = loss_fn(conv.weight.data, conv.bias.data)
        num_grad = np.zeros_like(conv.weight.data)
        it = np.nditer(conv.weight.data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            perturbed = conv.weight.data.copy()
            perturbed[idx] += eps
            num_grad[idx] = (loss_fn(perturbed, conv.bias.data) - base) / eps

        np.testing.assert_allclose(conv.weight.grad, num_grad, atol=1e-3)

    def test_bias_gradient_matches_finite_differences(self):
        np.random.seed(2)
        conv = Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1)
        x_data = np.random.randn(2, 2, 7, 7)
        x = Tensor(x_data.copy())

        def loss_fn(b):
            out = naive_conv2d(x_data, conv.weight.data, b, conv.stride, conv.padding)
            return (out ** 2).sum()

        loss = (conv.forward(x) ** 2).sum()
        loss.backward()

        eps = 1e-6
        base = loss_fn(conv.bias.data)
        num_grad = np.zeros_like(conv.bias.data)
        for i in range(len(num_grad)):
            perturbed = conv.bias.data.copy()
            perturbed[i] += eps
            num_grad[i] = (loss_fn(perturbed) - base) / eps

        np.testing.assert_allclose(conv.bias.grad, num_grad, atol=1e-3)

    def test_input_gradient_matches_finite_differences(self):
        np.random.seed(2)
        conv = Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=2, padding=1)
        x_data = np.random.randn(2, 2, 7, 7)
        x = Tensor(x_data.copy())

        def loss_fn(xd):
            out = naive_conv2d(xd, conv.weight.data, conv.bias.data, conv.stride, conv.padding)
            return (out ** 2).sum()

        loss = (conv.forward(x) ** 2).sum()
        loss.backward()

        eps = 1e-6
        base = loss_fn(x_data)
        num_grad = np.zeros_like(x_data)
        it = np.nditer(x_data, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            perturbed = x_data.copy()
            perturbed[idx] += eps
            num_grad[idx] = (loss_fn(perturbed) - base) / eps

        np.testing.assert_allclose(x.grad, num_grad, atol=1e-3)

    def test_weight_receives_nonzero_gradient(self):
        # regression test: weight_reshaped used to be a Tensor rebuilt from
        # self.weight.data with no children, so self.weight.grad stayed
        # zero forever -- the kernel would silently never train.
        conv = Conv2d(2, 3, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 2, 5, 5))
        loss = conv.forward(x).sum()
        loss.backward()
        assert np.any(conv.weight.grad != 0)

    def test_gradient_flows_to_upstream_layer(self):
        # regression test: im2col used to return a Tensor with no children
        # and no _backward_fn, disconnecting it from the graph -- any layer
        # feeding into a Conv2d would get zero gradient.
        lin = Linear(3, 2 * 5 * 5)
        x = Tensor(np.random.randn(4, 3))
        conv_input = lin.forward(x).reshape((4, 2, 5, 5))
        conv = Conv2d(2, 3, kernel_size=3, padding=1)
        loss = conv.forward(conv_input).sum()
        loss.backward()
        assert np.any(lin.weight.grad != 0)


class TestConv2dIntegration:
    def test_trains_inside_a_sequential_with_optimizer(self):
        np.random.seed(0)
        model = Sequential(Conv2d(1, 4, kernel_size=3, stride=1, padding=1), ReLU(), Flatten(), Linear(4 * 6 * 6, 1))
        opt = Adam(model.parameters(), lr=0.01)

        X = Tensor(np.random.randn(20, 1, 6, 6))
        y_true = Tensor(X.data.sum(axis=(1, 2, 3)).reshape(-1, 1))

        losses = []
        for _ in range(100):
            y_pred = model.forward(X)
            loss = ((y_pred - y_true) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.data))

        assert losses[-1] < losses[0] * 0.1
