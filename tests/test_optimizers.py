"""
Unit tests for src/simplenet/optimizers.py.

Each stateful optimizer (SGDMomentum, Adam, AdamW) is checked two ways:
(1) an exact hand-computed recurrence against the canonical update formula,
run step-by-step against a real Tensor, and (2) a convergence check on a
simple quadratic. (2) exists because a formula check on a single parameter
can pass while integration-level bugs -- wrong `enumerate()` order, state
arrays built for the wrong shape, a shared timestep incremented per
parameter instead of per step -- still break real training.
"""

import numpy as np
import pytest

from simplenet.tensor import Tensor
from simplenet.optimizers import Optimizer, SGD, SGDMomentum, Adam, AdamW


def t(data):
    return Tensor(np.array(data, dtype=np.float64))


class TestOptimizerBase:
    def test_zero_grad_resets_all_parameters(self):
        a, b = t([1.0, 2.0]), t([[1.0, 2.0], [3.0, 4.0]])
        a.grad = np.array([5.0, 5.0])
        b.grad = np.ones((2, 2))
        opt = SGD([a, b], lr=0.1)
        opt.zero_grad()
        np.testing.assert_allclose(a.grad, [0.0, 0.0])
        np.testing.assert_allclose(b.grad, np.zeros((2, 2)))

    def test_step_not_implemented_on_base(self):
        opt = Optimizer([t([1.0])], lr=0.1)
        with pytest.raises(NotImplementedError):
            opt.step()


class TestSGD:
    def test_step_updates_by_negative_gradient(self):
        p = t([1.0, 2.0])
        p.grad = np.array([0.5, -0.5])
        opt = SGD([p], lr=0.1)
        opt.step()
        np.testing.assert_allclose(p.data, [0.95, 2.05])

    def test_updates_multiple_parameters_independently(self):
        a, b = t([1.0]), t([10.0])
        a.grad = np.array([1.0])
        b.grad = np.array([2.0])
        opt = SGD([a, b], lr=0.1)
        opt.step()
        np.testing.assert_allclose(a.data, [0.9])
        np.testing.assert_allclose(b.data, [9.8])


class TestSGDMomentum:
    def test_matches_hand_computed_recurrence(self):
        p = t([10.0])
        opt = SGDMomentum([p], lr=0.1, beta=0.9)
        v, x = 0.0, 10.0
        for g in [2.0, 1.5, -1.0, 0.5]:
            p.grad = np.array([g])
            opt.step()
            v = 0.9 * v - 0.1 * g
            x = x + v
            assert p.data[0] == pytest.approx(x)
            assert opt.v[0][0] == pytest.approx(v)

    def test_velocity_persists_across_zero_grad(self):
        p = t([1.0])
        opt = SGDMomentum([p], lr=0.1, beta=0.5)
        p.grad = np.array([1.0])
        opt.step()
        v_after_first_step = opt.v[0][0]
        opt.zero_grad()
        assert p.grad[0] == 0.0
        assert opt.v[0][0] == pytest.approx(v_after_first_step)

    def test_state_shapes_match_each_parameter(self):
        weight, bias = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), t([1.0, 2.0])
        opt = SGDMomentum([weight, bias], lr=0.1)
        assert opt.v[0].shape == (2, 3)
        assert opt.v[1].shape == (2,)

    def test_converges_on_simple_quadratic(self):
        # minimize (x - 3)^2
        x = t([0.0])
        opt = SGDMomentum([x], lr=0.05, beta=0.5)
        for _ in range(200):
            x.grad = np.array([2 * (x.data[0] - 3)])
            opt.step()
        assert x.data[0] == pytest.approx(3.0, abs=1e-3)


class TestAdam:
    def test_matches_hand_computed_recurrence(self):
        p = t([1.0])
        opt = Adam([p], lr=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
        m, v, x = 0.0, 0.0, 1.0
        for i, g in enumerate([0.5, -0.3, 0.2, 0.1], start=1):
            p.grad = np.array([g])
            opt.step()
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            m_hat = m / (1 - 0.9 ** i)
            v_hat = v / (1 - 0.999 ** i)
            x = x - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
            assert p.data[0] == pytest.approx(x)

    def test_timestep_increments_once_per_step_call_not_per_parameter(self):
        weight, bias = t([[1.0, 2.0]]), t([1.0])
        opt = Adam([weight, bias], lr=0.01)
        weight.grad = np.ones((1, 2))
        bias.grad = np.ones(1)
        opt.step()
        assert opt.t == 1
        opt.step()
        assert opt.t == 2

    def test_state_shapes_match_each_parameter(self):
        weight, bias = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), t([1.0, 2.0])
        opt = Adam([weight, bias], lr=0.01)
        assert opt.m[0].shape == (2, 3) and opt.v[0].shape == (2, 3)
        assert opt.m[1].shape == (2,) and opt.v[1].shape == (2,)

    def test_converges_on_simple_quadratic(self):
        x = t([0.0])
        opt = Adam([x], lr=0.1)
        for _ in range(200):
            x.grad = np.array([2 * (x.data[0] - 3)])
            opt.step()
        assert x.data[0] == pytest.approx(3.0, abs=1e-3)


class TestAdamW:
    def test_matches_hand_computed_recurrence(self):
        p = t([1.0])
        opt = AdamW([p], lr=0.1, beta1=0.9, beta2=0.999, lambda_=0.01, epsilon=1e-8)
        m, v, x = 0.0, 0.0, 1.0
        for i, g in enumerate([0.5, -0.3, 0.2, 0.1], start=1):
            p.grad = np.array([g])
            opt.step()
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            m_hat = m / (1 - 0.9 ** i)
            v_hat = v / (1 - 0.999 ** i)
            x = x - 0.1 * (0.01 * x + m_hat / (np.sqrt(v_hat) + 1e-8))
            assert p.data[0] == pytest.approx(x)

    def test_decoupled_weight_decay_shrinks_with_zero_gradient(self):
        p = t([10.0])
        opt = AdamW([p], lr=0.1, lambda_=0.5)
        for _ in range(5):
            p.grad = np.array([0.0])
            opt.step()
        # with grad == 0, m_hat/v_hat contribute 0 -> pure decay: x *= (1 - lr*lambda)
        expected = 10.0 * (1 - 0.1 * 0.5) ** 5
        assert p.data[0] == pytest.approx(expected)

    def test_zero_lambda_behaves_like_plain_adam(self):
        p_adamw, p_adam = t([1.0]), t([1.0])
        opt_w = AdamW([p_adamw], lr=0.1, lambda_=0.0)
        opt_a = Adam([p_adam], lr=0.1)
        for g in [0.5, -0.3, 0.2]:
            p_adamw.grad = np.array([g])
            p_adam.grad = np.array([g])
            opt_w.step()
            opt_a.step()
        assert p_adamw.data[0] == pytest.approx(p_adam.data[0])

    def test_state_shapes_match_each_parameter(self):
        weight, bias = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), t([1.0, 2.0])
        opt = AdamW([weight, bias], lr=0.01)
        assert opt.m[0].shape == (2, 3) and opt.v[0].shape == (2, 3)
        assert opt.m[1].shape == (2,) and opt.v[1].shape == (2,)

    def test_converges_on_simple_quadratic(self):
        x = t([0.0])
        opt = AdamW([x], lr=0.1, lambda_=0.0)
        for _ in range(200):
            x.grad = np.array([2 * (x.data[0] - 3)])
            opt.step()
        assert x.data[0] == pytest.approx(3.0, abs=1e-3)
