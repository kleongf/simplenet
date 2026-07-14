"""
Unit tests for core/tensor.py.

For single-op tests, `_backward_fn` is called directly after seeding
`out.grad` so each op is checked in isolation. Multi-op tests use the
public `Tensor.backward()` (topological sort + reverse traversal).

NOTE: `softmax`'s backward pass builds a Jacobian with `np.diagflat` /
`np.outer`, which only makes sense for a single 1-D probability vector.
For batched (2-D) input it flattens across the batch and produces a
Jacobian of the wrong shape (see TestSoftmax.test_backward_batched_input_is_unsupported).
"""

import numpy as np
import pytest

from simplenet.tensor import Tensor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def t(data):
    return Tensor(np.array(data, dtype=np.float64))


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

class TestInit:
    def test_stores_data(self):
        a = t([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(a.data, np.array([1.0, 2.0, 3.0]))

    def test_grad_initialized_to_zero_with_same_shape(self):
        a = t([[1.0, 2.0], [3.0, 4.0]])
        assert a.grad.shape == a.data.shape
        np.testing.assert_array_equal(a.grad, np.zeros((2, 2)))

    def test_no_children_by_default(self):
        a = t([1.0])
        assert a._children == set()

    def test_children_recorded(self):
        a, b = t([1.0]), t([2.0])
        out = a + b
        assert out._children == {a, b}

    def test_backward_fn_initially_none_for_leaf(self):
        a = t([1.0])
        assert a._backward_fn is None


class TestGetShape:
    def test_scalar(self):
        a = t([1.0])
        assert a._get_shape(5) == ()

    def test_flat_list(self):
        a = t([1.0])
        assert a._get_shape([1, 2, 3]) == (3,)

    def test_nested_list(self):
        a = t([1.0])
        assert a._get_shape([[1, 2], [3, 4], [5, 6]]) == (3, 2)

    def test_empty_list(self):
        a = t([1.0])
        assert a._get_shape([]) == (0,)


# ---------------------------------------------------------------------------
# __add__
# ---------------------------------------------------------------------------

class TestAdd:
    def test_forward(self):
        a, b = t([1.0, 2.0, 3.0]), t([4.0, 5.0, 6.0])
        out = a + b
        np.testing.assert_allclose(out.data, [5.0, 7.0, 9.0])

    def test_backward_same_shape(self):
        a, b = t([1.0, 2.0, 3.0]), t([4.0, 5.0, 6.0])
        out = a + b
        out.grad = np.array([1.0, 1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(b.grad, [1.0, 1.0, 1.0])

    def test_backward_accumulates_into_existing_grad(self):
        a, b = t([1.0, 2.0]), t([3.0, 4.0])
        a.grad = np.array([10.0, 10.0])
        out = a + b
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [11.0, 11.0])

    def test_backward_broadcasting_sums_extra_dims(self):
        # a: (2, 3), b: (3,) -> b is broadcast to every row of a.
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = t([10.0, 20.0, 30.0])
        out = a + b
        out.grad = np.ones((2, 3))
        out._backward_fn()
        np.testing.assert_allclose(a.grad, np.ones((2, 3)))
        # b's gradient should be summed over the broadcast (row) dimension.
        np.testing.assert_allclose(b.grad, [2.0, 2.0, 2.0])


# ---------------------------------------------------------------------------
# __mul__
# ---------------------------------------------------------------------------

class TestMul:
    def test_forward(self):
        a, b = t([1.0, 2.0, 3.0]), t([4.0, 5.0, 6.0])
        out = a * b
        np.testing.assert_allclose(out.data, [4.0, 10.0, 18.0])

    def test_backward(self):
        a, b = t([2.0, 3.0]), t([5.0, 7.0])
        out = a * b
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        # d(a*b)/da = b, d(a*b)/db = a
        np.testing.assert_allclose(a.grad, [5.0, 7.0])
        np.testing.assert_allclose(b.grad, [2.0, 3.0])

    def test_backward_with_upstream_gradient(self):
        a, b = t([2.0, 3.0]), t([5.0, 7.0])
        out = a * b
        out.grad = np.array([2.0, 10.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [10.0, 70.0])
        np.testing.assert_allclose(b.grad, [4.0, 30.0])

    def test_backward_broadcasting(self):
        a = t([[1.0, 2.0], [3.0, 4.0]])
        b = t([2.0, 10.0])
        out = a * b
        out.grad = np.ones((2, 2))
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [[2.0, 10.0], [2.0, 10.0]])
        # d(a*b)/db summed over rows: sum(a, axis=0)
        np.testing.assert_allclose(b.grad, [4.0, 6.0])


# ---------------------------------------------------------------------------
# scalar arithmetic (tensor OP scalar and scalar OP tensor)
# ---------------------------------------------------------------------------

class TestScalarArithmetic:
    def test_add_scalar_forward(self):
        a = t([1.0, 2.0, 3.0])
        out = a + 5.0
        np.testing.assert_allclose(out.data, [6.0, 7.0, 8.0])

    def test_radd_scalar_forward(self):
        a = t([1.0, 2.0, 3.0])
        out = 5.0 + a
        np.testing.assert_allclose(out.data, [6.0, 7.0, 8.0])

    def test_add_scalar_backward(self):
        a = t([1.0, 2.0])
        out = a + 5.0
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [1.0, 1.0])

    def test_mul_scalar_forward(self):
        a = t([1.0, 2.0, 3.0])
        out = a * 2.0
        np.testing.assert_allclose(out.data, [2.0, 4.0, 6.0])

    def test_rmul_scalar_forward(self):
        a = t([1.0, 2.0, 3.0])
        out = 2.0 * a
        np.testing.assert_allclose(out.data, [2.0, 4.0, 6.0])

    def test_mul_scalar_backward(self):
        a = t([1.0, 2.0])
        out = a * 3.0
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [3.0, 3.0])

    def test_rmul_scalar_backward(self):
        a = t([1.0, 2.0])
        out = 3.0 * a
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [3.0, 3.0])

    def test_sub_scalar_forward(self):
        a = t([5.0, 6.0])
        out = a - 2.0
        np.testing.assert_allclose(out.data, [3.0, 4.0])

    def test_truediv_scalar_forward(self):
        a = t([6.0, 9.0])
        out = a / 3.0
        np.testing.assert_allclose(out.data, [2.0, 3.0])

    def test_scalar_op_chain_matches_finite_differences(self):
        a_data = np.array([1.0, 2.0, 3.0])
        eps = 1e-6

        def loss(a_arr):
            return ((2.0 * a_arr + 1.0) ** 2).sum()

        a = Tensor(a_data.copy())
        out = ((2.0 * a + 1.0) ** 2).sum()
        out.backward()

        num_grad = np.zeros_like(a_data)
        for i in range(len(a_data)):
            perturbed = a_data.copy()
            perturbed[i] += eps
            num_grad[i] = (loss(perturbed) - loss(a_data)) / eps

        np.testing.assert_allclose(a.grad, num_grad, atol=1e-4)


# ---------------------------------------------------------------------------
# __exp__  (note: not a real Python dunder hook; must be called explicitly)
# ---------------------------------------------------------------------------

class TestExp:
    def test_forward(self):
        a = t([0.0, 1.0, 2.0])
        out = a.__exp__()
        np.testing.assert_allclose(out.data, np.exp([0.0, 1.0, 2.0]))

    def test_backward(self):
        a = t([0.0, 1.0, 2.0])
        out = a.__exp__()
        out.grad = np.ones(3)
        out._backward_fn()
        # d(exp(x))/dx = exp(x)
        np.testing.assert_allclose(a.grad, np.exp([0.0, 1.0, 2.0]))

    def test_backward_scales_with_upstream_grad(self):
        a = t([1.0])
        out = a.__exp__()
        out.grad = np.array([3.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, 3.0 * np.exp([1.0]))


# ---------------------------------------------------------------------------
# __matmul__
# ---------------------------------------------------------------------------

class TestMatmul:
    def test_forward(self):
        a = t([[1.0, 2.0], [3.0, 4.0]])
        b = t([[5.0, 6.0], [7.0, 8.0]])
        out = a @ b
        np.testing.assert_allclose(out.data, [[19.0, 22.0], [43.0, 50.0]])

    def test_backward(self):
        a = t([[1.0, 2.0], [3.0, 4.0]])
        b = t([[5.0, 6.0], [7.0, 8.0]])
        out = a @ b
        out.grad = np.ones((2, 2))
        out._backward_fn()
        np.testing.assert_allclose(a.grad, np.ones((2, 2)) @ b.data.T)
        np.testing.assert_allclose(b.grad, a.data.T @ np.ones((2, 2)))

    def test_backward_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        a_data = rng.normal(size=(3, 4))
        b_data = rng.normal(size=(4, 2))
        eps = 1e-6

        def loss(a_arr, b_arr):
            return (a_arr @ b_arr).sum()

        a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
        out = a @ b
        out.grad = np.ones_like(out.data)
        out._backward_fn()

        num_grad_a = np.zeros_like(a_data)
        for i in range(a_data.shape[0]):
            for j in range(a_data.shape[1]):
                perturbed = a_data.copy()
                perturbed[i, j] += eps
                num_grad_a[i, j] = (loss(perturbed, b_data) - loss(a_data, b_data)) / eps

        np.testing.assert_allclose(a.grad, num_grad_a, atol=1e-4)


# ---------------------------------------------------------------------------
# reshape / transpose
# ---------------------------------------------------------------------------

class TestReshape:
    def test_forward(self):
        a = t([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = a.reshape((2, 3))
        assert out.data.shape == (2, 3)
        np.testing.assert_allclose(out.data, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_backward(self):
        a = t([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        out = a.reshape((2, 3))
        out.grad = np.arange(6).reshape(2, 3).astype(np.float64)
        out._backward_fn()
        np.testing.assert_allclose(a.grad, np.arange(6))


class TestTranspose:
    def test_forward(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.transpose()
        np.testing.assert_allclose(out.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_backward(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.transpose()
        out.grad = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------

class TestSum:
    def test_forward_no_axis(self):
        a = t([[1.0, 2.0], [3.0, 4.0]])
        out = a.sum()
        assert out.data == pytest.approx(10.0)

    def test_backward_no_axis(self):
        a = t([[1.0, 2.0], [3.0, 4.0]])
        out = a.sum()
        out.grad = np.array(1.0)
        out._backward_fn()
        np.testing.assert_allclose(a.grad, np.ones((2, 2)))

    def test_backward_with_upstream_scale(self):
        a = t([1.0, 2.0, 3.0])
        out = a.sum()
        out.grad = np.array(5.0)
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [5.0, 5.0, 5.0])

    def test_forward_axis_no_keepdims(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.sum(axis=0)
        np.testing.assert_allclose(out.data, [5.0, 7.0, 9.0])
        assert out.data.shape == (3,)

    def test_backward_axis_no_keepdims(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.sum(axis=0)
        out.grad = np.array([1.0, 2.0, 3.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    def test_forward_axis_keepdims(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.sum(axis=1, keepdims=True)
        assert out.data.shape == (2, 1)
        np.testing.assert_allclose(out.data, [[6.0], [15.0]])

    def test_backward_axis_keepdims(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.sum(axis=1, keepdims=True)
        out.grad = np.array([[1.0], [2.0]])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])


# ---------------------------------------------------------------------------
# mean
# ---------------------------------------------------------------------------

class TestMean:
    def test_forward_no_axis(self):
        a = t([[1.0, 2.0], [3.0, 4.0]])
        out = a.mean()
        assert out.data == pytest.approx(2.5)

    def test_backward_no_axis(self):
        a = t([1.0, 2.0, 3.0, 4.0])
        out = a.mean()
        out.grad = np.array(1.0)
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [0.25, 0.25, 0.25, 0.25])

    def test_forward_axis(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.mean(axis=1)
        np.testing.assert_allclose(out.data, [2.0, 5.0])

    def test_backward_axis(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = a.mean(axis=1)
        out.grad = np.array([3.0, 6.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    def test_backward_via_full_graph_matches_finite_differences(self):
        a_data = np.array([1.0, 2.0, 3.0, 4.0])
        eps = 1e-6

        def loss(a_arr):
            return (a_arr ** 2).mean()

        a = Tensor(a_data.copy())
        out = (a ** 2).mean()
        out.backward()

        num_grad = np.zeros_like(a_data)
        for i in range(len(a_data)):
            perturbed = a_data.copy()
            perturbed[i] += eps
            num_grad[i] = (loss(perturbed) - loss(a_data)) / eps

        np.testing.assert_allclose(a.grad, num_grad, atol=1e-4)


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------

class TestRelu:
    def test_forward(self):
        a = t([-2.0, -0.1, 0.0, 0.1, 2.0])
        out = a.relu()
        np.testing.assert_allclose(out.data, [0.0, 0.0, 0.0, 0.1, 2.0])

    def test_backward_positive_and_negative(self):
        a = t([-1.0, 2.0])
        out = a.relu()
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [0.0, 1.0])

    def test_backward_at_zero_is_zero(self):
        # Implementation documents 0-gradient at x == 0 (uses strict >).
        a = t([0.0])
        out = a.relu()
        out.grad = np.array([1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [0.0])


class TestSigmoid:
    def test_forward(self):
        a = t([0.0])
        out = a.sigmoid()
        np.testing.assert_allclose(out.data, [0.5])

    def test_backward(self):
        a = t([0.0, 1.0, -1.0])
        out = a.sigmoid()
        out.grad = np.ones(3)
        out._backward_fn()
        s = 1.0 / (1.0 + np.exp(-a.data))
        np.testing.assert_allclose(a.grad, s * (1.0 - s))

    def test_backward_matches_finite_differences(self):
        eps = 1e-6
        x0 = 0.37
        a = t([x0])
        out = a.sigmoid()
        out.grad = np.array([1.0])
        out._backward_fn()

        def sig(x):
            return 1.0 / (1.0 + np.exp(-x))

        numeric = (sig(x0 + eps) - sig(x0 - eps)) / (2 * eps)
        assert a.grad[0] == pytest.approx(numeric, abs=1e-5)


class TestTanh:
    def test_forward(self):
        a = t([0.0])
        out = a.tanh()
        np.testing.assert_allclose(out.data, [0.0])

    def test_backward(self):
        a = t([0.0, 0.5, -0.5])
        out = a.tanh()
        out.grad = np.ones(3)
        out._backward_fn()
        np.testing.assert_allclose(a.grad, 1.0 - np.tanh(a.data) ** 2)


class TestSoftmax:
    def test_forward_sums_to_one(self):
        a = t([1.0, 2.0, 3.0])
        out = a.softmax()
        assert out.data.sum() == pytest.approx(1.0)
        assert np.all(out.data > 0)

    def test_forward_is_shift_invariant(self):
        a = t([1.0, 2.0, 3.0])
        b = t([101.0, 102.0, 103.0])
        np.testing.assert_allclose(a.softmax().data, b.softmax().data, atol=1e-10)

    def test_forward_batched_axis(self):
        a = t([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        out = a.softmax(axis=-1)
        np.testing.assert_allclose(out.data.sum(axis=-1), [1.0, 1.0])
        np.testing.assert_allclose(out.data[1], [1 / 3, 1 / 3, 1 / 3])

    def test_backward_matches_analytic_jacobian_for_1d_vector(self):
        a = t([1.0, 2.0, 3.0])
        out = a.softmax()
        out.grad = np.array([1.0, 0.0, 0.0])
        out._backward_fn()
        s = out.data
        expected_jacobian = np.diagflat(s) - np.outer(s, s)
        expected = expected_jacobian @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(a.grad, expected)

    def test_backward_batched_input_is_unsupported(self):
        # Known limitation: the Jacobian is built with diagflat/outer on the
        # full (flattened) output, which is only valid for a single 1-D
        # probability vector. For 2-D (batched) input the resulting
        # Jacobian shape doesn't match out.grad's shape.
        a = t([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        out = a.softmax(axis=-1)
        out.grad = np.ones_like(out.data)
        with pytest.raises(ValueError):
            out._backward_fn()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_does_not_raise(self):
        a = t([1.0, 2.0, 3.0])
        repr(a)

    def test_repr_contains_data(self):
        a = t([1.0, 2.0, 3.0])
        assert "1." in repr(a) or "1.0" in repr(a)


# ---------------------------------------------------------------------------
# __pow__
# ---------------------------------------------------------------------------

class TestPow:
    def test_forward(self):
        a = t([2.0, 3.0, 4.0])
        out = a ** 2
        np.testing.assert_allclose(out.data, [4.0, 9.0, 16.0])

    def test_backward(self):
        a = t([2.0, 3.0])
        out = a ** 2
        out.grad = np.ones(2)
        out._backward_fn()
        # d(x^2)/dx = 2x
        np.testing.assert_allclose(a.grad, [4.0, 6.0])

    def test_forward_negative_power(self):
        a = t([2.0, 4.0])
        out = a ** -1
        np.testing.assert_allclose(out.data, [0.5, 0.25])

    def test_backward_negative_power(self):
        a = t([2.0, 4.0])
        out = a ** -1
        out.grad = np.ones(2)
        out._backward_fn()
        # d(x^-1)/dx = -x^-2
        np.testing.assert_allclose(a.grad, [-0.25, -0.0625])


# ---------------------------------------------------------------------------
# __neg__
# ---------------------------------------------------------------------------

class TestNeg:
    def test_forward(self):
        a = t([1.0, -2.0, 3.0])
        out = -a
        np.testing.assert_allclose(out.data, [-1.0, 2.0, -3.0])

    def test_backward(self):
        a = t([1.0, -2.0])
        out = -a
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [-1.0, -1.0])


# ---------------------------------------------------------------------------
# abs
# ---------------------------------------------------------------------------

class TestAbs:
    def test_forward(self):
        a = t([1.0, -2.0, 0.0, -3.5])
        out = a.abs()
        np.testing.assert_allclose(out.data, [1.0, 2.0, 0.0, 3.5])

    def test_backward(self):
        a = t([2.0, -3.0])
        out = a.abs()
        out.grad = np.array([1.0, 1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [1.0, -1.0])

    def test_backward_scales_with_upstream_grad(self):
        a = t([2.0, -3.0])
        out = a.abs()
        out.grad = np.array([5.0, 2.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [5.0, -2.0])

    def test_backward_at_zero_is_zero(self):
        # Implementation uses np.sign(x), which is 0 at x == 0.
        a = t([0.0])
        out = a.abs()
        out.grad = np.array([1.0])
        out._backward_fn()
        np.testing.assert_allclose(a.grad, [0.0])


# ---------------------------------------------------------------------------
# __sub__
# ---------------------------------------------------------------------------

class TestSub:
    def test_forward(self):
        a, b = t([5.0, 3.0]), t([2.0, 1.0])
        out = a - b
        np.testing.assert_allclose(out.data, [3.0, 2.0])

    def test_backward(self):
        a, b = t([5.0, 3.0]), t([2.0, 1.0])
        out = a - b
        out.backward()
        np.testing.assert_allclose(a.grad, [1.0, 1.0])
        np.testing.assert_allclose(b.grad, [-1.0, -1.0])

    def test_backward_broadcasting(self):
        a = t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = t([1.0, 1.0, 1.0])
        out = (a - b).sum()
        out.backward()
        np.testing.assert_allclose(a.grad, np.ones((2, 3)))
        np.testing.assert_allclose(b.grad, [-2.0, -2.0, -2.0])


# ---------------------------------------------------------------------------
# __truediv__
# ---------------------------------------------------------------------------

class TestTrueDiv:
    def test_forward(self):
        a, b = t([6.0, 9.0]), t([2.0, 3.0])
        out = a / b
        np.testing.assert_allclose(out.data, [3.0, 3.0])

    def test_backward(self):
        a, b = t([6.0, 9.0]), t([2.0, 3.0])
        out = a / b
        out.backward()
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        np.testing.assert_allclose(a.grad, [0.5, 1 / 3])
        np.testing.assert_allclose(b.grad, [-1.5, -1.0])

    def test_backward_matches_finite_differences(self):
        a_data = np.array([6.0, 9.0, -3.0])
        b_data = np.array([2.0, 3.0, 4.0])
        eps = 1e-6

        def loss(a_arr, b_arr):
            return (a_arr / b_arr).sum()

        a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
        out = (a / b).sum()
        out.backward()

        num_grad_b = np.zeros_like(b_data)
        for i in range(len(b_data)):
            perturbed = b_data.copy()
            perturbed[i] += eps
            num_grad_b[i] = (loss(a_data, perturbed) - loss(a_data, b_data)) / eps

        np.testing.assert_allclose(b.grad, num_grad_b, atol=1e-4)


# ---------------------------------------------------------------------------
# backward() -- topological sort + full-graph reverse traversal
# ---------------------------------------------------------------------------

class TestBackward:
    def test_seeds_root_grad_with_ones(self):
        a = t([1.0, 2.0, 3.0])
        out = a.sum()
        out.backward()
        assert out.grad == pytest.approx(1.0)

    def test_leaf_grad_simple_chain(self):
        a = t([1.0, 2.0, 3.0])
        out = a.sum()
        out.backward()
        np.testing.assert_allclose(a.grad, [1.0, 1.0, 1.0])

    def test_accumulates_when_same_tensor_used_twice(self):
        # out = a * a -> d(out)/da = 2a
        a = t([3.0])
        out = a * a
        out.backward()
        np.testing.assert_allclose(a.grad, [6.0])

    def test_multi_op_chain_matches_finite_differences(self):
        # loss = ((a * b) + a).sum()
        a_data = np.array([1.0, 2.0, 3.0])
        b_data = np.array([4.0, 5.0, 6.0])
        eps = 1e-6

        def loss(a_arr, b_arr):
            return (a_arr * b_arr + a_arr).sum()

        a, b = Tensor(a_data.copy()), Tensor(b_data.copy())
        out = (a * b + a).sum()
        out.backward()

        num_grad_a = np.zeros_like(a_data)
        for i in range(len(a_data)):
            perturbed = a_data.copy()
            perturbed[i] += eps
            num_grad_a[i] = (loss(perturbed, b_data) - loss(a_data, b_data)) / eps

        np.testing.assert_allclose(a.grad, num_grad_a, atol=1e-4)

    def test_repeated_calls_accumulate_grad(self):
        # Documents current behavior: backward() does not zero existing
        # grads first, so calling it twice in a row doubles leaf gradients.
        a = t([1.0, 2.0])
        out = a.sum()
        out.backward()
        out.backward()
        np.testing.assert_allclose(a.grad, [2.0, 2.0])
