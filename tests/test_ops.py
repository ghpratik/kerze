import pytest

from kerze.ndarray import Array
from kerze.tensor import Tensor
from kerze.ops import (
    add,
    sub,
    mul,
    div,
    neg,
    pow,
    exp,
    log,
    sqrt,
    sum,
    mean,
    max,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_data_close(actual, expected):
    assert actual.allclose(Array(expected))


# ---------------------------------------------------------------------------
# _ensure_tensor / constants
# ---------------------------------------------------------------------------

class TestEnsureTensor:
    def test_operation_accepts_python_values(self):
        x = Tensor([1.0, 2.0], requires_grad=True)

        out = add(x, [3.0, 4.0])

        assert out.data.data == [4.0, 6.0]
        assert out.requires_grad is True

    def test_constant_does_not_require_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)

        out = add(x, [3.0, 4.0])

        assert out._prev
        assert out.requires_grad is True


# ---------------------------------------------------------------------------
# ADD
# ---------------------------------------------------------------------------

class TestAdd:
    def test_forward(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)

        out = add(a, b)

        assert out.data.data == [4.0, 6.0]

    def test_backward(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)

        out = add(a, b)
        out.sum().backward()

        assert a.grad.data == [1.0, 1.0]
        assert b.grad.data == [1.0, 1.0]

    def test_broadcast_vector(self):
        a = Tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]],
            requires_grad=True,
        )

        b = Tensor(
            [10.0, 20.0, 30.0],
            requires_grad=True,
        )

        out = add(a, b)

        assert out.data.data == [
            11.0, 22.0, 33.0,
            14.0, 25.0, 36.0,
        ]

    def test_broadcast_vector_backward(self):
        a = Tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]],
            requires_grad=True,
        )

        b = Tensor(
            [10.0, 20.0, 30.0],
            requires_grad=True,
        )

        out = add(a, b).sum()
        out.backward()

        assert a.grad.data == [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]

        assert b.grad.data == [2.0, 2.0, 2.0]


# ---------------------------------------------------------------------------
# SUB
# ---------------------------------------------------------------------------

class TestSub:
    def test_forward(self):
        a = Tensor([5.0, 7.0])
        b = Tensor([2.0, 3.0])

        out = sub(a, b)

        assert out.data.data == [3.0, 4.0]

    def test_backward(self):
        a = Tensor([5.0, 7.0], requires_grad=True)
        b = Tensor([2.0, 3.0], requires_grad=True)

        out = sub(a, b).sum()
        out.backward()

        assert a.grad.data == [1.0, 1.0]
        assert b.grad.data == [-1.0, -1.0]

    def test_broadcast_backward(self):
        a = Tensor(
            [[5.0, 6.0],
             [7.0, 8.0]],
            requires_grad=True,
        )

        b = Tensor([1.0, 2.0], requires_grad=True)

        out = sub(a, b).sum()
        out.backward()

        assert a.grad.data == [
            1.0, 1.0,
            1.0, 1.0,
        ]

        assert b.grad.data == [-2.0, -2.0]


# ---------------------------------------------------------------------------
# MUL
# ---------------------------------------------------------------------------

class TestMul:
    def test_forward(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])

        out = mul(a, b)

        assert out.data.data == [8.0, 15.0]

    def test_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)

        out = mul(a, b).sum()
        out.backward()

        assert a.grad.data == [4.0, 5.0]
        assert b.grad.data == [2.0, 3.0]

    def test_broadcast_backward(self):
        a = Tensor(
            [[1.0, 2.0],
             [3.0, 4.0]],
            requires_grad=True,
        )

        b = Tensor([10.0, 20.0], requires_grad=True)

        out = mul(a, b).sum()
        out.backward()

        assert a.grad.data == [
            10.0, 20.0,
            10.0, 20.0,
        ]

        assert b.grad.data == [4.0, 6.0]


# ---------------------------------------------------------------------------
# DIV
# ---------------------------------------------------------------------------

class TestDiv:
    def test_forward(self):
        a = Tensor([6.0, 8.0])
        b = Tensor([2.0, 4.0])

        out = div(a, b)

        assert out.data.data == [3.0, 2.0]

    def test_backward(self):
        a = Tensor([6.0, 8.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)

        out = div(a, b).sum()
        out.backward()

        # d(a/b)/da = 1/b
        assert_data_close(
            a.grad,
            [0.5, 0.25],
        )

        # d(a/b)/db = -a/b²
        assert_data_close(
            b.grad,
            [-1.5, -0.5],
        )

    def test_broadcast_backward(self):
        a = Tensor(
            [[6.0, 8.0],
             [10.0, 12.0]],
            requires_grad=True,
        )

        b = Tensor([2.0, 4.0], requires_grad=True)

        out = div(a, b).sum()
        out.backward()

        assert_data_close(
            a.grad,
            [
                [0.5, 0.25],
                [0.5, 0.25],
            ],
        )

        assert_data_close(
            b.grad,
            [
                -(6.0 / 4.0) - (10.0 / 4.0),
                -(8.0 / 16.0) - (12.0 / 16.0),
            ],
        )


# ---------------------------------------------------------------------------
# NEG
# ---------------------------------------------------------------------------

class TestNeg:
    def test_forward(self):
        a = Tensor([2.0, -3.0])

        out = neg(a)

        assert out.data.data == [-2.0, 3.0]

    def test_backward(self):
        a = Tensor([2.0, -3.0], requires_grad=True)

        out = neg(a).sum()
        out.backward()

        assert a.grad.data == [-1.0, -1.0]


# ---------------------------------------------------------------------------
# POW
# ---------------------------------------------------------------------------

class TestPow:
    def test_forward(self):
        a = Tensor([2.0, 3.0])

        out = pow(a, 2)

        assert out.data.data == [4.0, 9.0]

    def test_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)

        out = pow(a, 2).sum()
        out.backward()

        assert a.grad.data == [4.0, 6.0]

    def test_fractional_power(self):
        a = Tensor([4.0, 9.0], requires_grad=True)

        out = pow(a, 0.5).sum()
        out.backward()

        assert_data_close(
            a.grad,
            [0.25, 1.0 / 6.0],
        )


# ---------------------------------------------------------------------------
# EXP
# ---------------------------------------------------------------------------

class TestExp:
    def test_forward(self):
        a = Tensor([0.0, 1.0])

        out = exp(a)

        assert_data_close(
            out.data,
            [1.0, 2.718281828459045],
        )

    def test_backward(self):
        a = Tensor([0.0, 1.0], requires_grad=True)

        out = exp(a).sum()
        out.backward()

        assert_data_close(
            a.grad,
            [1.0, 2.718281828459045],
        )


# ---------------------------------------------------------------------------
# LOG
# ---------------------------------------------------------------------------

class TestLog:
    def test_forward(self):
        a = Tensor([1.0, 2.0])

        out = log(a)

        assert_data_close(
            out.data,
            [0.0, 0.6931471805599453],
        )

    def test_backward(self):
        a = Tensor([1.0, 2.0], requires_grad=True)

        out = log(a).sum()
        out.backward()

        assert_data_close(
            a.grad,
            [1.0, 0.5],
        )


# ---------------------------------------------------------------------------
# SQRT
# ---------------------------------------------------------------------------

class TestSqrt:
    def test_forward(self):
        a = Tensor([1.0, 4.0, 9.0])

        out = sqrt(a)

        assert out.data.data == [1.0, 2.0, 3.0]

    def test_backward(self):
        a = Tensor([1.0, 4.0, 9.0], requires_grad=True)

        out = sqrt(a).sum()
        out.backward()

        assert_data_close(
            a.grad,
            [
                0.5,
                0.25,
                1.0 / 6.0,
            ],
        )


# ---------------------------------------------------------------------------
# SUM
# ---------------------------------------------------------------------------

class TestSum:
    def test_forward_full(self):
        a = Tensor(
            [[1.0, 2.0],
             [3.0, 4.0]]
        )

        out = sum(a)

        assert out.data.data == [10.0]

    def test_backward_full(self):
        a = Tensor(
            [[1.0, 2.0],
             [3.0, 4.0]],
            requires_grad=True,
        )

        out = sum(a)
        out.backward()

        assert a.grad.data == [
            1.0, 1.0,
            1.0, 1.0,
        ]

    def test_axis_0_backward(self):
        a = Tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]],
            requires_grad=True,
        )

        out = sum(a, axis=0)
        out.sum().backward()

        assert a.grad.data == [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]

    def test_axis_1_backward(self):
        a = Tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]],
            requires_grad=True,
        )

        out = sum(a, axis=1)
        out.sum().backward()

        assert a.grad.data == [
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]

    def test_keepdims_backward(self):
        a = Tensor(
            [[1.0, 2.0],
             [3.0, 4.0]],
            requires_grad=True,
        )

        out = sum(a, axis=1, keepdims=True)
        out.sum().backward()

        assert a.grad.data == [
            1.0, 1.0,
            1.0, 1.0,
        ]


# ---------------------------------------------------------------------------
# MEAN
# ---------------------------------------------------------------------------

class TestMean:
    def test_forward_full(self):
        a = Tensor(
            [[1.0, 2.0],
             [3.0, 4.0]]
        )

        out = mean(a)

        assert out.data.data == [2.5]

    def test_backward_full(self):
        a = Tensor(
            [[1.0, 2.0],
             [3.0, 4.0]],
            requires_grad=True,
        )

        out = mean(a)
        out.backward()

        assert a.grad.data == [
            0.25, 0.25,
            0.25, 0.25,
        ]

    def test_axis_0_backward(self):
        a = Tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]],
            requires_grad=True,
        )

        out = mean(a, axis=0)
        out.sum().backward()

        assert a.grad.data == [
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
        ]

    def test_axis_1_backward(self):
        a = Tensor(
            [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]],
            requires_grad=True,
        )

        out = mean(a, axis=1)
        out.sum().backward()

        assert a.grad.data == [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ]


# ---------------------------------------------------------------------------
# MAX
# ---------------------------------------------------------------------------

class TestMax:
    def test_forward_full(self):
        a = Tensor(
            [[1.0, 5.0],
             [3.0, 4.0]]
        )

        out = max(a)

        assert out.data.data == [5.0]

    def test_backward_full(self):
        a = Tensor(
            [[1.0, 5.0],
             [3.0, 4.0]],
            requires_grad=True,
        )

        out = max(a)
        out.backward()

        assert a.grad.data == [
            0.0, 1.0,
            0.0, 0.0,
        ]

    def test_axis_0_backward(self):
        a = Tensor(
            [[1.0, 5.0, 3.0],
             [4.0, 2.0, 6.0]],
            requires_grad=True,
        )

        out = max(a, axis=0)
        out.sum().backward()

        assert a.grad.data == [
            0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
        ]

    def test_axis_1_backward(self):
        a = Tensor(
            [[1.0, 5.0],
             [7.0, 4.0]],
            requires_grad=True,
        )

        out = max(a, axis=1)
        out.sum().backward()

        assert a.grad.data == [
            0.0, 1.0,
            1.0, 0.0,
        ]

    def test_keepdims_backward(self):
        a = Tensor(
            [[1.0, 5.0],
             [7.0, 4.0]],
            requires_grad=True,
        )

        out = max(a, axis=1, keepdims=True)
        out.sum().backward()

        assert a.grad.data == [
            0.0, 1.0,
            1.0, 0.0,
        ]