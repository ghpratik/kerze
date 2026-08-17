import pytest

from kerze.ndarray import Array
from kerze.tensor import Tensor


class TestConstruction:
    def test_construct_from_array(self):
        data = Array([[1.0, 2.0], [3.0, 4.0]])

        x = Tensor(data)

        assert x.data is data
        assert x.shape == (2, 2)
        assert x.requires_grad is False
        assert x.grad is None

    def test_construct_from_nested_list(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])

        assert x.shape == (2, 2)
        assert x.data.data == [1.0, 2.0, 3.0, 4.0]

    def test_requires_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)

        assert x.requires_grad is True
        assert x.grad is None

    def test_shape_property(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])

        assert x.shape == x.data.shape
        assert x.shape == (2, 2)


class TestRepr:
    def test_repr(self):
        x = Tensor([1.0, 2.0], requires_grad=True)

        assert repr(x) == "Tensor(shape=(2,), requires_grad=True)"


class TestZeroGrad:
    def test_zero_grad_creates_zero_array(self):
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        x.zero_grad()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.data == [0.0, 0.0, 0.0]

    def test_zero_grad_resets_existing_gradient(self):
        x = Tensor([1.0, 2.0], requires_grad=True)

        y = x * 2
        y.sum().backward()

        assert x.grad.data == [2.0, 2.0]

        x.zero_grad()

        assert x.grad.data == [0.0, 0.0]


class TestRequiresGrad:
    def test_operation_requires_grad_if_first_input_requires_grad(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0])

        z = x + y

        assert z.requires_grad is True

    def test_operation_requires_grad_if_second_input_requires_grad(self):
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0], requires_grad=True)

        z = x + y

        assert z.requires_grad is True

    def test_operation_does_not_require_grad_if_no_input_requires_grad(self):
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])

        z = x + y

        assert z.requires_grad is False


class TestGraph:
    def test_parent_nodes_are_recorded(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)

        z = x + y

        assert x in z._prev
        assert y in z._prev

    def test_operation_is_recorded(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = Tensor([3.0, 4.0], requires_grad=True)

        z = x + y

        assert z._op == "add"

    def test_unary_operation_records_parent(self):
        x = Tensor([1.0, 4.0], requires_grad=True)

        y = x.sqrt()

        assert x in y._prev
        assert y._op == "sqrt"


class TestBackward:
    def test_backward_seeds_output_gradient(self):
        x = Tensor([2.0], requires_grad=True)

        y = x * x

        y.backward()

        assert y.grad.data == [1.0]

    def test_backward_simple_add(self):
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)

        z = x + y

        z.backward()

        assert x.grad.data == [1.0]
        assert y.grad.data == [1.0]

    def test_backward_chain_rule(self):
        x = Tensor([2.0], requires_grad=True)

        y = x * x
        z = y * x

        z.backward()

        # z = x^3
        # dz/dx = 3x^2 = 12
        assert x.grad.allclose(Array([12.0]))

    def test_backward_accumulates_gradients(self):
        x = Tensor([2.0], requires_grad=True)

        y = x * x

        y.backward()
        first_grad = x.grad.data.copy()

        y.backward()

        assert x.grad.data == [
            value * 2 for value in first_grad
        ]

    def test_zero_grad_between_backward_passes(self):
        x = Tensor([2.0], requires_grad=True)

        y = x * x

        y.backward()
        assert x.grad.data == [4.0]

        x.zero_grad()

        y.backward()
        assert x.grad.data == [4.0]


class TestBroadcastingAutograd:
    def test_add_broadcast_gradient(self):
        x = Tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )

        y = Tensor([[10.0, 20.0]], requires_grad=True)

        out = (x + y).sum()

        out.backward()

        assert x.grad.data == [1.0, 1.0, 1.0, 1.0]
        assert y.grad.data == [2.0, 2.0]

    def test_scalar_broadcast_gradient(self):
        x = Tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )

        y = Tensor([10.0], requires_grad=True)

        out = (x + y).sum()

        out.backward()

        assert x.grad.data == [1.0, 1.0, 1.0, 1.0]
        assert y.grad.data == [4.0]


class TestTensorOperators:
    def test_add(self):
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])

        z = x + y

        assert z.data.data == [4.0, 6.0]

    def test_radd(self):
        x = Tensor([1.0, 2.0])

        z = 2 + x

        assert z.data.data == [3.0, 4.0]

    def test_sub(self):
        x = Tensor([5.0, 6.0])
        y = Tensor([2.0, 3.0])

        z = x - y

        assert z.data.data == [3.0, 3.0]

    def test_rsub(self):
        x = Tensor([2.0, 3.0])

        z = 10 - x

        assert z.data.data == [8.0, 7.0]

    def test_mul(self):
        x = Tensor([2.0, 3.0])
        y = Tensor([4.0, 5.0])

        z = x * y

        assert z.data.data == [8.0, 15.0]

    def test_rmul(self):
        x = Tensor([2.0, 3.0])

        z = 2 * x

        assert z.data.data == [4.0, 6.0]

    def test_neg(self):
        x = Tensor([2.0, -3.0])

        z = -x

        assert z.data.data == [-2.0, 3.0]

    def test_div(self):
        x = Tensor([6.0, 8.0])
        y = Tensor([2.0, 4.0])

        z = x / y

        assert z.data.data == [3.0, 2.0]

    def test_rdiv(self):
        x = Tensor([2.0, 4.0])

        z = 8 / x

        assert z.data.data == [4.0, 2.0]

    def test_pow(self):
        x = Tensor([2.0, 3.0])

        z = x ** 2

        assert z.data.data == [4.0, 9.0]


class TestTensorMath:
    def test_exp(self):
        x = Tensor([0.0, 1.0])

        y = x.exp()

        assert y.data.allclose(
            Array([1.0, 2.718281828459045])
        )

    def test_log(self):
        x = Tensor([1.0, 2.0])

        y = x.log()

        assert y.data.allclose(
            Array([0.0, 0.6931471805599453])
        )

    def test_sqrt(self):
        x = Tensor([1.0, 4.0, 9.0])

        y = x.sqrt()

        assert y.data.data == [1.0, 2.0, 3.0]


class TestTensorReductions:
    def test_sum(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])

        y = x.sum()

        assert y.data.data == [10.0]

    def test_mean(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])

        y = x.mean()

        assert y.data.data == [2.5]

    def test_max(self):
        x = Tensor([[1.0, 5.0], [3.0, 4.0]])

        y = x.max()

        assert y.data.data == [5.0]

    def test_sum_backward(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        y = x.sum()

        y.backward()

        assert x.grad.data == [
            1.0, 1.0,
            1.0, 1.0,
        ]

    def test_mean_backward(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        y = x.mean()

        y.backward()

        assert x.grad.data == [
            0.25, 0.25,
            0.25, 0.25,
        ]

    def test_max_backward(self):
        x = Tensor([[1.0, 5.0], [3.0, 4.0]], requires_grad=True)

        y = x.max()

        y.backward()

        assert x.grad.data == [
            0.0, 1.0,
            0.0, 0.0,
        ]


class TestEquality:
    def test_same_object_is_equal(self):
        x = Tensor([1.0, 2.0])

        assert x == x

    def test_different_tensors_are_not_equal(self):
        x = Tensor([1.0, 2.0])
        y = Tensor([1.0, 2.0])

        assert x != y