"""
ops.py

Forward + backward operation pairs for kerze.

Each function here takes one or more `Tensor` inputs, computes a forward
result (an `Array`), wraps it in a new output `Tensor`, and attaches a
`_backward` closure to that output describing how to push gradient back
onto the input Tensors.

Arithmetic ops (add, sub, mul, div) now support full broadcasting, not
just exact shape matches or plain scalars — e.g. adding a (2,3) Tensor
and a (3,) Tensor works, the same way it would in numpy/PyTorch. This
relies on two pieces of machinery in `Array`:
    - `broadcast_to`/the arithmetic dunders (forward): expand smaller
      operands up to the common output shape.
    - `unbroadcast` (backward): collapse a gradient computed at the
      broadcast output shape back down to each operand's original shape,
      by summing over exactly the dimensions that were broadcast.
"""

from __future__ import annotations
from .ndarray import Array

from .tensor import Tensor

import math


# ----------------------------------------------ARITHMETIC--------------------------------------------------

def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Addition of two Tensors: out = a + b. Broadcasting-aware — `a` and
    `b` do not need the same shape, as long as their shapes are
    broadcast-compatible (see `Array.broadcast_shapes`).

    Forward:
        out = broadcast(a) + broadcast(b), elementwise, at the common
        broadcast shape.

    Backward:
        d(out)/d(a) = 1, d(out)/d(b) = 1 — gradient passes through
        unchanged at the broadcast shape, then each operand's
        contribution is collapsed back to its own original shape via
        `unbroadcast` (summing over any dimensions that were stretched
        to produce `out`).

    Args:
        a: First operand.
        b: Second operand. Shapes must be broadcast-compatible with `a`.

    Returns:
        A new Tensor holding `a.data + b.data` (broadcast if needed),
        with `requires_grad=True` if either input requires grad, and a
        `_backward` closure wired to accumulate gradient into both
        `a` and `b`, correctly shaped for each.

    Raises:
        ValueError: If the shapes are not broadcast-compatible.

    Example:
        >>> a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # (2,3)
        >>> b = Tensor([10.0, 20.0, 30.0], requires_grad=True)                  # (3,)
        >>> out = add(a, b)
        >>> out.data.data
        [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        >>> out.backward()
        >>> b.grad.data   # gradient summed over the broadcast batch dim
        [2.0, 2.0, 2.0]
    """
    out_data = a.data + b.data  # Array.__add__ already handles broadcasting
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="add",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += out.grad.unbroadcast(a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            b.grad += out.grad.unbroadcast(b.shape)

    out._backward = _backward
    return out


def sub(a: Tensor, b: Tensor) -> Tensor:
    """
    Subtraction of two Tensors: out = a - b. Broadcasting-aware.

    Forward:
        out = broadcast(a) - broadcast(b), elementwise.

    Backward:
        d(out)/d(a) = 1, d(out)/d(b) = -1 — gradient passes through
        unchanged to `a` and negated to `b`, each then collapsed back
        to its original shape via `unbroadcast`.

    Args:
        a: First operand (minuend).
        b: Second operand (subtrahend). Shapes must be
            broadcast-compatible with `a`.

    Returns:
        A new Tensor holding `a.data - b.data` (broadcast if needed).

    Raises:
        ValueError: If the shapes are not broadcast-compatible.

    Example:
        >>> a = Tensor([1.0, 2.0], requires_grad=True)
        >>> b = Tensor([3.0, 4.0], requires_grad=True)
        >>> out = sub(a, b)
        >>> out.data.data
        [-2.0, -2.0]
        >>> out.backward()
        >>> a.grad.data
        [1.0, 1.0]
        >>> b.grad.data
        [-1.0, -1.0]
    """
    out_data = a.data - b.data
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="sub",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += out.grad.unbroadcast(a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            b.grad += (-out.grad).unbroadcast(b.shape)

    out._backward = _backward
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise multiplication of two Tensors: out = a * b.
    Broadcasting-aware.

    Forward:
        out = broadcast(a) * broadcast(b), elementwise.

    Backward (product rule, applied at the broadcast shape, then
    collapsed back to each operand's own shape):
        d(out)/d(a) = b, so the contribution to `a` is (b * out.grad),
        computed at the broadcast shape, then unbroadcast to a.shape.
        d(out)/d(b) = a, symmetric.

    Args:
        a: First operand.
        b: Second operand. Shapes must be broadcast-compatible with `a`.

    Returns:
        A new Tensor holding `a.data * b.data` (broadcast if needed).

    Raises:
        ValueError: If the shapes are not broadcast-compatible.

    Example:
        >>> a = Tensor([2.0, 3.0], requires_grad=True)
        >>> b = Tensor([4.0, 5.0], requires_grad=True)
        >>> out = mul(a, b)
        >>> out.data.data
        [8.0, 15.0]
        >>> out.backward()
        >>> a.grad.data   # d(out)/da = b
        [4.0, 5.0]
        >>> b.grad.data   # d(out)/db = a
        [2.0, 3.0]
    """
    out_data = a.data * b.data
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="mul",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            contribution = b.data * out.grad  # at broadcast shape
            a.grad += contribution.unbroadcast(a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            contribution = a.data * out.grad
            b.grad += contribution.unbroadcast(b.shape)

    out._backward = _backward
    return out


def div(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise division of two Tensors: out = a / b. Broadcasting-aware.

    Forward:
        out = broadcast(a) / broadcast(b), elementwise.

    Backward (quotient rule, applied at the broadcast shape, then
    collapsed back to each operand's own shape):
        d(out)/d(a) = 1/b, so the contribution to `a` is
        ((1/b) * out.grad), unbroadcast to a.shape.
        d(out)/d(b) = -a/b**2, so the contribution to `b` is
        ((-a/b**2) * out.grad), unbroadcast to b.shape.

    Args:
        a: First operand (numerator).
        b: Second operand (denominator). Shapes must be
            broadcast-compatible with `a`.

    Returns:
        A new Tensor holding `a.data / b.data` (broadcast if needed).

    Raises:
        ValueError: If the shapes are not broadcast-compatible.

    Example:
        >>> a = Tensor([2.0, 6.0], requires_grad=True)
        >>> b = Tensor([4.0, 3.0], requires_grad=True)
        >>> out = div(a, b)
        >>> out.data.data
        [0.5, 2.0]
    """
    out_data = a.data / b.data
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="div",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            contribution = (1 / b.data) * out.grad
            a.grad += contribution.unbroadcast(a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            contribution = (-a.data / (b.data ** 2)) * out.grad
            b.grad += contribution.unbroadcast(b.shape)

    out._backward = _backward
    return out


def neg(a: Tensor) -> Tensor:
    """
    Elementwise negation: out = -a. Unary — no broadcasting involved.

    Backward:
        d(out)/d(a) = -1, so gradient flowing into `a` is the incoming
        gradient (out.grad) negated.
    """
    out_data = -a.data
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="neg",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += -out.grad

    out._backward = _backward
    return out


def pow(a: Tensor, exp: float) -> Tensor:
    """
    Elementwise power: out = a ** exp. `exp` is a scalar, so this is
    unary at the Tensor level — no broadcasting between two Tensors
    involved.

    Forward:
        out[i] = a[i] ** exp for every element i.

    Backward:
        d(out)/d(a) = exp * a**(exp-1), computed here as
        exp * (out.data / a.data), which is algebraically equal to
        exp * a**(exp-1) and avoids a second full power computation.

    Args:
        a: The Tensor operand.
        exp: The scalar exponent (Python int/float).

    Returns:
        A new Tensor holding `a.data ** exp`.

    Example:
        >>> a = Tensor([2.0, 3.0], requires_grad=True)
        >>> out = pow(a, 2)
        >>> out.data.data
        [4.0, 9.0]
        >>> out.backward()
        >>> a.grad.data   # d(x^2)/dx = 2x
        [4.0, 6.0]
    """
    out_data = a.data ** exp
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="pow",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += exp * (out.data / a.data) * out.grad

    out._backward = _backward
    return out


# ---------------------------------------------MATH FUNCTIONS----------------------------------------------

def exp(a: Tensor) -> Tensor:
    """
    Elementwise exponential: out = e ** a. Unary — no broadcasting
    involved.

    Backward:
        d(out)/d(a) = e**a = out, so gradient flowing to `a` is
        (incoming_grad * out.data).

    Example:
        >>> a = Tensor([0.0, 1.0], requires_grad=True)
        >>> out = exp(a)
        >>> out.backward()
        >>> a.grad.data   # d(e^x)/dx = e^x
        [1.0, 2.718281828459045]
    """
    out_data = a.data.exp()
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="exp",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += out.data * out.grad

    out._backward = _backward
    return out


def log(a: Tensor) -> Tensor:
    """
    Elementwise natural logarithm: out = log(a). Unary — no broadcasting
    involved.

    Backward:
        d(out)/d(a) = 1/a, so gradient flowing to `a` is
        (incoming_grad * 1/a).

    Example:
        >>> a = Tensor([1.0, 2.0], requires_grad=True)
        >>> out = log(a)
        >>> out.backward()
        >>> a.grad.data   # d(log(x))/dx = 1/x
        [1.0, 0.5]
    """
    out_data = a.data.log()
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="log",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += (1 / a.data) * out.grad

    out._backward = _backward
    return out


def sqrt(a: Tensor) -> Tensor:
    """
    Elementwise square root: out = sqrt(a). Unary — no broadcasting
    involved.

    Backward:
        d(out)/d(a) = 1/(2*sqrt(a)) = 1/(2*out), so gradient flowing to
        `a` is (incoming_grad * 1/(2*out.data)).
    """
    out_data = a.data.sqrt()
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="sqrt",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += (1 / (2 * out.data)) * out.grad

    out._backward = _backward
    return out


# ---------------------------------------------Reduction Operations----------------------------------------------

def _expand_grad_for_reduction(
    grad: Array, axis, keepdims: bool, original_shape
) -> Array:
    """
    Expand a reduction's output gradient back to the shape of the
    original (pre-reduction) input, for use inside `sum`/`mean`
    backward closures.

    This is not part of the public API — it exists because reduction
    backward is subtly different from elementwise-broadcast backward:
    `unbroadcast` assumes size-1 dimensions are already in the correct
    position (aligned from the right), but when `sum(axis=k,
    keepdims=False)` drops a dimension entirely, that dimension needs to
    be reinserted at position `k` — not necessarily the rightmost
    position — before broadcasting the gradient back out. Reinserting
    it, then reusing `broadcast_to`, handles this correctly regardless
    of which axis was reduced.

    Args:
        grad: The gradient of the reduction's output, i.e. `out.grad`.
        axis: The axis that was reduced (None if the reduction was over
            the whole array).
        keepdims: Whether the forward reduction used keepdims=True (if
            so, `grad` already has a size-1 dim at `axis` and no
            reinsertion is needed).
        original_shape: The shape of the tensor *before* reduction —
            the shape to expand the gradient back out to.

    Returns:
        An Array of shape `original_shape`, where every element along
        the reduced axis (or every element, if axis is None) receives
        the same gradient value — the correct backward rule for both
        `sum` (derivative 1 per element) and `mean` (derivative 1/n per
        element, with the 1/n scaling applied by the caller before this
        function runs).
    """
    if axis is None:
        return grad.broadcast_to(original_shape)

    ax = axis if axis >= 0 else len(original_shape) + axis

    if keepdims:
        return grad.broadcast_to(original_shape)

    new_shape = list(grad.shape)
    new_shape.insert(ax, 1)
    return grad.reshape(tuple(new_shape)).broadcast_to(original_shape)


def sum(a: Tensor, axis: int = None, keepdims: bool = False) -> Tensor:
    """
    Sum of Tensor elements along an axis, or over the whole tensor.

    Forward:
        Delegates to `Array.sum`, matching its `axis`/`keepdims`
        semantics exactly (see `Array.sum` docstring for details).

    Backward:
        d(out)/d(a_i) = 1 for every element a_i that contributed to a
        given output position. So the gradient flowing back to `a` is
        the incoming gradient (out.grad), broadcast back out to `a`'s
        original shape — every element that was summed together
        receives an identical copy of the gradient at that output
        position.

    Args:
        a: The Tensor to reduce.
        axis: Which dimension to sum over (None = sum everything).
        keepdims: Whether to keep the reduced dimension as size 1 in
            the output shape.

    Returns:
        A new Tensor holding the sum, with shape determined by
        `axis`/`keepdims` exactly as in `Array.sum`.

    Example:
        >>> a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        >>> out = sum(a, axis=0)      # shape (3,)
        >>> out.data.data
        [5.0, 7.0, 9.0]
        >>> out.backward()
        >>> a.grad.data               # every element gets gradient 1
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    """
    out_data = a.data.sum(axis=axis, keepdims=keepdims)
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="sum",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            contribution = _expand_grad_for_reduction(
                out.grad, axis, keepdims, a.shape
            )
            a.grad += contribution

    out._backward = _backward
    return out


def mean(a: Tensor, axis: int = None, keepdims: bool = False) -> Tensor:
    """
    Mean of Tensor elements along an axis, or over the whole tensor.

    Forward:
        Delegates to `Array.mean`, matching its `axis`/`keepdims`
        semantics exactly.

    Backward:
        d(out)/d(a_i) = 1/n for every element a_i that contributed to a
        given output position, where n is the number of elements
        averaged together (the full size if axis=None, or the size of
        just that axis otherwise). So the gradient flowing back to `a`
        is (incoming_grad / n), broadcast back out to `a`'s original
        shape.

    Args:
        a: The Tensor to reduce.
        axis: Which dimension to average over (None = mean of everything).
        keepdims: Whether to keep the reduced dimension as size 1 in
            the output shape.

    Returns:
        A new Tensor holding the mean, with shape determined by
        `axis`/`keepdims` exactly as in `Array.mean`.

    Example:
        >>> a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        >>> out = mean(a, axis=1)     # shape (2,)
        >>> out.data.data
        [2.0, 5.0]
    """
    if axis is None:
        n = a.data.size
    else:
        ax = axis if axis >= 0 else a.data.ndim + axis
        n = a.data.shape[ax]

    out_data = a.data.mean(axis=axis, keepdims=keepdims)
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="mean",
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            scaled_grad = out.grad / n
            contribution = _expand_grad_for_reduction(
                scaled_grad, axis, keepdims, a.shape
            )
            a.grad += contribution

    out._backward = _backward
    return out