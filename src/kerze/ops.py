"""
ops.py

Forward + backward operation pairs for kerze.

Each function here takes one or more `Tensor` inputs, computes a forward
result (an `Array`), wraps it in a new output `Tensor`, and attaches a
`_backward` closure to that output describing how to push gradient back
onto the input Tensors.

This file currently only implements ops for same-shape operands (no
broadcasting yet — e.g. adding a (2,3) and a (2,3) is fine, but adding a
(2,3) and a (1,3) is not yet supported). Broadcasting-aware versions will
replace these once the `unbroadcast` helper is implemented.
"""

from __future__ import annotations

from .tensor import Tensor

import math


#----------------------------------------------ARITHMETIC--------------------------------------------------

def add(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise addition of two same-shaped Tensors: out = a + b.

    Forward:
        out[i] = a[i] + b[i]  for every element i

    Backward:
        d(out)/d(a) = 1, so gradient flows to `a` unchanged.
        d(out)/d(b) = 1, so gradient flows to `b` unchanged.

    Args:
        a: First operand.
        b: Second operand. Must have the same shape as `a` (no
            broadcasting support yet).

    Returns:
        A new Tensor holding `a.data + b.data`, with `requires_grad=True`
        if either input requires grad, and a `_backward` closure wired
        to accumulate gradient into both `a` and `b`.

    Raises:
        ValueError: If `a.shape != b.shape`.

    Example:
        >>> a = Tensor([1.0, 2.0], requires_grad=True)
        >>> b = Tensor([3.0, 4.0], requires_grad=True)
        >>> out = add(a, b)
        >>> out.data.data
        [4.0, 6.0]
        >>> out.backward()
        >>> a.grad.data
        [1.0, 1.0]
        >>> b.grad.data
        [1.0, 1.0]
    """
    if a.shape != b.shape:
        raise ValueError(
            f"add: shape mismatch {a.shape} vs {b.shape} "
            f"(broadcasting not yet supported)"
        )

    out_data = a.data + b.data
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="add",
    )

    def _backward() -> None:
        # Gradient of a sum passes through unchanged to each addend.
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += out.grad
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            b.grad += out.grad

    out._backward = _backward
    return out

def mul(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise multiplication of two same-shaped Tensors: out = a * b.

    Forward:
        out[i] = a[i] * b[i]  for every element i

    Backward (product rule, applied elementwise):
        d(out)/d(a) = b, so gradient flowing to `a` is (incoming_grad * b).
        d(out)/d(b) = a, so gradient flowing to `b` is (incoming_grad * a).

    Args:
        a: First operand.
        b: Second operand. Must have the same shape as `a` (no
            broadcasting support yet).

    Returns:
        A new Tensor holding `a.data * b.data` (elementwise), with
        `requires_grad=True` if either input requires grad, and a
        `_backward` closure wired to accumulate gradient into both
        `a` and `b` using the product rule.

    Raises:
        ValueError: If `a.shape != b.shape`.

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
    if a.shape != b.shape:
        raise ValueError(
            f"mul: shape mismatch {a.shape} vs {b.shape} "
            f"(broadcasting not yet supported)"
        )

    out_data = a.data * b.data
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="mul",
    )

    def _backward() -> None:
        # Product rule: gradient into `a` is scaled by `b`'s value, and
        # gradient into `b` is scaled by `a`'s value.
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += b.data * out.grad
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            b.grad += a.data * out.grad

    out._backward = _backward
    return out

def neg(a: Tensor) -> Tensor:
    """
    Elementwise negation: out = -a.

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

def sub(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise addition of two same-shaped Tensors: out = a - b.
    
    Forward:
        out[i] = a[i] - b[i]  for every element i

    Backward:
        d(out)/d(a) = 1, so gradient flows to `a` unchanged.
        d(out)/d(b) = -1, so gradient becomes `-1 * gradient of out`.

    Args:
        a: First operand.
        b: Second operand. Must have the same shape as `a` (no
            broadcasting support yet).

    Returns:
        A new Tensor holding `a.data - b.data`, with `requires_grad=True`
        if either input requires grad, and a `_backward` closure wired
        to accumulate gradient into both `a` and `b`.

    Raises:
        ValueError: If `a.shape != b.shape`.

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

    if a.shape != b.shape:
        raise ValueError(
            f"add: shape mismatch {a.shape} vs {b.shape} "
            f"(broadcasting not yet supported)"
        )
    out_data = a.data - b.data

    out = Tensor(
        data=out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a,b),
        _op='sub'
    )

    def _backward() -> None:
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += out.grad
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            b.grad -= out.grad  

    out._backward = _backward
    return out

def div(a: Tensor, b: Tensor) -> Tensor:
    """
    Elementwise division of two same-shaped Tensors: out = a / b.

    Forward:
        out[i] = a[i] / b[i]  for every element i

    Backward (quotient rule, applied elementwise):
        d(out)/d(a) = 1/b, so gradient flowing to `a` is (incoming_grad * 1/b).
        d(out)/d(b) = -a/b**2, so gradient flowing to `b` is (incoming_grad * -a/b**2).

    Args:
        a: First operand.
        b: Second operand. Must have the same shape as `a` (no
            broadcasting support yet).
    
    Returns:
        A new Tensor holding `a.data / b.data` (elementwise), with
        `requires_grad=True` if either input requires grad, and a
        `_backward` closure wired to accumulate gradient into both
        `a` and `b` using the quotient rule.

    Raises:
        ValueError: If `a.shape != b.shape`.

    Example:
        >>> a = Tensor([2.0, 6.0], requires_grad=True)
        >>> b = Tensor([4.0, 3.0], requires_grad=True)
        >>> out = div(a, b)
        >>> out.data.data
        [0.5, 2.0]
    """
    if a.shape != b.shape:
        raise ValueError(
            f"div: shape mismatch {a.shape} vs {b.shape} "
            f"(broadcasting not yet supported)"
        )

    out_data = a.data / b.data
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad or b.requires_grad),
        _children=(a, b),
        _op="div",
    )

    def _backward() -> None:
        # Quotient rule: gradient into `a` is scaled by `1/b`, and
        # gradient into `b` is scaled by `-a/b**2`.
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += (1 / b.data) * out.grad
        if b.requires_grad:
            if b.grad is None:
                b.zero_grad()
            b.grad += (-a.data / (b.data**2)) * out.grad

    out._backward = _backward
    return out

def pow(a: Tensor, exp: float) -> Tensor:
    """
    Elementwise power of Tensor: out = a**exp or a.pow(exp).

    Forward:
        out[i] = a[i]**(exp) for every element i

    Backward (product rule, applied elementwise):
        d(out)/d(a) = exp * a.pow(exp-1), so gradient flowing to `a` is (incoming_grad * exp * a.pow(exp-1)).

    Args:
        a: First operand.
        b: Scalar

    Returns:
        A new Tensor holding `a.data.pow(exp)` (elementwise), with
        `requires_grad=True` if either input requires grad, and a
        `_backward` closure wired to accumulate gradient into both
        `a` and `b` using the product rule.

    Raises:
        ValueError: If `a.shape != b.shape`.

    Example:
        >>> a = Tensor([2.0, 6.0], requires_grad=True)
        >>> b = Tensor([4.0, 3.0], requires_grad=True)
        >>> out = div(a, b)
        >>> out.data.data
        [0.5, 2.0]
        >>> out.backward()
        >>> a.grad.data   # d(out)/da = 1/b
        [0.25, 0.3333]
        >>> b.grad.data   # d(out)/db = -a/b**2
        [-0.125, -0.2222]
    """
    out_data = a.data**exp
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad),
        _children=(a,),
        _op="pow",
    )

    def _backward() -> None:
        # Product rule: gradient into `a` is scaled by `b`'s value, and
        # gradient into `b` is scaled by `a`'s value.
        if a.requires_grad:
            if a.grad is None:
                a.zero_grad()
            a.grad += exp * (out.data/a.data) * out.grad

    out._backward = _backward
    return out


#---------------------------------------------MATH FUNCTIONS----------------------------------------------

def exp(a: Tensor) -> Tensor:
    """
    Elementwise exponential of Tensor: out = exp(a) or e**a.

    Forward:
        out[i] = math.e**a[i] for every element i

    Backward (product rule, applied elementwise):
        d(out)/d(a) = e**a, so gradient flowing to `a` is (incoming_grad * e**self).

    Args:
        a: First operand.

    Returns:
        A new Tensor holding `a.data.pow(exp)` (elementwise), with
        `requires_grad=True` if either input requires grad, and a
        `_backward` closure wired to accumulate gradient into both
        `a` and `b` using the product rule.

    Raises:
        ValueError: If `a.shape != b.shape`.

    Example:
        >>> a = Tensor([2.0, 6.0], requires_grad=True)
        >>> b = Tensor([4.0, 3.0], requires_grad=True)
        >>> out = div(a, b)
        >>> out.data.data
        [0.5, 2.0]
        >>> out.backward()
        >>> a.grad.data   # d(out)/da = 1/b
        [0.25, 0.3333]
        >>> b.grad.data   # d(out)/db = -a/b**2
        [-0.125, -0.2222]
    """
    out_data = math.exp**a
    out = Tensor(
        out_data,
        requires_grad=(a.requires_grad),
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


