"""
functional.py

Stateless functions that operate on Tensors and return Tensors. No
learnable state lives here — that's what Module subclasses (Linear,
ReLU, ...) are for. This file is the "dumb computation" layer of nn,
exactly the way ops.py is the dumb computation layer under Tensor.

Module classes should be thin: hold Parameters, call one of these
functions, return the result. Keeping the actual math here (rather than
inline in each Module.forward) makes every op independently testable
and reusable outside a Module (e.g. F.mse_loss doesn't need a Module at
all).
"""

from __future__ import annotations
import math
from kerze.tensor import Tensor
from typing import Optional


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Affine transform: out = x @ weight.T + bias

    Shapes (matching PyTorch's nn.Linear convention):
        x:      (batch, in_features)
        weight: (out_features, in_features)
        bias:   (out_features,)
        out:    (batch, out_features)
    """
    out = x @ weight.T
    if bias is not None:
        out = out + bias  # broadcasts (batch, out) + (out,) — already supported
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Elementwise sigmoid: out = 1 / (1 + exp(-x))

    Fully expressible with existing ops.py primitives (neg, exp, add,
    div) — no new backward rule needed. Autograd correctness falls out
    for free from composing the existing ops.
    """
    return 1 / (1 + (-x).exp())


def tanh(x: Tensor) -> Tensor:
    """
    Elementwise tanh: out = (e^x - e^-x) / (e^x + e^-x)

    Also fully expressible with existing primitives. Numerically this
    naive form can overflow for large |x| (e^x grows unbounded before
    the division stabilizes it) — fine for a learning project, but flag
    it as a known simplification vs. a numerically-stable dedicated op.
    """
    ex = x.exp()
    e_neg_x = (-x).exp()
    return (ex - e_neg_x) / (ex + e_neg_x)


def relu(x: Tensor) -> Tensor:
    """
    Elementwise ReLU: out = max(x, 0). See ops.relu / Tensor.relu.
    """
    return x.relu()

def gelu(x: Tensor) -> Tensor:
    """
    GELU — tanh approximation (what PyTorch's nn.GELU(approximate='tanh')
    and most transformer implementations use; the exact form needs erf,
    which Array doesn't implement):

        out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Fully expressible from existing primitives — no new ops.py op needed.
    """
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * (x ** 3))
    return 0.5 * x * (1 + inner.tanh())


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean squared error: mean((pred - target) ** 2)

    Fully working today — only uses sub, mul, mean, all of which exist.

    Example:
        >>> pred = Tensor([1.0, 2.0], requires_grad=True)
        >>> target = Tensor([1.5, 2.5])
        >>> loss = mse_loss(pred, target)
        >>> loss.backward()
    """
    diff = pred - target
    return (diff * diff).mean()
