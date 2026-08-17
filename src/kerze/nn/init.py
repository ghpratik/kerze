"""
init.py

Weight-initialization helpers. Each function returns a raw `Array` (not
a Tensor/Parameter) — the calling Module wraps it in `Parameter(...)`.
Keeping init functions Array-in/Array-out (not Tensor-aware at all)
keeps them reusable and easy to unit test in isolation, same reasoning
as functional.py being Tensor-in/Tensor-out with no Module state.
"""

from __future__ import annotations
import math
import random
from typing import Tuple

from kerze.ndarray import Array


def _nested(shape: Tuple[int, ...], fill):
    """Recursively build a nested Python list of the given shape, calling
    `fill()` for every scalar leaf. Used by the random init functions
    below since Array's constructor accepts nested lists directly."""
    if len(shape) == 0:
        return fill()
    return [_nested(shape[1:], fill) for _ in range(shape[0])]


def zeros(shape: Tuple[int, ...]) -> Array:
    """All-zeros Array of the given shape. Thin wrapper for API symmetry
    with the other init functions — delegates straight to Array.zeros,
    which tensor.py already relies on."""
    return Array.zeros(shape)


def ones(shape: Tuple[int, ...]) -> Array:
    """All-ones Array of the given shape."""
    return Array.ones(shape)


def uniform(shape: Tuple[int, ...], low: float = -0.1, high: float = 0.1) -> Array:
    """Elementwise Uniform(low, high) — the simplest possible init,
    useful as a baseline / sanity check before reaching for kaiming."""
    return Array(_nested(shape, lambda: random.uniform(low, high)))


def kaiming_uniform(shape: Tuple[int, ...], a: float = math.sqrt(5)) -> Array:
    """
    Kaiming (He) uniform init — PyTorch's default for nn.Linear weights.

    bound = sqrt(6 / ((1 + a^2) * fan_in)), sampled from Uniform(-bound, bound).

    Args:
        shape: Weight shape, (out_features, in_features) for Linear.
        a: Negative slope of the rectifier used after this layer
           (default sqrt(5) matches PyTorch's nn.Linear default, which
           is a historical quirk more than a principled choice — for a
           layer actually followed by ReLU, a=0 is the textbook-correct
           value).
    """
    fan_in = shape[1] if len(shape) >= 2 else shape[0]
    bound = math.sqrt(6.0 / ((1 + a ** 2) * fan_in))
    return Array(_nested(shape, lambda: random.uniform(-bound, bound)))


def xavier_uniform(shape: Tuple[int, ...]) -> Array:
    """
    Xavier (Glorot) uniform init — better default for sigmoid/tanh
    layers than kaiming (which assumes a ReLU-like nonlinearity).

    bound = sqrt(6 / (fan_in + fan_out)), sampled from Uniform(-bound, bound).
    """
    fan_in = shape[1] if len(shape) >= 2 else shape[0]
    fan_out = shape[0]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return Array(_nested(shape, lambda: random.uniform(-bound, bound)))
