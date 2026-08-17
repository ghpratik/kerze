"""
optimizer.py — Optimizer

Base class for parameter-update rules. Optimizers are deliberately kept
outside kerze.nn: they act on a flat collection of Parameters (whatever
`model.parameters()` yields), not on the module tree itself, so they
don't need to know anything about Module/Sequential structure.
"""

from __future__ import annotations
from typing import Iterable
from kerze.nn.parameter import Parameter


class Optimizer:
    """
    Base class for optimizers.

    Args:
        params: An iterable of Parameters to optimize — typically
            `model.parameters()`. Materialized into a list immediately
            since `model.parameters()` is a generator and would
            otherwise be exhausted after the first use.
        lr: Learning rate.

    Subclasses must implement `step()`.
    """

    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset every managed parameter's gradient to None. Equivalent
        to calling model.zero_grad(), but useful when you're holding a
        raw parameter list rather than a Module (e.g. optimizing a
        subset of a model's parameters)."""
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """Apply one optimization step using currently-populated .grad
        values. Must be called after loss.backward()."""
        raise NotImplementedError
