"""
sgd.py — SGD

Vanilla / momentum stochastic gradient descent.
"""

from __future__ import annotations
from typing import Iterable, Optional, Dict
from kerze.nn.parameter import Parameter
from kerze.optim.optimizer import Optimizer
from kerze.ndarray import Array


class SGD(Optimizer):
    """
    Stochastic gradient descent, with optional momentum.

    Update rule (momentum=0, the default): w -= lr * grad
    Update rule (momentum=m > 0):
        v = m * v_prev + grad
        w -= lr * v

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor (0 disables momentum entirely).

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(self, params: Iterable[Parameter], lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self._velocity: Dict[int, Array] = {}  # id(param) -> velocity Array

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            if self.momentum > 0:
                v_prev = self._velocity.get(id(p))
                v = p.grad if v_prev is None else (self.momentum * v_prev + p.grad)
                self._velocity[id(p)] = v
                p.data = p.data - self.lr * v
            else:
                p.data = p.data - self.lr * p.grad
