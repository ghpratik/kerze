"""
adam.py — Adam

Adaptive moment estimation (Kingma & Ba, 2014). Kept separate from
sgd.py — one optimizer per file, same reasoning as one op per function
in ops.py: easy to test, easy to read, easy to add a new one without
touching existing ones.
"""

from __future__ import annotations
from typing import Iterable, Dict
from kerze.nn.parameter import Parameter
from kerze.optim.optimizer import Optimizer
from kerze.ndarray import Array


class Adam(Optimizer):
    """
    Adam optimizer.

    Maintains a per-parameter running mean (m) and running uncentered
    variance (v) of the gradient, bias-corrected for the fact that both
    are initialized at zero (which biases early estimates toward zero
    without correction).

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        betas: (beta1, beta2) — decay rates for the first and second
            moment estimates.
        eps: Small constant added to the denominator for numerical
            stability (avoids division by ~0 early in training).

    Example:
        >>> optimizer = Adam(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # global step counter, shared across all params, for bias correction
        self._m: Dict[int, Array] = {}
        self._v: Dict[int, Array] = {}

    def step(self) -> None:
        self.t += 1
        bias_correction1 = 1 - self.beta1 ** self.t
        bias_correction2 = 1 - self.beta2 ** self.t

        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad
            m_prev = self._m.get(id(p))
            v_prev = self._v.get(id(p))

            m = g if m_prev is None else (self.beta1 * m_prev + (1 - self.beta1) * g)
            v = (g * g) if v_prev is None else (self.beta2 * v_prev + (1 - self.beta2) * (g * g))

            self._m[id(p)] = m
            self._v[id(p)] = v

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            p.data = p.data - self.lr * m_hat / (v_hat.sqrt() + self.eps)
