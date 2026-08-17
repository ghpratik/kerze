"""
activation.py

Stateless, parameter-free layers. Each is a one-line wrapper around a
functional.py function, wrapped as a Module purely so it can sit inside
a Sequential/model alongside stateful layers like Linear and be called
uniformly as `layer(x)`.
"""

from __future__ import annotations
from kerze.nn.module import Module
from kerze.nn import functional as F


class ReLU(Module):
    """
    Elementwise ReLU: max(x, 0).

    ADAPT: F.relu currently raises NotImplementedError — needs a new
    `relu` primitive in ops.py first (see functional.py note).
    """

    def forward(self, x):
        return F.relu(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Elementwise sigmoid: 1 / (1 + exp(-x)). Fully working today."""

    def forward(self, x):
        return F.sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Elementwise tanh. Fully working today."""

    def forward(self, x):
        return F.tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"

class GELU(Module):
    """Gaussian Error Linear Unit (tanh approximation). Smoother than
    ReLU — the standard choice in transformer feedforward blocks."""
    def forward(self, x):
        return F.gelu(x)
    def __repr__(self):
        return "GELU()"
