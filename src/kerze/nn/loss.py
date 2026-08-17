"""
loss.py

Loss "layers" — thin Module wrappers around functional.py loss
functions. Kept as Modules (rather than bare functions) purely for API
consistency with PyTorch, so `criterion = MSELoss(); criterion(pred, target)`
reads the same way every other layer call does.
"""

from __future__ import annotations
from kerze.nn.module import Module
from kerze.nn import functional as F


class MSELoss(Module):
    """
    Mean squared error loss. Fully working today — see F.mse_loss.

    Example:
        >>> criterion = MSELoss()
        >>> loss = criterion(pred, target)
        >>> loss.backward()
    """

    def forward(self, pred, target):
        return F.mse_loss(pred, target)

    def __repr__(self) -> str:
        return "MSELoss()"


# ADAPT — NOT YET IMPLEMENTED:
# CrossEntropyLoss / NLLLoss need a way to *gather* per-row values at
# target class indices (out[i, target[i]]) out of a Tensor. None of your
# current ops support indexing/gather on a Tensor — that's a separate,
# nontrivial primitive (forward: pick elements by index; backward:
# scatter gradient back to just those positions, zero elsewhere).
# Softmax itself is buildable today (exp + sum(axis=-1) + div, all
# existing), it's specifically the "select the correct class's
# probability per row" step that's missing. Worth tackling after
# matmul/relu, once you're doing actual classification tasks.
