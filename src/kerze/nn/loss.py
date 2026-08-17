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


class NLLLoss(Module):
    """
    Negative log-likelihood loss. Expects log-probabilities as input
    (typically from `F.log_softmax`), and integer class targets — NOT
    a Tensor for `target`, matches PyTorch's convention.

    Example:
        >>> criterion = NLLLoss()
        >>> log_probs = F.log_softmax(logits)
        >>> loss = criterion(log_probs, target=[1, 0, 2])
        >>> loss.backward()
    """

    def forward(self, log_probs, target):
        return F.nll_loss(log_probs, target)

    def __repr__(self) -> str:
        return "NLLLoss()"


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss. Expects raw logits as input (applies
    log_softmax internally) and integer class targets.

    Example:
        >>> criterion = CrossEntropyLoss()
        >>> loss = criterion(logits, target=[1, 0, 2])
        >>> loss.backward()
    """

    def forward(self, logits, target):
        return F.cross_entropy(logits, target)

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"
