"""
linear.py — Linear

The fully-connected / affine layer: out = x @ weight.T + bias

Deliberately thin: all it does is own the two Parameters and delegate
the actual math to F.linear. This is the pattern every other layer in
this file should follow — Module subclasses hold state, functional.py
holds math.
"""

from __future__ import annotations
from typing import Optional

from kerze.nn.module import Module
from kerze.nn.parameter import Parameter
from kerze.nn import init
from kerze.nn import functional as F


class Linear(Module):
    """
    Applies a linear transformation: y = x @ W.T + b

    Args:
        in_features: Size of each input sample's last dimension.
        out_features: Size of each output sample's last dimension.
        bias: If False, the layer will not learn an additive bias.

    ADAPT: relies on F.linear, which currently requires matmul + .T
    support on Tensor (not yet implemented — see functional.py).

    Example:
        >>> layer = Linear(3, 4)
        >>> x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # (1, 3)
        >>> out = layer(x)  # (1, 4), once matmul exists
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform((out_features, in_features)))
        self.bias: Optional[Parameter] = (
            Parameter(init.zeros((out_features,))) if bias else None
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )
