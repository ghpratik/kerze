"""
parameter.py — Parameter

A Parameter is just a Tensor that is:
  1. Always requires_grad=True by default
  2. Tagged so Module.__setattr__ can auto-detect and register it

Keeping this as its own tiny file (rather than folding it into module.py)
mirrors the ndarray/ops split: one concept, one file, one job.
"""

from __future__ import annotations
from kerze.tensor import Tensor
from kerze.ndarray import Array, NestedList
from typing import Union


class Parameter(Tensor):
    """
    A Tensor marking a learnable parameter of a Module.

    Functionally identical to Tensor — this class exists purely as a
    type marker so Module can walk the object graph and auto-collect
    parameters without you manually registering them.

    Example:
        >>> w = Parameter(Array.zeros((3, 4)))
        >>> w.requires_grad
        True
    """

    def __init__(self, data: Union[Array, NestedList], requires_grad: bool = True) -> None:
        super().__init__(data, requires_grad=requires_grad)

    def __repr__(self) -> str:
        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"
