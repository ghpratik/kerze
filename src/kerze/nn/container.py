"""
container.py

Modules that hold other Modules and define how data flows between them,
rather than doing any math themselves.
"""

from __future__ import annotations
from typing import Iterator
from kerze.nn.module import Module


class Sequential(Module):
    """
    Chains a sequence of Modules, feeding each one's output into the
    next one's input, in the order given.

    Doesn't need matmul/relu to work as a *container* — it's fully
    functional today; it just can't run end-to-end through a Linear or
    ReLU layer until those land. Works fine right now with e.g.
    Sequential(Linear(...), Sigmoid(), Linear(...)) once Linear works.

    Example:
        >>> model = Sequential(Linear(3, 8), Sigmoid(), Linear(8, 1))
        >>> out = model(x)
    """

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        # setattr(self, str(i), layer) routes through Module.__setattr__,
        # which auto-registers each layer into self._modules — so
        # self.parameters() picks them up with zero extra bookkeeping.
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)
        self._layers = layers  # keep insertion order explicitly for forward()

    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return x

    def __iter__(self) -> Iterator[Module]:
        return iter(self._layers)

    def __len__(self) -> int:
        return len(self._layers)

    def __getitem__(self, idx: int) -> Module:
        return self._layers[idx]

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, layer in enumerate(self._layers):
            lines.append(f"  ({i}): {layer!r}")
        lines.append(")")
        return "\n".join(lines)
