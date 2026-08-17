"""
module.py — Module

Base class for every layer in kerze.nn (Linear, ReLU, Sequential, and any
model you build out of them).

The core trick — same one PyTorch uses — is overriding __setattr__ so that
when a subclass writes:

    self.weight = Parameter(...)
    self.layer1 = Linear(...)

the assignment is intercepted and the object is automatically registered
into self._parameters or self._modules. You never manually maintain a
list of params; Module walks the object graph for you.

This mirrors the Tensor/Array split at a higher level: Tensor tracks a
*computation* graph edge-by-edge via _prev/_backward; Module tracks a
*structural* graph (which submodules own which parameters) via the same
"register on assignment" idea.
"""

from __future__ import annotations
from typing import Dict, Iterator, Tuple

from kerze.nn.parameter import Parameter


class Module:
    """
    Base class for all neural network modules.

    Subclasses should:
      1. Call `super().__init__()` first in their own `__init__`.
      2. Assign any Parameters or child Modules as plain attributes
         (`self.weight = Parameter(...)`) — registration is automatic.
      3. Implement `forward(self, *args, **kwargs)`.

    Example:
        >>> class MyLayer(Module):
        ...     def __init__(self, in_f, out_f):
        ...         super().__init__()
        ...         self.weight = Parameter(Array.zeros((out_f, in_f)))
        ...
        ...     def forward(self, x):
        ...         return x  # placeholder
        >>> layer = MyLayer(3, 4)
        >>> list(layer.parameters())
        [Parameter(shape=(4, 3), requires_grad=True)]
    """

    def __init__(self) -> None:
        # NOTE: these must be set via object.__setattr__ directly (or by
        # relying on __setattr__'s setdefault below) to avoid infinite
        # recursion, since __setattr__ itself reads self._parameters.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value) -> None:
        """
        Intercepts every attribute assignment on a Module (and its
        subclasses). Parameters and child Modules get auto-registered
        into internal dicts; everything else falls through to normal
        attribute assignment.
        """
        if isinstance(value, Parameter):
            self.__dict__.setdefault("_parameters", {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault("_modules", {})[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        """Subclasses must override this with their actual computation."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ---------------------------- Parameter tree ----------------------------

    def parameters(self) -> Iterator[Parameter]:
        """
        Yield every Parameter owned by this module and all of its
        (recursively nested) child modules.

        Example:
            >>> for p in model.parameters():
            ...     print(p.shape)
        """
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        """
        Like `parameters()`, but yields (dotted_name, Parameter) pairs —
        e.g. "layer1.weight" — useful for debugging/printing/saving.
        """
        for name, p in self._parameters.items():
            yield f"{prefix}{name}", p
        for name, m in self._modules.items():
            yield from m.named_parameters(prefix=f"{prefix}{name}.")

    def modules(self) -> Iterator["Module"]:
        """Yield this module and every submodule, recursively."""
        yield self
        for m in self._modules.values():
            yield from m.modules()

    # ------------------------------ Training loop utils ------------------------------

    def zero_grad(self) -> None:
        """
        Reset every parameter's `.grad` to None before the next
        `.backward()` call.

        Set to None (not Array.zeros) rather than calling p.zero_grad()
        directly: ops.py's backward closures already do
        `if x.grad is None: x.zero_grad()` before accumulating, so this
        is equivalent but skips an unnecessary allocation on parameters
        that end up unused in a given forward pass.
        """
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        """
        Set this module and all submodules to training mode. Matters for
        layers that behave differently at train vs. eval time (Dropout,
        BatchNorm) — plumbed through now even though none of the initial
        layers need it yet.
        """
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        """Set this module and all submodules to evaluation mode."""
        return self.train(False)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for name, m in self._modules.items():
            child_repr = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({name}): {child_repr}")
        lines.append(")")
        return "\n".join(lines) if self._modules else f"{self.__class__.__name__}()"
