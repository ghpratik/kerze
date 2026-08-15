"""
tensor.py

The autograd layer of kerze.

`Tensor` wraps an `Array` (the raw flat-buffer storage) and adds:
    - `grad`: an Array of the same shape, accumulating gradients
    - `requires_grad`: whether this tensor should track gradients at all
    - `_prev`: the set of parent Tensors that produced this one (graph edges)
    - `_backward`: a closure that knows how to push this tensor's gradient
      back onto its parents, for exactly the operation that created it

Calling `.backward()` on a scalar-output Tensor walks the computation graph
in reverse topological order and accumulates gradients into every Tensor
that has `requires_grad=True`.
"""

from __future__ import annotations
from typing import Callable, List, Optional, Set, Tuple, Union

from .ndarray import Array, NestedList


class Tensor:
    """
    An autograd-tracked wrapper around an `Array`.

    A `Tensor` represents one node in a computation graph. It stores its
    own value (`data`), the gradient of some downstream scalar with
    respect to itself (`grad`, populated after calling `.backward()`).

    Attributes:
        data (Array): The actual numeric contents of this tensor.
        grad (Optional[Array]): The accumulated gradient of the final
            scalar (the one `.backward()` was called on) with respect to
            this tensor. `None` until `.backward()` has been run, or if
            `requires_grad` is False.
        requires_grad (bool): Whether this tensor should participate in
            gradient tracking. Leaf tensors created directly by the user
            (e.g. model weights) typically set this to True; intermediate
            tensors inherit it automatically from their inputs.
        shape (Tuple[int, ...]): Convenience passthrough to `self.data.shape`.
    """

    def __init__(
        self,
        data: Union[Array, NestedList],
        requires_grad: bool = False,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        """
        Construct a Tensor from raw nested-list data or an existing Array.

        Args:
            data: Either an `Array` instance, or nested-list/flat-list
                data that will be wrapped into an `Array` automatically.
            requires_grad: Whether this tensor should track gradients.
                Set True for leaf tensors you intend to optimize (e.g.
                weights, biases)..

        Example:
            >>> x = Tensor([[1, 2], [3, 4]], requires_grad=True)
            >>> x.shape
            (2, 2)
        """
        self.data: Array = data if isinstance(data, Array) else Array(data)
        self.grad: Optional[Array] = None
        self.requires_grad: bool = requires_grad

        # Graph bookkeeping — internal, not part of the public API.
        self._prev: Set["Tensor"] = set(_children)
        self._backward: Callable[[], None] = lambda: None
        self._op: str = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        """Convenience passthrough to `self.data.shape`."""
        return self.data.shape

    def zero_grad(self) -> None:
        """
        Reset this tensor's accumulated gradient to zero (same shape as
        `data`), without discarding the tensor or its graph position.

        Call this before each new `.backward()` pass in a training loop —
        PyTorch-style autograd *accumulates* gradients by default, so
        without zeroing, gradients from previous steps would silently
        add onto the current step's gradients.

        Example:
            >>> x = Tensor([1.0, 2.0], requires_grad=True)
            >>> x.zero_grad()
            >>> x.grad.data
            [0.0, 0.0]
        """
        self.grad = Array.zeros(self.shape)

    def backward(self) -> None:
        """
        Run backpropagation starting from this tensor.

        This tensor must be the final output of your computation graph
        (conventionally a scalar loss — an Array of shape `()` or `(1,)`).
        The seed gradient (gradient of the output with respect to itself)
        is implicitly 1.0 for every element of this tensor's data.

        Builds a topological ordering of the computation graph (so every
        Tensor is processed only after all Tensors that depend on it have
        already been processed), then walks that ordering in reverse,
        calling each Tensor's `_backward` closure to push gradient onto
        its parents (`_prev`).

        After this call, every Tensor in the graph with
        `requires_grad=True` will have `.grad` populated with the
        gradient of this tensor's value with respect to that Tensor.

        Raises:
            Nothing explicitly, but calling this on a non-scalar tensor
            without first reducing it (e.g. via `.sum()`) means the
            implicit seed gradient of all-ones is applied elementwise,
            which is usually not what you want for anything other than
            a scalar loss — this is the same convention PyTorch uses
            (`.backward()` requires a scalar unless you pass a gradient
            argument explicitly, which this minimal implementation does
            not yet support).

        Example:
            >>> x = Tensor([2.0], requires_grad=True)
            >>> y = x * x            # dy/dx = 2x = 4
            >>> y.backward()
            >>> x.grad.data
            [4.0]
        """
        topo_order: List["Tensor"] = []
        visited: Set["Tensor"] = set()

        def build_topo(node: "Tensor") -> None:
            """Depth-first traversal building a reverse-safe processing order."""
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo_order.append(node)

        build_topo(self)

        # Seed gradient: d(self)/d(self) = 1, for every element.
        self.grad = Array.ones(self.shape)

        # Process in reverse topological order: every node is guaranteed
        # to have received all of its incoming gradient contributions
        # from downstream nodes before we compute its own _backward.
        for node in reversed(topo_order):
            node._backward()

    # OPERATIONS
    def __add__(self, other: "Tensor") -> "Tensor":
        """
        Operator overload for `self + other`.
 
        Args:
            other: The Tensor to add to this one. Must have the same
                shape (see `ops.add` for details).
 
        Returns:
            A new Tensor, `self + other`, with gradient tracking wired up.
 
        Example:
            >>> a = Tensor([1.0, 2.0], requires_grad=True)
            >>> b = Tensor([3.0, 4.0], requires_grad=True)
            >>> (a + b).data.data
            [4.0, 6.0]
        """
        from .ops import add
        return add(self, other)
 
    def __mul__(self, other: "Tensor") -> "Tensor":
        """
        Operator overload for `self * other` (elementwise multiplication).
 
        Args:
            other: The Tensor to multiply with this one, elementwise.
                Must have the same shape (see `ops.mul` for details).
 
        Returns:
            A new Tensor, `self * other`, with gradient tracking wired up.
 
        Example:
            >>> a = Tensor([2.0, 3.0], requires_grad=True)
            >>> b = Tensor([4.0, 5.0], requires_grad=True)
            >>> (a * b).data.data
            [8.0, 15.0]
        """
        from .ops import mul
        return mul(self, other)

    def __neg__(self) -> Tensor:
        """
            Operator overload for `-self`

            Args: None

            Returns:
                    A new Tensor, `-1 * self`, with gradient tracking wired up.
             
            Example:
                >>> a = Tensor([2.0, 3.0], requires_grad=True)
                >>> (-a).data.data
                [-2.0, -3.0]
        """
        from .ops import neg
        return neg(self)

    def __sub__(self, other: "Tensor") -> "Tensor":
        """
        Operator overload for `self - other`.
 
        Args:
            other: The Tensor to subtract from this one. Must have the same
                shape (see `ops.` for details).
 
        Returns:
            A new Tensor, `self - other`, with gradient tracking wired up.
 
        Example:
            >>> a = Tensor([1.0, 2.0], requires_grad=True)
            >>> b = Tensor([3.0, 4.0], requires_grad=True)
            >>> (a - b).data.data
            [-2.0, -2.0]
        """
        from .ops import sub
        return sub(self, other)

    def __truediv__(self, other):
        """
        Operator overload for `self / other` (elementwise division).
    
        Args:
            other: The Tensor to divide with this one, elementwise.
                Must have the same shape (see `ops.div` for details).
    
        Returns:
            A new Tensor, `self / other`, with gradient tracking wired up.
    
        Example:
            >>> a = Tensor([2.0, 6.0], requires_grad=True)
            >>> b = Tensor([4.0, 3.0], requires_grad=True)
            >>> (a / b).data.data
            [0.5, 2.0]
        """
        from .ops import div
        return div(self, other)

    def __pow__(self, exponent: float):
        """
        Operator overload for `self / other` (elementwise division).
            
        Args:
            other: The Tensor to divide with this one, elementwise.
                Must have the same shape (see `ops.div` for details).
    
        Returns:
            A new Tensor, `self / other`, with gradient tracking wired up.
    
        Example:
            >>> a = Tensor([2.0, 6.0], requires_grad=True)
            >>> b = Tensor([4.0, 3.0], requires_grad=True)
            >>> (a / b).data.data
            [0.5, 2.0]
        """
        from .ops import pow
        return pow(self, exponent)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def __hash__(self) -> int:
        """
        Hash by object identity.

        Required because `_prev` is a `set[Tensor]` — Tensors need to be
        hashable to live in a set/dict. Two distinct Tensor objects with
        identical data are still treated as different graph nodes, which
        is the correct behavior (they occupy different positions in the
        computation graph even if their values coincide).
        """
        return id(self)

    #--------------------------------Math Functions--------------------------------
    def exp(self) -> "Tensor":
        """
        Elementwise exponential of this tensor.

        Returns:
            A new Tensor holding `math.exp(self.data)` (elementwise), with
            `requires_grad=True` if this tensor requires grad, and a
            `_backward` closure wired to accumulate gradient into this
            tensor using the chain rule.
        """
        from .ops import exp
        return exp(self)

    def log(self) -> "Tensor":
        """
        Elementwise natural logarithm of this tensor.

        Returns:
            A new Tensor holding `math.log(self.data)` (elementwise), with
            `requires_grad=True` if this tensor requires grad, and a
            `_backward` closure wired to accumulate gradient into this
            tensor using the chain rule.
        """
        from .ops import log
        return log(self)

    def sqrt(self) -> "Tensor":
        """
        Elementwise square root of this tensor.

        Returns:
            A new Tensor holding `math.sqrt(self.data)` (elementwise), with
            `requires_grad=True` if this tensor requires grad, and a
            `_backward` closure wired to accumulate gradient into this
            tensor using the chain rule.
        """
        from .ops import sqrt
        return sqrt(self)


    #----------------------------------Reduction Functions----------------------------------

    def sum(self) -> "Tensor":
        """
        Elementwise sum of this tensor.

        Returns:
            A new Tensor holding `self.data.sum()` (elementwise), with
            `requires_grad=True` if this tensor requires grad, and a
            `_backward` closure wired to accumulate gradient into this
            tensor using the chain rule.
        """
        from .ops import sum
        return sum(self)

    def mean(self) -> "Tensor":
        """
        Elementwise mean of this tensor.

        Returns:
            A new Tensor holding `self.data.mean()` (elementwise), with
            `requires_grad=True` if this tensor requires grad, and a
            `_backward` closure wired to accumulate gradient into this
            tensor using the chain rule.
        """
        from .ops import mean
        return mean(self)
    #-------------------------------Comparison--------------------------------

    def __eq__(self, other: object) -> bool:
        """
        Equality by object identity, matching `__hash__`.

        Note this means `Tensor([1,2]) == Tensor([1,2])` is False (they
        are different graph nodes), even though their underlying data is
        equal. Use `tensor.data.allclose(other.data)` to compare values.
        """
        return self is other