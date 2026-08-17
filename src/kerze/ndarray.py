"""
Array.py

A minimal, pure-Python N-dimensional array implementation.

Stores data as a flat 1D list plus shape/stride metadata, mirroring how
numpy and PyTorch represent tensors internally. This lets operations like
transpose and reshape be implemented as metadata changes instead of data
copies, without depending on numpy.
"""

from __future__ import annotations
from functools import reduce
from operator import mul
import math
import itertools
from typing import List, Tuple, Union


NestedList = Union[float, int, List["NestedList"]]


class Array:
    """
    A flat-buffer-backed N-dimensional array.

    Internally stores all elements in a single flat Python list (`self.data`)
    plus a `shape` tuple describing the logical dimensions and a `strides`
    tuple describing how many flat positions to skip to move one step along
    each dimension.

    Users interact with this class using normal nested lists (e.g.
    `[[1, 2], [3, 4]]`); flattening and shape inference happen automatically.

    Attributes:
        data (List[float]): The flat, 1D buffer holding every element in
            row-major (C-style) order. This is the actual storage — shape
            and strides are just metadata describing how to interpret it.
        shape (Tuple[int, ...]): The logical dimensions of the array, e.g.
            (2, 3) for a 2-row, 3-column matrix.
        strides (Tuple[int, ...]): The number of flat-buffer positions to
            skip to move one step along each axis. Used to translate
            (row, col, ...) coordinates into a single flat index.
    """

    def __init__(
        self,
        data: NestedList,
        shape: Tuple[int, ...] = None,
    ) -> None:
        """
        Construct an Array from nested-list data, with optional explicit shape.

        Args:
            data: The array contents. Can be a nested Python list (e.g.
                [[1, 2], [3, 4]]), in which case the shape is inferred and
                the data is flattened automatically. Can also be a flat
                list, in which case `shape` must be provided.
            shape: The intended shape of the array. If omitted, it is
                inferred from the nesting depth/lengths of `data`.

        Raises:
            ValueError: If `shape` is provided and the number of elements
                in `data` does not match `numel(shape)`.
        """
        if shape is None:
            shape = self._infer_shape(data)
            data = self._flatten(data)
        else:
            if self._is_nested(data):
                data = self._flatten(data)
            else:
                data = list(data)

            expected_size = self._numel(shape)
            actual_size = len(data)
            if actual_size != expected_size:
                raise ValueError(
                    f"Shape {shape} implies {expected_size} elements, "
                    f"but got {actual_size} elements."
                )

        self.data: List[float] = data
        self.shape: Tuple[int, ...] = tuple(shape)
        self.strides: Tuple[int, ...] = self._compute_strides(self.shape)

    @staticmethod
    def _numel(shape: Tuple[int, ...]) -> int:
        """Compute the total number of elements a given shape represents."""
        return reduce(mul, shape, 1)

    @staticmethod
    def _is_nested(data: NestedList) -> bool:
        """Check whether `data` is a nested list rather than a flat list."""
        return isinstance(data, list) and len(data) > 0 and isinstance(data[0], list)

    @staticmethod
    def _infer_shape(nested: NestedList) -> Tuple[int, ...]:
        """Determine the shape of a nested list by walking its structure."""
        shape = []
        while isinstance(nested, list):
            shape.append(len(nested))
            nested = nested[0]
        return tuple(shape)

    @classmethod
    def _flatten(cls, nested: NestedList) -> List[float]:
        """Recursively flatten a nested list into a single flat list."""
        if not isinstance(nested, list):
            return [nested]
        result: List[float] = []
        for item in nested:
            result.extend(cls._flatten(item))
        return result

    @classmethod
    def _unflatten(cls, flat: List[float], shape: Tuple[int, ...]) -> NestedList:
        """Recursively rebuild a nested list from a flat list and a shape."""
        if len(shape) == 0:
            return flat[0]
        if len(shape) == 1:
            return list(flat)
        chunk_size = cls._numel(shape[1:])
        outer_size = shape[0]
        return [
            cls._unflatten(flat[i * chunk_size : (i + 1) * chunk_size], shape[1:])
            for i in range(outer_size)
        ]

    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute row-major (C-style) strides for a given shape."""
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    def get(self, *indices: int) -> float:
        """Retrieve the value at the given logical coordinates."""
        flat_index = sum(i * s for i, s in zip(indices, self.strides))
        return self.data[flat_index]

    def set(self, value: float, *indices: int) -> None:
        """Write a value at the given logical coordinates."""
        flat_index = sum(i * s for i, s in zip(indices, self.strides))
        self.data[flat_index] = value

    @property
    def ndim(self) -> int:
        """The number of dimensions (axes) of this array."""
        return len(self.shape)
    
#----------------------------------Shape-----------------------------------------------------------
    @property
    def size(self) -> int:
        """The total number of elements in this array."""
        return self._numel(self.shape)
    
    def reshape(self, new_shape):
        """Reshape to a new shape with the same total element count."""
        if self._numel(new_shape) != self.size:
            raise ValueError(f"Cannot reshape {self.shape} into {new_shape}")
        return Array(self.data, shape=new_shape)

    def transpose(self):
        """Zero-copy transpose: reverses shape and strides, shares data buffer."""
        new_shape = tuple(reversed(self.shape))
        new_strides = tuple(reversed(self.strides))
        result = Array.__new__(Array)
        result.data = self.data
        result.shape = new_shape
        result.strides = new_strides
        return result

    def squeeze(self, axis: int = None) -> "Array":
        """
        Remove size-1 dimensions from the shape.

        If `axis` is None (default), removes all size-1 dimensions.
        If `axis` is an int, removes only that dimension if it has size 1,
        otherwise raises ValueError.

        Returns:
            A new Array with the squeezed shape. Shares the same data buffer.
        """
        if axis is None:
            new_shape = tuple(d for d in self.shape if d != 1)
        else:
            if axis < -self.ndim or axis >= self.ndim:
                raise ValueError(
                    f"Axis {axis} is out of bounds for array of dimension {self.ndim}"
                )

            if axis < 0:
                axis += self.ndim

            if self.shape[axis] != 1:
                raise ValueError(
                    f"Cannot squeeze axis {axis} with size {self.shape[axis]}"
                )

            new_shape = (
                self.shape[:axis] +
                self.shape[axis + 1:]
            )

        return Array(self.data, shape=new_shape)

    def unsqueeze(self, axis: int) -> "Array":
        """
        Add a size-1 dimension at the specified axis.

        Args:
            axis: The position to insert the new dimension. Can be negative
                to count from the end (e.g., -1 adds a new last dimension).

        Returns:
            A new Array with the unsqueezed shape. Shares the same data buffer.
        """
        if axis < 0:
            axis += self.ndim + 1

        if axis < 0 or axis > self.ndim:
            raise ValueError(
                f"Axis {axis} is out of bounds for array of dimension {self.ndim}"
            )

        new_shape = (
            self.shape[:axis] +
            (1,) +
            self.shape[axis:]
        )

        return Array(self.data, shape=new_shape)

    @classmethod
    def zeros(cls, shape):
        """Create an Array of the given shape, with every element set to 0.0."""
        return cls([0.0] * cls._numel(shape), shape=shape)

    @classmethod
    def ones(cls, shape):
        """Create an Array of the given shape, with every element set to 1.0."""
        return cls([1.0] * cls._numel(shape), shape=shape)

    @classmethod
    def full(cls, value: float, shape: Tuple[int, ...]) -> "Array":
        """Create an Array of the given shape, with every element set to `value`."""
        return cls([value] * cls._numel(shape), shape=shape)

    def allclose(self, other: "Array", tol=1e-6):
        """Check elementwise approximate equality within a tolerance."""
        return self.shape == other.shape and all(
            abs(a - b) < tol for a, b in zip(self.data, other.data)
        )

    # -------------------------------------------------------------------
    # Broadcasting
    # -------------------------------------------------------------------

    @staticmethod
    def broadcast_shapes(
        shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """
        Determine the resulting shape of broadcasting two shapes together,
        following numpy-style broadcasting rules.

        Shapes are aligned from the right (trailing dimensions first).
        Missing leading dimensions are treated as size 1. Two dimensions
        are compatible if they are equal, or if either of them is 1 (in
        which case the size-1 dimension is "stretched" to match the
        other). The output shape takes the larger size at each position.

        Args:
            shape_a: The first shape, e.g. (2, 3).
            shape_b: The second shape, e.g. (3,).

        Returns:
            The broadcast-compatible output shape, e.g. (2, 3) for the
            example above.

        Raises:
            ValueError: If the shapes are not broadcast-compatible at
                any aligned dimension.

        Example:
            >>> Array.broadcast_shapes((2, 3), (3,))
            (2, 3)
            >>> Array.broadcast_shapes((2, 1), (2, 3))
            (2, 3)
        """
        len_diff = len(shape_b) - len(shape_a)
        if len_diff > 0:
            shape_a = (1,) * len_diff + shape_a
        elif len_diff < 0:
            shape_b = (1,) * (-len_diff) + shape_b

        result = []
        for a, b in zip(shape_a, shape_b):
            if a == b or a == 1 or b == 1:
                result.append(max(a, b))
            else:
                raise ValueError(f"Shapes not broadcastable: {shape_a} vs {shape_b}")
        return tuple(result)

    def broadcast_to(self, target_shape: Tuple[int, ...]) -> "Array":
        """
        Expand this array to `target_shape` by replicating elements along
        any dimension where this array has size 1 (or is missing the
        dimension entirely, i.e. a leading dimension).

        This does not check that `target_shape` was actually produced by
        `broadcast_shapes` against some other array — it only checks that
        `self.shape` can be validly stretched into `target_shape`. Use
        `broadcast_shapes` first to compute the correct target shape when
        combining two arrays.

        Args:
            target_shape: The shape to expand into. Must have the same
                or more dimensions than `self.shape`, and every one of
                `self.shape`'s dimensions must either match the
                corresponding `target_shape` dimension or be 1.

        Returns:
            A new Array of shape `target_shape`. If `self.shape` already
            equals `target_shape`, returns `self` unchanged (no copy).

        Raises:
            ValueError: If `self.shape` cannot be broadcast into
                `target_shape`.

        Example:
            >>> v = Array([10, 20, 30], shape=(3,))
            >>> Array._unflatten(v.broadcast_to((2, 3)).data, (2, 3))
            [[10, 20, 30], [10, 20, 30]]
        """
        if self.shape == target_shape:
            return self

        pad = len(target_shape) - self.ndim
        if pad < 0:
            raise ValueError(f"Cannot broadcast {self.shape} to {target_shape}")

        padded_shape = (1,) * pad + self.shape
        for s, t in zip(padded_shape, target_shape):
            if s != t and s != 1:
                raise ValueError(f"Cannot broadcast {self.shape} to {target_shape}")

        result_data = []
        for idx in itertools.product(*[range(d) for d in target_shape]):
            # For each target position, use index 0 along any dimension
            # where the (padded) source has size 1 — that's the "stretch."
            src_idx_full = [
                0 if padded_shape[axis] == 1 else idx[axis]
                for axis in range(len(target_shape))
            ]
            # Drop the leading padded dimensions before indexing into
            # self, since self.get only expects self.ndim indices.
            result_data.append(self.get(*src_idx_full[pad:]))
        return Array(result_data, shape=target_shape)

    def unbroadcast(self, target_shape: Tuple[int, ...]) -> "Array":
        """
        Collapse this array back down to `target_shape`, assuming it was
        produced by broadcasting an array of `target_shape` up to
        `self.shape`.

        This is the backward-pass counterpart to `broadcast_to`: forward
        broadcasting replicates a smaller array's values to fill a bigger
        shape; the gradient flowing back through that operation must be
        **summed** back down along exactly the dimensions that were
        replicated, so each original element's gradient reflects the sum
        of contributions from every position it was copied into.

        Two kinds of collapsing happen, matching the two ways
        `broadcast_to` can expand:
            1. Extra leading dimensions (added via left-padding) are
               summed away entirely (axis=0, repeatedly).
            2. Dimensions that were size 1 in `target_shape` but got
               stretched to a larger size are summed back to size 1
               (keeping the dimension in place, via keepdims=True).

        Args:
            target_shape: The shape to collapse back down to — normally
                the original (pre-broadcast) shape of one operand in a
                broadcasting binary op.

        Returns:
            A new Array of shape `target_shape`.

        Example:
            >>> grad = Array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
            >>> grad.unbroadcast((3,)).data              # summed over axis 0
            [5.0, 7.0, 9.0]
            >>> grad.unbroadcast((2, 1)).data             # summed over axis 1
            [6.0, 15.0]
        """
        result = self
        while result.ndim > len(target_shape):
            result = result.sum(axis=0)
        for axis in range(len(target_shape)):
            if target_shape[axis] == 1 and result.shape[axis] != 1:
                result = result.sum(axis=axis, keepdims=True)
        return result.reshape(target_shape)

    def _broadcast_binop(self, other, op) -> "Array":
        """
        Shared implementation for broadcasting-aware elementwise binary
        operators (add, sub, mul, truediv). Not part of the public API —
        called internally by the dunder methods below, so the actual
        broadcasting logic lives in exactly one place (DRY).

        Converts a scalar `other` into a matching-shape Array, computes
        the broadcast output shape for the two operands, expands both to
        that shape, then applies `op` elementwise.

        Args:
            other: An Array or scalar (int/float) to combine with self.
            op: A function taking two scalars and returning one, e.g.
                `lambda x, y: x + y`.

        Returns:
            A new Array holding the elementwise result, with shape equal
            to the broadcast of `self.shape` and `other`'s shape.
        """
        other_arr = other if isinstance(other, Array) else Array.full(other, self.shape)
        out_shape = Array.broadcast_shapes(self.shape, other_arr.shape)
        a = self.broadcast_to(out_shape)
        b = other_arr.broadcast_to(out_shape)
        return Array([op(x, y) for x, y in zip(a.data, b.data)], shape=out_shape)

    # -------------------------------------------------------------------
    # Arithmetic operators — now broadcasting-aware
    # -------------------------------------------------------------------

    def __add__(self, other: "Array") -> "Array":
        """
        Addition: self + other. Supports full broadcasting (not just
        exact shape matches or plain scalars) — e.g. (2,3) + (3,) or
        (2,1) + (2,3) both work now.
        """
        return self._broadcast_binop(other, lambda x, y: x + y)

    def __radd__(self, other: "Array") -> "Array":
        """Supports `scalar + array`. Addition is commutative."""
        return self.__add__(other)

    def __mul__(self, other: "Array") -> "Array":
        """Elementwise multiplication: self * other. Broadcasting-aware."""
        return self._broadcast_binop(other, lambda x, y: x * y)

    def __rmul__(self, other: "Array") -> "Array":
        """Supports `scalar * array`. Multiplication is commutative."""
        return self.__mul__(other)

    def __neg__(self) -> "Array":
        """Elementwise negation: -self."""
        return Array([-x for x in self.data], shape=self.shape)

    def __sub__(self, other: "Array") -> "Array":
        """Elementwise subtraction: self - other. Broadcasting-aware."""
        return self._broadcast_binop(other, lambda x, y: x - y)

    def __rsub__(self, other: "Array") -> "Array":
        """Supports `scalar - array`. Subtraction is NOT commutative, so
        operand order is flipped inside the lambda (y - x, not x - y)."""
        return self._broadcast_binop(other, lambda x, y: y - x)

    def __truediv__(self, other: "Array") -> "Array":
        """Elementwise division: self / other. Broadcasting-aware."""
        return self._broadcast_binop(other, lambda x, y: x / y)

    def __rtruediv__(self, other: "Array") -> "Array":
        """Supports `scalar / array`. Division is NOT commutative, so
        operand order is flipped inside the lambda (y / x, not x / y)."""
        return self._broadcast_binop(other, lambda x, y: y / x)

    def __pow__(self, exp: float) -> "Array":
        """
        Elementwise power: self ** exp. `exp` is a scalar (int/float),
        applied independently to every element.
        """
        return Array([x ** exp for x in self.data], shape=self.shape)

    # -------------------------------------------------------------------
    # Elementwise math functions
    # -------------------------------------------------------------------

    def exp(self) -> "Array":
        """Elementwise e^x."""
        return Array([math.exp(x) for x in self.data], shape=self.shape)

    def log(self) -> "Array":
        """Elementwise natural log. Raises ValueError for elements <= 0."""
        return Array([math.log(x) for x in self.data], shape=self.shape)

    def sqrt(self) -> "Array":
        """Elementwise square root. Raises ValueError for negative elements."""
        return Array([math.sqrt(x) for x in self.data], shape=self.shape)

    def tanh(self) -> "Array":
        """Elementwise hyperbolic tangent."""
        return Array([math.tanh(x) for x in self.data], shape=self.shape)

    # -------------------------------------------------------------------
    # Reduction operations — now axis-aware
    # -------------------------------------------------------------------

    def sum(self, axis: int = None, keepdims: bool = False) -> "Array":
        """
        Sum elements along an axis, or over the whole array.

        Args:
            axis: Which dimension to sum over. If None (default), sums
                every element in the array down to a single value
                (shape (1,), or all-1s shape if keepdims=True). If an
                int, sums only along that axis, keeping the others.
                Negative axis values count from the end (-1 = last axis),
                matching numpy convention.
            keepdims: If True, the reduced axis is kept in the output
                shape with size 1 (e.g. summing a (2,3) array along
                axis=0 gives shape (1,3) instead of (3,)). Useful when
                you want the result to still broadcast naturally against
                the original array.

        Returns:
            A new Array with the specified axis reduced (or fully
            reduced if axis=None).

        Example:
            >>> a = Array([[1, 2, 3], [4, 5, 6]])   # shape (2, 3)
            >>> Array._unflatten(a.sum().data, a.sum().shape)
            [21]
            >>> s0 = a.sum(axis=0)
            >>> Array._unflatten(s0.data, s0.shape)
            [5, 7, 9]
            >>> s1 = a.sum(axis=1)
            >>> Array._unflatten(s1.data, s1.shape)
            [6, 15]
        """
        if axis is None:
            total = sum(self.data)
            out_shape = tuple(1 for _ in self.shape) if keepdims else (1,)
            return Array([total], shape=out_shape)

        if axis < 0:
            axis = self.ndim + axis

        reduced_shape = list(self.shape)
        reduced_shape[axis] = 1
        reduced_shape = tuple(reduced_shape)

        result = Array.zeros(reduced_shape)
        for idx in itertools.product(*[range(d) for d in self.shape]):
            out_idx = list(idx)
            out_idx[axis] = 0
            result.set(result.get(*out_idx) + self.get(*idx), *out_idx)

        if not keepdims:
            final_shape = tuple(d for i, d in enumerate(reduced_shape) if i != axis)
            if final_shape == ():
                final_shape = (1,)
            result = result.reshape(final_shape)
        return result

    def mean(self, axis: int = None, keepdims: bool = False) -> "Array":
        """
        Mean of elements along an axis, or over the whole array.

        Same `axis`/`keepdims` semantics as `sum` — this is implemented
        as `sum(axis, keepdims) / n`, where `n` is the number of elements
        that were averaged together (the full size if axis=None, or the
        size of just that axis otherwise).

        Example:
            >>> a = Array([[1, 2, 3], [4, 5, 6]])
            >>> m = a.mean(axis=1)
            >>> Array._unflatten(m.data, m.shape)
            [2.0, 5.0]
        """
        if axis is None:
            n = self.size
        else:
            ax = axis if axis >= 0 else self.ndim + axis
            n = self.shape[ax]
        s = self.sum(axis=axis, keepdims=keepdims)
        return s / n

    def max(self, axis: int = None, keepdims: bool = False) -> "Array":
        """
        Maximum of elements along an axis, or over the whole array.

        Same `axis`/`keepdims` semantics as `sum`. Implemented by iterating
        over all indices and keeping track of the maximum value for each
        output position.

        Example:
            >>> a = Array([[1, 2, 3], [4, 5, 6]])
            >>> m = a.max(axis=0)
            >>> Array._unflatten(m.data, m.shape)
            [4, 5, 6]
        """
        if axis is None:
            max_val = max(self.data)
            out_shape = tuple(1 for _ in self.shape) if keepdims else (1,)
            return Array([max_val], shape=out_shape)

        if axis < 0:
            axis = self.ndim + axis

        reduced_shape = list(self.shape)
        reduced_shape[axis] = 1
        reduced_shape = tuple(reduced_shape)

        result = Array.full(float("-inf"), reduced_shape)
        for idx in itertools.product(*[range(d) for d in self.shape]):
            out_idx = list(idx)
            out_idx[axis] = 0
            current_max = result.get(*out_idx)
            new_val = self.get(*idx)
            if new_val > current_max:
                result.set(new_val, *out_idx)

        if not keepdims:
            final_shape = tuple(d for i, d in enumerate(reduced_shape) if i != axis)
            if final_shape == ():
                final_shape = (1,)
            result = result.reshape(final_shape)
        return result

    # -------------------------------------------------------------------
    # Matrix multiplication - Linear Algebra
    # -------------------------------------------------------------------

    def _matmul_2d(self, other: "Array") -> "Array":
        """Core 2D matrix multiplication: self @ other, shape (m,k)@(k,n)->(m,n)."""
        m, k = self.shape
        k2, n = other.shape
        if k != k2:
            raise ValueError(
                f"Incompatible shapes for matmul: {self.shape} @ {other.shape}"
            )
        result = [0.0] * (m * n)
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += self.get(i, p) * other.get(p, j)
                result[i * n + j] = s
        return Array(result, shape=(m, n))

    def _batch_slice(self, b: int) -> "Array":
        """Extract the b-th 2D slice from a 3D array of shape (batch, m, n)."""
        _, m, n = self.shape
        chunk_size = m * n
        start = b * chunk_size
        end = start + chunk_size
        return Array(self.data[start:end], shape=(m, n))

    @staticmethod
    def stack(arrays: List["Array"]) -> "Array":
        """Stack a list of equal-shaped 2D Arrays into a single 3D Array."""
        if not arrays:
            raise ValueError("stack: cannot stack an empty list of arrays")
        shape0 = arrays[0].shape
        for a in arrays:
            if a.shape != shape0:
                raise ValueError(
                    f"stack: all arrays must have the same shape, "
                    f"got {shape0} and {a.shape}"
                )
        flat = []
        for a in arrays:
            flat.extend(a.data)
        return Array(flat, shape=(len(arrays), *shape0))

    def matmul(self, other: "Array") -> "Array":
        """
        Matrix multiplication: self @ other.

        Dispatches based on dimensionality: 2D@2D standard matmul, or
        matching-batch 3D@3D batched matmul. General N-D broadcasting
        matmul is not supported (deliberate scope limit).
        """
        if self.ndim == 2 and other.ndim == 2:
            return self._matmul_2d(other)
        elif self.ndim == 3 and other.ndim == 3:
            batch = self.shape[0]
            other_batch = other.shape[0]
            if batch != other_batch:
                raise ValueError(
                    f"matmul: batch size mismatch {batch} vs {other_batch}"
                )
            results = [
                self._batch_slice(b)._matmul_2d(other._batch_slice(b))
                for b in range(batch)
            ]
            return Array.stack(results)
        else:
            raise ValueError(
                f"matmul: unsupported ndim combination "
                f"self.ndim={self.ndim}, other.ndim={other.ndim} "
                f"(only 2D@2D and matching-batch 3D@3D are supported)"
            )

    def __matmul__(self, other: "Array") -> "Array":
        """Operator overload for `self @ other`. Delegates to `matmul`."""
        return self.matmul(other)

    def __repr__(self) -> str:
        """Concise developer-facing repr showing data and shape."""
        return f"Array(data={self._unflatten(self.data, self.shape)}, shape={self.shape})"

#---------------------------------------Comparison Operators-----------------------------------------------------------

    def __eq__(self, other) -> "Array":
        if not isinstance(other, Array):
            other = Array(other)

        if self.shape != other.shape:
            other = other.broadcast_to(self.shape)

        data = [
            1.0 if a == b else 0.0
            for a, b in zip(self.data, other.data)
        ]

        return Array(data, shape=self.shape)