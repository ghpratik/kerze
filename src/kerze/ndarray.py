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
                inferred from the nesting depth/lengths of `data`. If
                provided alongside nested data, the nested data is still
                flattened and then validated against this shape. If
                provided alongside flat data, the flat data's length is
                validated against this shape.

        Raises:
            ValueError: If `shape` is provided and the number of elements
                in `data` does not match `numel(shape)` (i.e. the product
                of all dimensions in `shape`).

        Example:
            >>> Array([[1, 2, 3], [4, 5, 6]])
            Array(shape=(2, 3))
            >>> Array([[1, 2, 3], [4, 5, 6], [3,4,2]], shape=(3, 3))
            Array(shape=(3, 3))
        """
        if shape is None:
            # No shape given: infer it from the nested list structure,
            # then flatten the nested list into self.data.
            shape = self._infer_shape(data)
            data = self._flatten(data)
        else:
            # Shape given explicitly: flatten if needed, then verify the
            # flat length actually matches what `shape` implies.
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
    def _numel(shape: Tuple[int, ...]) -> int: # Number of Elements
        """
        Compute the total number of elements a given shape represents.

        This is the product of all dimension sizes. E.g. shape (2, 3, 4)
        holds 2 * 3 * 4 = 24 elements.

        Args:
            shape: A tuple of dimension sizes.

        Returns:
            The total element count implied by `shape`.
        """
        return reduce(mul, shape, 1)

    @staticmethod
    def _is_nested(data: NestedList) -> bool:
        """
        Check whether `data` is a nested list (list of lists) rather than
        a flat list of numbers.

        Args:
            data: The candidate data, either flat (e.g. [1, 2, 3]) or
                nested (e.g. [[1, 2], [3, 4]]).

        Returns:
            True if `data` is a non-empty list whose first element is
            itself a list (i.e. it has at least one more level of nesting).
        """
        return isinstance(data, list) and len(data) > 0 and isinstance(data[0], list)

    @staticmethod
    def _infer_shape(nested: NestedList) -> Tuple[int, ...]:
        """
        Determine the shape of a nested list by walking its structure.

        Repeatedly measures the length of each nesting level and descends
        into the first element until a non-list (scalar) is reached.

        Args:
            nested: A nested Python list, e.g. [[1, 2, 3], [4, 5, 6]].

        Returns:
            A tuple of dimension sizes, e.g. (2, 3) for the example above.

        Note:
            Assumes the nested list is "rectangular" (every sub-list at a
            given depth has the same length). No validation is performed
            for ragged/jagged input.
        """
        shape = []
        while isinstance(nested, list):
            shape.append(len(nested))
            nested = nested[0]
        return tuple(shape)

    @classmethod
    def _flatten(cls, nested: NestedList) -> List[float]:
        """
        Recursively flatten a nested list into a single flat list, in
        row-major (C-style) order.

        Args:
            nested: A nested Python list, or a single scalar value.

        Returns:
            A flat list of all scalar values, in the order they appear
            when reading the nested structure left-to-right, outer-to-inner.

        Example:
            >>> Array._flatten([[1, 2], [3, 4]])
            [1, 2, 3, 4]
        """
        if not isinstance(nested, list):
            return [nested]
        result: List[float] = []
        for item in nested:
            result.extend(cls._flatten(item))
        return result

    @classmethod
    def _unflatten(
        cls, flat: List[float], shape: Tuple[int, ...]
    ) -> NestedList:
        """
        Recursively rebuild a nested list from a flat list and a shape.
 
        At each recursion level, splits `flat` into `shape[0]` equal-sized
        chunks (each chunk having size = product of the remaining
        dimensions), then recurses into each chunk using the remaining
        shape. The base case (empty shape) returns the single remaining
        scalar.
 
        Declared as a classmethod (rather than staticmethod) because it
        recurses on itself via `cls`, matching the pattern used by
        `_flatten`. This ensures the recursion resolves correctly against
        `type(self)` if `Array` is ever subclassed.
 
        Args:
            flat: The flat list of values to regroup. Must have exactly
                `numel(shape)` elements.
            shape: The target shape to rebuild, e.g. (2, 3).
 
        Returns:
            A nested list matching `shape`, or a single scalar if `shape`
            is empty (representing a 0-D array).
 
        Example:
            >>> Array._unflatten([1, 2, 3, 4, 5, 6], (2, 3))
            [[1, 2, 3], [4, 5, 6]]
        """
        if len(shape) == 0:
            # Base case: no dimensions left, this is a single scalar.
            return flat[0]
 
        if len(shape) == 1:
            # Innermost dimension: just return the chunk as-is.
            return list(flat)
 
        # Size of each sub-chunk = product of all dimensions after the first.
        chunk_size = cls._numel(shape[1:])
        outer_size = shape[0]
 
        return [
            cls._unflatten(
                flat[i * chunk_size : (i + 1) * chunk_size],
                shape[1:],
            )
            for i in range(outer_size)
        ]

    @staticmethod
    def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute row-major (C-style) strides for a given shape.

        The stride for a given axis is the number of flat-buffer positions
        you must skip to move one step along that axis. For row-major
        layout, the stride of the last axis is always 1, and each
        preceding axis's stride is the product of all following
        dimension sizes.

        Args:
            shape: The shape to compute strides for, e.g. (2, 3, 4).

        Returns:
            A tuple of strides, one per axis, e.g. (12, 4, 1) for shape
            (2, 3, 4).

        Example:
            >>> Array._compute_strides((2, 3))
            (3, 1)
        """
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)

    def get(self, *indices: int) -> float:
        """
        Retrieve the value at the given logical (row, col, ...) coordinates.

        Translates the multi-dimensional index into a single flat-buffer
        position using `self.strides`, then looks up that position in
        `self.data`.

        Args:
            *indices: One integer per dimension, e.g. `get(1, 2)` for a
                2D array's row 1, column 2. Must supply exactly
                `len(self.shape)` indices.

        Returns:
            The scalar value stored at the given coordinates.

        Example:
            >>> arr = Array([[1, 2, 3], [4, 5, 6]])
            >>> arr.get(1, 2)
            6
        """
        flat_index = sum(i * s for i, s in zip(indices, self.strides))
        return self.data[flat_index]

    def set(self, value: float, *indices: int) -> None:
        """
        Write a value at the given logical (row, col, ...) coordinates.

        Translates the multi-dimensional index into a single flat-buffer
        position using `self.strides`, then writes `value` at that
        position in `self.data`.

        Args:
            value: The scalar value to store.
            *indices: One integer per dimension, e.g. `set(9, 1, 2)` sets
                row 1, column 2 to 9 for a 2D array.

        Example:
            >>> arr = Array([[1, 2, 3], [4, 5, 6]])
            >>> arr.set(9, 1, 2)
            >>> arr.get(1, 2)
            9
        """
        flat_index = sum(i * s for i, s in zip(indices, self.strides))
        self.data[flat_index] = value

    @property
    def ndim(self) -> int:
        """
        The number of dimensions (axes) of this array.

        Equivalent to `len(self.shape)`. E.g. a matrix has ndim=2, a
        vector has ndim=1, a scalar wrapped as an Array has ndim=0.
        """
        return len(self.shape)

    @property
    def size(self) -> int:
        """
        The total number of elements in this array.

        Equivalent to the product of all values in `self.shape`, and
        always equal to `len(self.data)`.
        """
        return self._numel(self.shape)

    def reshape(self, new_shape):
        if self._numel(new_shape) != self.size:
            raise ValueError(f"Cannot reshape {self.shape} into {new_shape}")
        return Array(self.data, shape=new_shape)

    def transpose(self):
        new_shape = tuple(reversed(self.shape))
        new_strides = tuple(reversed(self.strides))
        result = Array.__new__(Array)   # bypass __init__'s flatten logic
        result.data = self.data
        result.shape = new_shape
        result.strides = new_strides
        return result

    @classmethod
    def zeros(cls, shape):
        return cls([0.0] * cls._numel(shape), shape=shape)

    @classmethod
    def ones(cls, shape):
        return cls([1.0] * cls._numel(shape), shape=shape)

    @classmethod
    def full(cls, value: float, shape: Tuple[int, ...]) -> "Array":
        """
        Create an Array of the given shape, with every element set to
        the same scalar value.
 
        Args:
            value: The scalar value every element should be set to.
            shape: The desired shape of the resulting array, e.g. (2, 3).
 
        Returns:
            A new Array of shape `shape`, with every element equal to
            `value`.
 
        Example:
            >>> Array.full(7, (2, 3))
            Array(data=[[7, 7, 7], [7, 7, 7]], shape=(2, 3))
        """
        return cls([value] * cls._numel(shape), shape=shape)


    def allclose(self, other: "Array", tol=1e-6):
        return self.shape == other.shape and all(
            abs(a - b) < tol for a, b in zip(self.data, other.data)
        )

    # -----------------------------------------------OPERATIONS---------------------------------------------------------------
    
    # Arithmetic Operators

    def __add__(self, other: "Array") -> "Array":
        """Elementwise addition: self + other. Shapes must match exactly."""
        other = other if isinstance(other, Array) else Array.full(other, self.shape)
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Array([x + y for x, y in zip(self.data, other.data)], shape=self.shape)

    def __radd__(self, other: "Array") -> "Array":
        """Supports `scalar + array` (Python tries __radd__ when the left
        operand's __add__ doesn't know how to handle an Array)."""
        return self.__add__(other)


    def __mul__(self, other: "Array") -> "Array":
        """Elementwise multiplication: self * other. Shapes must match exactly."""
        other = other if isinstance(other, Array) else Array.full(other, self.shape)
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Array(
            [x * y for x, y in zip(self.data, other.data)],
            shape=self.shape,
        )

    def __rmul__(self, other: "Array") -> "Array":
        """Supports `scalar * array` (Python tries __rmul__ when the left
        operand's __mul__ doesn't know how to handle an Array)."""
        return self.__mul__(other)

    def __neg__(self) -> "Array":
        """Elementwise negation: -self."""
        return Array([-x for x in self.data], shape=self.shape)

    def __sub__(self, other: "Array") -> "Array":
        """Elementwise subtraction: self - other. Shapes must match exactly."""
        other = other if isinstance(other, Array) else Array.full(other, self.shape)
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Array(
            [x - y for x, y in zip(self.data, other.data)],
            shape=self.shape,
        )

    def __rsub__(self, other: "Array") -> "Array":
        """Supports `scalar - array` (Python tries __rsub__ when the left
        operand's __sub__ doesn't know how to handle an Array)."""
        other = other if isinstance(other, Array) else Array.full(other, self.shape)
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Array(
            [y - x for x, y in zip(self.data, other.data)],
            shape=self.shape,
        )

    def __truediv__(self, other: "Array") -> "Array":
        """Elementwise division: self/other. Shapes must match exactly"""
        other = other if isinstance(other, Array) else Array.full(other, self.shape)
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Array(
            [x/y for x, y in zip(self.data, other.data)],
            shape=self.shape
        )

    def __rtruediv__(self, other: "Array") -> "Array":
        """Supports `scalar / array` (Python tries __rtruediv__ when the left
        operand's __truediv__ doesn't know how to handle an Array)."""
        other = other if isinstance(other, Array) else Array.full(other, self.shape)
        if self.shape != other.shape:
            raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
        return Array(
            [y/x for x, y in zip(self.data, other.data)],
            shape=self.shape
        )

    def __pow__(self, exp: float) -> "Array":
        """
        Elementwise power: self ** exp.

        Applies the exponent independently to every element. `exp` is
        a scalar (Python int/float), not another Array — this is
        "raise each element to this power," not elementwise
        array-to-array exponentiation.
        """
        return Array([x ** exp for x in self.data], shape=self.shape)

    # Math Functions

    def exp(self) -> "Array":
        """Elementwise e^x."""
        return Array([math.exp(x) for x in self.data], shape=self.shape)

    def log(self) -> "Array":
        """
        Elementwise natural log. Raises ValueError (via math.log) if any
        element is <= 0 — this is deliberate, not caught/suppressed, since
        silently producing NaN would hide a real bug (e.g. log of a
        negative loss value).
        """
        return Array([math.log(x) for x in self.data], shape=self.shape)

    def sqrt(self) -> "Array":
        """Elementwise square root. Raises ValueError for negative elements."""
        return Array([math.sqrt(x) for x in self.data], shape=self.shape)

    # Reduction Operations

    def sum(self) -> "Array":
        """Sum all elements, returning a shape-(1,) Array."""
        return Array([sum(self.data)], shape=(1,))

    def mean(self) -> "Array":
        """Mean of all elements, returning a shape-(1,) Array."""
        return Array([sum(self.data) / self.size], shape=(1,))

    ####### Matrix Multiplication ##################3
    def _matmul_2d(self, other: "Array") -> "Array":
        """
        Core 2D matrix multiplication: self @ other.
 
        Requires self.shape = (m, k) and other.shape = (k, n), producing
        a result of shape (m, n). This is the triple-nested-loop
        implementation (O(m*k*n)) that both plain 2D matmul and each
        batch slice of 3D matmul delegate to.
        """
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
        """
        Extract the b-th 2D slice from a 3D array of shape (batch, m, n).
 
        Because storage is row-major and `batch` is the outermost
        dimension, each batch's (m, n) sub-matrix is a contiguous chunk
        of the flat buffer — no strided/scattered indexing needed, just
        a plain slice.
 
        Args:
            b: The batch index to extract, 0 <= b < self.shape[0].
 
        Returns:
            A new (m, n) Array holding a copy of batch b's data.
        """
        _, m, n = self.shape
        chunk_size = m * n
        start = b * chunk_size
        end = start + chunk_size
        return Array(self.data[start:end], shape=(m, n))
 
    @staticmethod
    def stack(arrays: List["Array"]) -> "Array":
        """
        Stack a list of equal-shaped 2D Arrays into a single 3D Array.
 
        The resulting shape is (len(arrays), m, n), where (m, n) is the
        shape shared by every array in `arrays`. Because row-major layout
        places the batch dimension outermost, this is simply the
        concatenation of each array's flat data, in order.
 
        Args:
            arrays: A non-empty list of Arrays, all sharing the same 2D
                shape.
 
        Returns:
            A new 3D Array of shape (len(arrays), m, n).
 
        Raises:
            ValueError: If `arrays` is empty, or the arrays don't all
                share the same shape.
        """
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
 
        Dispatches based on dimensionality:
            - Both 2D (m,k) @ (k,n) -> (m,n): standard matrix multiply.
            - Both 3D (batch,m,k) @ (batch,k,n) -> (batch,m,n): batched
              matrix multiply, applying 2D matmul independently to each
              batch slice. Batch sizes must match.
 
        Args:
            other: The Array to multiply with. Must be 2D if self is 2D,
                or 3D with a matching batch size if self is 3D.
 
        Returns:
            The matrix product, as described above.
 
        Raises:
            ValueError: If shapes are incompatible for multiplication, or
                if the dimensionality combination isn't supported (e.g.
                one 2D and one 3D operand).
 
        Example:
            >>> a = Array([[1, 2], [3, 4]])       # shape (2, 2)
            >>> b = Array([[5, 6], [7, 8]])       # shape (2, 2)
            >>> a.matmul(b).unflatten()
            [[19, 22], [43, 50]]
 
            >>> batch_a = Array.stack([a, a])     # shape (2, 2, 2)
            >>> batch_b = Array.stack([b, b])     # shape (2, 2, 2)
            >>> batch_a.matmul(batch_b).shape
            (2, 2, 2)
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
        """
        Return a concise developer-facing string representation showing
        the array's shape, e.g. "Array(shape=(2, 3))".
        """
        return f"Array(data={self._unflatten(self.data, self.shape)}, shape={self.shape})"