# `ndarray`

`ndarray` provides a lightweight, pure-Python N-dimensional array implementation used as the numerical foundation of **kerze**.

Unlike NumPy, it does not depend on external numerical libraries. Data is stored in a flat Python list together with shape and stride metadata.

## Overview

The `Array` class provides:

- N-dimensional array storage
- Automatic shape inference from nested lists
- Flattening and unflattening
- Shape manipulation
- Broadcasting
- Elementwise arithmetic
- Elementwise mathematical functions
- Reductions
- Matrix multiplication
- Basic comparisons
- Array creation utilities

The implementation follows concepts used by NumPy and PyTorch while keeping the implementation intentionally small and easy to understand.

## Internal Representation

An `Array` stores its data using three pieces of information:

```text
data
shape
strides
```

For example:

```python
a = Array([
    [1, 2, 3],
    [4, 5, 6],
])
```

is internally represented approximately as:

```text
data    = [1, 2, 3, 4, 5, 6]
shape   = (2, 3)
strides = (3, 1)
```

The data is stored in **row-major (C-style) order**.

The shape describes the logical dimensions, while strides determine how logical coordinates map to positions in the flat buffer.

---

# `Array`

```python
class Array(data, shape=None)
```

The main N-dimensional array class.

### Parameters

#### `data`

Can be:

- A nested Python list
- A flat list when `shape` is provided
- A scalar

For example:

```python
Array([1, 2, 3])
```

or:

```python
Array([
    [1, 2],
    [3, 4],
])
```

or:

```python
Array([1, 2, 3, 4], shape=(2, 2))
```

### `shape`

Optional explicit shape.

If omitted, the shape is inferred from nested-list structure.

```python
Array([
    [1, 2],
    [3, 4],
]).shape
# (2, 2)
```

When an explicit shape is provided, the number of elements must match the requested shape.

---

# Basic Properties

## `ndim`

Returns the number of dimensions.

```python
a = Array([
    [1, 2, 3],
    [4, 5, 6],
])

a.ndim
# 2
```

## `size`

Returns the total number of elements.

```python
a.size
# 6
```

---

# Indexing

## `get()`

Retrieve an element using logical coordinates.

```python
a = Array([
    [1, 2],
    [3, 4],
])

a.get(1, 0)
# 3
```

## `set()`

Modify an element at the specified coordinates.

```python
a.set(10, 1, 0)

a.get(1, 0)
# 10
```

Internally, coordinates are converted into a flat-buffer index using the array's strides.

---

# Shape Operations

## `reshape()`

Changes the logical shape while preserving the total number of elements.

```python
a = Array([
    [1, 2],
    [3, 4],
])

b = a.reshape((4,))

b.shape
# (4,)
```

The number of elements must remain unchanged.

```python
a.reshape((3, 2))
# ValueError
```

---

## `transpose()`

Reverses the order of all axes.

```python
a = Array([
    [1, 2, 3],
    [4, 5, 6],
])

b = a.transpose()

b.shape
# (3, 2)
```

For an array with shape:

```text
(2, 3, 4)
```

transpose produces:

```text
(4, 3, 2)
```

Unlike `reshape`, `transpose()` currently materializes a reordered data buffer rather than creating a zero-copy view.

---

## `squeeze()`

Removes dimensions whose size is `1`.

```python
a = Array([[
    [1, 2, 3]
]])

a.shape
# (1, 1, 3)

b = a.squeeze()

b.shape
# (3,)
```

A specific axis can also be provided:

```python
a.squeeze(axis=0)
```

The selected axis must have size `1`.

---

## `unsqueeze()`

Adds a dimension of size `1`.

```python
a = Array([1, 2, 3])

b = a.unsqueeze(0)

b.shape
# (1, 3)
```

Negative axes are supported:

```python
a.unsqueeze(-1)
# shape: (3, 1)
```

---

# Array Creation

## `zeros()`

Creates an array filled with zeros.

```python
Array.zeros((2, 3))
```

## `ones()`

Creates an array filled with ones.

```python
Array.ones((2, 3))
```

## `full()`

Creates an array filled with a specified value.

```python
Array.full(5.0, (2, 3))
```

---

# Broadcasting

`Array` implements NumPy-style broadcasting for elementwise binary operations.

## `broadcast_shapes()`

Determines the compatible output shape of two arrays.

```python
Array.broadcast_shapes(
    (2, 3),
    (3,),
)
# (2, 3)
```

Broadcasting aligns dimensions from the right.

Dimensions are compatible when:

- They are equal, or
- One of them is `1`

For example:

```text
(2, 3)
(   3)
-----
(2, 3)
```

Incompatible shapes raise `ValueError`.

---

## `broadcast_to()`

Expands an array to a compatible target shape.

```python
a = Array([10, 20, 30])

b = a.broadcast_to((2, 3))
```

Conceptually:

```text
[10, 20, 30]

        ↓

[[10, 20, 30],
 [10, 20, 30]]
```

---

## `unbroadcast()`

Collapses a broadcasted array back to its original shape by summing replicated dimensions.

This operation is particularly important for the **autograd engine**.

For example:

```python
grad = Array([
    [1, 2, 3],
    [4, 5, 6],
])

grad.unbroadcast((3,))
```

produces:

```text
[5, 7, 9]
```

This corresponds to the gradient contributions generated when an array of shape `(3,)` was broadcast to `(2, 3)`.

---

# Elementwise Arithmetic

The following operators support broadcasting:

```python
+
-
*
/
```

For example:

```python
a = Array([
    [1, 2, 3],
    [4, 5, 6],
])

b = Array([10, 20, 30])

a + b
```

produces:

```text
[[11, 22, 33],
 [14, 25, 36]]
```

Scalar operations are also supported:

```python
a + 2
a * 2
a - 2
a / 2
```

Reverse scalar operations are supported as well:

```python
2 + a
2 * a
2 - a
2 / a
```

---

## Power

`**` applies a scalar exponent elementwise.

```python
a = Array([1, 2, 3])

a ** 2
# [1, 4, 9]
```

---

## Negation

```python
-a
```

negates every element.

---

# Elementwise Mathematical Functions

## `exp()`

Computes:

$$
e^x
$$

elementwise.

```python
a.exp()
```

## `log()`

Computes the natural logarithm elementwise.

```python
a.log()
```

Values less than or equal to zero raise `ValueError`.

## `sqrt()`

Computes the square root elementwise.

```python
a.sqrt()
```

Negative values raise `ValueError`.

## `tanh()`

Computes the hyperbolic tangent elementwise.

```python
a.tanh()
```

---

# Reductions

`Array` provides axis-aware reduction operations.

## `sum()`

Sum all elements:

```python
a.sum()
```

Sum along an axis:

```python
a.sum(axis=0)
a.sum(axis=1)
```

Negative axes are supported:

```python
a.sum(axis=-1)
```

`keepdims=True` keeps the reduced dimension with size `1`.

```python
a.sum(axis=0, keepdims=True)
```

---

## `mean()`

Computes the arithmetic mean.

```python
a.mean()
```

or along an axis:

```python
a.mean(axis=1)
```

It supports the same `axis` and `keepdims` semantics as `sum()`.

---

## `max()`

Computes the maximum value.

```python
a.max()
```

or along an axis:

```python
a.max(axis=0)
```

`keepdims` is also supported.

---

# Matrix Multiplication

## `matmul()`

Matrix multiplication is implemented through:

```python
a.matmul(b)
```

or:

```python
a @ b
```

The current implementation supports:

### 2D × 2D

```text
(m, k) @ (k, n) → (m, n)
```

For example:

```python
a = Array([
    [1, 2],
    [3, 4],
])

b = Array([
    [5, 6],
    [7, 8],
])

c = a @ b
```

### Batched 3D × 3D

Matching batch dimensions are supported:

```text
(batch, m, k) @ (batch, k, n)
→
(batch, m, n)
```

The current implementation does **not** support general N-dimensional NumPy-style matmul broadcasting.

---

# Utility Operations

## `allclose()`

Checks whether two arrays have the same shape and approximately equal values.

```python
a.allclose(b)
```

A tolerance can be specified:

```python
a.allclose(b, tol=1e-5)
```

This is particularly useful for numerical tests.

---

## `stack()`

Stacks equally shaped 2D arrays into a 3D array.

```python
Array.stack([a, b, c])
```

If the input arrays have shape:

```text
(2, 3)
```

the result has shape:

```text
(3, 2, 3)
```

All arrays must have identical shapes.

---

# Comparison

`Array` supports equality comparison:

```python
a == b
```

The result is an `Array` containing:

```text
1.0
```

where elements are equal and:

```text
0.0
```

where they are not.

Broadcasting is applied when the shapes differ.

---

# Design Philosophy

`Array` is intentionally kept independent from the autograd system.

Its responsibilities are primarily:

```text
Storage
   ↓
Shape
   ↓
Broadcasting
   ↓
Numerical operations
   ↓
Linear algebra
   ↓
Reductions
```

Autograd is implemented at the `Tensor`/operation level on top of `Array`.

This separation allows `kerze` to use `Array` as a lightweight numerical backend while `Tensor` adds computation graphs and automatic differentiation.
