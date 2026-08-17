# Tensor and Autograd

`kerze.tensor` provides the **autograd layer** of kerze.

The `Tensor` class wraps the lower-level `Array` class and adds automatic differentiation through a dynamic computation graph. Operations between tensors create graph nodes, and calling `.backward()` traverses that graph in reverse to compute gradients.

The design is intentionally small and PyTorch-like.

---

## Overview

The architecture separates numerical storage from automatic differentiation:

```text
Array
  │
  │ raw numerical data
  ▼
Tensor
  │
  ├── data
  ├── grad
  ├── requires_grad
  ├── _prev
  ├── _backward
  └── _op
```

### `Array`

`Array` is the low-level numerical container. It provides functionality such as:

- arithmetic
- broadcasting
- reshaping
- reductions
- matrix multiplication
- elementwise mathematical functions
- transpose

### `Tensor`

`Tensor` wraps an `Array` and adds:

- gradient tracking
- computation graph construction
- backward propagation
- gradient accumulation

This separation keeps the numerical layer independent from the autograd layer.

---

# Tensor

```python
class Tensor:
    ...
```

A `Tensor` represents one node in the computation graph.

A tensor contains both its numerical value and information about how that value was produced.

## Constructor

```python
Tensor(
    data,
    requires_grad=False,
    _children=(),
    _op="",
)
```

### Parameters

| Parameter       | Description                                            |
| --------------- | ------------------------------------------------------ |
| `data`          | An `Array` or nested/flat Python list                  |
| `requires_grad` | Whether gradients should be tracked                    |
| `_children`     | Internal parent tensors in the computation graph       |
| `_op`           | Internal name of the operation that created the tensor |

For normal user code, `_children` and `_op` should not need to be specified.

### Example

```python
x = Tensor(
    [[1.0, 2.0],
     [3.0, 4.0]],
    requires_grad=True
)
```

---

# Tensor Attributes

## `data`

```python
x.data
```

Contains the underlying `Array`.

```python
x = Tensor([1.0, 2.0, 3.0])

print(x.data)
```

The `Tensor` itself handles autograd, while `Array` handles numerical operations.

---

## `grad`

```python
x.grad
```

Stores the accumulated gradient of the final output with respect to the tensor.

Before backpropagation:

```python
x.grad is None
```

After:

```python
y.backward()
```

the gradient is populated for tensors that have `requires_grad=True`.

Example:

```python
x = Tensor([2.0], requires_grad=True)

y = x * x
y.backward()

print(x.grad.data)
```

Result:

```text
[4.0]
```

because:

[
y=x^2
]

and therefore:

[
\frac{dy}{dx}=2x=4
]

---

## `requires_grad`

```python
x = Tensor([2.0], requires_grad=True)
```

Determines whether the tensor participates in gradient computation.

Typically, model parameters use:

```python
requires_grad=True
```

while input data can use:

```python
requires_grad=False
```

Intermediate tensors automatically require gradients if at least one of their inputs does.

For example:

```python
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0])

z = x * y

print(z.requires_grad)
```

Result:

```text
True
```

---

## `shape`

```python
x.shape
```

A convenience property that exposes the shape of the underlying `Array`.

```python
x = Tensor([[1, 2, 3], [4, 5, 6]])

print(x.shape)
```

```text
(2, 3)
```

---

# Computation Graph

Every operation between tensors creates a new tensor and records its parents.

For example:

```python
x = Tensor([2.0], requires_grad=True)

y = x * x
z = y + x
```

Conceptually, the graph is:

```text
        x
       / \
      /   \
     ▼     ▼
    x*x    │
     │     │
     ▼     │
     y     │
      \    │
       \   │
        ▼  ▼
          z
```

Internally, each tensor stores its parents in:

```python
_tensor._prev
```

and the operation that created it in:

```python
_tensor._op
```

The backward function for the operation is stored in:

```python
_tensor._backward
```

---

# Backpropagation

Backpropagation is performed using:

```python
tensor.backward()
```

For example:

```python
x = Tensor([2.0], requires_grad=True)

y = x * x
y.backward()

print(x.grad.data)
```

Output:

```text
[4.0]
```

The implementation:

1. Traverses the computation graph.
2. Builds a topological ordering.
3. Seeds the final tensor's gradient with ones.
4. Traverses the graph in reverse order.
5. Executes each tensor's `_backward` function.
6. Accumulates gradients into parent tensors.

Conceptually:

```text
Forward:

x → operation → operation → loss


Backward:

x ← operation ← operation ← loss
```

---

# Gradient Accumulation

Gradients are accumulated rather than replaced.

For example:

```python
x = Tensor([2.0], requires_grad=True)

y = x * x
z = y + x

z.backward()
```

Since:

[
z=x^2+x
]

the derivative is:

[
\frac{dz}{dx}=2x+1
]

At `x = 2`:

```text
5
```

The implementation accumulates contributions from every path through the computation graph.

This is particularly important when a tensor is used multiple times.

---

# `zero_grad()`

```python
x.zero_grad()
```

Resets the tensor's accumulated gradient.

It creates a zero-filled `Array` with the same shape as the tensor.

```python
x = Tensor([1.0, 2.0], requires_grad=True)

x.zero_grad()

print(x.grad.data)
```

```text
[0.0, 0.0]
```

This should normally be called between training iterations.

Example:

```python
for batch in data:

    parameter.zero_grad()

    loss = model(batch)
    loss.backward()

    # update parameters
```

Without resetting gradients, successive backward passes accumulate into the existing gradient.

---

# Arithmetic Operations

`Tensor` supports standard arithmetic operators.

## Addition

```python
z = a + b
```

Equivalent to:

[
z=a+b
]

Backward:

[
\frac{\partial z}{\partial a}=1
]

[
\frac{\partial z}{\partial b}=1
]

Broadcasting is supported.

```python
a = Tensor(
    [[1, 2, 3],
     [4, 5, 6]],
    requires_grad=True
)

b = Tensor([10, 20, 30], requires_grad=True)

c = a + b
```

---

## Subtraction

```python
z = a - b
```

Backward:

[
\frac{\partial z}{\partial a}=1
]

[
\frac{\partial z}{\partial b}=-1
]

---

## Multiplication

```python
z = a * b
```

Elementwise multiplication.

Backward uses the product rule:

[
\frac{\partial z}{\partial a}=b
]

[
\frac{\partial z}{\partial b}=a
]

The implementation also handles broadcasting during backward propagation.

---

## Division

```python
z = a / b
```

Backward:

[
\frac{\partial z}{\partial a}=\frac{1}{b}
]

and

[
\frac{\partial z}{\partial b}
=============================

-\frac{a}{b^2}
]

---

## Negation

```python
z = -a
```

Backward:

[
\frac{dz}{da}=-1
]

---

## Power

```python
z = a ** 2
```

More generally:

[
z=a^p
]

Backward:

[
\frac{dz}{da}=p a^{p-1}
]

The exponent is currently a scalar Python value rather than a differentiable tensor.

---

# Broadcasting and Autograd

`kerze` supports NumPy-style broadcasting for arithmetic operations.

For example:

```python
a.shape == (2, 3)
b.shape == (3,)
```

allows:

```python
c = a + b
```

The output has shape:

```text
(2, 3)
```

During backward propagation, the gradient has the output shape:

```text
(2, 3)
```

but `b` originally has shape:

```text
(3,)
```

Therefore the gradient must be collapsed back to `(3,)`.

This is handled using:

```python
unbroadcast()
```

For example:

```text
Forward:

a (2,3) + b (3,)
          ↓
       output (2,3)


Backward:

grad (2,3)
    ↓
unbroadcast
    ↓
grad for b (3,)
```

This is required for broadcast-aware operations such as:

- addition
- subtraction
- multiplication
- division

---

# Mathematical Functions

`Tensor` exposes several elementwise mathematical functions.

## Exponential

```python
y = x.exp()
```

Computes:

[
e^x
]

Derivative:

[
\frac{d}{dx}e^x=e^x
]

The implementation can therefore use the output directly during backward propagation.

---

## Natural Logarithm

```python
y = x.log()
```

Computes:

[
\ln(x)
]

Derivative:

[
\frac{d}{dx}\ln(x)=\frac{1}{x}
]

---

## Square Root

```python
y = x.sqrt()
```

Computes:

[
\sqrt{x}
]

Derivative:

[
\frac{d}{dx}\sqrt{x}
====================

\frac{1}{2\sqrt{x}}
]

---

# Reduction Operations

`Tensor` provides:

```python
sum()
mean()
max()
```

These delegate numerical computation to `Array` while adding gradient propagation.

---

## Sum

```python
y = x.sum()
```

Computes:

[
y=\sum_i x_i
]

The derivative with respect to every contributing element is:

[
\frac{\partial y}{\partial x_i}=1
]

Therefore the incoming gradient is broadcast back to the original tensor shape.

Axis reductions are also supported:

```python
y = x.sum(axis=0)
```

and:

```python
y = x.sum(axis=1, keepdims=True)
```

---

## Mean

```python
y = x.mean()
```

Computes:

[
y=\frac{1}{n}\sum_i x_i
]

Derivative:

[
\frac{\partial y}{\partial x_i}=\frac{1}{n}
]

The gradient is expanded back to the original shape after applying the `1/n` scaling.

---

## Max

```python
y = x.max()
```

The gradient is propagated only through the element that produced the maximum.

For:

```text
[1, 5, 3]
```

the backward mask is:

```text
[0, 1, 0]
```

For an axis reduction, the mask is broadcast back to the original tensor shape.

---

# Matrix Multiplication

Matrix multiplication is available through:

```python
a.matmul(b)
```

or:

```python
a @ b
```

Example:

```python
a = Tensor(
    [[1.0, 2.0],
     [3.0, 4.0]],
    requires_grad=True
)

b = Tensor(
    [[5.0, 6.0],
     [7.0, 8.0]],
    requires_grad=True
)

c = a @ b
```

Forward:

[
C=AB
]

Backward:

[
\frac{\partial L}{\partial A}
=============================

\frac{\partial L}{\partial C}B^T
]

and:

[
\frac{\partial L}{\partial B}
=============================

A^T\frac{\partial L}{\partial C}
]

The current implementation supports:

- 2D × 2D
- matching-batch 3D × 3D

General N-dimensional broadcasting matmul is outside the current scope.

---

# Transpose

Transpose is available through:

```python
x.T
```

or internally through:

```python
transpose(x)
```

For a 2D tensor:

```text
(2, 3) → (3, 2)
```

Since transpose is its own inverse, the backward operation is also transpose:

[
\frac{\partial L}{\partial A}
=============================

\left(\frac{\partial L}{\partial A^T}\right)^T
]

---

# Activation Functions

The current Tensor API provides:

```python
x.relu()
x.tanh()
```

---

## ReLU

```python
y = x.relu()
```

Computes:

[
\operatorname{ReLU}(x)=\max(0,x)
]

Derivative:

[
\frac{d}{dx}\operatorname{ReLU}(x)
==================================

\begin{cases}
1 & x>0\
0 & x\leq0
\end{cases}
]

The implementation constructs a binary gradient mask.

Example:

```python
x = Tensor([-1.0, 0.0, 2.0], requires_grad=True)

y = x.relu()
y.backward()

print(x.grad.data)
```

```text
[0.0, 0.0, 1.0]
```

---

## Tanh

```python
y = x.tanh()
```

Computes:

[
\tanh(x)
]

Derivative:

[
\frac{d}{dx}\tanh(x)
====================

1-\tanh^2(x)
]

Since the output is already:

[
y=\tanh(x)
]

the backward pass can use:

[
1-y^2
]

directly.

The implementation delegates the forward computation to:

```python
Array.tanh()
```

rather than implementing it using exponentials. This avoids unnecessary overflow for large input magnitudes.

---

# Index Selection

`select()` provides row-wise selection:

```python
x.select(indices)
```

For a tensor:

```text
[
    [1, 2, 3],
    [4, 5, 6]
]
```

and:

```python
indices = [1, 2]
```

the result is:

```text
[2, 6]
```

This operation is primarily useful for classification losses such as:

- `CrossEntropyLoss`
- `NLLLoss`

The indices themselves are ordinary Python integers and are not differentiable.

During backward propagation, gradients are placed only at the selected positions.

---

# Operation Layer

The actual implementations of Tensor operations live in:

```text
kerze/ops.py
```

The `Tensor` methods primarily act as a public interface.

For example:

```python
def relu(self):
    from .ops import relu
    return relu(self)
```

The actual operation is implemented in:

```python
def relu(a: Tensor) -> Tensor:
    ...
```

This separation keeps the `Tensor` API clean while keeping forward and backward implementations together.

---

# Operation Structure

Each differentiable operation generally follows the same pattern:

```python
def operation(a: Tensor) -> Tensor:

    # 1. Forward computation
    out_data = ...

    # 2. Create output Tensor
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="operation",
    )

    # 3. Define backward
    def _backward():
        ...

    # 4. Attach backward function
    out._backward = _backward

    return out
```

For binary operations:

```text
        a ─────┐
               │
               ▼
           operation ───→ out
               ▲
               │
        b ─────┘
```

The operation records both `a` and `b` as parents.

---

# Current Operation Categories

The current Tensor/autograd layer contains the following categories.

### Arithmetic

```text
+
-
*
/
**
-
```

### Mathematical functions

```text
exp()
log()
sqrt()
```

### Reductions

```text
sum()
mean()
max()
```

### Linear algebra

```text
matmul()
@
T
```

### Activations

```text
relu()
tanh()
```

### Indexing

```text
select()
```

---

# Design Philosophy

The autograd system follows a deliberately minimal design:

```text
Array
  ↓
numerical operations
  ↓
Tensor
  ↓
computation graph
  ↓
backward closures
  ↓
gradients
```

Rather than implementing a separate symbolic differentiation system, every operation knows how to differentiate itself through its `_backward` closure.

For example, multiplication stores:

```python
def _backward():
    ...
```

which applies the product rule and accumulates the resulting gradients into its parents.

This makes adding new differentiable operations straightforward:

1. Implement the forward computation.
2. Create the output `Tensor`.
3. Register its parent tensors.
4. Implement the mathematical derivative.
5. Store the derivative logic in `_backward`.
6. Return the output tensor.

That pattern is the core of `kerze`'s autograd engine.
