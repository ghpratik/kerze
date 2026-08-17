# kerze

A lightweight, PyTorch-like machine learning library built from scratch in Python.

**kerze** combines a small NumPy-inspired multidimensional array implementation with an automatic differentiation engine and a neural-network API.

The project is designed to make the internals of numerical computing and deep learning easier to understand by keeping the implementation small, explicit, and readable.

> ⚠️ **Warning:** kerze is primarily an experimental project, not a replacement for NumPy, PyTorch, or other production-grade numerical computing frameworks and it currently supports **CPU execution only**. There is **no GPU/CUDA support**.

# Installation

## Install with pip

If kerze is published on PyPI, install it with:

```bash
pip install kerze
```

### From Source

```bash
git clone https://github.com/ghpratik/kerze.git
cd kerze
pip install -e .
```

## Overview

Kerze is organized into three main layers:

```text
Array
  ↓
Tensor + Autograd
  ↓
nn
```

- **`Array`** — NumPy-like multidimensional array and mathematical operations.
- **`Tensor`** — wraps `Array` and adds automatic differentiation.
- **`nn`** — PyTorch-like neural-network building blocks.

Example:

```python
from kerze import Tensor, nn

model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
y = model(x)

loss = y.mean()
loss.backward()
```

---

## Features

- NumPy-inspired multidimensional `Array`
- Broadcasting
- Elementwise arithmetic operations
- Matrix multiplication
- Reductions such as `sum`, `mean`, and `max`
- Reshape, transpose, squeeze, and other shape operations
- Automatic differentiation
- Reverse-mode backpropagation
- Computation graphs
- Gradient accumulation
- Gradient broadcasting / unbroadcasting
- Basic mathematical operations
- Activation functions
- Neural-network modules
- Learnable Parameters
- Sequential models
- Linear layers
- Loss functions
- Functional neural-network API
- Parameter initialization
- Training/evaluation modes
- PyTorch-like API design

---

## Neural Network API

Kerze provides a small PyTorch-inspired `nn` API:

```python
nn.Module
nn.Parameter
nn.Linear

nn.ReLU
nn.GELU
nn.Sigmoid
nn.Tanh

nn.MSELoss
nn.NLLLoss
nn.CrossEntropyLoss

nn.Sequential
```

Functional operations are available through:

```python
from kerze.nn import functional as F
```

## Autograd

Kerze builds a computation graph during tensor operations and traverses it backwards during:

```python
loss.backward()
```

Each operation defines its own backward rule, allowing gradients to propagate through arithmetic, reductions, activations, matrix multiplication, and other supported operations.

## Scope & Limitations

Kerze is intentionally **small and educational**.

- CPU only
- No GPU/CUDA support
- No optimized C/C++ kernels
- Limited NumPy compatibility
- Limited neural-network layers
- No optimizers yet / limited optimizer support
- No distributed training
- No production-scale performance guarantees
- Some operations support a smaller set of shapes/features than NumPy/PyTorch

The implementation prioritizes **clarity and understanding of the underlying mechanics** over performance and completeness.

## Documentation

Detailed documentation is available in [`docs/`](docs/):

- [`Array`](docs/ndarray.md)
- [`Tensor & Autograd`](docs/tensor.md)
- [`Neural Networks`](docs/nn.md)

## Status

Kerze is an experimental/learning project and is actively evolving.

## License

Kerze is licensed under the [MIT License](LICENSE).
