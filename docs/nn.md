# `kerze.nn` — Neural Network API

The `nn` package provides the higher-level neural-network building blocks of **kerze**. It sits on top of the autograd system provided by `Tensor` and `ops`.

The design follows a PyTorch-like separation of responsibilities:

```text
kerze
├── ndarray.py          Raw numerical array operations
├── tensor.py           Autograd-enabled Tensor
├── ops.py              Forward + backward primitives
│
└── nn/
    ├── __init__.py     Public neural-network API
    ├── module.py       Module base class
    ├── parameter.py    Learnable parameters
    ├── linear.py       Stateful neural-network layers
    ├── activation.py   Activation layers
    ├── loss.py         Loss modules
    ├── functional.py   Stateless neural-network functions
    ├── container.py    Module containers
    └── init.py         Parameter initialization
```

The `nn` layer is primarily responsible for **model structure, learnable parameters, layers, activations, losses, and reusable neural-network functions**.

---

## 1. Public API

### `nn/__init__.py`

The package exposes the main neural-network components through a single API:

```python
from kerze import nn

model = nn.Sequential(
    nn.Linear(3, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1)
)
```

The public API includes:

| Component          | Purpose                                  |
| ------------------ | ---------------------------------------- |
| `Module`           | Base class for neural-network components |
| `Parameter`        | Marks learnable tensors                  |
| `Linear`           | Fully-connected layer                    |
| `ReLU`             | ReLU activation                          |
| `GELU`             | GELU activation                          |
| `Sigmoid`          | Sigmoid activation                       |
| `Tanh`             | Hyperbolic tangent activation            |
| `MSELoss`          | Mean squared error                       |
| `NLLLoss`          | Negative log likelihood                  |
| `CrossEntropyLoss` | Cross entropy                            |
| `Sequential`       | Sequential module container              |
| `functional`       | Stateless neural-network functions       |
| `init`             | Parameter initialization utilities       |

---

# 2. `Module`

### `nn/module.py`

`Module` is the **base class for every neural-network component** in kerze.

Layers such as `Linear`, activation functions, losses, and user-defined models inherit from it.

```python
class MyModel(Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        ...
```

The central responsibility of `Module` is to maintain the **structural graph of a model**.

This is different from the computation graph maintained by `Tensor`.

```text
Tensor
  └── computation graph
      ├── operations
      ├── gradients
      └── backward functions


Module
  └── model structure
      ├── parameters
      └── child modules
```

---

## Automatic parameter registration

`Module` overrides `__setattr__`.

When a `Parameter` is assigned:

```python
self.weight = Parameter(...)
```

it is automatically registered in:

```python
self._parameters
```

Likewise, assigning another `Module`:

```python
self.layer1 = Linear(10, 20)
```

registers it in:

```python
self._modules
```

This means models don't need to manually maintain parameter lists.

For example:

```python
class Model(Module):
    def __init__(self):
        super().__init__()

        self.layer1 = Linear(10, 20)
        self.layer2 = Linear(20, 1)
```

The resulting structure is conceptually:

```text
Model
├── layer1
│   ├── weight
│   └── bias
└── layer2
    ├── weight
    └── bias
```

---

## `forward()`

Every concrete module is expected to implement:

```python
def forward(self, x):
    ...
```

`Module.forward()` itself raises `NotImplementedError`.

This provides the computation performed by the module.

---

## `__call__()`

Modules can be called directly:

```python
output = layer(x)
```

because:

```python
def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)
```

Therefore:

```python
layer(x)
```

is equivalent to:

```python
layer.forward(x)
```

---

## Parameter traversal

### `parameters()`

Returns every `Parameter` belonging to the module and its nested child modules.

```python
for parameter in model.parameters():
    print(parameter)
```

For a nested model, traversal is recursive.

---

### `named_parameters()`

Returns parameter names together with their `Parameter` objects.

For example:

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

can produce names such as:

```text
layer1.weight
layer1.bias
layer2.weight
layer2.bias
```

This is useful for debugging and inspecting model structure.

---

### `modules()`

Returns the module itself and all recursively nested modules.

```python
for module in model.modules():
    print(module)
```

---

# 3. Training utilities

### `zero_grad()`

Resets all parameter gradients:

```python
model.zero_grad()
```

The implementation sets:

```python
parameter.grad = None
```

This is important because kerze's autograd system **accumulates gradients** during backward propagation.

A typical training step therefore follows:

```python
model.zero_grad()

prediction = model(x)
loss = criterion(prediction, target)

loss.backward()
```

---

## Training and evaluation modes

Modules have a:

```python
training
```

attribute.

By default:

```python
training = True
```

### `train()`

Sets the module and all child modules to training mode:

```python
model.train()
```

### `eval()`

Sets the module and all child modules to evaluation mode:

```python
model.eval()
```

The infrastructure is present for layers whose behavior differs between training and evaluation, such as Dropout or BatchNorm.

---

# 4. `Parameter`

### `nn/parameter.py`

`Parameter` is a specialized `Tensor` used to represent **learnable model parameters**.

```python
weight = Parameter(...)
```

It behaves like a normal `Tensor`, but is identifiable by `Module` as something that should be registered as a model parameter.

By default:

```python
Parameter(...).requires_grad == True
```

For example:

```python
weight = Parameter(Array.zeros((3, 4)))
```

The distinction is primarily semantic and structural:

```text
Tensor
└── ordinary value participating in computation

Parameter
└── Tensor representing learnable model state
```

This allows:

```python
model.parameters()
```

to automatically discover the values that an optimizer should update.

---

# 5. `Linear`

### `nn/linear.py`

`Linear` implements a fully-connected affine transformation:

$$y = xW^T + b$$

where:

```text
x       = input
W       = weight
b       = bias
```

For the standard shapes:

```text
x      : (batch, in_features)
weight : (out_features, in_features)
bias   : (out_features,)
output : (batch, out_features)
```

Example:

```python
layer = nn.Linear(3, 4)

x = Tensor(
    [[1.0, 2.0, 3.0]],
    requires_grad=True
)

output = layer(x)
```

The layer owns two learnable parameters:

```text
Linear
├── weight
└── bias
```

unless `bias=False` is specified.

```python
layer = nn.Linear(3, 4, bias=False)
```

The actual mathematical computation is delegated to:

```python
F.linear(...)
```

This keeps `Linear` thin: the module owns the state while `functional.py` owns the computation.

---

# 6. Activation modules

### `nn/activation.py`

Activation modules are **parameter-free modules**.

They don't own learnable parameters. Their purpose is to provide a uniform module interface so activations can be placed inside models and `Sequential`.

For example:

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 2)
)
```

The activation modules currently provided are:

```text
ReLU
GELU
Sigmoid
Tanh
```

---

## ReLU

$$f(x)=\max(x,0)$$

```python
nn.ReLU()
```

Delegates to:

```python
F.relu(x)
```

---

## GELU

GELU uses the tanh approximation:

$$
GELU(x) =

\frac{1}{2}x
\left(
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)
\right)
\right)
$$

```python
nn.GELU()
```

It is implemented entirely by composing existing Tensor operations.

---

## Sigmoid

$$\sigma(x)=\frac{1}{1+e^{-x}}$$

```python
nn.Sigmoid()
```

The implementation is composed from existing operations:

```text
neg
  ↓
exp
  ↓
add
  ↓
div
```

Therefore it does not require a dedicated backward implementation.

---

## Tanh

$$
\tanh(x)
=

\frac{e^x-e^{-x}}
{e^x+e^{-x}}
$$

```python
nn.Tanh()
```

It is also constructed using existing Tensor operations.

---

# 7. Functional API

### `nn/functional.py`

`functional.py` contains **stateless neural-network operations**.

This is an important architectural separation.

A function here:

```python
F.linear(x, weight, bias)
```

does not own any state.

A module:

```python
nn.Linear(10, 20)
```

owns parameters and delegates its computation to the functional implementation.

The relationship is:

```text
nn.Linear
    │
    └── F.linear()
            │
            └── Tensor operations
                    │
                    └── ops.py
                            │
                            └── Array
```

This makes functional operations independently reusable and testable.

---

## Functional layers

### `F.linear`

Computes:

$$xW^T+b$$

and supports broadcasting of the bias.

---

### `F.sigmoid`

Computes the sigmoid activation through existing Tensor primitives.

---

### `F.tanh`

Computes the hyperbolic tangent using exponential operations.

---

### `F.gelu`

Computes the tanh approximation of GELU.

---

### `F.relu`

Delegates to:

```python
x.relu()
```

---

# 8. Loss functions

`functional.py` also contains stateless loss functions.

---

## MSE

```python
F.mse_loss(pred, target)
```

Computes:

$$MSE = \frac{1}{n}\sum_i(pred_i-target_i)^2$$

It is implemented using existing Tensor operations:

```text
sub
 ↓
mul
 ↓
mean
```

No dedicated autograd operation is required.

---

## Log Softmax

```python
F.log_softmax(x)
```

Computes:

$$\log\operatorname{softmax}(x)$$

using the log-sum-exp technique for numerical stability.

The implementation first subtracts the maximum value:

```text
x
 ↓
max
 ↓
x - max(x)
 ↓
exp
 ↓
sum
 ↓
log
```

This avoids directly computing potentially very large exponentials.

---

## Softmax

```python
F.softmax(x)
```

is implemented as:

```python
log_softmax(x).exp()
```

---

## NLL Loss

```python
F.nll_loss(log_probs, target)
```

expects **log-probabilities** and integer class indices.

For a batch:

```text
log_probs: (batch, classes)
target:    [class_0, class_1, ...]
```

the target class is selected from each row and the negative mean is returned.

Internally it uses the Tensor `select()` operation.

---

## Cross Entropy

```python
F.cross_entropy(logits, target)
```

combines:

```text
logits
  ↓
log_softmax
  ↓
nll_loss
```

Therefore `CrossEntropyLoss` expects **raw logits**, not already-softmaxed probabilities.

---

# 9. Loss Modules

### `nn/loss.py`

The loss modules are thin `Module` wrappers around functional loss functions.

This provides a PyTorch-like API:

```python
criterion = nn.MSELoss()

loss = criterion(prediction, target)
```

rather than:

```python
loss = F.mse_loss(prediction, target)
```

Available losses:

```text
MSELoss
NLLLoss
CrossEntropyLoss
```

---

## `MSELoss`

```python
criterion = nn.MSELoss()
loss = criterion(pred, target)
```

Delegates to:

```python
F.mse_loss(...)
```

---

## `NLLLoss`

```python
criterion = nn.NLLLoss()
loss = criterion(log_probs, target)
```

The input must contain log-probabilities.

---

## `CrossEntropyLoss`

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)
```

The input must contain raw logits.

It internally performs:

```text
logits
  ↓
log_softmax
  ↓
NLL loss
```

---

# 10. Module vs Functional API

The central distinction in `kerze.nn` is:

|             | Module                | Functional              |
| ----------- | --------------------- | ----------------------- |
| State       | Can own state         | Stateless               |
| Parameters  | Can own `Parameter`s  | Does not own parameters |
| Model tree  | Participates          | No                      |
| `forward()` | Yes                   | No                      |
| Callable    | Yes                   | Normal function         |
| Example     | `nn.Linear`           | `F.linear`              |
| Example     | `nn.ReLU`             | `F.relu`                |
| Example     | `nn.CrossEntropyLoss` | `F.cross_entropy`       |

A useful mental model is:

```text
Module
  = state + structure + callable interface

Functional
  = computation
```

For example:

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(...)
        self.bias = Parameter(...)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```

The `Linear` module owns the parameters, while `F.linear` performs the actual calculation.

---

# 11. Overall `nn` architecture

The complete flow is:

```text
                    kerze.nn
                       │
        ┌──────────────┴──────────────┐
        │                             │
     Modules                       Functional
        │                             │
   ┌────┼──────────────┐              │
   │    │              │              │
Linear Activation     Loss        F.linear()
   │    │              │          F.relu()
   │    │              │          F.gelu()
   │    │              │          F.softmax()
   │    │              │          F.cross_entropy()
   │    │              │
   └────┴──────────────┘
             │
             ▼
        Tensor operations
             │
             ▼
           ops.py
             │
             ▼
          ndarray.py
```

The most important design principle is that **each layer has one responsibility**:

- `Module` → model structure and parameter registration
- `Parameter` → learnable state
- `Linear` → stateful affine layer
- `activation.py` → activation modules
- `loss.py` → loss modules
- `functional.py` → stateless neural-network computations
- `container.py` → composition of modules
- `init.py` → parameter initialization
- `Tensor`/`ops.py` → autograd and differentiation
- `Array` → numerical storage and array operations

This gives kerze a clean progression from raw numerical computation to trainable neural networks:

```text
Array
  ↓
Tensor + Autograd
  ↓
Functional operations
  ↓
Modules + Parameters
  ↓
Sequential / Models
  ↓
Training
```
