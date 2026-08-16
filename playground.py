from kerze.ndarray import Array

from kerze.tensor import Tensor

from kerze.ops import add, mul, neg, sub

import math

y = Tensor(2.0, requires_grad=True)
print(y)

x = Tensor(
    Array([-2.0, -1.0, 0.0, 1.0, 2.0]),
    requires_grad=True,
)

y = x.gelu()
loss = y.sum()

loss.backward()

print(x.grad)

