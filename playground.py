from kerze.ndarray import Array

from kerze.tensor import Tensor

from kerze.ops import add, mul, neg, sub

x = Array([[1.0, 2.0], [5.0, 2.0]])
y = Array([[3.0, 4.0], [1.0, 1.0]])

a = Tensor(x, requires_grad=True)
b = Tensor(y, requires_grad=True)

out = a.sqrt()
# out = -c
print("Out Tensor: ", out.data)

out.backward()

print("Gradient of a: ", a.grad)  # [0.3333, 0.25][1., 1.,]
# print("Gradient of b: ", b.grad)  # [-1/9.0, -2/16], [-5., -2.] 
# print("Gradient of c: ", c.grad)  # [-1., -1]

print(out.data)

