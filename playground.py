from kerze.ndarray import Array

from kerze.tensor import Tensor

from kerze.ops import add, mul, neg, sub

x = Array([[1.0, 2.0], [5.0, 2.0]])
y = Array([[3.0, 4.0]])

z = x + y

print("Z: ", z)
a = Tensor(x, requires_grad=True)
b = Tensor(y, requires_grad=True)

c = a.sqrt()
d = c.mean()
e = d + b
out = e.sum()
# out = -c
print("Out Tensor: ", out.data)

out.backward()

print("Gradient of a: ", a.grad)
print("Gradient of b: ", b.grad)  
print("Gradient of c: ", c.grad)
print("Gradient of d: ", d.grad)
print("Gradient of e: ", e.grad)    # [1.0, 1.0]
print("Gradient of out: ", out.grad)    # [1.0, ]


print(out.data)

