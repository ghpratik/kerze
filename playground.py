from kerze.ndarray import Array

from kerze.tensor.tensor import Tensor

from kerze.ops import add, mul, neg, sub

data = Array([2, 4, 2, 5])

print(data.transpose())

myTensor = Tensor(data, requires_grad=True)

print(myTensor)


x = Array([[1.0, 2.0], [5.0, 2.0]])
y = Array([[3.0, 4.0], [1.0, 1.0]])
a = Tensor(x, requires_grad=True)
b = Tensor(y, requires_grad=True)
out = a**2 + b**2
# out = -c
print(out)
out.backward()
print("Gradient of a: ", a.grad, a.grad.shape, a.grad.strides)  # [0.3333, 0.25][1., 1.,]
print("Gradient of b: ", b.grad)  # [-1/9.0, -2/16], [-5., -2.] 
# print("Gradient of c: ", c.grad)  # [-1., -1]
print("Gradient of Out: ", out.grad) # [1., 1.]

print(out.data)

a = Array([[1, 2], [3, 4], [4, 2]])       # shape (2, 2)
b = Array([[5, 6], [7, 8]])       # shape (2, 2)
c = a @ b
print(c)
# Result: [[19, 22], [43, 50]]

batch_a = Array.stack([a, a])     # shape (2, 2, 2)
batch_b = Array.stack([b, b])     # shape (2, 2, 2)
res = batch_a.matmul(batch_b).shape
print(res)
# Result: (2, 2, 2)
