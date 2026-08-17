import random
random.seed(0)

from kerze.tensor import Tensor
from kerze.ndarray import Array
from kerze import nn
from kerze import optim

# --- Test 1: matmul + transpose gradient check (2x2 @ 2x2) ---
a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
out = a @ b
assert out.data.data == [19.0, 22.0, 43.0, 50.0], out.data.data
loss = out.sum()
loss.backward()
# d(sum(a@b))/da = ones(2,2) @ b.T ; d/db = a.T @ ones(2,2)
print("matmul a.grad:", a.grad.data, "expected:", (Array.ones((2,2)) @ b.data.transpose()).data)
print("matmul b.grad:", b.grad.data, "expected:", (a.data.transpose() @ Array.ones((2,2))).data)
assert a.grad.allclose(Array.ones((2,2)) @ b.data.transpose())
assert b.grad.allclose(a.data.transpose() @ Array.ones((2,2)))
print("[PASS] matmul forward+backward\n")

# --- Test 2: relu forward+backward ---
x = Tensor([-1.0, 0.0, 2.0, 3.5], requires_grad=True)
r = x.relu()
assert r.data.data == [0.0, 0.0, 2.0, 3.5]
r.sum().backward()
assert x.grad.data == [0.0, 0.0, 1.0, 1.0]
print("[PASS] relu forward+backward\n")

# --- Test 3: Linear layer forward shape + backward populates grads ---
lin = nn.Linear(3, 4)
xb = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # (batch=2, in=3)
out = lin(xb)
assert out.shape == (2, 4), out.shape
out.sum().backward()
assert lin.weight.grad is not None
assert lin.bias.grad is not None
print("[PASS] Linear forward shape (2,4) + backward populates weight/bias grads\n")

# --- Test 4: full Module.parameters() / Sequential / SGD training loop ---
# Fit y = 2x + 1 with a tiny MLP: Linear(1,8) -> ReLU -> Linear(8,1)
model = nn.Sequential(
    nn.Linear(1, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_params = sum(1 for _ in model.parameters())
print(f"model has {n_params} parameter tensors")
assert n_params == 4  # weight+bias for each of 2 Linear layers

xs = [[float(i)] for i in range(-5, 6)]       # -5..5
ys = [[2.0 * i[0] + 1.0] for i in xs]

losses = []
for epoch in range(200):
    x_batch = Tensor(xs, requires_grad=False)
    y_batch = Tensor(ys, requires_grad=False)

    pred = model(x_batch)
    loss = criterion(pred, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.data.data[0])

print("first loss:", losses[0])
print("last loss:", losses[-1])
assert losses[-1] < losses[0], "loss did not decrease!"
assert losses[-1] < 1.0, f"loss did not converge, final={losses[-1]}"
print("[PASS] full training loop: loss decreased from", losses[0], "to", losses[-1])

# --- Test 5: Adam optimizer runs without error ---
model2 = nn.Sequential(nn.Linear(1, 4), nn.Sigmoid(), nn.Linear(4, 1))
opt2 = optim.Adam(model2.parameters(), lr=0.01)
x_batch = Tensor(xs, requires_grad=False)
y_batch = Tensor(ys, requires_grad=False)
for _ in range(20):
    pred = model2(x_batch)
    loss = criterion(pred, y_batch)
    opt2.zero_grad()
    loss.backward()
    opt2.step()
print("[PASS] Adam ran 20 steps without error, final loss:", loss.data.data[0])

print("\nALL TESTS PASSED")
