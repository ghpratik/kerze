"""
demo.py — a guided tour of kerze.

Runs top to bottom, printing as it goes. Each section is self-contained
and exercises a different layer of the library:

    1. Array         — raw N-D array math (kerze/ndarray.py)
    2. Tensor         — autograd: ops, backward, matmul, activations
    3. gradcheck      — proving a *new* custom op's backward is correct
    4. Regression     — Linear + ReLU + MSELoss + SGD, end to end
    5. Classification — a hand-written nn.Module, GELU, CrossEntropyLoss,
                         Adam, train()/eval(), parameter introspection

No external dependencies — everything here, including the synthetic
datasets, is built from the standard library and kerze itself.

Run with:  python3 demo.py
"""

import math
import random

from kerze.ndarray import Array
from kerze.tensor import Tensor
from kerze import nn
from kerze import optim
from kerze.testing import gradcheck


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. Array — the raw math layer, no autograd involved
# ============================================================

def demo_array():
    section("1. Array — raw N-D array operations")

    a = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2,3)
    print("a =", a)
    print("a.shape:", a.shape, " a.ndim:", a.ndim, " a.size:", a.size)

    # reshape / transpose
    print("a.reshape((3,2)):", a.reshape((3, 2)))
    print("a.transpose():", a.transpose(), " shape:", a.transpose().shape)

    # broadcasting arithmetic
    row = Array([10.0, 20.0, 30.0])  # shape (3,)
    print("a + row (broadcast):", a + row)
    print("a * 2:", a * 2)

    # elementwise math
    print("a.exp() (first row):", a.exp().reshape((2, 3)).get(0, 0), "...")
    print("a.sqrt():", a.sqrt())

    # reductions
    print("a.sum():", a.sum(), " a.sum(axis=0):", a.sum(axis=0), " a.sum(axis=1):", a.sum(axis=1))
    print("a.mean(axis=1):", a.mean(axis=1))
    print("a.max(axis=0):", a.max(axis=0))

    # matmul
    b = Array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3,2)
    print("a @ b:", a @ b)  # (2,3)@(3,2) -> (2,2)

    # squeeze / unsqueeze round trip
    c = Array([1.0, 2.0, 3.0]).unsqueeze(0)  # (1,3)
    print("unsqueeze(0):", c.shape, " squeeze():", c.squeeze().shape)


# ============================================================
# 2. Tensor — autograd: build a graph, call backward(), inspect grads
# ============================================================

def demo_tensor_autograd():
    section("2. Tensor — autograd fundamentals")

    x = Tensor([2.0, -1.0, 3.0], requires_grad=True)
    y = Tensor([1.0, 4.0, 0.5], requires_grad=True)

    # a small expression: z = sum((x * y + x^2) )
    z = (x * y + x ** 2).sum()
    z.backward()
    print("x:", x.data, " y:", y.data)
    print("z = sum(x*y + x^2) =", z.data.data[0])
    print("dz/dx:", x.grad.data, " (expected y + 2x =", (y.data + 2 * x.data).data, ")")
    print("dz/dy:", y.grad.data, " (expected x =", x.data.data, ")")

    section("2b. Tensor — math functions, activations, matmul/transpose")

    a = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    print("a.exp():", a.exp().data.data)
    print("a.relu():", a.relu().data.data)
    print("a.tanh():", a.tanh().data.data)
    print("sigmoid(a):", nn.functional.sigmoid(a).data.data)
    print("gelu(a):", nn.functional.gelu(a).data.data)

    m = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    n = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    out = (m @ n.T).sum()
    out.backward()
    print("m @ n.T =", (m.data @ n.data.transpose()).data, " sum backward populated m.grad:", m.grad.data)

    section("2c. Tensor — reductions with axis, matching Array's API")

    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    row_sums = t.sum(axis=1)
    row_sums.sum().backward()
    print("t.sum(axis=1):", row_sums.data.data, " -> t.grad:", t.grad.data)


# ============================================================
# 3. gradcheck — verify a brand-new custom op before trusting it
# ============================================================

def swish(x: Tensor) -> Tensor:
    """
    A custom activation NOT built into kerze.nn — x * sigmoid(x).
    Built purely by composing existing ops (mul, sigmoid), the same way
    kerze.nn.functional.gelu is. Demonstrates that extending kerze with
    a new activation needs zero changes to ops.py, as long as it's
    expressible from what's already there.
    """
    return x * nn.functional.sigmoid(x)


def demo_gradcheck():
    section("3. gradcheck — proving a custom op's gradient is correct")

    x = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    ok = gradcheck(lambda t: swish(t).sum(), [x], verbose=True)
    print(f"swish(x) = x * sigmoid(x) gradcheck passed: {ok}")

    # gradcheck also catches mistakes — demonstrate on a deliberately
    # wrong "custom op" to show what a failure looks like
    def buggy_double(t: Tensor) -> Tensor:
        # correct forward, WRONG backward: pretends d(2x)/dx = 1 instead of 2
        out = Tensor(t.data * 2, requires_grad=t.requires_grad, _children=(t,), _op="buggy")
        def _backward():
            if t.requires_grad:
                if t.grad is None:
                    t.zero_grad()
                t.grad += out.grad  # bug: should be 2 * out.grad
        out._backward = _backward
        return out

    bad_x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    ok_bad = gradcheck(lambda t: buggy_double(t).sum(), [bad_x])
    print(f"deliberately buggy op gradcheck passed: {ok_bad}  (correctly caught as False)")


# ============================================================
# 4. Regression — Linear + ReLU + MSELoss + SGD
# ============================================================

def make_regression_data(n=21):
    """y = 2x + 1 plus small noise."""
    random.seed(0)
    xs = [[float(i)] for i in range(-(n // 2), n // 2 + 1)]
    ys = [[2.0 * x[0] + 1.0 + random.uniform(-0.3, 0.3)] for x in xs]
    return xs, ys


def demo_regression():
    section("4. Regression — Sequential(Linear, ReLU, Linear) + MSELoss + SGD")

    xs, ys = make_regression_data()
    x_batch, y_batch = Tensor(xs), Tensor(ys)

    model = nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    print(model)
    print("parameter count:", sum(1 for _ in model.parameters()))

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(300):
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 299:
            print(f"  epoch {epoch:3d}  loss {loss.data.data[0]:.4f}")

    # sanity check: model should now roughly predict y = 2x + 1
    test_x = Tensor([[10.0]])
    pred = model(test_x)
    print(f"model(10.0) = {pred.data.data[0]:.2f}  (target ~= {2*10+1})")


# ============================================================
# 5. Classification — hand-written Module, GELU, CrossEntropyLoss, Adam
# ============================================================

def make_blobs(n_per_class=20, n_classes=3, spread=0.6, seed=1):
    """n_classes 2D gaussian blobs arranged in a ring, for a toy
    classification problem — no numpy, just random.gauss."""
    random.seed(seed)
    xs, ys = [], []
    for c in range(n_classes):
        angle = 2 * math.pi * c / n_classes
        cx, cy = 4.0 * math.cos(angle), 4.0 * math.sin(angle)
        for _ in range(n_per_class):
            xs.append([random.gauss(cx, spread), random.gauss(cy, spread)])
            ys.append(c)
    return xs, ys


class MLP(nn.Module):
    """
    A hand-written Module (rather than nn.Sequential) — demonstrates
    manual Parameter/submodule registration via plain attribute
    assignment, which Module.__setattr__ picks up automatically.
    """

    def __init__(self, in_features, hidden, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes, bias=False)  # no-bias layer, for variety

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def accuracy(logits: Tensor, targets) -> float:
    n = logits.shape[0]
    correct = 0
    for i in range(n):
        row = [logits.data.get(i, c) for c in range(logits.shape[1])]
        pred = row.index(max(row))
        correct += int(pred == targets[i])
    return correct / n


def demo_classification():
    section("5. Classification — hand-written Module, GELU, CrossEntropyLoss, Adam")

    xs, targets = make_blobs()
    x_batch = Tensor(xs)

    model = MLP(in_features=2, hidden=32, n_classes=3)
    print(model)
    print("named_parameters:")
    for name, p in model.named_parameters():
        print(f"  {name:12s} shape={p.shape}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    model.train()  # explicit, even though it's the default — shows the API
    for epoch in range(150):
        logits = model(x_batch)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 149:
            acc = accuracy(logits, targets)
            print(f"  epoch {epoch:3d}  loss {loss.data.data[0]:.4f}  accuracy {acc:.2%}")

    # eval mode — no-op for this architecture (no Dropout/BatchNorm yet),
    # but demonstrates the train()/eval() switch that Module already supports
    model.eval()
    final_logits = model(x_batch)
    final_acc = accuracy(final_logits, targets)
    print(f"final eval-mode accuracy: {final_acc:.2%}")

    section("5b. Same loss, decomposed: log_softmax + NLLLoss vs CrossEntropyLoss")
    logits = model(x_batch)
    ce = nn.CrossEntropyLoss()(logits, targets)
    log_probs = nn.functional.log_softmax(logits)
    nll = nn.NLLLoss()(log_probs, targets)
    print(f"CrossEntropyLoss: {ce.data.data[0]:.6f}   log_softmax+NLLLoss: {nll.data.data[0]:.6f}"
          f"   (should match)")


if __name__ == "__main__":
    demo_array()
    demo_tensor_autograd()
    demo_gradcheck()
    demo_regression()
    demo_classification()

    section("Done")
    print("Every module exercised: kerze.ndarray, kerze.tensor, kerze.nn "
          "(Module/Parameter/Linear/Sequential/activations/losses), "
          "kerze.optim (SGD/Adam), kerze.testing (gradcheck).")