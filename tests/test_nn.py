"""
tests/test_nn.py

Correctness tests for kerze.nn — functional ops via gradcheck, and
Module/training-loop level integration tests (parameter registration,
Sequential composition, an actual converging training loop for both
regression and classification).
"""

import random
import pytest
from kerze.tensor import Tensor
from kerze import nn
from kerze import optim
from kerze.testing import gradcheck


def T(data, requires_grad=True):
    return Tensor(data, requires_grad=requires_grad)


class TestFunctionalGradcheck:
    def test_sigmoid(self):
        a = T([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert gradcheck(lambda x: nn.functional.sigmoid(x), [a])

    def test_tanh(self):
        a = T([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert gradcheck(lambda x: nn.functional.tanh(x), [a])

    def test_gelu(self):
        a = T([-2.0, -0.5, 0.0, 0.5, 2.0])
        assert gradcheck(lambda x: nn.functional.gelu(x), [a])

    def test_relu_away_from_kink(self):
        a = T([-2.0, -0.5, 0.5, 2.0])
        assert gradcheck(lambda x: nn.functional.relu(x), [a])

    def test_mse_loss(self):
        pred = T([1.0, 2.0, 3.0])
        target = T([1.5, 1.8, 3.3], requires_grad=False)
        assert gradcheck(lambda p: nn.functional.mse_loss(p, target), [pred])

    def test_linear(self):
        x = T([[1.0, -2.0, 0.5], [0.3, 0.1, -0.7]])  # (2,3)
        w = T([[1.0, 0.5, -1.0], [0.2, -0.3, 0.4], [0.0, 1.0, 1.0], [-0.5, 0.5, 0.5]])  # (4,3)
        b = T([0.1, -0.2, 0.3, 0.0])
        assert gradcheck(lambda x_, w_, b_: nn.functional.linear(x_, w_, b_).sum(), [x, w, b])

    def test_log_softmax_sums_to_zero_in_prob_space(self):
        x = Tensor([[1.0, 2.0, 0.5], [0.1, 0.1, 0.1]])
        log_probs = nn.functional.log_softmax(x)
        probs = log_probs.exp()
        row_sums = probs.sum(axis=1).data.data
        assert all(abs(s - 1.0) < 1e-6 for s in row_sums)

    def test_cross_entropy_gradcheck(self):
        logits = T([[1.0, 2.0, 0.5, -1.0], [0.1, -0.2, 3.0, 0.4], [2.0, 2.0, 2.0, 2.1]])
        target = [1, 2, 3]
        assert gradcheck(lambda x: nn.functional.cross_entropy(x, target), [logits])

    def test_nll_loss_gradcheck(self):
        log_probs = T([[-0.5, -1.5, -2.0], [-1.0, -1.0, -1.5]])
        target = [0, 2]
        assert gradcheck(lambda x: nn.functional.nll_loss(x, target), [log_probs])


class TestModuleStructure:
    def test_linear_registers_parameters(self):
        layer = nn.Linear(3, 4)
        params = list(layer.parameters())
        assert len(params) == 2  # weight, bias
        assert layer.weight.shape == (4, 3)
        assert layer.bias.shape == (4,)

    def test_linear_no_bias(self):
        layer = nn.Linear(3, 4, bias=False)
        assert layer.bias is None
        assert len(list(layer.parameters())) == 1

    def test_sequential_collects_nested_parameters(self):
        model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2))
        params = list(model.parameters())
        assert len(params) == 4  # 2 Linear layers x (weight, bias)

    def test_zero_grad_clears_all(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
        x = Tensor([[1.0, 2.0]])
        out = model(x)
        out.sum().backward()
        assert all(p.grad is not None for p in model.parameters())
        model.zero_grad()
        assert all(p.grad is None for p in model.parameters())

    def test_train_eval_propagates_to_children(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        model.eval()
        assert model.training is False
        assert model._modules["0"].training is False
        model.train()
        assert model.training is True
        assert model._modules["0"].training is True


class TestTrainingLoops:
    def test_regression_converges(self):
        random.seed(0)
        model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        xs = [[float(i)] for i in range(-5, 6)]
        ys = [[2.0 * i[0] + 1.0] for i in xs]
        x_batch, y_batch = Tensor(xs), Tensor(ys)

        first_loss = None
        for _ in range(200):
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            if first_loss is None:
                first_loss = loss.data.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.data.data[0] < first_loss
        assert loss.data.data[0] < 1.0

    def test_classification_converges(self):
        random.seed(0)
        # 3-class, linearly separable-ish toy dataset in 2D
        xs = [
            [0.0, 0.0], [0.1, -0.1], [-0.1, 0.1],   # class 0, near origin
            [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],     # class 1
            [-5.0, 5.0], [-4.9, 5.1], [-5.1, 4.9],  # class 2
        ]
        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05)

        x_batch = Tensor(xs)
        first_loss = None
        for _ in range(300):
            logits = model(x_batch)
            loss = criterion(logits, targets)
            if first_loss is None:
                first_loss = loss.data.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss.data.data[0] < first_loss

        # check it actually learned to classify correctly
        logits = model(x_batch)
        preds = []
        for i in range(len(xs)):
            row = [logits.data.get(i, c) for c in range(3)]
            preds.append(row.index(max(row)))
        accuracy = sum(p == t for p, t in zip(preds, targets)) / len(targets)
        assert accuracy == 1.0, f"expected perfect accuracy on toy data, got {accuracy}"
