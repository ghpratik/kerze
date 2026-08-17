"""
functional.py

Stateless functions that operate on Tensors and return Tensors. No
learnable state lives here — that's what Module subclasses (Linear,
ReLU, ...) are for. This file is the "dumb computation" layer of nn,
exactly the way ops.py is the dumb computation layer under Tensor.

Module classes should be thin: hold Parameters, call one of these
functions, return the result. Keeping the actual math here (rather than
inline in each Module.forward) makes every op independently testable
and reusable outside a Module (e.g. F.mse_loss doesn't need a Module at
all).
"""

from __future__ import annotations
import math
from kerze.tensor import Tensor
from typing import Optional


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    Affine transform: out = x @ weight.T + bias

    Shapes (matching PyTorch's nn.Linear convention):
        x:      (batch, in_features)
        weight: (out_features, in_features)
        bias:   (out_features,)
        out:    (batch, out_features)
    """
    out = x @ weight.T
    if bias is not None:
        out = out + bias  # broadcasts (batch, out) + (out,) — already supported
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Elementwise sigmoid: out = 1 / (1 + exp(-x))

    Fully expressible with existing ops.py primitives (neg, exp, add,
    div) — no new backward rule needed. Autograd correctness falls out
    for free from composing the existing ops.
    """
    return 1 / (1 + (-x).exp())


def tanh(x: Tensor) -> Tensor:
    """
    Elementwise tanh: out = (e^x - e^-x) / (e^x + e^-x)

    Also fully expressible with existing primitives. Numerically this
    naive form can overflow for large |x| (e^x grows unbounded before
    the division stabilizes it) — fine for a learning project, but flag
    it as a known simplification vs. a numerically-stable dedicated op.
    """
    ex = x.exp()
    e_neg_x = (-x).exp()
    return (ex - e_neg_x) / (ex + e_neg_x)


def gelu(x: Tensor) -> Tensor:
    """
    Gaussian Error Linear Unit — tanh approximation (the same one
    PyTorch's `nn.GELU(approximate='tanh')` and most transformer
    implementations use, since the exact form needs erf, which neither
    Array nor ops.py implements):

        out = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Fully expressible with existing primitives (pow, mul, add, tanh) —
    no new ops.py primitive needed, same as sigmoid. Smoother than ReLU
    (no kink at 0), which is why it's the default in transformer
    feedforward blocks.
    """
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * (x ** 3))
    return 0.5 * x * (1 + inner.tanh())


def relu(x: Tensor) -> Tensor:
    """
    Elementwise ReLU: out = max(x, 0). See ops.relu / Tensor.relu.
    """
    return x.relu()


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    Mean squared error: mean((pred - target) ** 2)

    Fully working today — only uses sub, mul, mean, all of which exist.

    Example:
        >>> pred = Tensor([1.0, 2.0], requires_grad=True)
        >>> target = Tensor([1.5, 2.5])
        >>> loss = mse_loss(pred, target)
        >>> loss.backward()
    """
    diff = pred - target
    return (diff * diff).mean()


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Numerically-stable log-softmax along `axis`:

        log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))

    Subtracting the row max before exponentiating prevents overflow for
    large x_i (a bare `exp(x).sum().log()` would blow up); this is the
    standard "log-sum-exp trick." Fully expressible from existing ops —
    max, sub, exp, sum, log — no new primitive needed. The max
    subtraction is mathematically a no-op (softmax is translation
    invariant: softmax(x) == softmax(x - c) for any constant c), and
    since it's differentiated through like everything else here rather
    than detached, autograd correctly produces the same gradient either
    way — verified in tests/test_nn.py via gradcheck.

    Args:
        x: Tensor of shape (..., num_classes).
        axis: Which axis represents the class dimension (default: last).

    Returns:
        Tensor of the same shape as x, where values along `axis` are
        log-probabilities (i.e. exp(out).sum(axis) == 1).
    """
    x_max = x.max(axis=axis, keepdims=True)
    shifted = x - x_max
    log_sum_exp = shifted.exp().sum(axis=axis, keepdims=True).log()
    return shifted - log_sum_exp


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """Softmax along `axis`: exp(log_softmax(x)). See `log_softmax` for
    the numerically-stable implementation this is built on."""
    return log_softmax(x, axis=axis).exp()


def nll_loss(log_probs: Tensor, target) -> Tensor:
    """
    Negative log-likelihood loss, given log-probabilities (typically
    the output of `log_softmax`) and integer class targets.

    loss = -mean(log_probs[i, target[i]] for each row i)

    Args:
        log_probs: Tensor of shape (batch, num_classes).
        target: A plain list/tuple of `batch` ints (class indices),
            NOT a Tensor — matches PyTorch's `nn.NLLLoss` target
            convention (class indices, not one-hot).

    Kept separate from `cross_entropy` (rather than only exposing the
    combined version) because it's occasionally useful on its own, e.g.
    if you've already computed log-probabilities via a different path.
    """
    picked = log_probs.select(target)  # shape (batch,)
    return -picked.mean()


def cross_entropy(logits: Tensor, target) -> Tensor:
    """
    Cross-entropy loss from raw (unnormalized) logits — combines
    `log_softmax` + `nll_loss` in one call, matching PyTorch's
    `nn.CrossEntropyLoss` (which expects logits, not probabilities —
    do NOT pass already-softmaxed input here).

    Args:
        logits: Tensor of shape (batch, num_classes), raw scores.
        target: A plain list/tuple of `batch` ints (class indices).
    """
    return nll_loss(log_softmax(logits, axis=-1), target)
