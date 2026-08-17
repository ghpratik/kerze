"""
gradcheck.py

Numerical gradient checking for kerze.

The single most important piece of infrastructure an autograd library
can have: a way to *prove* a backward pass is correct rather than just
assert it. `gradcheck` compares the analytical gradient (produced by
`.backward()`) against a numerical gradient (produced by perturbing
each input element by ±eps and measuring the change in output — central
finite differences) for every requires_grad input.

If they agree within tolerance, the backward implementation is almost
certainly correct — a wrong backward rule essentially never happens to
match a numerical gradient by coincidence. This is the same technique
PyTorch's own `torch.autograd.gradcheck` uses internally.
"""

from __future__ import annotations
from typing import Callable, List, Optional

from kerze.tensor import Tensor
from kerze.ndarray import Array


def _reduce_to_scalar(out: Tensor) -> Tensor:
    """
    gradcheck needs a single scalar to call .backward() on. If `func`
    naturally returns a scalar (shape (1,)), use it as-is; otherwise sum
    it. Summing is a deliberate, safe choice: it's a real differentiable
    op already verified elsewhere, and d(sum(out))/d(x) folds every
    output element's gradient contribution into x in exactly the way we
    want to check.
    """
    return out if out.data.size == 1 else out.sum()


def numerical_gradient(
    func: Callable[..., Tensor],
    inputs: List[Tensor],
    eps: float = 1e-5,
) -> List[Optional[Array]]:
    """
    Compute the numerical gradient of `func(*inputs)` (reduced to a
    scalar via `_reduce_to_scalar`) with respect to every Tensor in
    `inputs` that has `requires_grad=True`, via central finite
    differences: (f(x+eps) - f(x-eps)) / (2*eps), applied independently
    to every element of every such input.

    Args:
        func: Callable taking `*inputs` and returning a Tensor (any
            shape).
        inputs: The Tensors to differentiate with respect to. Tensors
            with `requires_grad=False` are skipped (None in the output).
        eps: Finite-difference step size. 1e-5 is a reasonable default
            for float64-precision Python floats; too small and you hit
            floating-point cancellation error, too large and you hit
            truncation error from the function's curvature.

    Returns:
        A list the same length as `inputs`: an `Array` of the same
        shape as the corresponding input holding its numerical
        gradient, or `None` for inputs that don't require grad.

    Note:
        This mutates each input Tensor's underlying data in place
        during the sweep (perturb, evaluate, restore) — by the time
        this returns, every input's `.data` is back to its original
        value. Safe to call repeatedly.
    """
    results: List[Optional[Array]] = []

    for target in inputs:
        if not target.requires_grad:
            results.append(None)
            continue

        original = list(target.data.data)  # copy — this is what gets restored
        grad_flat = [0.0] * len(original)

        for i in range(len(original)):
            target.data.data[i] = original[i] + eps
            f_plus = _reduce_to_scalar(func(*inputs)).data.data[0]

            target.data.data[i] = original[i] - eps
            f_minus = _reduce_to_scalar(func(*inputs)).data.data[0]

            target.data.data[i] = original[i]  # restore before next element

            grad_flat[i] = (f_plus - f_minus) / (2 * eps)

        results.append(Array(grad_flat, shape=target.shape))

    return results


def gradcheck(
    func: Callable[..., Tensor],
    inputs: List[Tensor],
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    verbose: bool = False,
) -> bool:
    """
    Check that `func`'s analytical (autograd) gradient matches its
    numerical (finite-difference) gradient, for every input Tensor with
    `requires_grad=True`.

    Args:
        func: Callable taking `*inputs` (Tensors) and returning a
            Tensor. Any non-Tensor arguments `func` needs (e.g. integer
            class labels for a loss function) should be captured via a
            closure/lambda — see the classification example below.
        inputs: List of Tensor objects to check gradients for.
        eps: Finite-difference step size, passed to `numerical_gradient`.
        atol, rtol: Absolute/relative tolerance, matching numpy's
            `allclose` convention: a mismatch is flagged when
            `abs(analytical - numerical) > atol + rtol * abs(numerical)`.
        verbose: If True, print every mismatching element. If False,
            only a one-line summary per failing input is printed.

    Returns:
        True if every requires_grad input's gradient matches within
        tolerance across every element, False otherwise.

    Example:
        >>> a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        >>> b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        >>> gradcheck(lambda x, y: (x @ y).sum(), [a, b])
        True

        # Non-Tensor extra args (e.g. classification targets) via closure:
        >>> logits = Tensor([[1.0, 2.0, 0.5]], requires_grad=True)
        >>> target = [1]
        >>> gradcheck(lambda x: F.cross_entropy(x, target), [logits])
        True
    """
    # 1. Analytical gradient: one real forward + backward pass.
    for t in inputs:
        t.grad = None
    out = _reduce_to_scalar(func(*inputs))
    out.backward()
    analytical = [t.grad for t in inputs]

    # 2. Numerical gradient: many forward-only passes.
    numerical = numerical_gradient(func, inputs, eps=eps)

    ok = True
    for idx, (a_grad, n_grad, t) in enumerate(zip(analytical, numerical, inputs)):
        if not t.requires_grad:
            continue
        if a_grad is None:
            print(f"[gradcheck] input {idx}: requires_grad=True but .grad is None "
                  f"after backward() — this input never received gradient.")
            ok = False
            continue

        mismatches = []
        for j, (av, nv) in enumerate(zip(a_grad.data, n_grad.data)):
            if abs(av - nv) > atol + rtol * abs(nv):
                mismatches.append((j, av, nv))

        if mismatches:
            ok = False
            print(f"[gradcheck] input {idx} (shape {t.shape}): "
                  f"{len(mismatches)}/{len(a_grad.data)} elements mismatched")
            if verbose:
                for j, av, nv in mismatches:
                    print(f"    elem {j}: analytical={av:.8f} numerical={nv:.8f} "
                          f"diff={abs(av - nv):.2e}")

    return ok
