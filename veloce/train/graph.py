"""A training graph that compiles through Veloce.

The idea: models are described once, in Python, as a graph of *tensor ops*. From
that single description the framework produces

  1. a NumPy forward + analytic gradients (fast reference, used by tests), and
  2. Veloce DSL source for `forward` and `backward`, which the existing compiler
     turns into C — so training and inference share one definition.

Scope of this first slice is deliberately small: matmul, add, mul, transpose,
rmsnorm, silu, softmax, mean-pool, embedding and cross-entropy.  No attention
mask, no loop-carried state, no trit weights in the backward (trit is an
*inference* storage format; training keeps a float latent that is ternarised on
export).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

Array = np.ndarray


class TrainError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
@dataclass
class Node:
    op: str
    inputs: List["Node"]
    attrs: dict = field(default_factory=dict)
    value: Array | None = None        # forward result
    grad: Array | None = None         # accumulated dL/dvalue
    name: str = ""                    # DSL variable name, filled by emit()

    @property
    def shape(self) -> tuple:
        return tuple(self.value.shape) if self.value is not None else ()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{self.op} {self.shape}>"


@dataclass
class Param(Node):
    """A leaf that receives gradients (float latent weights)."""

    def __init__(self, name: str, value: Array):
        super().__init__(op="param", inputs=[], attrs={"name": name}, value=value, name=name)


# ---------------------------------------------------------------------------
# ops (numpy forward)
# ---------------------------------------------------------------------------
def param(name: str, value: Array) -> Param:
    return Param(name, np.asarray(value, dtype=np.float32))


def matmul(a: Node, b: Node) -> Node:
    return Node("matmul", [a, b], value=(a.value @ b.value).astype(np.float32))


def add(a: Node, b: Node) -> Node:
    return Node("add", [a, b], value=(a.value + b.value).astype(np.float32))


def mul(a: Node, b: Node) -> Node:
    return Node("mul", [a, b], value=(a.value * b.value).astype(np.float32))


def transpose(a: Node) -> Node:
    return Node("transpose", [a], value=np.swapaxes(a.value, -1, -2).astype(np.float32))


def rmsnorm(x: Node, w: Node, eps: float = 1e-5) -> Node:
    v = x.value
    ms = np.mean(v * v, axis=-1, keepdims=True) + eps
    inv = (1.0 / np.sqrt(ms)).astype(np.float32)
    return Node("rmsnorm", [x, w], {"eps": eps, "inv": inv},
                value=(v * inv * w.value).astype(np.float32))


def silu(x: Node) -> Node:
    v = x.value
    s = 1.0 / (1.0 + np.exp(-v))
    return Node("silu", [x], {"sig": s.astype(np.float32)}, value=(v * s).astype(np.float32))


def softmax(x: Node) -> Node:
    v = x.value
    z = v - v.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    return Node("softmax", [x], {"p": p}, value=p)


def mean_rows(x: Node) -> Node:
    """[B, T, D] -> [B, D] (mean over the sequence axis)."""
    return Node("mean_rows", [x], value=x.value.mean(axis=1).astype(np.float32))


def embedding(table: Node, ids: Array) -> Node:
    ids = np.asarray(ids, dtype=np.int64)
    return Node("embedding", [table], {"ids": ids}, value=table.value[ids].astype(np.float32))


def sum_sq(a: Node) -> Node:
    """Scalar loss: sum of squares (used by the rmsnorm gradient check)."""
    return Node("sum_sq", [a], value=np.asarray(np.sum(a.value ** 2), dtype=np.float32))


def cross_entropy(logits: Node, labels: Array) -> Node:
    labels = np.asarray(labels, dtype=np.int64)
    v = logits.value
    z = v - v.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    b = v.shape[0]
    loss = float(-np.log(p[np.arange(b), labels] + 1e-12).mean())
    return Node("ce", [logits], {"labels": labels, "p": p}, value=np.asarray(loss, dtype=np.float32))


# ---------------------------------------------------------------------------
# reverse-mode differentiation (analytic, per op)
# ---------------------------------------------------------------------------
def backward(loss: Node) -> None:
    """Accumulate dL/dvalue on every node reachable from `loss`."""
    order: List[Node] = []
    seen = set()

    def visit(n: Node) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        for i in n.inputs:
            visit(i)
        order.append(n)

    visit(loss)
    for n in order:
        n.grad = None
    loss.grad = np.ones_like(loss.value, dtype=np.float32)

    for n in reversed(order):
        g = n.grad
        if g is None:
            continue
        if n.op == "param":
            continue
        if n.op == "matmul":
            a, b = n.inputs
            _acc(a, g @ np.swapaxes(b.value, -1, -2))
            _acc(b, np.swapaxes(a.value, -1, -2) @ g)
        elif n.op == "add":
            a, b = n.inputs
            _acc(a, g)
            _acc(b, g)
        elif n.op == "mul":
            a, b = n.inputs
            _acc(a, g * b.value)
            _acc(b, g * a.value)
        elif n.op == "transpose":
            _acc(n.inputs[0], np.swapaxes(g, -1, -2))
        elif n.op == "rmsnorm":
            x, w = n.inputs
            inv = n.attrs["inv"]
            gx = g * inv * w.value
            d_inv = np.sum(g * x.value * w.value, axis=-1, keepdims=True)
            gx = gx - (d_inv * inv ** 3 / x.value.shape[-1]) * x.value
            _acc(x, gx)
            _acc(w, np.sum(g * x.value * inv, axis=tuple(range(g.ndim - 1))))
        elif n.op == "silu":
            x = n.inputs[0]
            sig = n.attrs["sig"]
            _acc(x, g * (sig * (1.0 + x.value * (1.0 - sig))))
        elif n.op == "softmax":
            p = n.attrs["p"]
            _acc(n.inputs[0], p * (g - np.sum(g * p, axis=-1, keepdims=True)))
        elif n.op == "mean_rows":
            x = n.inputs[0]
            t = x.value.shape[1]
            _acc(x, np.broadcast_to((g / t)[:, None, :], x.value.shape).copy())
        elif n.op == "embedding":
            table = n.inputs[0]
            ids = n.attrs["ids"]
            gt = np.zeros_like(table.value)
            np.add.at(gt, ids, g)
            _acc(table, gt)
        elif n.op == "sum_sq":
            x = n.inputs[0]
            _acc(x, (2.0 * x.value * float(np.asarray(g))).astype(np.float32))
        elif n.op == "ce":
            p = n.attrs["p"]
            labels = n.attrs["labels"]
            b = p.shape[0]
            d = p.copy()
            d[np.arange(b), labels] -= 1.0
            _acc(n.inputs[0], (d / b) * float(g))
        else:  # pragma: no cover
            raise TrainError(f"no gradient rule for op {n.op!r}")


def _acc(n: Node, g: Array) -> None:
    g = np.asarray(g, dtype=np.float32)
    n.grad = g.copy() if n.grad is None else (n.grad + g).astype(np.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def topo(loss: Node) -> List[Node]:
    order, seen = [], set()

    def visit(n: Node) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        for i in n.inputs:
            visit(i)
        order.append(n)

    visit(loss)
    return order


def params_of(loss: Node) -> List[Param]:
    return [n for n in topo(loss) if isinstance(n, Param)]


def grad_of(loss: Node, p: Node) -> Array:
    if p.grad is None:
        raise TrainError(f"no gradient for {p.name!r} — is it on the path to the loss?")
    return p.grad


def finite_difference(fn, arr: Array, idx: Sequence[int], eps: float = 1e-3) -> float:
    """d f / d arr[idx] by central difference — the test oracle."""
    base = arr.copy()
    plus, minus = arr.copy(), arr.copy()
    plus[idx] += eps
    minus[idx] -= eps
    hi, lo = fn(plus), fn(minus)
    base[...] = arr
    return float((hi - lo) / (2 * eps))
