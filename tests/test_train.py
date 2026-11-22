"""Gradient checks for the Veloce training graph.

Each analytic backward is compared with central finite differences on the NumPy
forward.  A wrong gradient rule silently ruins training, so this is the gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from veloce.train import graph as G  # noqa: E402

TESTS = []
FAILS = []


def test(fn):
    TESTS.append(fn)
    return fn


def numeric_grad(loss_fn, p: G.Param, idx, eps=1e-3) -> float:
    orig = float(p.value[idx])
    p.value[idx] = orig + eps
    hi = float(loss_fn().value)
    p.value[idx] = orig - eps
    lo = float(loss_fn().value)
    p.value[idx] = orig
    return (hi - lo) / (2 * eps)


def check(loss_fn, p: G.Param, samples=8, tol=2e-2) -> float:
    loss = loss_fn()
    G.backward(loss)
    ana = G.grad_of(loss, p)
    rng = np.random.default_rng(0)
    flat = np.arange(p.value.size)
    worst = 0.0
    for i in rng.choice(flat, size=min(samples, flat.size), replace=False):
        idx = np.unravel_index(int(i), p.value.shape)
        num = numeric_grad(loss_fn, p, idx)
        scale = max(1.0, abs(num), float(np.max(np.abs(ana))))
        worst = max(worst, abs(float(ana[idx]) - num) / scale)
    assert worst < tol, f"{p.name}: worst rel err {worst:.3e} (tol {tol:g})"
    return worst


@test
def test_mlp_single_block():
    B, T, D, F, V = 4, 5, 8, 16, 11
    rng = np.random.default_rng(1)
    emb = G.param("emb", rng.standard_normal((V, D)).astype(np.float32) * 0.3)
    w1 = G.param("w1", rng.standard_normal((F, D)).astype(np.float32) * 0.3)
    w2 = G.param("w2", rng.standard_normal((V, F)).astype(np.float32) * 0.3)
    nw = G.param("nw", (1.0 + 0.05 * rng.standard_normal(D)).astype(np.float32))
    ids = rng.integers(0, V, size=(B, T))
    labels = rng.integers(0, V, size=B)

    def loss_fn():
        x = G.embedding(emb, ids)                 # [B,T,D]
        h = G.mean_rows(x)                        # [B,D]
        hn = G.rmsnorm(h, nw)                     # [B,D]
        a = G.silu(G.matmul(hn, G.transpose(w1)))  # [B,F]
        logits = G.matmul(a, G.transpose(w2))     # [B,V]
        return G.cross_entropy(logits, labels)

    worst = 0.0
    for p in (emb, w1, w2, nw):
        worst = max(worst, check(loss_fn, p))
    print(f"  worst {worst:.2e}")
    return worst


@test
def test_softmax_transpose_mul():
    B, D = 3, 6
    rng = np.random.default_rng(2)
    x = G.param("x", rng.standard_normal((B, D)).astype(np.float32))
    w = G.param("w", rng.standard_normal((D, D)).astype(np.float32))
    labels = rng.integers(0, D, size=B)

    def loss_fn():
        h = G.matmul(x, G.transpose(w))
        h = G.mul(h, h)
        return G.cross_entropy(G.softmax(h), labels)

    worst = 0.0
    for p in (x, w):
        worst = max(worst, check(loss_fn, p))
    print(f"  worst {worst:.2e}")
    return worst


@test
def test_rmsnorm_rows():
    B, D = 2, 7
    rng = np.random.default_rng(3)
    x = G.param("x", rng.standard_normal((B, D)).astype(np.float32))
    w = G.param("w", (1 + 0.1 * rng.standard_normal(D)).astype(np.float32))
    tgt = G.param("tgt", rng.standard_normal((B, D)).astype(np.float32))

    def loss_fn():
        diff = G.add(G.rmsnorm(x, w), G.mul(tgt, G.param("neg", np.float32(-1.0))))
        return G.Node("sum_sq", [diff], value=np.asarray(np.sum(diff.value ** 2), dtype=np.float32))

    worst = 0.0
    for p in (x, w):
        worst = max(worst, check(loss_fn, p, tol=5e-2))
    print(f"  worst {worst:.2e}")
    return worst


def main() -> int:
    print(f"running {len(TESTS)} gradient checks\n" + "-" * 60)
    for t in TESTS:
        try:
            w = t()
            print(f"ok    {t.__name__}  (worst rel {w:.2e})")
        except Exception as e:  # noqa: BLE001
            FAILS.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")
    print("-" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        return 1
    print(f"all {len(TESTS)} gradient checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
