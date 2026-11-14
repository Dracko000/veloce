#!/usr/bin/env python3
"""End-to-end test of the Veloce transformer-block example.

Pipeline: compile `transformer_block.vel` -> C -> executable; feed it
deterministic ternary (trit) weights packed exactly as the runtime expects;
run the real compiled kernel; compare against a NumPy reference computed
from the *same* weights.  A near-zero max error proves the whole chain
(front end, static arena plan, trit packing, matmul-free lowering, kernels)
is correct.
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import veloce  # noqa: E402

T, D, F = 4, 16, 32
TRIT = (-1, 0, 1)


def pack_trits(arr):
    """Pack a flat int8 array of {-1,0,1} 5-per-byte, base-3 (matches runtime)."""
    flat = [int(v) for v in arr.reshape(-1)]
    out = bytearray()
    for b in range((len(flat) + 4) // 5):
        w, p = 0, 1
        for k in range(5):
            idx = b * 5 + k
            v = flat[idx] if idx < len(flat) else 0
            w += (v + 1) * p
            p *= 3
        out.append(w)
    return bytes(out)


def c_float_array(name, arr):
    vals = ", ".join(f"{float(v):.8f}f" for v in arr.reshape(-1))
    return f"static float {name}[{arr.size}] = {{ {vals} }};"


def c_byte_array(name, data):
    vals = ", ".join(str(b) for b in data)
    return f"static uint8_t {name}[{len(data)}] = {{ {vals} }};"


def rmsnorm(x, w):
    ss = (x * x).sum(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ss / x.shape[-1] + 1e-5)
    return x * inv * w


def tritmm(x, wtr):
    return x @ wtr.T.astype(np.float64)


def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def reference(x, nw, wq, wk, wv, wo, w1, w2):
    xn = rmsnorm(x, nw)
    q = tritmm(xn, wq)
    k = tritmm(xn, wk)
    v = tritmm(xn, wv)
    scores = q @ k.T
    probs = softmax(scores * 0.25)
    attn = probs @ v
    proj = tritmm(attn, wo)
    x1 = x + proj
    xn2 = rmsnorm(x1, nw)
    h = tritmm(xn2, w1)
    a = silu(h)
    ffn = tritmm(a, w2)
    return x1 + ffn


def main():
    # 1. compile the DSL to C
    res = veloce.compile_file(os.path.join(HERE, "transformer_block.vel"))
    cpath = "/tmp/tb_run.c"
    with open(cpath, "w") as f:
        f.write(res.c_source)
    print(res.report())

    # 2. deterministic weights
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((T, D)).astype(np.float64)
    nw = (rng.standard_normal(D)).astype(np.float64)
    wq = rng.integers(-1, 2, size=(D, D)).astype(np.int8)
    wk = rng.integers(-1, 2, size=(D, D)).astype(np.int8)
    wv = rng.integers(-1, 2, size=(D, D)).astype(np.int8)
    wo = rng.integers(-1, 2, size=(D, D)).astype(np.int8)
    w1 = rng.integers(-1, 2, size=(F, D)).astype(np.int8)
    w2 = rng.integers(-1, 2, size=(D, F)).astype(np.int8)

    # 3. driver
    parts = ["#include \"veloce_runtime.h\"", "#include <stdio.h>",
             "extern float *veloce_transformer_layer(float*,float*,"
             "uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*);"]
    parts.append(c_float_array("X", x))
    parts.append(c_float_array("NW", nw))
    parts.append(c_byte_array("WQ", pack_trits(wq)))
    parts.append(c_byte_array("WK", pack_trits(wk)))
    parts.append(c_byte_array("WV", pack_trits(wv)))
    parts.append(c_byte_array("WO", pack_trits(wo)))
    parts.append(c_byte_array("W1", pack_trits(w1)))
    parts.append(c_byte_array("W2", pack_trits(w2)))
    parts.append("int main(void){")
    parts.append("  float *out = veloce_transformer_layer(X,NW,WQ,WK,WV,WO,W1,W2);")
    parts.append(f"  for (int i=0;i<{T*D};i++) printf(\"%.9g\\n\", out[i]);")
    parts.append("  return 0; }")
    drv = "/tmp/tb_driver.c"
    with open(drv, "w") as f:
        f.write("\n".join(parts))

    # 4. build + run
    exe = "/tmp/tb_run"
    cc = os.environ.get("CC", "cc")
    cmd = [cc, "-std=c11", "-O3", "-march=native", "-ffast-math",
           "-I", os.path.join(ROOT, "runtime"), drv, cpath, "-lm", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("compile failed")
    r = subprocess.run([exe], capture_output=True, text=True)
    got = np.array([float(v) for v in r.stdout.split()]).reshape(T, D)

    # 5. verify
    ref = reference(x, nw, wq, wk, wv, wo, w1, w2)
    err = float(np.max(np.abs(got - ref)))
    print(f"\nmax |veloce - numpy| = {err:.3e}")
    if err < 1e-4:
        print("PASS: compiled Veloce transformer layer matches NumPy reference")
        return 0
    print("FAIL: numeric mismatch")
    print("veloce:\n", got)
    print("numpy:\n", ref)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
