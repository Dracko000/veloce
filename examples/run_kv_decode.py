#!/usr/bin/env python3
"""End-to-end KV-cache decoding: compiled Veloce kernels vs. a NumPy reference.

Compiles examples/kv_decode.vel, builds a C host that keeps a caller-owned
cache across five decode steps, and checks every returned vector and the final
attention output against a float64 NumPy implementation.

    python3 examples/run_kv_decode.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import util  # noqa: E402
from veloce import weights as vw  # noqa: E402

CT, D = 8, 16
STEPS = 5


def cf(v):
    """C float literal that stays a float even for integral values."""
    s = f"{float(v):.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def carr(name, ctype, vals):
    body = (", ".join(cf(v) for v in vals) if ctype == "float"
            else ", ".join(str(int(v)) for v in vals))
    return f"{ctype} {name}[{len(vals)}] = {{ {body} }};"


def rmsnorm(x, w):
    ss = float((x.astype(np.float64) ** 2).sum())
    inv = 1.0 / np.sqrt(ss / D + 1e-5)
    return x.astype(np.float64) * inv * w.astype(np.float64)


def main():
    rng = np.random.default_rng(42)
    nw = rng.standard_normal(D).astype(np.float32)
    wk = rng.integers(-1, 2, size=(D, D)).astype(np.int8)
    wv = rng.integers(-1, 2, size=(D, D)).astype(np.int8)
    xs = [rng.standard_normal((1, D)).astype(np.float32) for _ in range(STEPS)]

    src = open(os.path.join(ROOT, "examples", "kv_decode.vel")).read()
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_decode_step(float*,float*,float*,float*,uint8_t*,uint8_t*,int32_t);
float *veloce_attend(float*,float*,float*);
int main(void) {{
    static float ck[{CT*D}] = {{0}};
    static float cv[{CT*D}] = {{0}};
    {carr("nw", "float", nw)}
    {carr("wk", "uint8_t", vw.pack_trits(wk))}
    {carr("wv", "uint8_t", vw.pack_trits(wv))}
""" + "\n".join(
        f'    {carr(f"x{i}", "float", x.flatten())}\n'
        f'    float *v{i} = veloce_decode_step(ck, cv, x{i}, nw, wk, wv, {i});\n'
        f'    for (int j = 0; j < {D}; j++) printf("v{i} %a\\n", v{i}[j]);'
        for i, x in enumerate(xs)) + f"""
    float *o = veloce_attend(ck, cv, x{STEPS-1});
    for (int j = 0; j < {D}; j++) printf("attn %a\\n", o[j]);
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    if rc != 0:
        print(err)
        return 1

    got = {}
    for line in out.splitlines():
        tag, hexval = line.split()
        got.setdefault(tag, []).append(float.fromhex(hexval))

    # ---- reference: same arithmetic in float64 -------------------------
    ck = np.zeros((CT, D))
    cv = np.zeros((CT, D))
    worst = 0.0
    for i, x in enumerate(xs):
        xn = rmsnorm(x[0], nw)
        ck[i] = xn @ wk.astype(np.float64).T
        cv[i] = xn @ wv.astype(np.float64).T
        worst = max(worst, np.abs(np.array(got[f"v{i}"]) - cv[i]).max())

    q = xs[-1][0].astype(np.float64)
    s = q @ ck.T
    s = np.exp(s - s.max())
    ref_attn = (s / s.sum()) @ cv
    attn_err = np.abs(np.array(got["attn"]) - ref_attn).max()

    print(f"KV-cache decode: {STEPS} tokens into a {CT}x{D} caller-owned cache")
    print(f"  per-token value   max |err| = {worst:.3e}")
    print(f"  cached attention  max |err| = {attn_err:.3e}")
    print(f"  cache rows written         = {int(np.count_nonzero(ck)) // D}/{CT}")
    ok = worst < 1e-4 and attn_err < 1e-4
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
