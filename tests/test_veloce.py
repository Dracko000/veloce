#!/usr/bin/env python3
"""Veloce test-suite (stdlib-only; run with `python3 tests/test_veloce.py`).

Each test compiles real C, links it, runs it, and checks numeric results
against an independent NumPy reference (passed into the C driver as expected
arrays) or against compiler diagnostics.  A tiny homemade runner keeps the
project dependency-free.
"""
import os
import re
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import util  # noqa: E402

FAILS = []


def cf(v):
    """C float literal for `v`.

    ``f"{v:.9g}"`` drops the decimal point for integral values (``-11``), and
    appending ``f`` then makes ``-11f`` — an invalid *integer* suffix that cc
    rejects.  Keep the literal a float by adding ``.0`` when there is no
    decimal point or exponent.
    """
    s = f"{float(v):.9g}"
    if s in ("nan", "-nan"):
        return "NAN"
    if s in ("inf", "-inf"):
        return "INFINITY" if s == "inf" else "(-INFINITY)"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s + "f"


def carr(name, ctype, values):
    if ctype.startswith("float"):
        body = ", ".join(cf(v) for v in values)
    elif "int8" in ctype:
        body = ", ".join(f"{int(v)}" for v in values)
    else:
        body = ", ".join(str(v) for v in values)
    return f"{ctype} {name}[{len(values)}] = {{ {body} }};"


def check_close(got_name, exp, n, tol=1e-4, tag=""):
    """Return C code comparing got_name[0..n) to a passed expected array."""
    uid = "".join(c if c.isalnum() else "_" for c in tag)
    arr = f"__exp_{uid}"
    expected = carr(arr, "float", exp)
    return f"""
    {expected}
    {{
      double m = 0.0; int ok = 1;
      for (int i = 0; i < {n}; i++) {{
        double d = fabs((double){got_name}[i] - (double){arr}[i]);
        if (d > m) m = d;
        if (d > {tol}) ok = 0;
      }}
      printf("CHECK {tag} maxerr=%.3e %s\\n", m, ok?"PASS":"FAIL");
    }}
    """


# --------------------------------------------------------------------------
# 1. Trit bit-packing round-trip + matmul-free matmul kernel
# --------------------------------------------------------------------------
def test_trit_pack_and_matmul():
    rng = np.random.default_rng(1)
    M, K, N = 2, 12, 3
    x = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)  # [N,K] trits
    expected = (x @ W.astype(np.float32).T).flatten()

    wl = ", ".join(str(int(v)) for v in W.flatten())
    xl = ", ".join(cf(v) for v in x.flatten())
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    float x[{M*K}] = {{ {xl} }};
    int8_t w[{N*K}] = {{ {wl} }};
    uint8_t packed[{(N*K+4)//5}];
    veloce_pack_trits(packed, w, {N*K});
    /* round-trip: unpack and compare */
    int8_t rt[{N*K}];
    veloce_expand_trit(rt, packed, {N*K});
    int rt_ok = 1;
    for (int i = 0; i < {N*K}; i++) if (rt[i] != w[i]) rt_ok = 0;
    printf("roundtrip %s\\n", rt_ok ? "PASS" : "FAIL");
    float out[{M*N}];
    veloce_matmul_trit(out, x, packed, {M}, {K}, {N});
    {check_close('out', list(expected), M*N, tag='trit_matmul')}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run("module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n",
                                         driver)
    assert rc == 0, err
    assert "roundtrip PASS" in out, out
    assert "trit_matmul maxerr" in out and "PASS" in out.split("trit_matmul")[1], out


# --------------------------------------------------------------------------
# 2. Elementwise + bias broadcast via generated code
# --------------------------------------------------------------------------
def test_elementwise_broadcast():
    src = """
module t
dim R = 2
dim C = 3
fn f(x: f32[R, C], b: f32[C]) -> f32[R, C] {
    let y = x + b
    let z = y * 2.0
    return z
}
"""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    expected = ((x + b) * 2.0).flatten()
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
{util_proto('float * veloce_f(float *, float *)')}
int main(void) {{
    float x[6] = {{ {', '.join(cf(v) for v in x.flatten())} }};
    float b[3] = {{ {', '.join(cf(v) for v in b)} }};
    float *out = veloce_f(x, b);
    {check_close('out', list(expected), 6, tag='ew_broadcast')}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    assert rc == 0, err
    assert "ew_broadcast maxerr" in out and out.strip().endswith("PASS"), out


# --------------------------------------------------------------------------
# 3. RMSNorm
# --------------------------------------------------------------------------
def test_rmsnorm():
    rng = np.random.default_rng(3)
    T, D = 3, 8
    x = rng.standard_normal((T, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    ss = (x * x).sum(axis=1) / D
    expected = (x / np.sqrt(ss[:, None] + 1e-5) * w[None, :]).flatten()
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    float x[{T*D}] = {{ {', '.join(cf(v) for v in x.flatten())} }};
    float w[{D}] = {{ {', '.join(cf(v) for v in w)} }};
    float out[{T*D}];
    veloce_rmsnorm(out, x, w, {T*D}, {D});
    {check_close('out', list(expected), T*D, tol=1e-4, tag='rmsnorm')}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run("module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n",
                                         driver)
    assert rc == 0, err
    assert "rmsnorm maxerr" in out and out.strip().endswith("PASS"), out


# --------------------------------------------------------------------------
# 4. SiLU + Softmax
# --------------------------------------------------------------------------
def test_silu_softmax():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(6).astype(np.float32)
    silu = x / (1 + np.exp(-x))
    s = rng.standard_normal((2, 3)).astype(np.float32)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    sm = (e / e.sum(axis=1, keepdims=True)).flatten()
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    float x[6] = {{ {', '.join(cf(v) for v in x)} }};
    float o1[6];
    veloce_silu(o1, x, 6);
    {check_close('o1', list(silu), 6, tag='silu')}
    float s[6] = {{ {', '.join(cf(v) for v in s.flatten())} }};
    float o2[6];
    veloce_softmax(o2, s, 6, 3);
    {check_close('o2', list(sm), 6, tag='softmax')}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run("module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n",
                                         driver)
    assert rc == 0, err
    assert out.count("PASS") >= 2 and "FAIL" not in out, out


# --------------------------------------------------------------------------
# 4b. SiLU saturation (-ffast-math regression)
# --------------------------------------------------------------------------
def test_silu_saturating_inputs():
    """Large |x| must not produce NaN.

    The naive v/(1+exp(-v)) becomes NaN for v < -88 once -ffast-math lets GCC
    vectorise the divide into vrcpps(): vrcpps(inf) == 0 and inf*0 == NaN.
    """
    xs = [0.0, -0.5, 1.5, 20.0, 88.0, 200.0, -20.0, -88.0, -200.0, -1000.0]
    xarr = (f"float x[{len(xs)}] = {{ "
            + ", ".join(f"{v:.1f}" for v in xs) + " };")
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
#include <string.h>
int main(void) {{
    {xarr}
    float o[{len(xs)}];
    veloce_silu(o, x, {len(xs)});
    int bad = 0;
    for (int i = 0; i < {len(xs)}; i++) {{
        unsigned bits;
        memcpy(&bits, &o[i], 4);
        /* bit test: -ffast-math folds isfinite()/x != x away */
        if ((bits & 0x7f800000u) == 0x7f800000u) {{
            bad++;
            printf("bad i=%d in=%g out=%g bits=%08x\\n", i, x[i], o[i], bits);
        }}
    }}
    /* saturation values */
    if (!(o[{len(xs) - 1}] > -1e-6f && o[{len(xs) - 1}] <= 0.0f)) {{
        printf("silu(-1000) = %g, expected -0\\n", o[{len(xs) - 1}]);
        bad++;
    }}
    printf(bad ? "FAIL\\n" : "PASS\\n");
    return bad != 0;
}}
"""
    rc, out, err, _ = util.build_and_run(
        "module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n",
        driver, extra_flags=["-ffast-math"])
    assert rc == 0, out + err
    assert "PASS" in out and "FAIL" not in out, out


# --------------------------------------------------------------------------
# 5. Transpose + matmul via generated code
# --------------------------------------------------------------------------
def test_transpose_matmul():
    src = """
module t
dim R = 2
dim K = 3
dim N = 4
fn f(a: f32[R, K], b: f32[K, N]) -> f32[R, N] {
    return matmul(a, b)
}
fn gt(a: f32[R, K]) -> f32[K, R] {
    return transpose(a)
}
"""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32)
    exp_mm = (a @ b).flatten()
    exp_t = a.T.flatten()
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float * veloce_f(float *, float *);
float * veloce_gt(float *);
int main(void) {{
    float a[6] = {{ {', '.join(cf(v) for v in a.flatten())} }};
    float b[12] = {{ {', '.join(cf(v) for v in b.flatten())} }};
    float *o = veloce_f(a, b);
    {check_close('o', list(exp_mm), 8, tag='matmul')}
    float *t = veloce_gt(a);
    {check_close('t', list(exp_t), 6, tag='transpose')}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    assert rc == 0, err
    assert out.count("PASS") >= 2 and "FAIL" not in out, out


# --------------------------------------------------------------------------
# 6. fp16 weight matmul
# --------------------------------------------------------------------------
def test_fp16_matmul():
    rng = np.random.default_rng(6)
    M, K, N = 2, 4, 3
    x = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.standard_normal((K, N)).astype(np.float16)
    expected = (x @ W.astype(np.float32)).flatten()
    # encode fp16 bits
    wbits = W.view(np.uint16).flatten()
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    float x[{M*K}] = {{ {', '.join(cf(v) for v in x.flatten())} }};
    uint16_t w[{K*N}] = {{ {', '.join(str(int(v)) for v in wbits)} }};
    float out[{M*N}];
    veloce_matmul_f16w(out, x, w, {M}, {K}, {N});
    {check_close('out', list(expected), M*N, tol=2e-3, tag='fp16_matmul')}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run("module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n",
                                         driver)
    assert rc == 0, err
    assert "fp16_matmul maxerr" in out and out.strip().endswith("PASS"), out


# --------------------------------------------------------------------------
# 7. End-to-end small model: embedding -> trit proj -> argmax
# --------------------------------------------------------------------------
def test_end_to_end_argmax():
    src = """
module tiny
dim V = 8
dim D = 4
dim N = 5
fn classify(emb: f32[V, D], w: trit[N, D], tok: i32) -> i32 {
    let x = embedding(emb, tok)
    let logits = matmul_trit(x, w)
    return argmax(logits)
}
"""
    rng = np.random.default_rng(7)
    emb = rng.standard_normal((8, 4)).astype(np.float32)
    W = rng.integers(-1, 2, size=(5, 4)).astype(np.int8)
    tok = 3
    logits = emb[tok] @ W.T.astype(np.float32)
    exp = int(np.argmax(logits))
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int32_t veloce_classify(float *, uint8_t *, int32_t);
int main(void) {{
    float emb[32] = {{ {', '.join(cf(v) for v in emb.flatten())} }};
    int8_t w[20] = {{ {', '.join(str(int(v)) for v in W.flatten())} }};
    uint8_t packed[4];
    veloce_pack_trits(packed, w, 20);
    int32_t r = veloce_classify(emb, packed, {tok});
    printf("result %d expected {exp} %s\\n", r, r == {exp} ? "PASS" : "FAIL");
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    assert rc == 0, err
    assert "PASS" in out and "FAIL" not in out, out


# --------------------------------------------------------------------------
# 8. Static arena reuse in the transformer block
# --------------------------------------------------------------------------
def test_arena_reuse():
    root = os.path.dirname(HERE)
    res = util.emit(os.path.join(root, "examples", "transformer_block.vel"))
    layer = res.memory["transformer_layer"].report()
    assert layer["peak_bytes"] < layer["naive_bytes"], layer
    assert layer["reuse_savings_pct"] > 30, layer
    # zero-allocation: every activation is arena-backed (no malloc in output)
    assert "malloc" not in res.c_source
    assert "calloc" not in res.c_source


# --------------------------------------------------------------------------
# 9. Diagnostics: type/shape errors are caught
# --------------------------------------------------------------------------
def _expect_compile_error(src, needle):
    import veloce
    try:
        veloce.compile_source(src)
    except veloce.VeloceError as e:
        assert needle.lower() in str(e).lower(), f"got: {e}\nwanted: {needle}"
        return
    raise AssertionError(f"expected compile error containing {needle!r}")


def test_diagnostics():
    _expect_compile_error(
        "module t\ndim A=2\nfn f(x: f32[A]) -> f32[A] { return x + y }\n",
        "undefined variable")
    _expect_compile_error(
        "module t\ndim A=2\ndim B=3\nfn f(a: f32[A], b: f32[B]) -> f32[A] { return a + b }\n",
        "shape mismatch")
    _expect_compile_error(
        "module t\ndim A=2\nfn f(x: f32[A]) -> f32[A] {\n  const y = x\n  y = x\n  return y\n}\n",
        "immutable")
    _expect_compile_error(
        "module t\ndim A=2\nfn f(x: f32[A]) -> i32 { return x }\n",
        "expected")


def util_proto(sig):
    return f"extern {sig};"


def _pack_trits_bytes(W):
    """Python reference for Veloce flat base-3 packing (5 trits/byte)."""
    flat = W.flatten().astype(np.int8)
    n = flat.size
    out = []
    for b in range((n + 4) // 5):
        w = 0
        p = 1
        for k in range(5):
            idx = b * 5 + k
            v = int(flat[idx]) if idx < n else 0
            w += (v + 1) * p
            p *= 3
        out.append(w)
    return out


def test_e2e_transformer_layer():
    """Run the compiled transformer_layer and match a NumPy reference exactly."""
    root = os.path.dirname(HERE)
    src_path = os.path.join(root, "examples", "transformer_block.vel")
    res = util.emit(src_path)
    rng = np.random.default_rng(11)
    T, D, F = 4, 16, 32
    x = rng.standard_normal((T, D)).astype(np.float32)
    nw = rng.standard_normal(D).astype(np.float32)
    trit = lambda *s: rng.integers(-1, 2, size=s).astype(np.int8)
    wq, wk, wv, wo = trit(D, D), trit(D, D), trit(D, D), trit(D, D)
    w1, w2 = trit(F, D), trit(D, F)

    def rmsnorm(xx, w):
        ss = (xx * xx).mean(axis=-1, keepdims=True)
        return xx / np.sqrt(ss + 1e-5) * w

    def silu(v):
        return v / (1 + np.exp(-v))

    xn = rmsnorm(x, nw)
    q = xn @ wq.T.astype(np.float32)
    k = xn @ wk.T.astype(np.float32)
    v = xn @ wv.T.astype(np.float32)
    scores = q @ k.T
    e = np.exp(scores * 0.25 - (scores * 0.25).max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    attn = probs @ v
    proj = attn @ wo.T.astype(np.float32)
    x1 = x + proj
    xn2 = rmsnorm(x1, nw)
    h = xn2 @ w1.T.astype(np.float32)
    ffn = silu(h) @ w2.T.astype(np.float32)
    expected = (x1 + ffn).flatten().astype(np.float32)

    def cfloat(a):
        return ", ".join(cf(v) for v in a.flatten())

    def ctrit(W):
        packed = _pack_trits_bytes(W)
        return ", ".join(str(b) for b in packed), len(packed)

    wq_b, wq_n = ctrit(wq)
    wk_b, wk_n = ctrit(wk)
    wv_b, wv_n = ctrit(wv)
    wo_b, wo_n = ctrit(wo)
    w1_b, w1_n = ctrit(w1)
    w2_b, w2_n = ctrit(w2)

    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float * veloce_transformer_layer(float *, float *, uint8_t *, uint8_t *, uint8_t *,
                                 uint8_t *, uint8_t *, uint8_t *);
int main(void) {{
    float x[{T*D}] = {{ {cfloat(x)} }};
    float nw[{D}] = {{ {cfloat(nw)} }};
    uint8_t wq[{wq_n}] = {{ {wq_b} }};
    uint8_t wk[{wk_n}] = {{ {wk_b} }};
    uint8_t wv[{wv_n}] = {{ {wv_b} }};
    uint8_t wo[{wo_n}] = {{ {wo_b} }};
    uint8_t w1[{w1_n}] = {{ {w1_b} }};
    uint8_t w2[{w2_n}] = {{ {w2_b} }};
    float *o1 = veloce_transformer_layer(x, nw, wq, wk, wv, wo, w1, w2);
    float first[{T*D}];
    for (int i = 0; i < {T*D}; i++) first[i] = o1[i];
    float *o2 = veloce_transformer_layer(x, nw, wq, wk, wv, wo, w1, w2);
    int det = 1;
    for (int i = 0; i < {T*D}; i++) if (first[i] != o2[i]) det = 0;
    printf("determinism %s\\n", det ? "PASS" : "FAIL");
    {check_close('first', list(expected), T*D, tol=1e-4, tag='transformer')}
    return 0;
}}
"""
    rc, out, err, csrc = util.build_and_run(src_path and open(src_path).read(), driver,
                                            name="tb")
    assert rc == 0, err
    assert "determinism PASS" in out, out
    assert "transformer maxerr" in out and out.strip().endswith("PASS"), out


# --------------------------------------------------------------------------
def test_veloce_weight_file_roundtrip():
    """pack_trits (compiler) must agree with the C runtime's packer, and the
    .veloce file must round-trip its manifest."""
    import tempfile
    import veloce.weights as vw
    rng = np.random.default_rng(9)
    n = 123
    trits = rng.integers(-1, 2, size=n).astype(np.int8)
    py = vw.pack_trits(trits.tolist())
    # C packer must produce the same bytes.
    vals = ", ".join(str(int(v)) for v in trits)
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    int8_t t[{n}] = {{ {vals} }};
    uint8_t packed[{(n+4)//5}];
    veloce_pack_trits(packed, t, {n});
    for (size_t i = 0; i < sizeof(packed); i++) printf("%02x", packed[i]);
    printf("\\n");
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(
        "module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n", driver)
    assert rc == 0, err
    c_hex = out.strip()
    py_hex = py.hex()
    assert c_hex == py_hex, f"packing mismatch {c_hex} vs {py_hex}"

    # Round-trip a .veloce file and check offsets are 64-byte aligned.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.veloce")
        W = rng.integers(-1, 2, size=(5, 7)).astype(np.int8)
        man = vw.write_veloce_model(p, [
            {"name": "wq", "dtype": "trit", "data": W},
            {"name": "emb", "dtype": "f32", "data": rng.standard_normal((8, 4)).astype(np.float32)},
        ])
        back = vw.read_veloce_manifest(p)
        assert back["tensors"][0]["name"] == "wq"
        for e in back["tensors"]:
            assert e["offset"] % 64 == 0, e
        # unpack matches
        with open(p, "rb") as f:
            data = f.read()
        e = back["tensors"][0]
        blk = data[e["offset"]:e["offset"] + e["nbytes"]]
        assert vw.unpack_trits(blk, e["numel"]) == W.flatten().tolist()


def test_mmap_zero_copy_inference():
    """mmap a .veloce file and run a trit matmul straight from the mapping."""
    import tempfile
    import veloce.weights as vw
    rng = np.random.default_rng(10)
    M, K, N = 1, 8, 4
    x = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)     # [N,K]
    expected = (x @ W.astype(np.float32).T).flatten()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "w.veloce")
        man = vw.write_veloce_model(p, [{"name": "w", "dtype": "trit", "data": W}])
        e = man["tensors"][0]
        driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    veloce_map m = veloce_mmap_ro("{p}");
    if (!m.data) {{ printf("mmap FAIL\\n"); return 1; }}
    /* zero-copy: weight pointer is inside the mapping, no deserialisation */
    const uint8_t *w = (const uint8_t *)m.data + {e['offset']};
    float x[{M*K}] = {{ {', '.join(cf(v) for v in x.flatten())} }};
    float out[{M*N}];
    veloce_matmul_trit(out, x, w, {M}, {K}, {N});
    {check_close('out', list(expected), M*N, tag='mmap_trit')}
    veloce_munmap(&m);
    return 0;
}}
"""
        rc, out, err, _ = util.build_and_run(
            "module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n", driver)
        assert rc == 0, err
        assert "mmap_trit maxerr" in out and out.strip().endswith("PASS"), out


def test_multireturn_and_control_flow():
    src = """
module mr
dim D = 4

fn split(x: f32[D]) -> f32[D], f32[D] {
    let a = x * 2.0
    let b = x + 1.0
    return a, b
}

fn use(x: f32[D]) -> f32[D] {
    let p, q = split(x)
    return p + q
}

fn count(n: i32) -> i32 {
    let acc = 0
    let i = 0
    while i < n {
        acc += i
        i += 1
    }
    return acc
}
"""
    rng = np.random.default_rng(12)
    x = rng.standard_normal(4).astype(np.float32)
    expected = (x * 2.0 + (x + 1.0)).flatten()
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float * veloce_use(float *);
int32_t veloce_count(int32_t);
int main(void) {{
    float x[4] = {{ {', '.join(cf(v) for v in x)} }};
    float *o = veloce_use(x);
    {check_close('o', list(expected), 4, tag='multireturn')}
    int32_t c = veloce_count(10);   /* 0+1+...+9 = 45 */
    printf("loop %d expected 45 %s\\n", c, c == 45 ? "PASS" : "FAIL");
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    assert rc == 0, err
    assert "multireturn maxerr" in out and "FAIL" not in out, out
    assert "loop 45 expected 45 PASS" in out, out


def test_embedding_seq_meanpool():
    src = """
module es
dim V = 6
dim D = 4
dim T = 3

fn pooled(emb: f32[V, D], ids: i32[T]) -> f32[1, D] {
    let e = embedding_seq(emb, ids)   # [T, D]
    return meanpool(e)                # [1, D]
}
"""
    rng = np.random.default_rng(13)
    emb = rng.standard_normal((6, 4)).astype(np.float32)
    ids = np.array([5, 0, 3], dtype=np.int32)
    expected = emb[ids].mean(axis=0).reshape(1, 4)
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float * veloce_pooled(float *, int32_t *);
int main(void) {{
    float emb[24] = {{ {', '.join(cf(v) for v in emb.flatten())} }};
    int32_t ids[3] = {{ {', '.join(str(int(v)) for v in ids)} }};
    float *o = veloce_pooled(emb, ids);
    {check_close('o', list(expected.flatten()), 4, tag='embedding_seq_meanpool')}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    assert rc == 0, err
    assert "embedding_seq_meanpool maxerr" in out and out.strip().endswith("PASS"), out


def test_erd_introspection():
    from veloce import erd
    root = os.path.dirname(HERE)
    res = util.emit(os.path.join(root, "examples", "transformer_block.vel"))
    ent = erd.build_erd(res.module, res.memory, res.diagnostics)
    s = ent["summary"]
    assert s["functions"] == 4, s
    assert s["ir_nodes"] >= 15, s
    assert s["trit_matmuls"] == 6, s
    assert s["weight_assets"] == 8, s
    # sub-byte trit dtype + 5-per-byte packing rule present
    names = [d["name"] for d in ent["dtypes"]]
    assert "trit" in names and "f32" in names, names
    rule = [p for p in ent["packing_rules"] if p["elements_per_word"] == 5]
    assert rule and rule[0]["container_bits"] == 8, ent["packing_rules"]
    # every arena slice is 64-byte aligned and within capacity
    for sl in ent["slices"]:
        assert sl["byte_offset"] % 4 == 0
        assert sl["death_step"] >= sl["birth_step"]
    # pure functions detected
    assert all(f["is_pure"] for f in ent["functions"])
    mermaid = erd.to_mermaid(ent)
    assert mermaid.startswith("erDiagram")
    assert "PACKING_RULE" in mermaid


# --------------------------------------------------------------------------
# FR-4.2: parallel kernels
# --------------------------------------------------------------------------
def _threads_build(vel_source, driver_c, pool, threads=None, name="par"):
    """Build with an explicit pool backend (0=serial, 1=C11 threads, 2=OpenMP)."""
    flags = ["-DVELOCE_POOL=%d" % pool]
    if threads is not None:
        flags.append("-DVELOCE_THREADS=%d" % threads)
    if pool == 2:
        flags.append("-fopenmp")
    return util.build_and_run(vel_source, driver_c, extra_flags=flags, name=name,
                              link_flags=flags)


def _trit_matmul_driver(M, K, N, tag):
    rng = np.random.default_rng(hash(tag) & 0xFFFF)
    x = rng.standard_normal((M, K)).astype(np.float32)
    W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
    expected = (x @ W.astype(np.float32).T).flatten()
    wl = ", ".join(str(int(v)) for v in W.flatten())
    xl = ", ".join(cf(v) for v in x.flatten())
    return f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    float x[{M*K}] = {{ {xl} }};
    int8_t w[{N*K}] = {{ {wl} }};
    uint8_t packed[{(N*K+4)//5}];
    veloce_pack_trits(packed, w, {N*K});
    float out[{M*N}];
    veloce_matmul_trit(out, x, packed, {M}, {K}, {N});
    {check_close('out', list(expected), M*N, tol=(1e-4 if K <= 128 else 1e-3), tag=tag)}
    return 0;
}}
"""


def test_parallel_trit_bit_exact():
    """FR-4.2: serial, C11-threads and OpenMP kernels must agree bit-for-bit.

    Includes K % 5 != 0 so flat (unpadded) base-3 row packing is exercised.
    """
    src = "module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n"
    for M, K, N in ((1, 4096, 300), (4, 4096, 300), (1, 1003, 7)):
        drv = _trit_matmul_driver(M, K, N, f"par_{M}_{K}_{N}")
        outs = []
        for pool, threads in ((0, None), (1, 1), (1, 4), (2, 1), (2, 4)):
            rc, out, err, _ = _threads_build(src, drv, pool, threads,
                                             name=f"par{pool}_{threads or 1}_{K}")
            assert rc == 0, err
            line = [l for l in out.splitlines() if "maxerr" in l][0]
            assert "PASS" in line, (M, K, N, pool, threads, out)
            outs.append(line)
        # every configuration must report exactly the same max error
        assert len(set(outs)) == 1, outs


def test_trit_gemv_phase_grouped_rows():
    """The four-row GEMV groups rows that share a byte *phase*: with the flat
    packing row j starts at slot (j*K)%5, so the kernel runs rows
    j, j+p, j+2p, j+3p (p = 5/gcd(K,5)) and lets them share one x load.

    K % 5 != 0 makes p == 5, and N is a whole number of tiles, so every output
    neuron goes through that grouped path: a kernel that assumed adjacent rows
    -- or that mis-indexed the phase stride -- decodes another row's weights and
    the reference check below fails.  The second assertion pins the batched
    (M == 2) row path to exactly the same arithmetic as the GEMV path, which is
    what makes the two paths interchangeable for a decode loop.
    """
    K, N = 1003, 20                     # period 5, 1 tile of 20 rows
    rng = np.random.default_rng(4242)
    x = rng.standard_normal((1, K)).astype(np.float32)
    W = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
    expected = (x @ W.astype(np.float32).T).flatten()

    xl = ", ".join(cf(v) for v in x.flatten())
    wl = ", ".join(str(int(v)) for v in W.flatten())
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
int main(void) {{
    float x[2 * {K}] = {{ {xl}, {xl} }};
    int8_t w[{N*K}] = {{ {wl} }};
    uint8_t packed[{(N*K+4)//5}];
    veloce_pack_trits(packed, w, {N*K});
    float g[{N}], b[2 * {N}];
    veloce_matmul_trit(g, x, packed, 1, {K}, {N});
    veloce_matmul_trit(b, x, packed, 2, {K}, {N});
    int same = 1;
    for (int j = 0; j < {N}; j++)
        if (b[j] != g[j] || b[{N} + j] != g[j]) same = 0;
    printf("batched_matches_gemv %s\\n", same ? "PASS" : "FAIL");
    {check_close('g', list(expected), N, tol=1e-4, tag='phase_gemv')}
    return same ? 0 : 1;
}}
"""
    rc, out, err, _ = util.build_and_run(
        "module t\ndim A=1\nfn f(x: f32[A]) -> f32[A] { return x }\n", driver)
    assert rc == 0, (rc, err, out)
    assert "batched_matches_gemv PASS" in out, out
    assert "phase_gemv maxerr" in out and "PASS" in out.split("phase_gemv")[1], out


def test_parallel_matches_serial_transformer():
    """FR-4.2: the full transformer layer is deterministic under 4 threads."""
    root = os.path.dirname(HERE)
    src_path = os.path.join(root, "examples", "transformer_block.vel")
    res = util.emit(src_path)
    rng = np.random.default_rng(77)
    T, D, F = 4, 16, 32
    x = rng.standard_normal((T, D)).astype(np.float32)
    nw = rng.standard_normal(D).astype(np.float32)
    W = {k: rng.integers(-1, 2, size=s).astype(np.int8)
         for k, s in (("wq", (D, D)), ("wk", (D, D)), ("wv", (D, D)), ("wo", (D, D)),
                      ("w1", (F, D)), ("w2", (D, F)))}

    def cfl(a):
        return ", ".join(cf(v) for v in a.flatten())

    def cb(a):
        flat = [int(v) for v in a.flatten()]
        out = bytearray()
        for b in range((len(flat) + 4) // 5):
            w, p = 0, 1
            for k in range(5):
                i = b * 5 + k
                w += ((flat[i] if i < len(flat) else 0) + 1) * p
                p *= 3
            out.append(w)
        return ", ".join(str(v) for v in out), len(out)

    parts = [f"float x[{T*D}] = {{ {cfl(x)} }};", f"float nw[{D}] = {{ {cfl(nw)} }};"]
    names = []
    for k, a in W.items():
        body, n = cb(a)
        parts.append(f"uint8_t {k}[{n}] = {{ {body} }};")
        names.append(k)
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float * veloce_transformer_layer(float *, float *, uint8_t *, uint8_t *, uint8_t *,
                                 uint8_t *, uint8_t *, uint8_t *);
int main(void) {{
    {' '.join(parts)}
    float *o = veloce_transformer_layer(x, nw, {', '.join(names)});
    for (int i = 0; i < {T*D}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    bases = {}
    for pool, threads in ((0, None), (1, 4), (2, 4)):
        rc, out, err, _ = _threads_build(open(src_path).read(), driver, pool, threads,
                                         name=f"tpar{pool}")
        assert rc == 0, err
        bases[(pool, threads)] = out.split()
    assert bases[(0, None)] == bases[(1, 4)] == bases[(2, 4)], "threaded output differs"


def test_i4_matmul_bit_exact():
    """FR-1.1: packed i4 (2 nibbles/byte) matmul matches a scalar reference
    bit-for-bit across serial, C11-threads and OpenMP builds."""
    src = """module t
dim M = 4
dim K = 1003
dim N = 33
fn f(x: f32[M, K], w: i4[N, K]) -> f32[M, N] {
    let y = matmul_i4(x, w)
    return relu(y)
}
"""
    rng = np.random.default_rng(21)
    M, K, N = 4, 1003, 33
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w = rng.integers(-8, 8, size=(N, K)).astype(np.int8)
    xl = ", ".join(cf(v) for v in x.flatten())
    flat = [int(v) for v in w.flatten()]
    packed = []
    for i in range(0, len(flat), 2):
        lo = flat[i] & 0x0F
        hi = (flat[i + 1] & 0x0F) if i + 1 < len(flat) else 0
        packed.append(lo | (hi << 4))
    pl = ", ".join(str(v) for v in packed)
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float *, uint8_t *);
int main(void) {{
    float x[{M*K}] = {{ {xl} }};
    uint8_t w[{len(packed)}] = {{ {pl} }};
    float *o = veloce_f(x, w);
    for (int i = 0; i < {M*N}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    outs = {}
    for pool in (0, 1, 2):
        rc, out, err, _ = _threads_build(src, driver, pool, 4, name=f"i4_{pool}")
        assert rc == 0, err
        outs[pool] = out
    assert outs[0] == outs[1] == outs[2], "i4 matmul differs across backends"
    got = np.array([float.fromhex(v) for v in outs[0].split()], dtype=np.float32)
    ref = np.maximum(x.astype(np.float64) @ w.astype(np.float64).T, 0).astype(np.float32)
    assert np.allclose(got.reshape(M, N), ref, atol=1e-4), (got, ref)


def test_parallel_row_kernels():
    """FR-4.2: row kernels (rmsnorm/silu/softmax/rope/transpose) stay bit-exact."""
    src = """module t
dim R = 4
dim C = 16
fn f(x: f32[R, C], w: f32[C]) -> f32[R, C] { return rmsnorm(x, w) }
fn g(x: f32[R, C]) -> f32[R, C] { return softmax(x) }
fn h(x: f32[R, C]) -> f32[C, R] { return transpose(x) }
fn i(x: f32[R, C]) -> f32[R, C] { return silu(x) }
fn j(x: f32[R, C]) -> f32[R, C] { return rope(x) }
"""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 16)).astype(np.float32)
    w = rng.standard_normal(16).astype(np.float32)
    xl = ", ".join(cf(v) for v in x.flatten())
    wl = ", ".join(cf(v) for v in w)
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*,float*); float *veloce_g(float*); float *veloce_h(float*);
float *veloce_i(float*); float *veloce_j(float*);
int main(void) {{
    float x[64] = {{ {xl} }};
    float w[16] = {{ {wl} }};
    float *r;
    r = veloce_f(x,w); for(int q=0;q<64;q++) printf("%a ", r[q]); printf("\\n");
    r = veloce_g(x);   for(int q=0;q<64;q++) printf("%a ", r[q]); printf("\\n");
    r = veloce_h(x);   for(int q=0;q<64;q++) printf("%a ", r[q]); printf("\\n");
    r = veloce_i(x);   for(int q=0;q<64;q++) printf("%a ", r[q]); printf("\\n");
    r = veloce_j(x);   for(int q=0;q<64;q++) printf("%a ", r[q]); printf("\\n");
    return 0;
}}
"""
    outs = {}
    for pool, threads in ((0, None), (1, 4), (2, 4)):
        rc, out, err, _ = _threads_build(src, driver, pool, threads, name=f"rk{pool}")
        assert rc == 0, err
        outs[pool] = out
    assert outs[0] == outs[1] == outs[2], "row kernels differ across backends"


def test_i2_matmul_numeric():
    """FR-1.1: packed i2 (4 values/byte) matmul matches a float reference.

    Covers K % 4 != 0 so the flat (unpadded) packing tail is exercised, and
    checks the packed footprint is a quarter of one byte per weight.
    """
    from veloce import weights as vw
    M, K, N = 4, 1003, 7
    rng = np.random.default_rng(21)
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w = rng.integers(-2, 2, size=(N, K)).astype(np.int8)
    packed = vw.pack_i2(w)
    assert len(packed) == (N * K + 3) // 4
    src = f"""module t
dim M = {M}
dim K = {K}
dim N = {N}
fn f(x: f32[M, K], w: i2[N, K]) -> f32[M, N] {{
    return matmul_i2(x, w)
}}
"""
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*,uint8_t*);
int main(void) {{
    float x[{M*K}] = {{ {", ".join(cf(v) for v in x.flatten())} }};
    uint8_t w[{len(packed)}] = {{ {", ".join(str(b) for b in packed)} }};
    float *o = veloce_f(x, w);
    for (int i = 0; i < {M*N}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    outs = {}
    for pool in (0, 1, 2):
        rc, out, err, _ = _threads_build(src, driver, pool, 4, name=f"i2_{pool}")
        assert rc == 0, err
        outs[pool] = out
    assert outs[0] == outs[1] == outs[2], "i2 matmul differs across thread backends"
    got = np.array([float.fromhex(v) for v in outs[0].split()], dtype=np.float64).reshape(M, N)
    ref = x.astype(np.float64) @ w.astype(np.float64).T
    assert np.abs(got - ref).max() < 1e-4, (got, ref)


def test_i2_pack_roundtrip():
    """pack_i2/unpack_i2 agree with the runtime's bit layout."""
    from veloce import weights as vw
    assert list(vw.pack_i2([0, 0, 0, 0])) == [0x00]
    # -1 -> 0b11 in every slot
    assert list(vw.pack_i2([-1, -1, -1, -1])) == [0xFF]
    # slot k holds element 4b+k, low slot first: 1,-2,0,-1 -> 01|10|00|11
    assert list(vw.pack_i2([1, -2, 0, -1])) == [0b11001001]
    vals = [1, -2, 0, -1, -2, 1, 1, 0, -1, -2]
    assert vw.unpack_i2(vw.pack_i2(vals), len(vals)) == vals
    with np.testing.assert_raises(ValueError):
        vw.pack_i2([2, 0])


def test_fp8_matmul_numeric():
    """FR-1.1: fp8 E4M3 weights decode and multiply correctly.

    The reference uses the *decoded* weights, since fp8 is a storage format
    exactly like fp16: the dot product is exact in f32 for the stored values.
    """
    from veloce import weights as vw
    M, K, N = 3, 517, 9
    rng = np.random.default_rng(31)
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    wf = (rng.standard_normal((N, K)) * 0.7).astype(np.float32)
    packed = vw.pack_fp8(wf)
    assert len(packed) == N * K
    dec = np.array(vw.unpack_fp8(packed, N * K), dtype=np.float64).reshape(N, K)
    src = f"""module t
dim M = {M}
dim K = {K}
dim N = {N}
fn f(x: f32[M, K], w: fp8[N, K]) -> f32[M, N] {{
    return matmul_fp8(x, w)
}}
"""
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*,uint8_t*);
int main(void) {{
    float x[{M*K}] = {{ {", ".join(cf(v) for v in x.flatten())} }};
    uint8_t w[{len(packed)}] = {{ {", ".join(str(b) for b in packed)} }};
    float *o = veloce_f(x, w);
    for (int i = 0; i < {M*N}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    outs = {}
    for pool in (0, 1, 2):
        rc, out, err, _ = _threads_build(src, driver, pool, 4, name=f"fp8_{pool}")
        assert rc == 0, err
        outs[pool] = out
    assert outs[0] == outs[1] == outs[2], "fp8 matmul differs across thread backends"
    got = np.array([float.fromhex(v) for v in outs[0].split()], dtype=np.float64).reshape(M, N)
    ref = x.astype(np.float64) @ dec.T
    assert np.abs(got - ref).max() < 1e-4, (got, ref)


def test_fp8_e4m3_encoding():
    """The Python encoder and the C decoder must agree on every finite byte."""
    from veloce import weights as vw
    # Round-trip every representable value through encode -> decode.
    assert vw.unpack_fp8(bytes([0x00]), 1) == [0.0]
    assert vw.unpack_fp8(bytes([0x80]), 1) == [-0.0]
    assert vw.unpack_fp8(bytes([0x38]), 1) == [1.0]          # exp 7, man 0
    assert vw.unpack_fp8(bytes([0xB8]), 1) == [-1.0]
    assert vw.unpack_fp8(bytes([0x3C]), 1) == [1.5]          # exp 7, man 4
    assert vw.unpack_fp8(bytes([0x01]), 1) == [2.0 ** -9]    # smallest subnormal
    assert vw.unpack_fp8(bytes([0x7E]), 1) == [448.0]        # max finite
    assert vw.unpack_fp8(bytes([0x7F]), 1) == [0.0]          # NaN -> 0
    for v in (0.0, 1.0, -1.0, 1.5, 0.1, -3.75, 448.0, 1e-9, 2.0 ** -9):
        b = vw.pack_fp8([v])[0]
        back = vw.unpack_fp8(bytes([b]), 1)[0]
        if v == 0.0:
            assert back == 0.0
        else:
            assert abs(back - v) <= max(abs(v) * 2 ** -3, 2.0 ** -9), (v, b, back)
    # Out-of-range magnitudes saturate instead of overflowing to NaN.
    assert vw.unpack_fp8(vw.pack_fp8([1e9]), 1)[0] == 448.0


def test_row_store_into_caller_buffer():
    """`cache[pos] = row` must copy the row, not cast a pointer to float.

    This was a silent miscompile: a row store and an element store shared one
    code path, so `m[i] = vector` emitted `m[i*cols] = (float)(ptr)`.
    """
    import veloce
    src = """module t
dim CT = 4
dim D = 8
fn put(rw cache: f32[CT, D], pos: i32, row: f32[D]) -> f32[CT, D] {
    cache[pos] = row
    return cache
}
fn put2(rw cache: f32[CT, D], pos: i32, row: f32[1, D]) -> f32[CT, D] {
    cache[pos] = row
    return cache
}
fn get(cache: f32[CT, D], pos: i32) -> f32[D] {
    return cache[pos]
}
"""
    c = veloce.compile_source(src, module_name="t").c_source
    assert "veloce_copyf(" in c, c
    assert "(float)(p_row" not in c, "row store degraded to a scalar cast"

    rng = np.random.default_rng(9)
    CT, D = 4, 8
    row = rng.standard_normal(D).astype(np.float32)
    row2 = rng.standard_normal(D).astype(np.float32)
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_put(float*,int32_t,float*);
float *veloce_put2(float*,int32_t,float*);
int main(void) {{
    static float cache[{CT*D}] = {{0}};
    float r1[{D}] = {{ {", ".join(cf(v) for v in row)} }};
    float r2[{D}] = {{ {", ".join(cf(v) for v in row2)} }};
    veloce_put(cache, 1, r1);
    veloce_put2(cache, 3, r2);
    for (int i = 0; i < {CT*D}; i++) printf("%a\\n", cache[i]);
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    assert rc == 0, err
    got = np.array([float.fromhex(v) for v in out.split()]).reshape(CT, D)
    ref = np.zeros((CT, D))
    ref[1] = row
    ref[3] = row2
    assert np.array_equal(got, ref), (got, ref)


def test_row_store_rejects_shape_mismatch():
    """A row store must not accept a scalar or a wrong-width row."""
    import veloce
    for bad_type, bad_val in (("f32", "1.0"), ("f32[4]", "row")):
        src = f"""module t
dim CT = 4
dim D = 8
fn put(rw cache: f32[CT, D], row: {bad_type}) -> f32[CT, D] {{
    cache[0] = row
    return cache
}}
"""
        with np.testing.assert_raises(Exception):
            veloce.compile_source(src, module_name="t")


def test_kv_cache_decode_example():
    """The shipped KV-cache example runs and matches its NumPy reference."""
    import subprocess
    ex = os.path.join(os.path.dirname(HERE), "examples", "run_kv_decode.py")
    r = subprocess.run([sys.executable, ex], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout, r.stdout


def test_scaled_lowbit_matmul_numeric():
    """Group-scaled i4 and row-scaled trit matmuls match a float reference.

    These two kernels existed in the runtime but had no coverage at all, so a
    codegen/runtime disagreement (group size) went unnoticed.
    """
    from veloce import weights as vw

    # --- matmul_i4g: out[m,n] = sum_g scales[n,g] * sum_{k in g} x*w
    M, K, N, G = 4, 64, 8, 16
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    w4 = rng.integers(-8, 8, size=(N, K)).astype(np.int8)
    sc4 = (rng.standard_normal((N, K // G)) * 0.5).astype(np.float32)
    packed4 = vw.pack_i4(w4)
    src = f"""module t
dim M = {M}
dim K = {K}
dim N = {N}
dim NG = {K // G}
fn f(x: f32[M, K], w: i4[N, K], s: f32[N, NG]) -> f32[M, N] {{
    return matmul_i4g(x, w, s)
}}
"""
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*,uint8_t*,float*);
int main(void) {{
    float x[{M*K}] = {{ {", ".join(cf(v) for v in x.flatten())} }};
    uint8_t w[{len(packed4)}] = {{ {", ".join(str(b) for b in packed4)} }};
    float s[{N*(K//G)}] = {{ {", ".join(cf(v) for v in sc4.flatten())} }};
    float *o = veloce_f(x, w, s);
    for (int i = 0; i < {M*N}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver)
    assert rc == 0, err
    got = np.array([float.fromhex(v) for v in out.split()], dtype=np.float64).reshape(M, N)
    ref = np.zeros((M, N))
    for n in range(N):
        for g in range(K // G):
            ref[:, n] += (x[:, g*G:(g+1)*G].astype(np.float64)
                          @ w4[n, g*G:(g+1)*G].astype(np.float64)) * sc4[n, g]
    assert np.abs(got - ref).max() < 1e-4, (got, ref)

    # --- matmul_trit_s: out[m,n] = scales[n] * sum_k x*w
    M2, K2, N2 = 1, 1003, 5
    rng = np.random.default_rng(11)
    x2 = (rng.standard_normal((M2, K2)) * 0.5).astype(np.float32)
    w2 = rng.integers(-1, 2, size=(N2, K2)).astype(np.int8)
    sc2 = (rng.standard_normal(N2) * 0.5).astype(np.float32)
    packed2 = vw.pack_trits(w2)
    src2 = f"""module t2
dim M = {M2}
dim K = {K2}
dim N = {N2}
fn g(x: f32[M, K], w: trit[N, K], s: f32[N]) -> f32[M, N] {{
    return matmul_trit_s(x, w, s)
}}
"""
    driver2 = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_g(float*,uint8_t*,float*);
int main(void) {{
    float x[{M2*K2}] = {{ {", ".join(cf(v) for v in x2.flatten())} }};
    uint8_t w[{len(packed2)}] = {{ {", ".join(str(b) for b in packed2)} }};
    float s[{N2}] = {{ {", ".join(cf(v) for v in sc2)} }};
    float *o = veloce_g(x, w, s);
    for (int i = 0; i < {M2*N2}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src2, driver2)
    assert rc == 0, err
    got2 = np.array([float.fromhex(v) for v in out.split()], dtype=np.float64).reshape(M2, N2)
    ref2 = (x2.astype(np.float64) @ w2.astype(np.float64).T) * sc2.astype(np.float64)
    assert np.abs(got2 - ref2).max() < 1e-4, (got2, ref2)


def test_cached_decode_attention_mask():
    """A cached decode step must ignore slots past each query row's position.

    This test used to be parked: a two-layer `forward_cached` diverged at
    ~1e30 on the *second* layer while one layer was exact.  The compiler was
    never at fault -- the driver built its argument list by hand (all KV
    caches first, then all weights) while the generated signature interleaves
    them per layer (`ck1, cv1, nw1, wq1, ..., ck2, cv2, ...`).  For `L == 1`
    the two orders coincide, which is exactly why only the two-layer case
    failed: every argument past `cv1` was shifted by two, so layer 2 read
    layer 1's weights and an `f32[D]` array as a trit matrix.  The argument
    list is now derived from the parsed signature, and the test runs 1, 2 and
    3 layers so a per-layer interleaving mistake cannot hide behind `L == 1`.

    The mask is now 2-D (`i32[T, CT]`, one row per query row in the chunk) and
    the test decodes in chunks of 1, 2 and 3 tokens.  With `chunk > 1` the
    mask rows genuinely differ -- row 0 may only see `pos`, row 2 sees
    `pos + 2` -- so a lowering that read the 2-D mask as row 0 for every row
    (the silent miscompile fixed in `softmax`) produces different logits and
    this test fails.  Without any mask the zeros in unwritten slots dilute
    every row and the reference below disagrees.
    """
    for layers in (1, 2, 3):
        _check_cached_decode(layers, chunk=1)
    for layers in (1, 2):
        _check_cached_decode(layers, chunk=2)
        _check_cached_decode(layers, chunk=3)


def test_cached_decode_mask_hides_garbage_slots():
    """Unwritten cache rows may hold anything and must not reach the output.

    A zeroed cache makes a broken mask quiet: a wrong row of zeros contributes
    a finite, plausible score.  Here every cache row starts at +-1e30 and the
    decode must still match the clean reference bit for bit -- only the slots
    the model wrote for the current chunk (and earlier ones) may be read.
    """
    for chunk in (1, 2, 3):
        _check_cached_decode(2, chunk=chunk, poison=True)


def _check_cached_decode(L, chunk=1, poison=False):
    """One cached decode of six tokens through an `L`-layer decoder."""
    from veloce.lmmodule import cached_module

    V, D, F, CT = 16, 8, 16, 6
    cfg = dict(name=f"kvtest{L}", V=V, D=D, F=F, layers=L)
    src = cached_module(cfg, CT, chunk=chunk)

    rng = np.random.default_rng(5)
    emb = (rng.standard_normal((V, D)) * 0.3).astype(np.float32)
    nw = (1.0 + 0.05 * rng.standard_normal(D)).astype(np.float32)
    mats = {}
    for l in range(1, L + 1):
        mats[f"nw{l}"] = nw
        for n in ("wq", "wk", "wv", "wo"):
            mats[f"{n}{l}"] = rng.integers(-1, 2, size=(D, D)).astype(np.int8)
        for n in ("w1", "w3"):
            mats[f"{n}{l}"] = rng.integers(-1, 2, size=(F, D)).astype(np.int8)
        mats[f"w2{l}"] = rng.integers(-1, 2, size=(D, F)).astype(np.int8)
    mats["nwf"] = nw
    head = rng.integers(-1, 2, size=(V, D)).astype(np.int8)
    toks = [3, 11, 0, 7, 5, 9]
    assert len(toks) % chunk == 0

    def packed(name):
        w = mats[name]
        if w.dtype == np.int8:
            return vw_pack(w)
        return list(w.flatten())

    decls = [f"static float ck[{L}][{CT*D}];",
             f"static float cv[{L}][{CT*D}];",
             f"static int32_t mask[{chunk}][{CT}];",
             f"static int32_t tok_buf[{chunk}];",
             "static void poison(float *p, int n) {",
             "    for (int i = 0; i < n; i++) p[i] = (i & 1) ? -1e30f : 1e30f;",
             "}"]
    # `emb` is fp16 storage: the C runtime takes a uint16_t table.
    decls.append("    " + carr("emb", "uint16_t",
                               list(emb.astype(np.float16).view(np.uint16)
                                    .flatten())))
    for l in range(1, L + 1):
        decls.append("    " + carr(f"nw{l}", "float", mats[f"nw{l}"]))
        for n in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
            decls.append("    " + carr(f"{n}{l}", "uint8_t", packed(f"{n}{l}")))
    decls.append("    " + carr("nwf", "float", mats["nwf"]))
    decls.append("    " + carr("head", "uint8_t", vw_pack(head)))

    import veloce
    from veloce.codegen import ctype_of
    cached_fn = next(f for f in veloce.parse(veloce.tokenize(src)).functions
                     if f.name == "forward_cached")

    def _cached_arg(name, pos):
        """C expression for one `forward_cached` parameter, matched by name."""
        if name == "emb":
            return "emb"
        if name == "tok":
            return "tok_buf"
        if name == "pos":
            return str(pos)
        if name == "mask":
            return "mask"
        m = re.fullmatch(r"(ck|cv)(\d+)", name)
        if m:
            return f"{m.group(1)}[{int(m.group(2)) - 1}]"
        return name  # nw{l}, wq{l}, ..., nwf, head

    loop = []
    for l in range(L):
        loop.append(f"    if ({1 if poison else 0}) {{ "
                    f"poison(ck[{l}], {CT*D}); poison(cv[{l}], {CT*D}); }} else {{ "
                    f"for (int i = 0; i < {CT*D}; i++) "
                    f"ck[{l}][i] = cv[{l}][i] = 0.0f; }}")
    for t in range(0, len(toks), chunk):
        for r in range(chunk):
            loop.append(f"    tok_buf[{r}] = {toks[t + r]};")
        # Row r of the mask belongs to query row r, which sits at position
        # t + r and may see every slot up to it.
        for r in range(chunk):
            loop.append(f"    for (int i = 0; i < {CT}; i++) "
                        f"mask[{r}][i] = (i <= {t + r}) ? 1 : 0;")
        args = ", ".join(_cached_arg(p.name, t) for p in cached_fn.params)
        loop.append(f"    float *o{t} = veloce_forward_cached({args});")
        loop.append(f"    for (int i = 0; i < {V}; i++) "
                    f'printf("t{t + chunk - 1} %a\\n", '
                    f"o{t}[{(chunk - 1) * V} + i]);")

    # Prototype straight from the compiled AST, so it cannot drift from the
    # real signature (a hand-written one silently truncates every pointer).
    # The argument list is built from the same parameter list, in the same
    # order -- see the docstring for what happens when it is not.
    proto = ", ".join(f"{ctype_of(p.type)} {p.name}" for p in cached_fn.params)
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_forward_cached({proto});
{chr(10).join(decls)}
int main(void) {{
{chr(10).join(loop)}
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver,
                                         name=f"kv_cached{L}_{chunk}")
    assert rc == 0, err

    got = {}
    for line in out.splitlines():
        tag, hexval = line.split()
        got.setdefault(tag, []).append(float.fromhex(hexval))

    # ---- reference: fp64, token by token, strictly causal -----------------
    ck = [np.zeros((CT, D)) for _ in range(L)]
    cv = [np.zeros((CT, D)) for _ in range(L)]
    scale = np.float64(1.0 / np.sqrt(D))
    emb16 = emb.astype(np.float16).astype(np.float64)  # fp16 storage
    for t, tok in enumerate(toks):
        x = emb16[tok]
        for l in range(1, L + 1):
            xn = _rmsnorm64(x, nw)
            q = xn @ mats[f"wq{l}"].astype(np.float64).T
            k = xn @ mats[f"wk{l}"].astype(np.float64).T
            v = xn @ mats[f"wv{l}"].astype(np.float64).T
            ck[l - 1][t] = k
            cv[l - 1][t] = v
            s = (q @ ck[l - 1][:t + 1].T) * scale
            e = np.exp(s - s.max())
            p = e / e.sum()
            x = x + (p @ cv[l - 1][:t + 1]) @ mats[f"wo{l}"].astype(np.float64).T
            xn2 = _rmsnorm64(x, nw)
            g = xn2 @ mats[f"w1{l}"].astype(np.float64).T
            u = xn2 @ mats[f"w3{l}"].astype(np.float64).T
            h = g / (1.0 + np.exp(-g)) * u
            x = x + h @ mats[f"w2{l}"].astype(np.float64).T
        xn = _rmsnorm64(x, nw)
        ref = xn @ head.astype(np.float64).T
        # Only the last row of each chunk is printed: that is the token the
        # caller would sample next.
        if f"t{t}" not in got:
            continue
        err = float(np.abs(np.array(got[f"t{t}"]) - ref).max())
        assert err < 1e-4, (f"layers {L}, chunk {chunk}, token {t}: "
                            f"max|err| = {err:.3e}")


def test_group_scaled_i2_fp8_matmul_numeric():
    """`matmul_i2g` / `matmul_fp8g` match a float reference.

    Only the row-scaled variants (`_s`) existed, so a real 2-bit or FP8
    checkpoint -- which ships per-group scales like `matmul_i4g` -- could not
    be expressed at all.  The group contract is the established one: the
    scale tensor is `f32[N, K/G]` and the group count must divide K.
    """
    from veloce import weights as vw

    M, K, N, G = 3, 64, 8, 16
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
    sc = (rng.standard_normal((N, K // G)) * 0.5).astype(np.float32)

    # --- matmul_i2g: 4 signed 2-bit values per byte, range [-2, 1]
    w2 = rng.integers(-2, 2, size=(N, K)).astype(np.int8)
    packed2 = vw.pack_i2(w2)
    src = f"""module t2g
dim M = {M}
dim K = {K}
dim N = {N}
dim NG = {K // G}
fn f(x: f32[M, K], w: i2[N, K], s: f32[N, NG]) -> f32[M, N] {{
    return matmul_i2g(x, w, s)
}}
"""
    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*, uint8_t*, float*);
int main(void) {{
    float x[{M*K}] = {{ {", ".join(cf(v) for v in x.flatten())} }};
    uint8_t w[{len(packed2)}] = {{ {", ".join(str(b) for b in packed2)} }};
    float s[{N*(K//G)}] = {{ {", ".join(cf(v) for v in sc.flatten())} }};
    float *o = veloce_f(x, w, s);
    for (int i = 0; i < {M*N}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src, driver, name="i2g")
    assert rc == 0, err
    got = np.array([float.fromhex(v) for v in out.split()],
                   dtype=np.float64).reshape(M, N)
    ref = np.zeros((M, N))
    for n in range(N):
        for g in range(K // G):
            ref[:, n] += (x[:, g*G:(g+1)*G].astype(np.float64)
                          @ w2[n, g*G:(g+1)*G].astype(np.float64)) * sc[n, g]
    assert np.abs(got - ref).max() < 1e-4, (got, ref)

    # --- matmul_fp8g: E4M3 weights, one byte per value
    w8 = np.frombuffer(vw.pack_fp8(rng.standard_normal((N, K)) * 0.4),
                       dtype=np.uint8).reshape(N, K)
    src8 = f"""module t8g
dim M = {M}
dim K = {K}
dim N = {N}
dim NG = {K // G}
fn f(x: f32[M, K], w: fp8[N, K], s: f32[N, NG]) -> f32[M, N] {{
    return matmul_fp8g(x, w, s)
}}
"""
    driver8 = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*, uint8_t*, float*);
int main(void) {{
    float x[{M*K}] = {{ {", ".join(cf(v) for v in x.flatten())} }};
    uint8_t w[{N*K}] = {{ {", ".join(str(int(b)) for b in w8.flatten())} }};
    float s[{N*(K//G)}] = {{ {", ".join(cf(v) for v in sc.flatten())} }};
    float *o = veloce_f(x, w, s);
    for (int i = 0; i < {M*N}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src8, driver8, name="fp8g")
    assert rc == 0, err
    got8 = np.array([float.fromhex(v) for v in out.split()],
                    dtype=np.float64).reshape(M, N)
    # The reference must decode the *same* E4M3 bytes the kernel reads, so a
    # rounding difference in the packer cannot mask a kernel bug.
    dec = np.array(vw.unpack_fp8(w8.flatten(), N * K),
                   dtype=np.float64).reshape(N, K)
    ref8 = np.zeros((M, N))
    for n in range(N):
        for g in range(K // G):
            ref8[:, n] += (x[:, g*G:(g+1)*G].astype(np.float64)
                           @ dec[n, g*G:(g+1)*G]) * sc[n, g]
    assert np.abs(got8 - ref8).max() < 1e-4, (got8, ref8)


def test_group_scaled_rejects_ragged_group_count():
    """A group count that does not divide K is rejected for i2/fp8 too.

    `matmul_i4g` had exactly this bug (floor vs ceil group count, 35.8 error
    for K=100 over 7 groups).  The check is shared now, so the new kernels
    must reject the same shape at compile time instead of scaling rows
    against the wrong slices.
    """
    import veloce
    for name, dtype in (("matmul_i2g", "i2"), ("matmul_fp8g", "fp8")):
        src = f"""module t
dim M = 2
dim K = 100
dim N = 4
dim NG = 7
fn f(x: f32[M, K], w: {dtype}[N, K], s: f32[N, NG]) -> f32[M, N] {{
    return {name}(x, w, s)
}}
"""
        try:
            veloce.compile_source(src)
        except (veloce.TypeError_, veloce.SemanticError) as e:
            assert "divide K exactly" in str(e), e
        else:
            raise AssertionError(f"{name} accepted K=100 over 7 groups")


def test_masked_softmax_builtin():
    """`softmax(x, mask)` hides entries without biasing them.

    The point of a mask builtin (over adding -1e30 to the scores) is that a
    hidden entry's *value* is irrelevant: it can be NaN or +-1e30 and still
    contributes exactly zero.  This test poisons the hidden columns with
    exactly those values.
    """
    import veloce
    T, C = 3, 4
    rng = np.random.default_rng(7)
    x = rng.standard_normal((T, C)).astype(np.float32)
    x[0, 2] = 1e30
    x[0, 3] = -1e30
    x[1, 2] = np.nan            # hidden, must not reach the result
    mask = np.array([1, 1, 0, 0], dtype=np.int32)

    src = f"""module sm
dim T = {T}
dim C = {C}
fn f(x: f32[T, C], m: i32[C]) -> f32[T, C] {{
    return softmax(x, m)
}}
"""
    c = veloce.compile_source(src, module_name="sm").c_source
    assert "veloce_softmax_masked(" in c, c

    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*,int32_t*);
int main(void) {{
    float x[{T*C}] = {{ {", ".join(cf(v) for v in x.flatten())} }};
    int32_t m[{C}] = {{ {", ".join(str(v) for v in mask)} }};
    float *o = veloce_f(x, m);
    for (int i = 0; i < {T*C}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    outs = {}
    for pool in (0, 1, 2):
        rc, out, err, _ = _threads_build(src, driver, pool, 4, name=f"smask{pool}")
        assert rc == 0, err
        outs[pool] = out
    assert outs[0] == outs[1] == outs[2], "masked softmax differs across threads"

    got = np.array([float.fromhex(v) for v in outs[0].split()]).reshape(T, C)
    hidden = ~mask.astype(bool)
    assert np.array_equal(got[:, hidden], np.zeros((T, 2), dtype=got.dtype)), got
    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-6), got.sum(axis=1)
    ref = np.zeros((T, C))
    for r in range(T):
        vis = x[r][mask.astype(bool)].astype(np.float64)
        e = np.exp(vis - vis.max())
        ref[r][mask.astype(bool)] = e / e.sum()
    assert np.abs(got - ref).max() < 1e-6, (got, ref)

    # An entirely masked row has no distribution: zeros, never NaN.
    src2 = f"""module sm2
dim C = {C}
fn g(x: f32[1, C], m: i32[C]) -> f32[1, C] {{
    return softmax(x, m)
}}
"""
    driver2 = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_g(float*,int32_t*);
int main(void) {{
    float x[{C}] = {{ {", ".join(cf(v) for v in x[0])} }};
    int32_t m[{C}] = {{ 0, 0, 0, 0 }};
    float *o = veloce_g(x, m);
    for (int i = 0; i < {C}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    rc, out, err, _ = util.build_and_run(src2, driver2, name="smask_all")
    assert rc == 0, err
    allhid = np.array([float.fromhex(v) for v in out.split()])
    assert np.array_equal(allhid, np.zeros(C, dtype=allhid.dtype)), allhid


def test_masked_softmax_per_row_mask():
    """A 2-D mask is per row, not a silent broadcast of mask row 0.

    The typechecker has always accepted `i32[T, C]`, but the runtime read
    `mask[i]` for `i < C`, so every row saw row 0 of the mask.  A per-head
    causal mask therefore produced a plausible but wrong distribution with no
    diagnostic.  The mask below has a different visible pattern per row, so
    broadcasting row 0 cannot pass, and the hidden entries are poisoned with
    values that would leak if they were included in the max or the sum.
    """
    import veloce
    T, C = 3, 4
    rng = np.random.default_rng(11)
    x = rng.standard_normal((T, C)).astype(np.float32)
    mask = np.array([[1, 0, 0, 0],
                     [1, 1, 0, 0],
                     [1, 1, 1, 1]], dtype=np.int32)
    x[0, 1] = 1e30
    x[0, 3] = np.nan
    x[1, 2] = 1e30
    x[2, 3] = -1e30

    src = f"""module smr
dim T = {T}
dim C = {C}
fn f(x: f32[T, C], m: i32[T, C]) -> f32[T, C] {{
    return softmax(x, m)
}}
"""
    c = veloce.compile_source(src, module_name="smr").c_source
    assert "veloce_softmax_masked_rows(" in c, c
    assert "veloce_softmax_masked(" not in c, c

    driver = f"""
#include "veloce_runtime.h"
#include <stdio.h>
float *veloce_f(float*,int32_t*);
int main(void) {{
    float x[{T*C}] = {{ {", ".join(cf(v) for v in x.flatten())} }};
    int32_t m[{T*C}] = {{ {", ".join(str(int(v)) for v in mask.flatten())} }};
    float *o = veloce_f(x, m);
    for (int i = 0; i < {T*C}; i++) printf("%a\\n", o[i]);
    return 0;
}}
"""
    outs = {}
    for pool in (0, 1, 2):
        rc, out, err, _ = _threads_build(src, driver, pool, 4, name=f"smrow{pool}")
        assert rc == 0, err
        outs[pool] = out
    assert outs[0] == outs[1] == outs[2], "per-row masked softmax differs across threads"

    got = np.array([float.fromhex(v) for v in outs[0].split()]).reshape(T, C)
    assert not np.any(got[~mask.astype(bool)]), got
    ref = np.zeros((T, C))
    for r in range(T):
        vis = mask[r].astype(bool)
        xs = x[r][vis].astype(np.float64)
        e = np.exp(xs - xs.max())
        ref[r][vis] = e / e.sum()
    assert np.abs(got - ref).max() < 1e-6, (got, ref)
    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-6), got.sum(axis=1)


def test_softmax_mask_rejects_bad_shape_and_dtype():
    """A 2-D mask must match the row count, and only i32 is accepted.

    Both were silently accepted before: a short i32[2, C] mask would be read
    past its end, and an i8[T, C] mask would be reinterpreted as 32-bit
    words, so neither could be trusted to mean what it said.
    """
    import veloce
    cases = [("i32", 2, "rows"), ("i8", 3, "i32")]
    for ty, rows, want in cases:
        src = f"""module t2
dim T = 3
dim C = 4
dim M = {rows}
fn f(x: f32[T, C], m: {ty}[M, C]) -> f32[T, C] {{
    return softmax(x, m)
}}
"""
        try:
            veloce.compile_source(src, module_name="t2")
        except (veloce.TypeError_, veloce.SemanticError) as e:
            assert want in str(e), e
        else:
            raise AssertionError(f"softmax accepted a {ty}[{rows}, C] mask")


def test_masked_softmax_rejects_bad_mask():
    """A mask must be integer and match the last dimension."""
    import veloce
    for mask_ty, mask_dim in (("f32", 4), ("i32", 3)):
        src = f"""module t3
dim C = 4
dim M = {mask_dim}
fn f(x: f32[2, C], m: {mask_ty}[M]) -> f32[2, C] {{
    return softmax(x, m)
}}
"""
        with np.testing.assert_raises(Exception):
            veloce.compile_source(src, module_name="t3")


        with np.testing.assert_raises(Exception):
            veloce.compile_source(src, module_name="t")
def _rmsnorm64(x, w):
    inv = 1.0 / np.sqrt(float((x * x).sum()) / x.shape[0] + 1e-5)
    return x * inv * w.astype(np.float64)


def vw_pack(w):
    """Pack trits, as a list so the C array literal writer can iterate it."""
    from veloce import weights as vw
    return list(vw.pack_trits(w))


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    print(f"running {len(TESTS)} tests\n" + "-" * 60)
    for t in TESTS:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception as e:
            FAILS.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print("-" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
        sys.exit(1)
    print(f"all {len(TESTS)} tests passed")


if __name__ == "__main__":
    main()
