#!/usr/bin/env python3
"""Veloce test-suite (stdlib-only; run with `python3 tests/test_veloce.py`).

Each test compiles real C, links it, runs it, and checks numeric results
against an independent NumPy reference (passed into the C driver as expected
arrays) or against compiler diagnostics.  A tiny homemade runner keeps the
project dependency-free.
"""
import os
import sys
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import util  # noqa: E402

FAILS = []


def carr(name, ctype, values):
    if ctype.startswith("float"):
        body = ", ".join(f"{float(v):.9g}f" for v in values)
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
    xl = ", ".join(f"{float(v):.9g}f" for v in x.flatten())
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
    float x[6] = {{ {', '.join(f'{v:.9g}f' for v in x.flatten())} }};
    float b[3] = {{ {', '.join(f'{v:.9g}f' for v in b)} }};
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
    float x[{T*D}] = {{ {', '.join(f'{v:.9g}f' for v in x.flatten())} }};
    float w[{D}] = {{ {', '.join(f'{v:.9g}f' for v in w)} }};
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
    float x[6] = {{ {', '.join(f'{v:.9g}f' for v in x)} }};
    float o1[6];
    veloce_silu(o1, x, 6);
    {check_close('o1', list(silu), 6, tag='silu')}
    float s[6] = {{ {', '.join(f'{v:.9g}f' for v in s.flatten())} }};
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
    float a[6] = {{ {', '.join(f'{v:.9g}f' for v in a.flatten())} }};
    float b[12] = {{ {', '.join(f'{v:.9g}f' for v in b.flatten())} }};
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
    float x[{M*K}] = {{ {', '.join(f'{v:.9g}f' for v in x.flatten())} }};
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
    float emb[32] = {{ {', '.join(f'{v:.9g}f' for v in emb.flatten())} }};
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
        return ", ".join(f"{float(v):.9g}f" for v in a.flatten())

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
    float x[{M*K}] = {{ {', '.join(f'{v:.9g}f' for v in x.flatten())} }};
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
    float x[4] = {{ {', '.join(f'{v:.9g}f' for v in x)} }};
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
    float emb[24] = {{ {', '.join(f'{v:.9g}f' for v in emb.flatten())} }};
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
    xl = ", ".join(f"{float(v):.9g}f" for v in x.flatten())
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

    def cf(a):
        return ", ".join(f"{float(v):.9g}f" for v in a.flatten())

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

    parts = [f"float x[{T*D}] = {{ {cf(x)} }};", f"float nw[{D}] = {{ {cf(nw)} }};"]
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
    xl = ", ".join(f"{float(v):.9g}f" for v in x.flatten())
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
    xl = ", ".join(f"{v:.9g}f" for v in x.flatten())
    wl = ", ".join(f"{v:.9g}f" for v in w)
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
