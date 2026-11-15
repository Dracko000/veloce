#!/usr/bin/env python3
"""Veloce throughput benchmark — how far are we from the PRD targets?

Measures real wall-clock throughput of the compiled runtime kernels on this
machine and compares against the two quantitative PRD goals:

    PRD 3.1  Memory:   >= 80% smaller than an fp16 PyTorch baseline
    PRD 3.2  Speed:    >= 15 tokens/sec for a 7B-class ternary model on CPU
    PRD 3.3  Zero runtime heap allocation in the inference loop
    PRD 3.4  Compile standard transformer architectures in < 2 s

Everything runs compiled C through the same `veloce_runtime.h` kernels that
`velocec` emits, so the numbers are what a deployed Veloce module would see.

Usage:
    python3 benchmarks/bench_throughput.py              # full report
    python3 benchmarks/bench_throughput.py --quick      # fewer repetitions
    python3 benchmarks/bench_throughput.py --json out.json

No numpy is required: the benchmark uses the runtime directly from C.
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "runtime")
CC = os.environ.get("CC", "cc")

# PRD targets
PRD_TOKENS_PER_SEC = 15.0
PRD_MODEL = "7B-class ternary"
PRD_COMPILE_SECONDS = 2.0
PRD_MEMORY_REDUCTION = 0.80

# A 7B-class ternary decoder: 32 layers, d_model=4096, ffn=11008, vocab 32000.
# Per token, weight bytes touched (trit packed) for the dominant matmuls:
#   attention q/k/v/o: 4 * 4096*4096
#   MLP gate/up/down:  3 * 4096*11008
# The benchmark uses one representative layer and scales by depth.
MODEL = dict(layers=32, d_model=4096, ffn=11008, vocab=32000, heads=32)
BYTES_PER_TRIT = 1.0 / 5.0


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def have_openmp():
    with tempfile.TemporaryDirectory() as td:
        r = sh([CC, "-fopenmp", "-x", "c", "-", "-o", os.path.join(td, "p")],
               input="int main(void){return 0;}\n")
    return r.returncode == 0


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def nproc():
    return os.cpu_count() or 1


def build_and_run(name, source, timeout=600, extra_flags=None, gen_c=None):
    """Compile a C benchmark against the runtime, run it, return (stdout, seconds).

    If `gen_c` is given (output of the Veloce compiler), it is compiled as a
    separate translation unit and linked with the driver, exactly like a real
    deployment.
    """
    with tempfile.TemporaryDirectory() as td:
        cpath = os.path.join(td, name + ".c")
        exe = os.path.join(td, name)
        with open(cpath, "w") as f:
            f.write(source)
        extra = list(extra_flags or [])
        base = [CC, "-std=c11", "-O3", "-march=native", "-ffast-math",
                "-pthread", "-I", RUNTIME, *extra]
        objs = []
        if gen_c is not None:
            gpath = os.path.join(td, name + "_gen.c")
            gobj = os.path.join(td, name + "_gen.o")
            with open(gpath, "w") as f:
                f.write(gen_c)
            r = sh([*base, "-c", gpath, "-o", gobj])
            if r.returncode != 0:
                raise RuntimeError(f"gen build failed:\n{r.stderr}")
            objs.append(gobj)
        dobj = os.path.join(td, name + "_drv.o")
        r = sh([*base, "-c", cpath, "-o", dobj])
        if r.returncode != 0:
            raise RuntimeError(f"build failed:\n{r.stderr}")
        objs.append(dobj)
        # -fopenmp / -pthread must also be present on the link line
        r = sh([CC, *objs, *extra, "-lm", "-o", exe])
        if r.returncode != 0:
            raise RuntimeError(f"link failed:\n{r.stderr}")
        t0 = time.perf_counter()
        p = sh([exe], timeout=timeout)
        dt = time.perf_counter() - t0
        if p.returncode != 0:
            raise RuntimeError(f"run failed:\n{p.stderr}")
        return p.stdout, dt


def bench_trit_gemv(quick, pool_flags):
    """Core primitive: M=1 trit GEMV at 7B-layer shape (4096 x 4096).

    Builds one binary containing three copies of the same kernel, compiled
    with 1, 2 and 4 threads (a runtime switch is intentionally not offered by
    FR-4.2). Each copy is warmed and timed round-robin so external CPU load
    hits all configurations equally; best time per configuration is kept.
    """
    rounds = 40 if quick else 120
    impl = r"""
#define _POSIX_C_SOURCE 199309L
#include "veloce_runtime.h"
#include <stdlib.h>
#include <time.h>
#ifndef RUN_FN
#error "RUN_FN must be defined"
#endif
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}
double RUN_FN(int reps) {
    const size_t K = 4096, N = 4096;
    static float *x, *o; static uint8_t *w;
    if (!x) {
        x = malloc(K * sizeof(float)); o = malloc(N * sizeof(float));
        w = malloc((N * K + 4) / 5);
        for (size_t i = 0; i < K; i++) x[i] = (float)((int)(i % 7) - 3) * 0.1f;
        for (size_t i = 0; i < (N * K + 4) / 5; i++) w[i] = (uint8_t)(i * 131 + 17);
    }
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t = now();
        veloce_matmul_trit(o, x, w, 1, K, N);
        t = now() - t; if (t < best) best = t;
    }
    return best;
}
"""
    driver = r"""
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
double run_t1(int), run_t2(int), run_t4(int);
int main(void) {
    run_t1(1); run_t2(1); run_t4(1);
    double b1 = 1e9, b2 = 1e9, b4 = 1e9;
    for (int r = 0; r < __ROUNDS__; r++) {
        double a = run_t1(1); if (a < b1) b1 = a;
        double b = run_t2(1); if (b < b2) b2 = b;
        double c = run_t4(1); if (c < b4) b4 = c;
    }
    const double m = 4096.0 * 4096.0;
    printf("T1_MS %.4f\nT2_MS %.4f\nT4_MS %.4f\n"
           "T1_GMAC %.3f\nT2_GMAC %.3f\nT4_GMAC %.3f\n",
           b1 * 1e3, b2 * 1e3, b4 * 1e3,
           m / b1 / 1e9, m / b2 / 1e9, m / b4 / 1e9);
    return 0;
}
""".replace("__ROUNDS__", str(rounds))
    with tempfile.TemporaryDirectory() as td:
        impl_path = os.path.join(td, "gemv_impl.c")
        drv_path = os.path.join(td, "gemv_drv.c")
        with open(impl_path, "w") as f:
            f.write(impl)
        with open(drv_path, "w") as f:
            f.write(driver)
        base = [CC, "-std=c11", "-O3", "-march=native", "-ffast-math",
                "-pthread", "-I", RUNTIME, *pool_flags]
        objs = []
        for n in (1, 2, 4):
            obj = os.path.join(td, f"t{n}.o")
            r = sh([*base, f"-DVELOCE_THREADS={n}", f"-DRUN_FN=run_t{n}",
                    "-c", impl_path, "-o", obj])
            if r.returncode != 0:
                raise RuntimeError(f"gemv t{n} build failed:\n{r.stderr}")
            objs.append(obj)
        dobj = os.path.join(td, "drv.o")
        r = sh([*base, "-c", drv_path, "-o", dobj])
        if r.returncode != 0:
            raise RuntimeError(f"gemv driver build failed:\n{r.stderr}")
        exe = os.path.join(td, "gemv")
        r = sh([CC, *objs, dobj, *pool_flags, "-lm", "-o", exe])
        if r.returncode != 0:
            raise RuntimeError(f"gemv link failed:\n{r.stderr}")
        p = sh([exe], timeout=600)
        if p.returncode != 0:
            raise RuntimeError(f"gemv run failed:\n{p.stderr}")
        vals = {k: float(v) for k, v in (l.split() for l in p.stdout.splitlines())}
        vals["THREADS"] = float(min(4, nproc()))
        return vals


def bench_attention(quick, pool_flags):
    """Attention-shaped work: QK^T + softmax + AV at 32 heads x 4096."""
    reps = 5 if quick else 20
    src = f"""
#define _POSIX_C_SOURCE 199309L
#include "veloce_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}}
int main(void){{
    const size_t T=256,D=4096;
    const size_t TD=T*D, TT=T*T;   /* size_t: T*D*4 overflows 32-bit int */
    float *q=malloc(TD*sizeof(float)),*k=malloc(TD*sizeof(float)),*v=malloc(TD*sizeof(float));
    float *kt=malloc(TD*sizeof(float)),*s=malloc(TT*sizeof(float)),*p=malloc(TT*sizeof(float)),*a=malloc(TD*sizeof(float));
    for(size_t i=0;i<T*D;i++){{q[i]=0.01f*(i%17);k[i]=0.01f*(i%13);v[i]=0.01f*(i%11);}}
    veloce_transpose(kt,k,T,D);
    double best=1e9;
    for(int r=0;r<{reps};r++){{
        double t=now();
        veloce_matmul_f32(s,q,kt,T,D,T);      /* [T,D] x [D,T] -> [T,T] */
        veloce_softmax(p,s,T*T,T);
        veloce_matmul_f32(a,p,v,T,T,D);       /* [T,T] x [T,D] -> [T,D] */
        t=now()-t; if(t<best)best=t;
    }}
    printf("BEST_MS %.4f\\nFLOPS %.3f\\n",best*1e3,(2.0*T*T*D+2.0*T*T*D)/best/1e9);
    return 0;
}}
"""
    out, _ = build_and_run("attn", src, extra_flags=pool_flags)
    return {k: float(v) for k, v in (l.split() for l in out.splitlines())}


def bench_membw(quick):
    """Streaming read bandwidth: the hard ceiling for weight-streaming decode."""
    n = 200 * 1024 * 1024
    reps = 5 if quick else 10
    src = f"""
#define _POSIX_C_SOURCE 199309L
#include "veloce_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}}
volatile unsigned long long sink;
int main(void){{
    const size_t n = {n};
    unsigned char *a = malloc(n);
    if (!a) {{ printf("ALLOC_FAIL\\n"); return 1; }}
    for (size_t i = 0; i < n; i++) a[i] = (unsigned char)i;
    double best = 1e9;
    for (int r = 0; r < {reps}; r++) {{
        double t = now(); unsigned long long acc = 0;
        for (size_t i = 0; i < n; i++) acc += a[i];
        sink = acc; t = now() - t; if (t < best) best = t;
    }}
    printf("GBPS %.3f\\n", (double)n / best / 1e9);
    free(a);
    return 0;
}}
"""
    out, _ = build_and_run("membw", src)
    for line in out.splitlines():
        if line.startswith("GBPS"):
            return float(line.split()[1])
    raise RuntimeError("membw failed: " + out)


def bench_compiled_layer(quick, pool_flags):
    """Compile a real 7B-shaped transformer layer through velocec and time it.

    This is a genuine end-to-end measurement: the Veloce compiler plans the
    arena, emits C, and the C driver generates deterministic trit weights,
    then runs `veloce_layer()` repeatedly. It captures kernel + call overhead
    but not the rest of the 32-layer stack.
    """
    sys.path.insert(0, ROOT)
    import veloce
    src = """
module bench_layer
dim D = 4096
dim F = 11008
fn layer(x: f32[1, D], wq: trit[D, D], wk: trit[D, D], wv: trit[D, D],
         wo: trit[D, D], w1: trit[F, D], w2: trit[D, F]) -> f32[1, D] {
    let q = matmul_trit(x, wq)
    let k = matmul_trit(x, wk)
    let v = matmul_trit(x, wv)
    let a = q + k + v
    let o = matmul_trit(a, wo)
    let h = matmul_trit(o, w1)
    let s = silu(h)
    return matmul_trit(s, w2)
}
"""
    reps = 3 if quick else 8
    driver = f"""
#define _POSIX_C_SOURCE 199309L
#include "veloce_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void){{struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}}
float *veloce_layer(float *x, uint8_t *wq, uint8_t *wk, uint8_t *wv,
                    uint8_t *wo, uint8_t *w1, uint8_t *w2);
static void fill(uint8_t *p, size_t nbytes, unsigned seed) {{
    for (size_t i = 0; i < nbytes; i++) p[i] = (uint8_t)((i * 131u + seed * 17u) & 0xFFu);
}}
int main(void) {{
    const size_t D = 4096, F = 11008;
    const size_t bytes_dd = (D * D + 4) / 5, bytes_fd = (F * D + 4) / 5, bytes_df = (D * F + 4) / 5;
    float *x = malloc(D * sizeof(float));
    uint8_t *wq = malloc(bytes_dd), *wk = malloc(bytes_dd), *wv = malloc(bytes_dd);
    uint8_t *wo = malloc(bytes_dd), *w1 = malloc(bytes_fd), *w2 = malloc(bytes_df);
    if (!x || !wq || !wk || !wv || !wo || !w1 || !w2) {{ printf("ALLOC_FAIL\\n"); return 1; }}
    for (size_t i = 0; i < D; i++) x[i] = 0.01f * (float)((int)(i % 13) - 6);
    fill(wq, bytes_dd, 1); fill(wk, bytes_dd, 2); fill(wv, bytes_dd, 3);
    fill(wo, bytes_dd, 4); fill(w1, bytes_fd, 5); fill(w2, bytes_df, 6);
    float *out = veloce_layer(x, wq, wk, wv, wo, w1, w2);   /* warm + first-touch */
    double best = 1e9;
    for (int r = 0; r < {reps}; r++) {{
        double t = now();
        out = veloce_layer(x, wq, wk, wv, wo, w1, w2);
        t = now() - t; if (t < best) best = t;
    }}
    /* MACs per layer: 4*D*D attention projections + 3*D*F MLP */
    double macs = 4.0 * D * D + 3.0 * D * F;
    printf("BEST_MS %.4f\\nGMAC %.3f\\nOUT %g\\n", best * 1e3, macs / best / 1e9, out[0]);
    return 0;
}}
"""
    res = veloce.compile_source(src, module_name="bench_layer")
    out, _ = build_and_run("layer", driver, extra_flags=pool_flags,
                           gen_c=res.c_source)
    vals = {}
    for line in out.splitlines():
        k, v = line.split()
        vals[k] = float(v)
    if vals.get("OUT") != vals.get("OUT"):      # NaN guard for strict JSON
        vals["OUT"] = None
    vals["macs_per_layer"] = 4.0 * 4096 * 4096 + 3.0 * 4096 * 11008
    return vals


def bench_memory():
    """Packed trit vs fp16 vs f32 footprint for the 7B-class model."""
    L, D, F, V = MODEL["layers"], MODEL["d_model"], MODEL["ffn"], MODEL["vocab"]
    attn = 4 * D * D
    mlp = 3 * D * F
    per_layer = attn + mlp
    total = L * per_layer + V * D            # + embedding (float, not counted)
    trit_bytes = total * BYTES_PER_TRIT
    fp16_bytes = total * 2
    f32_bytes = total * 4
    return {
        "params": total,
        "trit_bytes": trit_bytes,
        "fp16_bytes": fp16_bytes,
        "f32_bytes": f32_bytes,
        "vs_fp16_pct": 100.0 * (1.0 - trit_bytes / fp16_bytes),
        "vs_f32_pct": 100.0 * (1.0 - trit_bytes / f32_bytes),
    }


def bench_compile():
    """Compile the transformer example (in-process, no C compiler)."""
    sys.path.insert(0, ROOT)
    import veloce
    path = os.path.join(ROOT, "examples", "transformer_block.vel")
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        veloce.compile_file(path)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_zero_alloc():
    """Inspect generated C for heap allocation calls."""
    sys.path.insert(0, ROOT)
    import veloce
    bad = []
    for name in ("transformer_block", "tiny_lm", "tiny_lm_cls"):
        res = veloce.compile_file(os.path.join(ROOT, "examples", name + ".vel"))
        for token in ("malloc(", "calloc(", "realloc(", "free("):
            if token in res.c_source:
                bad.append((name, token))
    return bad


def estimate_tokens_per_sec(gemv_best_ms, attn_ms, threads):
    """Project single-stream decode throughput for the 7B-class model.

    Decode of one token streams every weight matrix once (GEMV, M=1), so the
    projection is simply total weight MACs / measured GEMV rate. Attention at
    decode time is O(context) per token and is <2% of the matmul work at
    context 2048, so it is excluded here and measured separately in [2].

    This is a projection from measured kernels, not a full model run; it is
    mildly optimistic (no launch overhead, no KV-cache traffic).
    """
    L, D, F = MODEL["layers"], MODEL["d_model"], MODEL["ffn"]
    attn_macs = L * 4 * D * D          # q, k, v, o projections
    mlp_macs = L * 3 * D * F           # gate, up, down
    total_macs = attn_macs + mlp_macs
    rate = (4096.0 * 4096.0) / (gemv_best_ms / 1e3)   # MAC/s, measured
    total_s = total_macs / rate
    return 1.0 / total_s, total_s, rate / 1e9, total_macs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer repetitions")
    ap.add_argument("--json", metavar="PATH", help="write machine-readable results")
    args = ap.parse_args()

    omp = have_openmp()
    pool_flags = ["-fopenmp"] if omp else []
    report = {
        # Stamp the run: these numbers are only meaningful for the host state
        # they were measured on, and an undated file invites comparing a fresh
        # figure against a stale one.
        "measured_at": datetime.datetime.now(datetime.timezone.utc)
                               .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": {"cpu": cpu_model(), "cores": nproc(), "cc": CC,
                    "openmp": omp, "python": platform.python_version()},
        "prd": {"tokens_per_sec": PRD_TOKENS_PER_SEC, "model": PRD_MODEL,
                "compile_seconds": PRD_COMPILE_SECONDS,
                "memory_reduction": PRD_MEMORY_REDUCTION},
    }

    print("=" * 72)
    print("Veloce throughput benchmark")
    print("=" * 72)
    print(f"CPU      : {cpu_model()}")
    print(f"Cores    : {nproc()}")
    print(f"Compiler : {CC}  (OpenMP: {'yes' if omp else 'no'})")
    print(f"PRD goal : >= {PRD_TOKENS_PER_SEC:g} tok/s on a {PRD_MODEL} model "
          f"({MODEL['layers']}L d={MODEL['d_model']} ffn={MODEL['ffn']})")
    print()

    # --- 1. core kernel throughput -------------------------------------
    gemv = bench_trit_gemv(args.quick, pool_flags)
    speedup = gemv["T4_GMAC"] / gemv["T1_GMAC"] if gemv["T1_GMAC"] else 0.0
    print("[1] trit GEMV 4096x4096 (the decoder's dominant kernel)")
    print(f"    1 thread : {gemv['T1_MS']:7.3f} ms   {gemv['T1_GMAC']:5.2f} GMAC/s")
    print(f"    2 threads: {gemv['T2_MS']:7.3f} ms   {gemv['T2_GMAC']:5.2f} GMAC/s   "
          f"({gemv['T2_GMAC']/gemv['T1_GMAC']:.2f}x)")
    print(f"    4 threads: {gemv['T4_MS']:7.3f} ms   {gemv['T4_GMAC']:5.2f} GMAC/s   "
          f"({speedup:.2f}x)")
    report["trit_gemv"] = gemv
    report["trit_gemv_speedup"] = speedup
    print()

    # --- 2. attention ---------------------------------------------------
    attn = bench_attention(args.quick, pool_flags)
    print("[2] attention block: QK^T + softmax + AV  (T=256, d=4096)")
    print(f"    best  : {attn['BEST_MS']:.2f} ms   {attn['FLOPS']:.2f} GFLOP/s")
    report["attention"] = attn
    print()

    mem = bench_memory()

    # --- 2b. memory bandwidth (roofline input) --------------------------
    bw_gbs = bench_membw(args.quick)
    print(f"[2b] streaming read bandwidth: {bw_gbs:.2f} GB/s")
    report["membw_gbps"] = bw_gbs
    print()

    # --- 3. compiled 7B-shaped layer (real end-to-end) -------------------
    layer_runs = [bench_compiled_layer(args.quick, pool_flags) for _ in range(2)]
    layer = min(layer_runs, key=lambda r: r["BEST_MS"])
    layers = MODEL["layers"]
    layer_tps = 1.0 / (layer["BEST_MS"] / 1e3 * layers)
    print("[3] compiled 7B-shaped transformer layer (velocec -> C -> run)")
    print(f"    {layer['BEST_MS']:.2f} ms/layer   {layer['GMAC']:.2f} GMAC/s   "
          f"-> {layer_tps:.2f} tok/s for {layers} layers")
    report["compiled_layer"] = layer
    report["compiled_layer_tokens_per_sec"] = layer_tps
    print()

    # --- 3b. model-level estimate --------------------------------------
    tps, total_s, rate_g, total_macs = estimate_tokens_per_sec(
        gemv["T4_MS"], attn["BEST_MS"], gemv["THREADS"])
    # roofline: decode must stream all weights once per token
    weight_bytes = mem["trit_bytes"]
    roof_s = weight_bytes / (bw_gbs * 1e9)
    roof_tps = 1.0 / roof_s
    gemv_bytes = 4096 * 4096 / 5.0          # one 4096x4096 trit matrix
    achieved_bw = gemv_bytes / (gemv["T4_MS"] / 1e3) / 1e9
    print("[3b] projected 7B-class ternary decode (single stream)")
    print(f"    GEMV rate     : {rate_g:.2f} GMAC/s")
    print(f"    matmul work   : {total_macs/1e9:.2f} GMAC/token")
    print(f"    per token     : {total_s*1e3:.1f} ms  ->  {tps:.2f} tokens/s")
    gap = PRD_TOKENS_PER_SEC / tps if tps else float("inf")
    print(f"    PRD target    : {PRD_TOKENS_PER_SEC:g} tok/s  "
          f"-> {'MET' if tps >= PRD_TOKENS_PER_SEC else f'{gap:.1f}x short'}")
    print(f"    roofline      : {bw_gbs:.2f} GB/s measured read BW "
          f"-> {roof_tps:.2f} tok/s ceiling on this machine")
    print(f"    GEMV BW       : {achieved_bw:.2f} GB/s "
          f"({100.0*achieved_bw/bw_gbs:.0f}% of streaming peak)")
    report["projection"] = {"tokens_per_sec": tps, "ms_per_token": total_s * 1e3,
                            "gemv_gmacs": rate_g, "macs_per_token": total_macs,
                            "gap_x": (PRD_TOKENS_PER_SEC / tps) if tps else None,
                            "roofline_tokens_per_sec": roof_tps,
                            "achieved_gbps": achieved_bw, "peak_gbps": bw_gbs}
    print()

    # --- 4. memory ------------------------------------------------------
    print("[4] memory footprint, 7B-class ternary weights")
    print(f"    params        : {mem['params']/1e9:.1f} B")
    print(f"    trit packed   : {mem['trit_bytes']/1e9:.1f} GB")
    print(f"    fp16 baseline : {mem['fp16_bytes']/1e9:.1f} GB  "
          f"({mem['vs_fp16_pct']:.1f}% saved)")
    print(f"    f32 baseline  : {mem['f32_bytes']/1e9:.1f} GB  "
          f"({mem['vs_f32_pct']:.1f}% saved)")
    print(f"    PRD target    : >= 80% vs fp16 -> "
          f"{'MET' if mem['vs_fp16_pct'] >= 80 else 'NOT MET'}")
    report["memory"] = mem
    print()

    # --- 5. compile speed / zero alloc ---------------------------------
    compile_s = bench_compile()
    bad = bench_zero_alloc()
    print("[5] compiler")
    print(f"    transformer_block.vel compile: {compile_s*1e3:.1f} ms  "
          f"(PRD < {PRD_COMPILE_SECONDS:g} s -> "
          f"{'MET' if compile_s < PRD_COMPILE_SECONDS else 'NOT MET'})")
    print(f"    heap allocation in generated C: "
          f"{'none (zero-allocation)' if not bad else bad}")
    report["compile"] = {"seconds": compile_s, "heap_calls": bad}
    print()

    print("=" * 72)
    print("SUMMARY vs PRD")
    print("=" * 72)
    rows = [
        ("FR-4.2 parallel speedup > 1", speedup > 1.05, f"{speedup:.2f}x on {nproc()} cores"),
        ("Memory >=80% smaller than fp16", mem["vs_fp16_pct"] >= 80,
         f"{mem['vs_fp16_pct']:.1f}%"),
        ("Zero heap allocation", not bad, "no malloc/calloc/realloc/free"),
        ("Compile < 2 s", compile_s < PRD_COMPILE_SECONDS, f"{compile_s*1e3:.0f} ms"),
        (f"Speed >= {PRD_TOKENS_PER_SEC:g} tok/s (7B-class)", tps >= PRD_TOKENS_PER_SEC,
         f"{tps:.2f} tok/s projected / {layer_tps:.2f} compiled (HW ceiling {roof_tps:.2f})"),
    ]
    for label, ok, detail in rows:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<40s} {detail}")
    report["summary"] = [{"requirement": l, "pass": bool(o), "measured": d}
                         for l, o, d in rows]

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
