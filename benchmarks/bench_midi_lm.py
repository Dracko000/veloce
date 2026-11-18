#!/usr/bin/env python3
"""End-to-end mid-size ternary LM: real weights, real mmap, measured tok/s.

This is the honest answer to "how fast is Veloce actually, on a real model?".
It builds a genuine 275.6M-ternary-parameter decoder (6 layers, d_model=1536,
ffn=6144, vocab=32000), writes it to a ``.veloce`` file with the production
packing code, renders + compiles the module with ``veloce.lmmodule`` and runs
the *generated C* against the mmap'd weights.  Every number printed is wall-clock measured on
this machine -- nothing is projected from a synthetic microbenchmark.

What is measured
----------------
* tok/s for a single-token forward pass (T=1, no KV cache), median of N runs
* the same binary at VELOCE_THREADS=1/2/4 (FR-4.2: bit-exact across counts)
* bytes streamed per token vs. the file's packed weight bytes
* optional NumPy reference check (``--verify``) of the produced logits
* the same model decoding N tokens through a per-layer KV cache (``--cached``,
  64 slots / 32 tokens by default); ``make bench-midi`` always passes it so the
  committed JSON carries both the eager and the cached numbers

Usage
-----
    python3 benchmarks/bench_midi_lm.py                  # full run
    python3 benchmarks/bench_midi_lm.py --iters 5        # fewer timed runs
    python3 benchmarks/bench_midi_lm.py --rebuild        # regenerate weights
    python3 benchmarks/bench_midi_lm.py --verify         # NumPy cross-check
    python3 benchmarks/bench_midi_lm.py --json out.json
    python3 benchmarks/bench_midi_lm.py --cached        # + KV-cache decode
    python3 benchmarks/bench_midi_lm.py --config D=2048,F=5632,layers=7

``--config`` takes ``key=value`` pairs (``V``, ``D``, ``F``, ``layers``,
``scale``, ``name``) and ``--model-dir`` sends both the generated module and
its ``.veloce`` weights there, so several shapes can be measured from one
command without a hand-written ``.vel`` file per shape.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import veloce  # noqa: E402
from veloce import weights as vw  # noqa: E402
from veloce.codegen import ctype_of  # noqa: E402
from veloce.lmmodule import (DEFAULT_CONFIG, attn_scale,  # noqa: E402
                             cache_buffers, cached_module, decoder_module,
                             normalize_config)

CC = os.environ.get("CC", "cc")
RUNTIME = os.path.join(ROOT, "runtime")
BUILD = os.path.join(ROOT, "build")

MODEL = dict(DEFAULT_CONFIG)
EPS = 1e-5

CFLAGS = ["-std=c11", "-O3", "-march=native", "-ffast-math", "-pthread"]


# ---------------------------------------------------------------------------
# deterministic weight generation
# ---------------------------------------------------------------------------
def rng_for(name):
    """Stable per-tensor RNG (Python's hash() is salted, crc32 is not)."""
    return np.random.default_rng(zlib.crc32(name.encode("utf-8")))


def rand_trits(name, shape):
    return rng_for(name).integers(-1, 2, size=shape, dtype=np.int8)


def rand_norm(name, D):
    return (1.0 + 0.02 * rng_for(name).standard_normal(D)).astype(np.float32)


def rand_emb(name, V, D):
    a = rng_for(name).standard_normal((V, D), dtype=np.float32)
    return (a * np.float32(0.02)).astype(np.float16)


def tensor_specs(cfg):
    """Yield (name, dtype, factory) in the order the model file stores them."""
    V, D, F, L = cfg["V"], cfg["D"], cfg["F"], cfg["layers"]
    yield ("emb", "fp16", lambda: rand_emb("emb", V, D))
    for l in range(1, L + 1):
        yield (f"nw{l}", "f32", lambda l=l: rand_norm(f"nw{l}", D))
        for n in ("wq", "wk", "wv", "wo"):
            yield (f"{n}{l}", "trit", lambda n=n, l=l: rand_trits(f"{n}{l}", (D, D)))
        yield (f"w1{l}", "trit", lambda l=l: rand_trits(f"w1{l}", (F, D)))
        yield (f"w2{l}", "trit", lambda l=l: rand_trits(f"w2{l}", (D, F)))
        yield (f"w3{l}", "trit", lambda l=l: rand_trits(f"w3{l}", (F, D)))
    yield ("nwf", "f32", lambda: rand_norm("nwf", D))
    yield ("head", "trit", lambda: rand_trits("head", (V, D)))


def param_counts(cfg):
    V, D, F, L = cfg["V"], cfg["D"], cfg["F"], cfg["layers"]
    per_layer = 4 * D * D + 3 * F * D
    return {
        "ternary": per_layer * L + V * D,
        "embedding": V * D,
        "norm": D * (L + 1),
        "layer_ternary": per_layer,
        "head_ternary": V * D,
    }


def model_card_name(cfg):
    """Stable card name derived from the shape, e.g. ``midi-lm-275m-demo``."""
    trits = param_counts(cfg)["ternary"]
    m = trits / 1e6
    tag = f"{m:.0f}m" if m >= 10 else f"{m:.1f}m"
    return f"midi-lm-{tag}-demo"


def parse_config(pairs):
    """``["D=2048", "layers=7"]`` -> cfg dict merged onto the defaults."""
    cfg = dict(MODEL)
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--config expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k == "name":
            cfg["name"] = v.strip()
        elif k in ("V", "D", "F", "layers"):
            cfg[k] = int(v)
        elif k == "scale":
            cfg["scale"] = float(v)
        else:
            raise SystemExit(f"unknown config key {k!r} "
                             f"(known: V, D, F, layers, scale, name)")
    return normalize_config(cfg)


def build_weights(path, cfg, rebuild=False):
    """Generate + write the .veloce file. Returns (manifest, seconds, rebuilt)."""
    if os.path.exists(path) and not rebuild:
        man = vw.read_veloce_manifest(path)
        if man.get("model_card") is not None:
            return man, 0.0, False
        # Older cache without a model card: rewrite it once.
        rebuild = True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.perf_counter()
    from veloce.capabilities import make_model_card
    card = make_model_card(
        model_card_name(cfg),
        model_type="ternary-transformer (untrained throughput demonstrator)",
        params={
            "ternary": param_counts(cfg)["ternary"],
            "embedding": param_counts(cfg)["embedding"],
            "norm": param_counts(cfg)["norm"],
        },
        training={
            "status": "not trained",
            "weights": "deterministic pseudo-random (per-tensor RNG)",
            "purpose": "end-to-end throughput measurement, not quality",
        },
    )
    vw.save_veloce(path, ((n, fn(), d) for (n, d, fn) in tensor_specs(cfg)),
                   model_card=card)
    dt = time.perf_counter() - t0
    return vw.read_veloce_manifest(path), dt, True


# ---------------------------------------------------------------------------
# C driver + generated module
# ---------------------------------------------------------------------------
def write_offsets_header(man, path):
    lines = ["/* generated by benchmarks/bench_midi_lm.py -- do not edit */",
             "#ifndef MIDI_LM_OFFSETS_H", "#define MIDI_LM_OFFSETS_H"]
    for e in man["tensors"]:
        lines.append(f"#define OFF_{e['name']} {e['offset']}ULL")
        lines.append(f"#define NB_{e['name']} {e['nbytes']}ULL")
    lines.append("#endif")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def extern_decl(module, fname="forward"):
    """C prototype for the module's `fname`, derived from the parsed AST."""
    fn = next(f for f in module.functions if f.name == fname)
    params = ", ".join(f"{ctype_of(p.type)} {p.cname}" for p in fn.params)
    return f"{ctype_of(fn.ret_types[0])} veloce_{fname}({params});"


def call_args(module):
    """Pointer expressions into the mmap'd file, in `forward` param order."""
    fn = next(f for f in module.functions if f.name == "forward")
    out = []
    for p in fn.params:
        if isinstance(p.type, veloce.ast_mod.TensorType):
            out.append(f"({ctype_of(p.type)})(base + OFF_{p.name})")
        else:
            out.append("1")          # tok
    return ",\n        ".join(out)


DRIVER_TEMPLATE = r"""
#define _GNU_SOURCE
#include "veloce_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "midi_lm_offsets.h"

__EXTERN__

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static uint64_t fnv1a(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 7;
    const char *wpath = argc > 2 ? argv[2] : "build/midi_lm.veloce";
    const char *dump = argc > 3 ? argv[3] : "-";
    const int V = __V__;

    int fd = open(wpath, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    uint8_t *base = (uint8_t *)mmap(NULL, (size_t)st.st_size, PROT_READ,
                                    MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    /* Two untimed warmups: fault the mapping in and warm the worker pool. */
    float *out = NULL;
    for (int i = 0; i < 2; i++)
        out = veloce_forward(
        __ARGS__);

    double *t = (double *)malloc(sizeof(double) * (size_t)iters);
    if (!t) return 1;
    uint64_t hash = 0;
    for (int i = 0; i < iters; i++) {
        double t0 = now_ns();
        out = veloce_forward(
        __ARGS__);
        double t1 = now_ns();
        t[i] = t1 - t0;
        hash = fnv1a(out, (size_t)V * sizeof(float));
        if (i == 0 && strcmp(dump, "-") != 0) {
            FILE *f = fopen(dump, "wb");
            if (f) { fwrite(out, sizeof(float), (size_t)V, f); fclose(f); }
        }
    }

    /* Determinism: same token twice must be byte-identical. */
    float *snap = (float *)malloc((size_t)V * sizeof(float));
    float *a = veloce_forward(
        __ARGS__);
    memcpy(snap, a, (size_t)V * sizeof(float));
    float *b = veloce_forward(
        __ARGS__);
    int det = memcmp(snap, b, (size_t)V * sizeof(float)) == 0;

    int nonfinite = 0, argmax = 0;
    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        /* bit-level check: -ffast-math folds isfinite() away */
        unsigned bits;
        memcpy(&bits, &snap[i], 4);
        if ((bits & 0x7f800000u) == 0x7f800000u) nonfinite++;
        sum += (double)snap[i];
        if (snap[i] > snap[argmax]) argmax = i;
    }

    double *sorted = (double *)malloc(sizeof(double) * (size_t)iters);
    memcpy(sorted, t, sizeof(double) * (size_t)iters);
    qsort(sorted, (size_t)iters, sizeof(double), cmp_double);
    double median = sorted[iters / 2];
    double mean = 0.0;
    for (int i = 0; i < iters; i++) mean += t[i];
    mean /= iters;
    double best = sorted[0];

    printf("BENCH threads=%s iters=%d median_ms=%.4f best_ms=%.4f mean_ms=%.4f "
           "tok_s=%.2f hash=%016llx det=%d argmax=%d sum=%.6f nonfinite=%d "
           "file_bytes=%lld\n",
           getenv("VELOCE_THREADS") ? getenv("VELOCE_THREADS") : "auto",
           iters, median / 1e6, best / 1e6, mean / 1e6, 1e9 / median,
           (unsigned long long)hash, det, argmax, sum, nonfinite,
           (long long)st.st_size);
    return det && nonfinite == 0 ? 0 : 2;
}
"""


def build_and_run(module, c_source, man, cfg, wpath, iters, dump, extra_flags,
                  workdir=None):
    """Compile the generated module + a C driver and time it.

    `workdir` isolates artefacts per shape, so repeated ``--config`` runs do
    not overwrite each other's offsets header or binary.
    """
    work = workdir or BUILD
    os.makedirs(work, exist_ok=True)
    hdr = os.path.join(work, "midi_lm_offsets.h")
    write_offsets_header(man, hdr)

    gen_c = os.path.join(work, "midi_lm.c")
    with open(gen_c, "w") as f:
        f.write(c_source)

    src = (DRIVER_TEMPLATE
           .replace("__EXTERN__", extern_decl(module))
           .replace("__ARGS__", call_args(module))
           .replace("__V__", str(cfg["V"])))
    drv = os.path.join(work, "midi_lm_driver.c")
    with open(drv, "w") as f:
        f.write(src)

    exe = os.path.join(work, "midi_lm_bench")
    cmd = [CC, *CFLAGS, *extra_flags, "-I", RUNTIME, "-I", work,
           drv, gen_c, "-lm", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("build failed:\n" + r.stderr)

    runs = []
    for threads in ("1", "2", "3", "4"):
        env = dict(os.environ, VELOCE_THREADS=threads)
        r = subprocess.run([exe, str(iters), wpath, dump], capture_output=True,
                           text=True, env=env)
        if r.returncode not in (0, 2):
            raise SystemExit(f"run failed (threads={threads}):\n{r.stderr}")
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("BENCH")), None)
        if line is None:
            raise SystemExit(f"no BENCH line:\n{r.stdout}\n{r.stderr}")
        rec = {}
        for kv in line.split()[1:]:
            k, v = kv.split("=", 1)
            rec[k] = v
        rec["threads"] = threads
        for k in ("median_ms", "best_ms", "mean_ms", "tok_s", "sum",
                  "file_bytes"):
            rec[k] = float(rec[k])
        for k in ("iters", "det", "argmax", "nonfinite"):
            rec[k] = int(rec[k])
        rec["hash"] = int(rec["hash"], 16)
        runs.append(rec)
    return runs, exe


CACHED_DRIVER_TEMPLATE = r"""
#define _GNU_SOURCE
#include "veloce_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "midi_lm_offsets.h"

__EXTERN__

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static uint64_t fnv1a(const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

__CACHE_DECLS__

static int32_t mask_buf[__CT__];

static const int toks[__NTOK__] = { __TOKS__ };

int main(int argc, char **argv) {
    const char *wpath = argc > 1 ? argv[1] : "build/midi_lm.veloce";
    const char *dump = argc > 2 ? argv[2] : "-";
    const int V = __V__, CT = __CT__;

    int fd = open(wpath, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("fstat"); return 1; }
    uint8_t *base = (uint8_t *)mmap(NULL, (size_t)st.st_size, PROT_READ,
                                    MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); return 1; }

    /* mask_buf[i] != 0 marks a slot that participates in softmax; hidden
       slots are excluded from the max and the normaliser entirely, so their
       content can never affect a score. */
    for (int i = 0; i < CT; i++) mask_buf[i] = 0;

    float *out = NULL;
    /* Warmup: two steps at pos 0, which writes the same row twice. */
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < CT; i++) mask_buf[i] = (i <= 0) ? 1 : 0;
        out = veloce_forward_cached(
        __CARGS__);
    }

    /* Untimed pass for bit-exact determinism of the whole decode. */
    __CACHE_ZERO__
    for (int i = 0; i < CT; i++) mask_buf[i] = 0;
    for (int t = 0; t < __NTOK__; t++) {
        for (int i = 0; i < CT; i++) mask_buf[i] = (i <= t) ? 1 : 0;
        out = veloce_forward_cached(
        __CARGS__);
    }
    float *ref = (float *)malloc((size_t)V * sizeof(float));
    memcpy(ref, out, (size_t)V * sizeof(float));

    __CACHE_ZERO__
    for (int i = 0; i < CT; i++) mask_buf[i] = 0;
    double t0 = now_ns();
    for (int t = 0; t < __NTOK__; t++) {
        for (int i = 0; i < CT; i++) mask_buf[i] = (i <= t) ? 1 : 0;
        out = veloce_forward_cached(
        __CARGS__);
    }
    double t1 = now_ns();
    int det = memcmp(ref, out, (size_t)V * sizeof(float)) == 0;

    int nonfinite = 0, argmax = 0;
    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        unsigned bits;
        memcpy(&bits, &out[i], 4);
        if ((bits & 0x7f800000u) == 0x7f800000u) nonfinite++;
        sum += (double)out[i];
        if (out[i] > out[argmax]) argmax = i;
    }
    if (strcmp(dump, "-") != 0) {
        FILE *f = fopen(dump, "wb");
        if (f) { fwrite(out, sizeof(float), (size_t)V, f); fclose(f); }
    }

    printf("CACHED threads=%s tokens=%d ms=%.4f per_token_ms=%.4f tok_s=%.2f "
           "hash=%016llx det=%d argmax=%d sum=%.6f nonfinite=%d\n",
           getenv("VELOCE_THREADS") ? getenv("VELOCE_THREADS") : "auto",
           __NTOK__, (t1 - t0) / 1e6, (t1 - t0) / 1e6 / __NTOK__,
           __NTOK__ * 1e9 / (t1 - t0),
           (unsigned long long)fnv1a(out, (size_t)V * sizeof(float)),
           det, argmax, sum, nonfinite);
    return det && nonfinite == 0 ? 0 : 2;
}
"""


def cached_artefacts(module, cfg, ct):
    """(C decls for the KV caches, arg list, zero-fill statement)."""
    bufs = cache_buffers(cfg, ct)
    decls = "\n".join(f"static float {n}[{ct * cfg['D']}] "
                      f"__attribute__((aligned(64)));" for n, _, _ in bufs)
    zero = "\n    ".join(f"memset({n}, 0, sizeof({n}));" for n, _, _ in bufs)
    host = {n for n, _, _ in bufs} | {"mask"}

    fn = next(f for f in module.functions if f.name == "forward_cached")
    args = []
    for p in fn.params:
        if p.name in host:
            args.append("mask_buf" if p.name == "mask" else p.name)
        elif isinstance(p.type, veloce.ast_mod.TensorType):
            args.append(f"({ctype_of(p.type)})(base + OFF_{p.name})")
        elif p.name == "tok":
            args.append("toks[t]")
        elif p.name == "pos":
            args.append("t")
        else:
            raise SystemExit(f"unexpected forward_cached parameter {p.name!r}")
    return decls, ",\n        ".join(args), zero


def build_and_run_cached(man, cfg, wpath, ct, ntok, dump, workdir):
    """Build and time the cached decoder: N tokens through a real KV cache."""
    import random
    rnd = random.Random(1234)
    toks = [rnd.randrange(cfg["V"]) for _ in range(ntok)]

    src = cached_module(cfg, ct)
    vel_path = os.path.join(workdir, cfg["name"] + "_cached.vel")
    with open(vel_path, "w") as f:
        f.write(src)
    res = veloce.compile_file(vel_path)

    os.makedirs(workdir, exist_ok=True)
    hdr = os.path.join(workdir, "midi_lm_offsets.h")
    write_offsets_header(man, hdr)
    gen_c = os.path.join(workdir, "midi_lm_cached.c")
    with open(gen_c, "w") as f:
        f.write(res.c_source)

    decls, cargs, zero = cached_artefacts(res.module, cfg, ct)
    drv_src = (CACHED_DRIVER_TEMPLATE
               .replace("__EXTERN__", extern_decl(res.module, "forward_cached"))
               .replace("__CACHE_DECLS__", decls)
               .replace("__CACHE_ZERO__", zero)
               .replace("__CARGS__", cargs)
               .replace("__CT__", str(ct))
               .replace("__NTOK__", str(ntok))
               .replace("__TOKS__", ", ".join(str(t) for t in toks))
               .replace("__V__", str(cfg["V"])))
    drv = os.path.join(workdir, "midi_lm_cached_driver.c")
    with open(drv, "w") as f:
        f.write(drv_src)

    exe = os.path.join(workdir, "midi_lm_cached")
    cmd = [CC, *CFLAGS, "-I", RUNTIME, "-I", workdir,
           drv, gen_c, "-lm", "-o", exe]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("cached build failed:\n" + r.stderr)

    runs = []
    for threads in ("1", "2", "3", "4"):
        env = dict(os.environ, VELOCE_THREADS=threads)
        r = subprocess.run([exe, wpath, dump], capture_output=True, text=True,
                           env=env)
        if r.returncode not in (0, 2):
            raise SystemExit(f"cached run failed (threads={threads}):\n{r.stderr}")
        line = next((ln for ln in r.stdout.splitlines()
                     if ln.startswith("CACHED")), None)
        if line is None:
            raise SystemExit(f"no CACHED line:\n{r.stdout}\n{r.stderr}")
        rec = {"threads": threads}
        for kv in line.split()[1:]:
            k, v = kv.split("=", 1)
            rec[k] = v
        for k in ("ms", "per_token_ms", "tok_s", "sum"):
            rec[k] = float(rec[k])
        for k in ("tokens", "det", "argmax", "nonfinite"):
            rec[k] = int(rec[k])
        rec["hash"] = int(rec["hash"], 16)
        runs.append(rec)
    return runs, toks, res


def reference_cached_logits(vf, toks, cfg, ct):
    """NumPy reference for the whole cached decode (masked attention)."""
    V, D, L = cfg["V"], cfg["D"], cfg["layers"]
    scale = np.float32(attn_scale(cfg))
    ck = [np.zeros((ct, D)) for _ in range(L)]
    cv = [np.zeros((ct, D)) for _ in range(L)]
    head = vf.numpy("head")
    x = None
    for t, tok in enumerate(toks):
        x = vf.numpy("emb")[tok].astype(np.float64)
        for l in range(1, L + 1):
            nw = vf.numpy(f"nw{l}").astype(np.float64)
            xn = _rmsnorm(x, nw, D)
            k = _tritmm(xn, vf.numpy(f"wk{l}"))
            v = _tritmm(xn, vf.numpy(f"wv{l}"))
            q = _tritmm(xn, vf.numpy(f"wq{l}"))
            ck[l - 1][t] = k
            cv[l - 1][t] = v
            scores = (q @ ck[l - 1][:t + 1].T) * scale
            e = np.exp(scores - scores.max())
            p = e / e.sum()
            a = p @ cv[l - 1][:t + 1]
            o = _tritmm(a, vf.numpy(f"wo{l}"))
            x = x + o
            xn2 = _rmsnorm(x, nw, D)
            g = _tritmm(xn2, vf.numpy(f"w1{l}"))
            u = _tritmm(xn2, vf.numpy(f"w3{l}"))
            x = x + _tritmm(_silu(g) * u, vf.numpy(f"w2{l}"))
    xn = _rmsnorm(x, vf.numpy("nwf").astype(np.float64), D)
    return _tritmm(xn, head)


# ---------------------------------------------------------------------------
# optional NumPy reference
# ---------------------------------------------------------------------------
def _rmsnorm(x, w, D):
    ss = float(np.dot(x, x))
    inv = 1.0 / np.sqrt(ss / D + EPS)
    return x * np.float32(inv) * w


def _tritmm(x, w):
    return x @ w.T.astype(np.float32)


def _silu(x):
    # Numerically stable SiLU: split by sign so exp() never overflows.
    # For x >= 0: x / (1 + e^-x);  for x < 0: x * e^x / (1 + e^x).
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = x[~pos] * ex / (1.0 + ex)
    return out


def reference_logits(vf, tok, cfg):
    V, D = cfg["V"], cfg["D"]
    cfg_scale = attn_scale(cfg)
    x = vf.numpy("emb")[tok].astype(np.float32)
    for l in range(1, cfg["layers"] + 1):
        nw = vf.numpy(f"nw{l}")
        xn = _rmsnorm(x, nw, D)
        q = _tritmm(xn, vf.numpy(f"wq{l}"))
        k = _tritmm(xn, vf.numpy(f"wk{l}"))
        v = _tritmm(xn, vf.numpy(f"wv{l}"))
        scores = (np.array([float(q @ k)], dtype=np.float32)
                  * np.float32(cfg_scale))
        probs = np.exp(scores - scores.max())
        probs = probs / probs.sum()
        # T=1: probs has shape (1,); keep an explicit (1,1) @ (1,D) product
        attn = (probs.reshape(1, 1) @ v.reshape(1, D))[0]
        x = x + _tritmm(attn, vf.numpy(f"wo{l}"))
        xn2 = _rmsnorm(x, nw, D)
        g = _tritmm(xn2, vf.numpy(f"w1{l}"))
        u = _tritmm(xn2, vf.numpy(f"w3{l}"))
        h = _silu(g) * u
        x = x + _tritmm(h, vf.numpy(f"w2{l}"))
    xn = _rmsnorm(x, vf.numpy("nwf"), D)
    return _tritmm(xn, vf.numpy("head"))


def verify(wpath, dump, cfg):
    if not os.path.exists(dump):
        print("verify: no logits dump, skipped")
        return None
    got = np.fromfile(dump, dtype=np.float32)
    with vw.VeloceFile(wpath) as vf:
        ref = reference_logits(vf, 1, cfg)
    abs_err = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(ref))) or 1.0
    rel = abs_err / scale
    ok = rel < 2e-2
    print(f"verify: max|veloce-numpy| = {abs_err:.3e}  "
          f"(relative {rel:.2e}, logit scale {scale:.3f})  "
          f"{'PASS' if ok else 'FAIL'}")
    return {"abs_err": abs_err, "rel_err": rel, "scale": scale, "pass": bool(ok)}


# ---------------------------------------------------------------------------
def _pread_range(fd, start, end, chunk=8 << 20):
    pos, n = start, 0
    while pos < end:
        b = os.pread(fd, min(chunk, end - pos), pos)
        if not b:
            break
        pos += len(b)
        n += len(b)
    return n


def measure_read_bw(path, threads, reps=3):
    """Best-of-N sequential read bandwidth of `path`, using `threads` readers."""
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            if threads <= 1:
                _pread_range(fd, 0, size)
            else:
                step = (size + threads - 1) // threads
                ths = [threading.Thread(target=_pread_range,
                                        args=(fd, i * step, min(size, (i + 1) * step)))
                       for i in range(threads)]
                for t in ths:
                    t.start()
                for t in ths:
                    t.join()
            dt = time.perf_counter() - t0
            if best is None or dt < best:
                best = dt
        return size / best / 1e9
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=7, help="timed forward passes")
    ap.add_argument("--rebuild", action="store_true", help="regenerate weights")
    ap.add_argument("--verify", action="store_true", help="NumPy reference check")
    ap.add_argument("--json", help="write results to this JSON file")
    ap.add_argument("--weights", default=None,
                    help=".veloce weight file (default: <model-dir>/<name>.veloce)")
    ap.add_argument("--model-dir", default=BUILD,
                    help="where the generated module and weights live")
    ap.add_argument("--config", action="append", metavar="KEY=VAL",
                    help="override V/D/F/layers/scale/name (repeatable)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="write module + weights, skip building/running")
    ap.add_argument("--cached", action="store_true",
                    help="also decode N tokens through a real KV cache")
    ap.add_argument("--cache-slots", type=int, default=64,
                    help="KV-cache length for --cached (default 64)")
    ap.add_argument("--tokens", type=int, default=32,
                    help="tokens to decode in --cached mode (default 32)")
    args = ap.parse_args()

    cfg = parse_config(args.config)
    os.makedirs(args.model_dir, exist_ok=True)
    model_file = os.path.join(args.model_dir, cfg["name"] + ".veloce")
    if args.weights is None:
        args.weights = model_file
    vel_path = os.path.join(args.model_dir, cfg["name"] + ".vel")
    with open(vel_path, "w") as f:
        f.write(decoder_module(cfg))
    print(f"module           : {vel_path} ({cfg['layers']} layers)")
    if args.prepare_only:
        man, gen_s, rebuilt = build_weights(args.weights, cfg, args.rebuild)
        print(f"weights          : {man['total_bytes']/1e6:.1f} MB"
              f"{' [generated]' if rebuilt else ' [cached]'}")
        return 0
    counts = param_counts(cfg)

    t0 = time.perf_counter()
    man, gen_s, rebuilt = build_weights(args.weights, cfg, args.rebuild)
    file_bytes = man["total_bytes"]

    bytes_by_dtype = {}
    for e in man["tensors"]:
        bytes_by_dtype[e["dtype"]] = bytes_by_dtype.get(e["dtype"], 0) + e["nbytes"]
    trit_bytes = bytes_by_dtype.get("trit", 0)
    emb_row_bytes = cfg["D"] * 2
    streamed = trit_bytes + emb_row_bytes + bytes_by_dtype.get("f32", 0)

    print("=" * 78)
    print("Veloce mid-size ternary LM -- end-to-end measured throughput")
    print("=" * 78)
    print(f"model            : {cfg['layers']} layers, d_model={cfg['D']}, "
          f"ffn={cfg['F']}, vocab={cfg['V']}, T=1 (no KV cache)")
    print(f"ternary params   : {counts['ternary']/1e6:.1f}M trits "
          f"({counts['layer_ternary']*cfg['layers']/1e6:.1f}M layers + "
          f"{counts['head_ternary']/1e6:.1f}M head)")
    print(f"fp16 embedding   : {counts['embedding']/1e6:.1f}M values "
          f"({bytes_by_dtype.get('fp16', 0)/1e6:.1f} MB)")
    print(f"packed weights   : {trit_bytes/1e6:.1f} MB trits + "
          f"{bytes_by_dtype.get('f32', 0)/1e6:.2f} MB f32 norms")
    print(f"model file       : {file_bytes/1e6:.1f} MB "
          f"({args.weights}{' [generated]' if rebuilt else ' [cached]'})")
    if rebuilt:
        print(f"weight generation: {gen_s:.1f} s (packing + .veloce write)")
    print(f"streamed / token : {streamed/1e6:.1f} MB (all trits + one emb row)")
    print()

    res = veloce.compile_file(vel_path)
    print(res.report())
    print()

    work = os.path.join(BUILD, cfg["name"])
    dump = os.path.join(work, "midi_lm_logits.bin")
    runs, exe = build_and_run(res.module, res.c_source, man, cfg,
                              args.weights, args.iters, dump, [],
                              workdir=work)

    def show(title, runs):
        print(f"-- {title} --")
        print(f"{'threads':>8} {'median ms':>10} {'best ms':>9} {'tok/s':>8} "
              f"{'hash':>18} {'det':>4}")
        for r in runs:
            print(f"{r['threads']:>8} {r['median_ms']:>10.3f} {r['best_ms']:>9.3f} "
                  f"{r['tok_s']:>8.2f} {r['hash']:016x} {r['det']:>4}")
        print()

    show("default runtime", runs)

    hashes = {r["hash"] for r in runs}
    det_ok = all(r["det"] == 1 for r in runs) and len(hashes) == 1
    print(f"determinism: {'PASS' if det_ok else 'FAIL'} -- "
          f"{len(hashes)} distinct logit hash(es) across {len(runs)} runs, "
          f"same-token reruns byte-identical")

    best = max(r["tok_s"] for r in runs)
    stream_gbs = streamed * best / 1e9
    print(f"best measured    : {best:.2f} tok/s "
          f"({stream_gbs:.2f} GB/s effective read bandwidth)")

    # Same-run streaming probe: how fast can this host read the model file
    # right now?  Without it a tok/s number is uninterpretable on a shared VM.
    probe = {}
    for th in sorted({int(r["threads"]) for r in runs}):
        probe[th] = measure_read_bw(args.weights, th)
    print()
    print("-- streaming read probe (same run, page cache warm) --")
    for th in sorted(probe):
        print(f"{th:>8} threads {probe[th]:>7.2f} GB/s")
    print()
    print(f"{'threads':>8} {'model GB/s':>11} {'probe GB/s':>11} {'% of probe':>11}")
    for r in runs:
        gbs = streamed * r["tok_s"] / 1e9
        p = probe.get(int(r["threads"]), 0.0)
        r["effective_gbs"] = gbs
        r["probe_gbs"] = p
        r["pct_of_probe"] = 100.0 * gbs / p if p else 0.0
    for r in runs:
        print(f"{r['threads']:>8} {r['effective_gbs']:>11.2f} "
              f"{r['probe_gbs']:>11.2f} {r['pct_of_probe']:>10.1f}%")
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = float("nan")
    print(f"host load average: {load:.2f} (1 min) on {os.cpu_count()} CPUs")

    ver = verify(args.weights, dump, cfg) if args.verify else None

    cached = None
    if args.cached:
        ct = args.cache_slots
        if args.tokens > ct:
            raise SystemExit(f"--tokens {args.tokens} exceeds --cache-slots {ct}")
        cdump = os.path.join(work, "midi_lm_cached_logits.bin")
        cruns, toks, cres = build_and_run_cached(man, cfg, args.weights, ct,
                                                 args.tokens, cdump, work)
        print()
        print(f"-- cached decode: {args.tokens} tokens through a "
              f"{ct}-slot KV cache per layer --")
        print(f"{'threads':>8} {'total ms':>10} {'ms/token':>10} {'tok/s':>8} "
              f"{'hash':>18} {'det':>4}")
        for r in cruns:
            print(f"{r['threads']:>8} {r['ms']:>10.2f} {r['per_token_ms']:>10.3f} "
                  f"{r['tok_s']:>8.2f} {r['hash']:016x} {r['det']:>4}")
        cbest = max(r["tok_s"] for r in cruns)
        chashes = {r["hash"] for r in cruns}
        cdet = all(r["det"] == 1 for r in cruns) and len(chashes) == 1
        cnonfinite = sum(r["nonfinite"] for r in cruns)
        print(f"cached best      : {cbest:.2f} tok/s "
              f"({1000.0 / cbest:.2f} ms/token, {args.tokens} tokens)")
        print(f"cached determinism: {'PASS' if cdet else 'FAIL'} -- "
              f"{len(chashes)} distinct hash(es), nonfinite={cnonfinite}")
        cver = None
        if args.verify:
            ref = reference_cached_logits(
                vw.VeloceFile(args.weights), toks, cfg, ct)
            got = np.fromfile(cdump, dtype=np.float32)
            cerr = float(np.max(np.abs(got - ref)))
            cscale = float(np.max(np.abs(ref))) or 1.0
            cok = (cerr / cscale) < 2e-2
            print(f"cached verify    : max|veloce-numpy| = {cerr:.3e} "
                  f"(relative {cerr / cscale:.2e}, logit scale {cscale:.3f})  "
                  f"{'PASS' if cok else 'FAIL'}")
            cver = {"abs_err": cerr, "rel_err": cerr / cscale,
                    "scale": cscale, "pass": bool(cok)}
        print(f"cached vs eager  : {cbest:.2f} vs {best:.2f} tok/s "
              f"({cbest / best:.2f}x)")
        cached = {"slots": ct, "tokens": args.tokens, "runs": cruns,
                  "best_tok_s": cbest, "deterministic": cdet,
                  "nonfinite": cnonfinite, "verify": cver,
                  "speedup_vs_eager": cbest / best}

    if args.json:
        payload = {
            # These numbers only mean something for the host state they were
            # measured on; an undated file invites stale comparisons.
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": cfg,
            "params": counts,
            "file_bytes": file_bytes,
            "trit_bytes": trit_bytes,
            "streamed_per_token": streamed,
            "weight_generation_s": gen_s,
            "runs": runs,
            "best_tok_s": best,
            "effective_gbs": stream_gbs,
            "probe_gbs": probe,
            "host_loadavg_1m": load,
            "deterministic": det_ok,
            "verify": ver,
            "cached": cached,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"json: {args.json}")
    return 0 if det_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
