#!/usr/bin/env python3
"""Where does the trit GEMV actually spend its time?

The implementation notes used to claim the kernel sits at the memory roofline.
This benchmark tests that claim three ways on the *same* packed bytes:

    probe   pure streaming read of the packed weights (no decode, no FLOPs)
    ceiling an 8-wide f32 FMA over dense weights -- the best a dot product
            can do on this machine when decoding is free
    kernel  the shipped trit GEMV

Run it and read the ratios:

    python3 benchmarks/bench_gemv_roofline.py

If `kernel` is close to `probe`, the kernel is bandwidth bound and nothing can
be won in the kernel.  If `kernel` is far below `ceiling`, it is limited by the
decode (load ports and wasted lanes), and there is headroom.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(ROOT, "runtime")
CC = os.environ.get("CC", "cc")

C_SRC = r'''
#define _POSIX_C_SOURCE 199309L
#include "veloce_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* pure read: touches every packed byte, produces no useful value */
static unsigned long long probe_read(const uint8_t *w, size_t n) {
    unsigned long long s = 0;
    size_t i = 0;
#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
    __m256i acc = _mm256_setzero_si256();
    for (; i + 32 <= n; i += 32)
        acc = _mm256_add_epi8(acc, _mm256_loadu_si256((const __m256i *)(w + i)));
    __m128i lo = _mm256_castsi256_si128(acc), hi = _mm256_extracti128_si256(acc, 1);
    __m128i q = _mm_add_epi8(lo, hi);
    unsigned char t[16]; _mm_storeu_si128((__m128i *)t, q);
    for (int k = 0; k < 16; k++) s += t[k];
#endif
    for (; i < n; i++) s += w[i];
    return s;
}

/* one output row: 8-wide f32 FMA with four accumulators */
static float fma_row(const float *x, const float *w, size_t K) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    size_t k = 0;
    for (; k + 32 <= K; k += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k),      _mm256_loadu_ps(w + k),      a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 8),  _mm256_loadu_ps(w + k + 8),  a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 16), _mm256_loadu_ps(w + k + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + k + 24), _mm256_loadu_ps(w + k + 24), a3);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 lo = _mm256_castps256_ps128(s), hi = _mm256_extractf128_ps(s, 1);
    __m128 q = _mm_add_ps(lo, hi);
    q = _mm_hadd_ps(q, q); q = _mm_hadd_ps(q, q);
    float acc = _mm_cvtss_f32(q);
    for (; k < K; k++) acc += x[k] * w[k];
    return acc;
}

int main(int argc, char **argv) {
    size_t K = argc > 1 ? (size_t)atol(argv[1]) : 4096;
    size_t N = argc > 2 ? (size_t)atol(argv[2]) : 2048;
    size_t rowb = veloce_trit_rowbytes(K);
    size_t nbytes = N * rowb;

    float *x = malloc(K * sizeof(float));
    float *wf = malloc(K * sizeof(float));
    uint8_t *w = malloc(nbytes);
    for (size_t i = 0; i < K; i++) {
        x[i] = (float)((int)(i % 17) - 8) * 0.031f;
        wf[i] = (float)((int)(i % 7) - 3) * 0.11f;
    }
    for (size_t i = 0; i < nbytes; i++) w[i] = (uint8_t)(i * 131 + 7);

    int reps = 200;
    volatile float sink = 0;
    volatile unsigned long long usink = 0;

    double t0 = now_s();
    for (int r = 0; r < reps; r++) usink += probe_read(w, nbytes);
    double tp = (now_s() - t0) / reps;

    t0 = now_s();
    for (int r = 0; r < reps; r++)
        for (size_t j = 0; j < N; j++) sink += fma_row(x, wf, K);
    double tc = (now_s() - t0) / reps;

    t0 = now_s();
    for (int r = 0; r < reps; r++)
        for (size_t j = 0; j < N; j++) sink += veloce_trit_dot(x, w, j * K, K);
    double tk = (now_s() - t0) / reps;

    double macs = (double)N * K;
    double mb = nbytes / 1e6;
    printf("K=%zu N=%zu  packed weights %.2f MB\n", K, N, mb);
    printf("  probe (read only)  : %8.3f ms  %7.2f GB/s\n", tp * 1e3, mb / 1e3 / tp);
    printf("  ceiling (dense FMA): %8.3f ms  %7.2f GMAC/s  %7.2f GB/s\n",
           tc * 1e3, macs / 1e9 / tc, (double)N * K * 4 / 1e9 / tc);
    printf("  kernel (trit GEMV) : %8.3f ms  %7.2f GMAC/s  %7.2f GB/s\n",
           tk * 1e3, macs / 1e9 / tk, mb / 1e3 / tk);
    printf("  kernel / ceiling   : %6.2fx   (>1 means the decode costs more than it saves)\n",
           tc / tk);
    printf("  kernel vs probe    : %6.2fx slower than a pure read of the same bytes\n",
           tk / tp);
    (void)sink; (void)usink;
    return 0;
}
'''


def main():
    cases = [(4096, 2048), (1536, 8192), (1024, 4096)]
    with tempfile.TemporaryDirectory(prefix="veloce_roofline_") as d:
        src = os.path.join(d, "bench.c")
        exe = os.path.join(d, "bench")
        with open(src, "w") as f:
            f.write(C_SRC)
        flags = ["-std=c11", "-O3", "-march=native", "-ffast-math", "-pthread",
                 "-I", RUNTIME]
        r = subprocess.run([CC, *flags, src, "-lm", "-o", exe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr)
            return 1
        for K, N in cases:
            out = subprocess.run([exe, str(K), str(N)], capture_output=True,
                                 text=True)
            print(out.stdout.rstrip())
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
