<p align="center">
  <img src="assets/logo.svg" alt="Veloce" width="420">
</p>

# Veloce

A **low-bit, zero-allocation programming language and AOT compiler for LLM
inference**, implementing the PRD and ERD in this repository.

Veloce treats sub-byte quantization as a first-class language feature, removes
imperative loops from tensor code, plans every buffer at compile time into a
static arena, and lowers ternary (1.58-bit) matrix multiplies to integer
add/sub. The compiler emits portable **C11** that builds into a self-contained
object with no runtime dependency.

```
 .vel source ──► lexer ─► parser ─► type/shape checker ─► static arena planner ─► C11 codegen ─► cc -O3 -march=native
                     │                 │                        │                      │
                 declarative      sub-byte dtype          liveness + buffer        static inline kernels,
                 tensor algebra   & shape inference       reuse (peak memory)      mmap'd weight blobs
```

## Quick start

```bash
# Compile a module to C
./velocec emit examples/transformer_block.vel -o build/tb.c

# Inspect the plan (trit matmuls, arena reuse)
./velocec check examples/transformer_block.vel

# Dump the compiler's internal entity graph (ERD.md entities)
./velocec erd examples/transformer_block.vel --format mermaid

# Build & run the tiny-LM inference demo
make demo

# Run the full test-suite (compiles + runs real C, checks vs. NumPy)
# (first run auto-creates .venv and installs NumPy from requirements-test.txt)
make test

# End-to-end examples: transformer vs NumPy, KV-cache decoding, and
# train -> ternarize -> deploy
make examples
make kv
make train

# Measure real kernel/model throughput against the PRD targets
make bench-throughput

# Print the capability model card embedded in a .veloce export
./velocec model-card examples/tiny_lm.veloce
```

## Brand assets

`assets/` holds the mark, the lockup and the icon set. Everything there is
generated — edit `assets/make_logo.py` and re-run it, never the SVGs by hand:

```bash
python3 assets/make_logo.py     # writes all of assets/, stdlib only
```

* `logo.svg` / `logo-light.svg` — lockup (mark + wordmark + tagline)
* `logo-mark.svg` / `logo-mark-light.svg` — mark alone, for avatars
* `logo-wordmark.svg` — wordmark alone
* `logo.png` (1024²), `logo-256.png`, `favicon.ico` (16–64 px)

The mark is a bold amber V — "veloce" — on the project's near-black slate.
A five-tick "one packed byte" motif was tried twice (below the vertex, then
inside its counter) and dropped both times: at every size it read as a separate
UI progress indicator or collided with the stroke, which made the mark weaker,
not richer.

Requirements: **Python 3.10+** and a **C11 compiler** (`cc`/`gcc`). The
compiler itself is stdlib-only; the test/example targets use NumPy as an
independent reference and run in an auto-created `.venv`.

## A taste of Veloce

```veloce
module transformer_block

dim T = 4
dim D = 16
dim F = 32

fn attention(q: f32[T, D], k: f32[T, D], v: f32[T, D]) -> f32[T, D] {
    let kt = transpose(k)            # [D, T]
    let scores = matmul(q, kt)       # [T, T]
    let probs = softmax(scores * 0.25)
    return matmul(probs, v)          # [T, D]
}

fn mlp(x: f32[T, D], w1: trit[F, D], w2: trit[D, F]) -> f32[T, D] {
    let h = matmul_trit(x, w1)       # [T, F]  — add/sub only, no multiply
    let a = silu(h)
    return matmul_trit(a, w2)        # [T, D]
}
```

No `for`/`while` loops in tensor code. `trit` weights are stored packed
(5 trits per byte); `matmul_trit` lowers to integer addition/subtraction.

## What's implemented

| Area | Feature | Status |
|------|---------|--------|
| **Language** | functions, typed `let`/`const`, multi-return, scalar `if`/`while`, casts, `extern` | ✅ |
| | first-class dtypes: `trit`, `fp16`, `f32`, `i32`, `i64`, `i8`/`u8`, `i4` (packed 2/byte) | ✅ |
| | declarative tensor ops: `matmul`, `matmul_trit`, `matmul_i4`, `rmsnorm`, `silu`, `relu`, `softmax`, `rope`, `embedding`, `embedding_seq`, `meanpool`, `transpose`, `argmax`, `+ - * /`, `@` | ✅ |
| | bias/row broadcasting, `dim` size bindings, `target`/`cache` directives | ✅ |
| **FR-1.1–1.3** | sub-byte types, automatic bit-packing (5 trits/byte, 2 `i4`/byte), matmul-free trit lowering | ✅ |
| **FR-2.1–2.2** | loop-free tensor algebra, elementwise operator fusion (single register loop) | ✅ |
| **FR-2.3** | implicit SIMD vectorization (delegated to the C compiler's autovectorizer) | ✅ (via `-O3 -march=native`) |
| **FR-3.1–3.2** | static liveness analysis + in-place arena buffer reuse | ✅ |
| **FR-3.3** | zero-copy `.veloce` mmap weight loading | ✅ |
| **FR-4.1** | portable C11 multi-target codegen | ✅ (C11 as the portable target) |
| **FR-4.2** | implicit cache-aware CPU parallelism | ✅ OpenMP / C11-threads backend, cache-aware split, bit-identical at any thread count |
| **NFR** | 100% deterministic execution; C ABI (`extern "C"`); static arena, no heap `malloc` | ✅ |

## Performance (measured, 4-vCPU dev VM)

`make bench-throughput` runs compiled C against the real `veloce_runtime.h`
kernels. Representative result:

| Metric | Measured | PRD target |
|--------|----------|------------|
| trit GEMV 4096×4096 (1 thread → 4 threads) | 8.8–9.3 → 19.7–23.0 GMAC/s (**2.2–2.5× speedup**) | — |
| trit GEMV read bandwidth | 3.9–4.6 GB/s = **22–31% of peak** (decode-bound, see below) | — |
| trit GEMV, four rows per pass vs one neuron at a time | **1.30–1.34×** median of 3 runs (K = 1024, 1536, 4096) | — |
| attention block (QK^T + softmax + AV) | ~31–35 GFLOP/s | — |
| **end-to-end 275.6M ternary decoder** (6L, d=1536, ffn=6144, V=32000) | **23.6 tok/s** (1 thread, 42.3 ms median; 25.9 tok/s best) | — |
| same model, 2 threads | 28–36 tok/s (28–36 ms median; host-load dependent) | — |
| same model, **KV-cached** decode (32 tokens, 64 slots/layer, 2 tokens/call) | **25.6 tok/s** (39.0 ms/token, 0.99× eager, verify rel err 1.3e-6) | — |
| projected 7B-class decode | **~2.4–3.0 tok/s** (compiled layer: 3.3–4.3 tok/s) | ≥ 15 tok/s (**5–6× short**) |
| weight footprint (7B-class) | 1.3 GB vs 13.2 GB fp16 (90% saved) | ≥ 80% smaller ✅ |
| compile time | ~4 ms | < 2 s ✅ |
| heap allocation in generated C | none | zero ✅ |

The 7B-class speed gap is NOT purely memory bandwidth, and the earlier claim
that the kernel sat at the bandwidth roofline was measurement error. The old
"streaming peak" was a scalar single-threaded byte-sum loop; the real ceiling
this VM can stream is **12.6–20.8 GB/s at 4 threads** (5.7 GB/s at 1, both
host-load dependent). Against that the trit GEMV achieves **3.9–4.6 GB/s —
22–31% of peak**, so it is *decode* bound, not bandwidth bound: every packed
byte needs a byte load plus a 32-byte LUT load for only 5 useful values.

The kernel now works on four output rows per pass. Flat base-3 packing puts
row j at byte `(j*K)/5` slot `(j*K)%5`, so rows only share an `x` window when
they are in the same byte phase; the kernel runs rows `j, j+p, j+2p, j+3p`
(`p = 5/gcd(K,5)`) and feeds one pair of 32-byte `x` loads to four LUT loads
and four FMAs instead of re-reading `x` for every output neuron. Same-process
back-to-back measurement puts that at **1.30–1.34×** (median of 3 runs at
K = 1024/1536/4096), and end to end at 23.6 tok/s versus 22.3. What remains is
one byte load plus one 32-byte table load per five MACs — about three load
slots per five MACs on this EPYC, whose 128-bit FP units split 32-byte loads in
two — so ~3.3 MAC/cycle is the current wall.
`benchmarks/bench_gemv_roofline.py` measures all of this directly (a dense f32
FMA runs 1.5–2× the GEMV's MAC rate while reading 4× the bytes). Closing the
rest is real optimization headroom, not a hardware wall; the 9.5–15.8 tok/s
roofline printed above is only reachable once the decode stops being the
bottleneck.

The mid-size number is a *real* end-to-end run, not a projection:
`benchmarks/bench_midi_lm.py` generates a 275.6M-ternary-parameter decoder with
deterministic weights, writes it as a `.veloce` file, renders the module with
`veloce/lmmodule.py` (the same text as `examples/midi_lm.vel`, which a test
keeps in sync), compiles it and times the generated C against the mmap'd file.
`--config D=2048 --config F=5632 --config layers=7` measures a second,
0.5B-shaped network from the same command; `make bench-midi-05b` does it. On
one thread the model sustains ~7.4 GMAC/s — the same rate as the standalone
kernel — so it runs at the kernel's compute roofline, and streams 55 MB of
packed trits per token at ~1.42 GB/s (25.7 tok/s single-threaded on a quiet
host). Decoding 32 tokens through a 64-slot per-layer KV cache runs at
25.6 tok/s (39.0 ms/token, 0.99× the eager path) with a 1.3e-6 relative error
against the NumPy reference. The cached driver decodes **two tokens per call**,
so the attention mask carries one row per query row (`i32[T, CT]`) and
different rows hide different slots -- a mask read as a single broadcast row
would change the logits. Four tokens per call measured 24.9 tok/s, i.e. no
further gain. Multi-thread scaling is limited by this shared
VM (load average ~1-2.7 on 4 vCPUs); 2 threads help, 4 usually hurt.

The VM is shared, so numbers vary run to run; every benchmark reports medians
and best-of-N, plus a same-run streaming-bandwidth probe so a tok/s figure can
be read against what the host can actually deliver.

`benchmarks/results.json` and `benchmarks/midi_lm_results.json` hold the last
full runs; the midi_lm file carries the eager *and* cached (KV-cache) numbers
in one place. All are stamped with `measured_at` (UTC) and the CPU they were
taken on, because a figure is only comparable against the host state that
produced it. `make bench-midi` regenerates the midi_lm artefact with
`--cached --cached-chunk 2`, so the cache block cannot go missing again and
the recorded decode goes through the per-row mask kernel rather than the
broadcast one.

## Serving & export scope

Two export/serving paths exist and both are exercised by a `make` target:

| Path | Scale | Command |
|------|-------|---------|
| QAT train → ternarize → pack → compile → mmap-deploy | 26.9k params (92% ternary) | `make train` |
| Production packing → `.veloce` → mmap'd forward pass | 275.6M ternary params | `make bench-midi` |

`make train` is the full quantization-aware-training loop: a NumPy trainer
trains with a straight-through ternary estimator, `weights.save_veloce` writes
the packed file, `velocec` compiles the `.vel` module to C, and the resulting
binary memory-maps the weights and serves predictions (32/32 match the NumPy
reference). `make bench-midi` takes the same export path at 275.6M parameters
and measures real decode throughput from the memory map.

Not implemented: QAT at 100M+ parameter scale. Training a 275M-parameter
model needs GPUs and days; the mid-size harness therefore uses deterministic
pseudo-random weights and measures *throughput*, not quality. The serving
runtime, packing, and mmap path are identical either way, so a QAT-trained
checkpoint at that scale would drop into the same `.veloce` file format.

## Model capability defaults

Every `.veloce` export carries a **capability model card**. The default profile
defines four pillars — deeper specialist knowledge, richer world
understanding, stronger coding and agentic behaviour, and real-world task
focus — each with a status of `declared`, `trained`, or `verified`. `trained`
and `verified` require attached evidence; `declared` is explicitly not a
capability claim. The compiler enforces this when writing and reading cards:

```bash
./velocec model-card examples/tiny_lm.veloce   # prints the embedded card
```

Both current checkpoints (the 27k-parameter QAT classifier and the 275.6M
throughput demonstrator) declare all four pillars at `declared`. Veloce is the
inference compiler — the capabilities themselves come from the trained model,
not from the language. See [`docs/model-capabilities.md`](docs/model-capabilities.md)
for the full contract and the metrics each pillar is evaluated with.

See [`docs/implementation-status.md`](docs/implementation-status.md) for the
exact FR-by-FR mapping and [`docs/language.md`](docs/language.md) for the
language reference.

## Repository layout

```
velocec                 compiler driver (build / emit / check / erd / ast)
veloce/                 the compiler
  lexer.py parser.py ast.py
  typecheck.py          resolution, type & shape checking, purity
  memplan.py            static liveness + arena allocator
  codegen.py            C11 lowering (+ operator fusion)
  lmmodule.py           renders a decoder .vel module from a shape config
  erd.py                internal-entity introspection (ERD.md)
  weights.py            .veloce weight-file format (zero-copy) + VeloceFile reader
  capabilities.py       default capability profile + model-card builder/validator
runtime/veloce_runtime.h  static-inline kernels: arena, trit unpack, matmul, parallel work splitting (FR-4.2), …
examples/               hello.vel, transformer_block.vel, tiny_lm.vel, midi_lm.vel
                        (generated by veloce/lmmodule.py), kv_decode.vel,
                        tiny_lm_cls.vel (trained classifier)
  run_transformer.py    transformer layer vs NumPy reference
  train_tiny_lm.py      train (QAT) -> ternarize -> pack -> compile -> mmap-deploy
demo/tiny_lm.c          runnable inference demo driver
benchmarks/bench_throughput.py  real throughput vs PRD targets (GMAC/s, tok/s, roofline)
benchmarks/bench_midi_lm.py     end-to-end 275.6M ternary decoder (--config for other shapes)
tests/test_veloce.py    end-to-end test-suite (31 tests, incl. FR-4.2)
tests/test_compiler.py  front-end / codegen / packing / model-card suite (40 tests)
requirements-test.txt   NumPy pin for the auto-created .venv
```

## Design notes

* **Compute dtype is `f32`; `trit`/`fp16` are storage dtypes** — activations
  stay in float, weights are quantized on disk and decoded/used in the kernel.
  This mirrors real low-bit inference engines.
* **Trit packing is flat, base-3, 5 per byte** — the same layout the C runtime
  writes and the `.veloce` file stores, so mmap'd weights are used in place.
* **`matmul_trit(x, w)` takes `w` shaped `[out, in]`** (the LLM weight layout)
  so the reduction runs over `in` and lowers to `acc += x[k]` / `acc -= x[k]`.
* **The arena is per-function `static`** — a direct realisation of "0 bytes
  allocated on the heap during inference".
