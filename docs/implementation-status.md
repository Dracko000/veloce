# Veloce — Implementation Status vs. PRD / ERD

This document maps each requirement in `PRD.md` / `ERD.md` to its concrete
implementation in this repository, with the file that provides it. Where a
requirement is intentionally scoped differently for v1.0, that is stated
plainly.

## Functional requirements

| ID | Requirement | Implementation | Where |
|----|-------------|----------------|-------|
| FR-1.1 | First-class sub-byte primitives (`trit`, `i2`, `i4`, `fp8`, `fp16`) | All five are packed storage types with their own kernels: `trit` (5/byte, add/sub lowering), `i4` (2 signed nibbles/byte), `i2` (4 signed values/byte, values in [-2,1]), `fp8` (E4M3, 1 byte/value, decoded to f32 during the dot) and `fp16`. Each has a plain and a row-scaled (`_s`) matmul, plus a group-scaled variant for `i4`/`trit` | `lexer.py`, `ast.py`, `typecheck.py`, `codegen.py`, `weights.py`, `runtime/veloce_runtime.h` |
| FR-1.2 | Automatic bit-packing (5 trits → 1 byte) | Base-3 flat packing, shared by compiler (`pack_trits`), C runtime (`veloce_pack_trits`), and `.veloce` files; `i4` packs two signed nibbles per byte, `i2` four signed 2-bit values per byte, and `fp8` one E4M3 byte per value, all with the same on-disk-equals-in-memory contract | `weights.py`, `runtime/veloce_runtime.h` |
| FR-1.3 | MatMul-free lowering of trit matmuls to ADD/SUB | `matmul_trit` → `veloce_matmul_trit` (sign in {-1,0,1}: `acc += x[k]` / `acc -= x[k]`, no FP multiply) | `codegen.py`, `runtime/veloce_runtime.h` |
| FR-2.1 | Loop-free declarative tensor algebra, einsum/matmul/`+ * @` | No `for` statement exists; `matmul`, `@`, elementwise ops, `transpose`, `softmax`, etc. | `parser.py`, `typecheck.py` |
| FR-2.2 | Automatic operator fusion of elementwise/reduction chains | Pure elementwise chains collapse to a single register loop `for i: d[i] = <expr(i)>` with no intermediate buffers | `codegen.py` (`_is_pure_ew`, `_build_ew`) |
| FR-2.3 | Implicit vectorization (AVX2/AVX-512/NEON) | Delegated to the C compiler's autovectorizer; build uses `-O3 -march=native -ffast-math` | `velocec`, `Makefile` |
| FR-3.1 | Static liveness analysis of tensor lifetimes | `birth_step`/`death_step` computed per activation from statement order and last use | `memplan.py` |
| FR-3.2 | In-place buffer reuse in a static Arena | Greedy first-fit allocator reuses freed 64-byte-aligned slots; peak = largest simultaneously-live set | `memplan.py` |
| FR-3.3 | Zero-copy mmap loading of `.veloce` weights | On-disk layout == packed in-memory layout; `veloce_mmap_ro` maps read-only; tensors are cast in place; `weights.VeloceFile` exposes zero-copy `raw()` views | `weights.py`, `runtime/veloce_runtime.h` |
| FR-4.1 | Multi-target codegen to a portable lower-level target | C11 output (portable across x86_64/ARM64; SIMD via `-march`); LLVM/Vulkan/WebGPU are future backends | `codegen.py` |
| FR-4.2 | Implicit cache-aware CPU parallelism | **Implemented.** Every hot kernel splits independent output elements across cores through an internal `veloce_parallel_for`; OpenMP backend with `-fopenmp`, C11 `<threads.h>` pool otherwise, serial fallback. Chunking is cache-aware: the grain is a byte budget (a quarter of L2, expressed in items and capped by `VELOCE_PAR_GRAIN` elements), and no item size can veto a split on its own, so a 1.9 MB loop of 6 KB tiles still forks. Each output element is accumulated by exactly one worker in fixed order, so results are **bit-identical for every thread count** | `runtime/veloce_runtime.h` (`veloce_parallel_for`, `veloce_worth_parallel`, `veloce_cache_bytes`), `velocec` (`-fopenmp` auto-detect) |

## Non-functional requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| Determinism (bit-identical runs) | ✅ within a build | Verified in `test_e2e_transformer_layer` (runs twice, compares) and `test_parallel_trit_bit_exact` / `test_parallel_matches_serial_transformer` / `test_parallel_row_kernels` (serial vs 1/2/4-thread OpenMP vs C11-threads: identical hashes). **Caveat:** this is per *build*. `-ffast-math` lets GCC re-contract FMAs when inlining or flags change, so two builds of the same source can differ in the low bits (~1e-7 relative, inside the 1e-6 verify tolerance, argmax unchanged) — reproduced on the cached decoder with `-ffp-contract=off` / `-fno-tree-vectorize` (three hashes, `det=1` each). Exact bit-for-bit reproduction needs identical source, flags and compiler. |
| Minimal toolchain / no runtime deps | ✅ | Runtime is `static inline` header; serial builds link with only libm. Parallel builds use the toolchain's OpenMP (`-fopenmp`) or the C11 `<threads.h>` pool; both are optional and auto-detected. |
| C ABI interoperability | ✅ | All functions are `veloce_<name>` with C types. |
| Zero runtime heap allocation | ✅ | Per-function `static` arena; generated C contains no `malloc`/`calloc` (asserted in tests). |

## ERD entities — where they live

| Entity | Realised as |
|--------|-------------|
| `MODULE` | `ast.Module` (`name`, `target_arch`, `cache_line_bytes`); surfaced by `velocec erd` |
| `FUNCTION` | `ast.FnDecl`; `is_pure` computed by purity scan; C symbol `veloce_<name>` |
| `MEMORY_ARENA` / `MEMORY_SLICE` | `memplan.Allocation` list with offsets/lengths/lifetimes; the emitted `static float __arena[]` |
| `IR_NODE` | Tensor ops discovered in the expression graph (`erd._iter_tensor_ops`), with `op_code` and `topological_order` |
| `FUSED_KERNEL` | A fused elementwise chain (one emitted `for` loop) / library kernel binding |
| `TENSOR_DESCRIPTOR` | `ast.TensorType` (dtype + shape) and each arena allocation |
| `DATA_TYPE` / `PACKING_RULE` | dtype table + 5-per-byte / 8-bit rule in `erd._dtype_records` |
| `WEIGHT_ASSET` / `PACKED_BLOCK` | tensor params of storage dtype `trit`/`fp16`; `.veloce` blob layout in `weights.py` |

`./velocec erd <file> --format json|mermaid` materialises these entities for a
real module (see `test_erd_introspection`).

## Honest scope notes

* **v1.0 compute model**: weights are quantized (`trit`/`fp16`) and decoded
  during the kernel; activations are `f32`. Arithmetic in `f32` is not an
  "approximation" of the PRD — it is the standard low-bit inference design
  (quantize storage, accumulate in float).
* **Every PRD dtype is implemented end to end.** `i4`: `pack_i4`/`veloce_pack_i4`
  store two signed nibbles per byte (low nibble = element 2k, high = 2k+1),
  `veloce_matmul_i4` lowers through a 256×4 signed-nibble LUT.
  `i2`: `pack_i2` stores four signed 2-bit values per byte (values in [-2,1]),
  `veloce_matmul_i2` decodes through a 256×4 LUT with an AVX2 path.
  `fp8`: `pack_fp8` stores one E4M3 byte per value (bias 7, max 448); the
  runtime decodes with a lazily built 256-entry table, so an E4M3 checkpoint
  maps zero-copy and is expanded during the dot product.
  `test_i4_matmul_bit_exact`, `test_i2_matmul_numeric` and
  `test_fp8_matmul_numeric` check each one is bit-identical across 1/2/4
  threads and across the `.veloce` roundtrip. Measured max |err| vs a float64
  reference: 5.6e-6 (i2, K=1003) and 4.3e-6 (fp8, K=517).
* **Group- and row-scaled variants** exist for every low-bit dtype.
  `matmul_i4g`/`matmul_i2g`/`matmul_fp8g` take `f32[N, groups]` and share one
  typechecker rule and one lowering path; `matmul_trit_s`/`matmul_i2_s`/
  `matmul_fp8_s` take one scale per output row. The group count must divide
  the inner dimension exactly — a ragged count has no single group size, so it
  is rejected at compile time rather than silently scaling the wrong slices
  (`test_group_scaled_rejects_ragged_groups` covers all three kernels).
* **KV cache (FR-2.1 scope)**: a tensor parameter declared `rw` is a
  caller-owned mutable buffer, and `cache[pos] = row` stores a whole row into
  it, so a host loop can keep keys/values across calls. `examples/kv_decode.vel`
  plus `examples/run_kv_decode.py` decode five tokens this way (max |err|
  7.7e-7 per token, 1.0e-6 for the cached attention). Attention masking is a
  first-class builtin: `softmax(scores, mask)` with an `i32` mask (non-zero =
  participates) excludes hidden entries from both the row maximum and the
  normaliser, so a hidden entry contributes exactly zero whatever it holds —
  a stale cache row, `NaN` or `±1e30` cannot reach the result
  (`test_masked_softmax_builtin` poisons the hidden columns with exactly
  those values). A fully masked row returns zeros, not `NaN`.
  A 1-D mask is broadcast to every row; a 2-D `i32[T,N]` mask is *per row*
  (mask row `r` for scores row `r`, i.e. a per-head causal mask), which the
  lowering selects by mask rank — reading a 2-D mask as 1-D used to apply
  row 0 to every row and produce a plausible but wrong distribution. The
  mask dtype is `i32` only; `i8`/`i64`/`i2` are rejected instead of being
  reinterpreted as 32-bit words (`test_masked_softmax_per_row_mask`,
  `test_softmax_mask_rejects_bad_shape_and_dtype`).
  The cached decoder generated by `veloce/lmmodule.py` now passes a **2-D**
  mask (`i32[T, CT]`, one row per query row in the chunk) and decodes `T`
  tokens per call, writing `ck[pos + r] = k[r]`/`cv[pos + r] = v[r]` for every
  row. With `T > 1` the mask rows genuinely differ (row `r` may see `pos + r`,
  row `r-1` may not), so the per-row kernel is reached end to end and a
  broadcast mask is observable in the output. `test_cached_decode_attention_mask`
  decodes six tokens in chunks of 1, 2 and 3 through a 1-, 2- and 3-layer
  decoder against a strictly causal float64 reference (forcing the broadcast
  kernel makes it fail at `max|err| = 8.07`), and
  `test_cached_decode_mask_hides_garbage_slots` starts every cache row at
  `±1e30` and still matches that reference bit for bit (`max|err| = 5.80` if
  the mask stops hiding the unwritten slots).
* **FR-4.2 threading policy**: work is split over *independent output*
  elements only (rows for matmul/rmsnorm/softmax, elements for elementwise
  ops), never over a reduction axis, which is why bit-exact determinism is
  preserved. Kernels below the L2-cache threshold stay serial to avoid team
  launch overhead. `VELOCE_POOL=0` forces a serial build; `VELOCE_THREADS=n`
  pins the worker count, `VELOCE_PAR_MIN` (default 1024 output elements) sets
  the smallest problem that is worth a fork-join, and `VELOCE_PAR_GRAIN`
  (default 512 elements) is the largest chunk one worker may be handed: the
  grain is a *byte* budget (L2/4) expressed in items, so a loop of 308
  six-kilobyte tiles splits four ways instead of collapsing onto one lane.
  Applying that floor in elements alone was wrong for exactly this reason --
  see the 2026-11-20 entry above.
* **Measured performance (4-vCPU shared VM, load average ~2.2)**: the AVX2
  trit GEMV decodes 5 trits per byte with a 256-entry float LUT and works on
  four output rows per pass, reaching ~8.8–9.3 GMAC/s on one thread and
  19.7–23.0 GMAC/s on four (2.2–2.5×), i.e. 3.9–4.6 GB/s of packed weights:
  **22–31% of the host's real streaming peak (12.6–20.8 GB/s at 4 threads)**,
  not the 81–97% an earlier revision claimed — that figure compared a
  single-threaded scalar probe against a four-thread SIMD kernel.

  The four-row pass is what bought that. Flat base-3 packing puts row j at byte
  `(j*K)/5`, slot `(j*K)%5`, so rows only share an `x` window when they are in
  the same byte phase; the kernel therefore runs rows `j, j+p, j+2p, j+3p`
  (`p = 5/gcd(K,5)`) and hands one pair of 32-byte `x` loads to four LUT loads
  and four FMAs, instead of re-reading the whole `x` vector once per output
  neuron. Measured back to back in one process by
  `benchmarks/bench_gemv_roofline.py`, that is 1.33× (K=4096), 1.30× (K=1536)
  and 1.34× (K=1024), medians of three runs, over the one-neuron-at-a-time
  kernel; end to end it is 23.6 tok/s / 42.3 ms per token against 22.3 tok/s /
  45–50 ms before.

  **Two tokens per weight pass.**  A chunked decode multiplies the same packed
  matrix by two x vectors; running the one-row kernel twice streamed the
  matrix twice.  `veloce_trit_dot4x2_avx2` loads each packed byte and its
  32-byte LUT row once and feeds both query rows from it (same four-phase
  row groups as the one-row kernel), and `veloce_matmul_trit` dispatches
  `M == 2` onto it.  Measured with the same microbench that produced the
  one-row numbers, 100 reps, one thread: 4096² **7.59 → 10.71 GMAC/s per
  token** (2.21 → 1.57 ms), 1536×4096 5.4 → 6.6-9.4 GMAC/s, i.e. 1.25-1.41×
  per token.  Each (query row, neuron) keeps the one-row kernel's two
  accumulator chains and reduction order, so the result is **bit-identical
  to two `M == 1` passes** -- `test_trit_matmul_two_rows_share_one_weight_pass`
  checks that with `memcmp` on two different `x` rows, at K = 1003 (5-row
  phase stride) and K = 1024 (single phase), plus a float64 reference.  The
  single-stream projection is unchanged by this (it decodes one token at a
  time); it pays off when a caller decodes a chunk, as the cached decoder
  does.

  **Four tokens per weight pass.**  The same idea cannot be extended by simply
  widening the row group: four output rows * four query rows is 32 vector
  accumulators (4 queries × 4 rows × 2 chains), twice what AVX2 has, and `M ==
  3/4` therefore fell back to the per-row path, streaming the whole matrix once
  per token.  `veloce_trit_dot2x4_avx2` instead pairs **two** output rows with
  **four** query rows (16 accumulators) and shares each packed byte and its LUT
  row across all four; `veloce_matmul_trit` dispatches `M == 4` onto it, with a
  work item covering 2*period rows instead of 4*period.  Weight traffic per
  token is a quarter of the per-row path and a half of the two-query kernel.
  Measured on 4096², three runs, one thread (the stable configuration; four
  threads on this shared VM vary by 2× run to run): **1.94 ms per token for
  `M == 1` vs 1.19 ms for the four-query pass, 8.6 → 13.5 GMAC/s per token
  (1.63×)** and 1.6× the matrix bytes per second the one-row kernel streams.  At
  four threads the best-of-10 pairs measured 0.50 vs 0.65 ms/token (1.3×), with
  higher spread.  Per (query row, neuron) the two accumulator chains and
  reduction order are unchanged, so `M == 4` is **bit-identical to four `M == 1`
  passes** -- `test_trit_matmul_four_rows_share_one_weight_pass` pins that with
  `memcmp` at K = 1003/512/1024/1536 over whole tiles, plus a float64
  reference, and fails if the second row of a pass decodes the first row's
  bytes (mutation-verified).

  **Three tokens per weight pass.**  `M == 3` used to fall back to the per-row
  path for the same register reason.  `veloce_trit_dot2x3_avx2` pairs two
  output rows with three query rows (3 queries × 2 rows × 2 chains = 12
  accumulators) and shares each packed byte and its LUT row across all three;
  `veloce_matmul_trit` dispatches `M == 3` onto it, with a work item covering
  2*period rows.  Weight traffic per token is a third of the per-row path.
  Measured 4096², three runs, one thread: the three-query pass runs at
  **12.3–12.6 GMAC/s per token vs 8.7–9.0 for three one-row passes (1.40–1.42×,
  5.58 → 4.02 ms/token)**; 1536² is 1.30–1.36×.  Each (query row, neuron)
  keeps the two accumulator chains and reduction order of the one-row kernel,
  so `M == 3` is **bit-identical to three `M == 1` passes** --
  `test_trit_matmul_three_rows_share_one_weight_pass` pins that with `memcmp`
  at K = 1003/512/1024/1536 over whole tiles, plus a float64 reference, and
  fails if a query row reads the first row's `x` (mutation-verified).

  The kernel is still *decode* bound, and the reason is now narrower: every
  packed byte costs a byte load plus a 32-byte LUT load for five useful values,
  and on this EPYC (128-bit FP units, 32-byte loads split in two) that is about
  three load slots per five MACs — ~3.3 MAC/cycle, which is where the kernel
  sits. A dense f32 FMA still reaches 1.5–2× the GEMV's MAC rate, and a pure
  read of the same packed bytes is 23–36× faster. A 7B-shaped layer compiled by
  `velocec` runs at 35.2 GMAC/s (5.44 tok/s for 32 layers) and the projected
  single-stream decode is 4.03 tok/s. A correct bandwidth roofline on this VM
  is 14.8 tok/s (19.5 GB/s measured), so the PRD's 15 tok/s target is only
  reachable *if the decode overhead is removed* — it is not a hardware wall,
  and the honest status is "FAIL today, real headroom".
* **End-to-end mid-size model** (`benchmarks/bench_midi_lm.py`): a genuine
  275.6M-ternary-parameter decoder (6 layers, d=1536, ffn=6144, V=32000,
  T=1 without KV cache) built with the production packing code, written as a
  153.5 MB `.veloce` file and run from a memory map. The module itself is
  rendered by `veloce/lmmodule.py` from a shape config, so the same command
  measures other sizes (`--config D=2048 --config F=5632 --config layers=7`,
  i.e. `make bench-midi-05b`); `examples/midi_lm.vel` is that renderer's
  default-config output and `test_default_render_matches_example` fails if the
  two drift. Measured in the committed artefact (`measured_at`
  2026-11-20T21:30Z, host load 1.5): **33.7 tok/s on one thread** and
  **62.9 tok/s on four** for the eager path, **63.6 tok/s** on four threads
  with the KV cache (32 tokens, chunks of two, 15.7 ms/token, rel err 1.3e-6,
  1.01x the eager path in the same run). Until the chunk-grain fix of the same
  day, four threads ran the whole decode inside one worker and were *slower*
  than one (15.4 vs 30.5 tok/s); the split now really happens, so the cached
  decode scales 1.89x from one thread to four. The logits match a NumPy
  reference to 1.4e-6 relative error and hash identically at every thread
  count.

* **The parallel split was silently a single worker** (fixed 2026-11-20).
  `veloce_parallel_for_sized` computed `chunk = ceil(n/lanes)` and then clamped
  it up to `VELOCE_PAR_GRAIN` (512 *elements*), so any loop with fewer than
  512*lanes items handed every item to worker 0 and left the caller empty:
  a two-row 4096² trit GEMV with `VELOCE_THREADS=4` ran at **1.06 CPUs
  utilised** and was 1.5× *slower* than one thread. The same term vetoed
  1536-wide trit tiles outright (`bytes_per_item < 8192`), so most of the
  275.6M decoder never forked at all. The grain is now a byte budget (L2/4,
  expressed in items, capped by the element floor) and the per-item size no
  longer vetoes a split; `test_parallel_for_hands_every_lane_work` counts
  distinct workers and asserts every item ran exactly once (with the old
  formula it reports `lanes=1 covered=205` and fails). Measured on the
  committed artefacts: trit GEMV 4096² one→four threads **9.19 → 26.13 GMAC/s
  (2.84×)**, the 275.6M decoder **25.1 tok/s on one thread and 62.9 on four**
  (the previous artefact measured 15.4 on four threads, i.e. *slower* than its
  own 22.3 one-thread run), and its KV-cached decode **33.7 → 63.6 tok/s
  (1.89×)**; the 7B-class projection moved from 3.04 to **4.03 tok/s** against
  a 14.8 tok/s bandwidth ceiling.

* **A 7B layer's time is 96–98% matmul, so the speed gap is inside the
  kernel** (measured 2026-11-22). Two attempts to isolate the non-matmul cost
  by fitting end-to-end times had been inconclusive, so the compiled layer is
  now instrumented instead: `bench_throughput.py --layer-profile` compiles the
  layer module, times it whole, then times the same seven trit matmuls one by
  one through `veloce_matmul_trit` and the elementwise work (`silu` + multiply
  over F) in the same process, with means over ten interleaved repetitions.
  Measured at 1 thread: full layer 23.4–27.9 ms, of which matmul 23.4–26.7 ms
  and elementwise 0.05–0.10 ms; across six runs the non-matmul share is
  1.2–4.7% (one run read −0.7%, i.e. below measurement resolution). At four
  threads it is 1.6–4.7%. There is no hidden second workload: the 4× gap to
  15 tok/s has to come out of the one-row inner loop. The same
  instrumentation caught a stale number — the compiled layer ran a
  **two**-matmul MLP while the projection in [3b] assumed three, so
  `compiled_layer.GMAC` was 29% too high (202.6M vs 157.3M MACs per layer).
  The benchmark layer now runs the real three-matmul SwiGLU shape (`w1` gate,
  `w3` up, `w2` down), so its per-token work matches the projection — and that
  same ~29% of extra real arithmetic is why the compiled figure in the
  regenerated artefact is **1.6 tok/s**, not the pre-fix 3.3–4.4: projected
  **3.78 tok/s**, compiled layer 1.6 tok/s in the same run (host-load
  dependent).
  `benchmarks/results.json` regenerated (`measured_at` 2026-11-22T21:24Z,
  host load 1.5).

## Model capability defaults

Every `.veloce` export carries a capability model card (see
`docs/model-capabilities.md`). The default profile declares four pillars —
deeper specialist knowledge, richer world understanding, stronger coding and
agentic behaviour, and real-world task focus — each with status `declared`,
`trained`, or `verified`. `trained`/`verified` require attached evidence, and
`veloce.capabilities.validate_model_card` rejects overclaims. This is a
declaration contract, not a capability claim: both shipped checkpoints (the
27k-parameter QAT classifier and the 275.6M throughput demonstrator) remain at
`declared` for all four pillars, because Veloce is the inference compiler and
the capabilities come from the trained model.

## Runnable verification

| Command | What it proves |
|---------|----------------|
| `make test` | 41 end-to-end tests (`test_veloce.py`, incl. 4 FR-4.2 thread-consistency tests, a lane-coverage test that fails if a fork sends every item to one worker, numeric coverage for every low-bit kernel, chunked cached-decode mask checks (per-row and garbage-slot), a bit-exactness check between the one-, two-, three- and four-row trit kernels and an `-ffast-math` SiLU saturation regression) + 49 front-end/codegen/packing tests (`test_compiler.py`, incl. 5 for the decoder-module renderer, 5 for the cached-module renderer and 2 guarding the committed benchmark artefact) — compiles and runs real C, checks vs. NumPy. Runs inside an auto-created `.venv` with NumPy installed |
| `make examples` | `run_transformer.py`: compiled transformer layer matches a NumPy reference (max err ~1e-5) |
| `make kv` | `run_kv_decode.py`: five tokens decoded into a caller-owned KV cache, per-token and cached-attention outputs match NumPy (max err ~1e-6) |
| `make train` | `train_tiny_lm.py`: trains a ternary classifier with QAT, packs a `.veloce` (92% ternary params), compiles `tiny_lm_cls.vel`, and mmap-deploys it — 32/32 predictions match NumPy |
| `make demo` | tiny-LM inference demo: ternary weights are 89.9% smaller than fp16 (PRD ≥80% target) |
| `make bench-roofline` | where the trit GEMV's time actually goes: a pure read of the packed bytes, a dense-f32 FMA ceiling, one neuron at a time, and the shipped four-row kernel, on identical data — shows the kernel is decode-bound (22–31% of real peak), not at a memory wall |
| `make bench-throughput` | real measured throughput vs the PRD targets: trit GEMV GMAC/s, attention GFLOP/s, projected 7B-class tokens/s, memory-bandwidth roofline, footprint and compile-time checks |
| `make bench-midi` | end-to-end 275.6M-ternary decoder: real `.veloce` weights, mmap'd forward pass, measured tok/s, determinism hash and optional NumPy verification |
| `velocec model-card FILE.veloce` | prints the capability model card embedded in a `.veloce` export; `make train` and `make bench-midi` both write one |

The `.veloce` reader/writer is exposed as `weights.save_veloce(path, tensors)`
and `weights.VeloceFile(path)` (zero-copy `mmap`; `.raw(name)` returns packed
bytes, `.numpy(name)` decodes a copy), alongside the lower-level
`write_veloce_model`/`read_veloce_manifest`.
