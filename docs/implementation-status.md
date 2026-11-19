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
| FR-4.2 | Implicit cache-aware CPU parallelism | **Implemented.** Every hot kernel splits independent output elements across cores through an internal `veloce_parallel_for`; OpenMP backend with `-fopenmp`, C11 `<threads.h>` pool otherwise, serial fallback. Chunking is cache-aware (L2-sized per-worker tiles), and each output element is accumulated by exactly one worker in fixed order, so results are **bit-identical for every thread count** | `runtime/veloce_runtime.h` (`veloce_parallel_for`, `veloce_worth_parallel`, `veloce_cache_bytes`), `velocec` (`-fopenmp` auto-detect) |

## Non-functional requirements

| Requirement | Status | Notes |
|-------------|--------|-------|
| Determinism (bit-identical runs) | ✅ | Verified in `test_e2e_transformer_layer` (runs twice, compares) and `test_parallel_trit_bit_exact` / `test_parallel_matches_serial_transformer` / `test_parallel_row_kernels` (serial vs 1/2/4-thread OpenMP vs C11-threads: identical hashes). |
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
  the smallest problem that is worth a fork-join.
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

  The kernel is still *decode* bound, and the reason is now narrower: every
  packed byte costs a byte load plus a 32-byte LUT load for five useful values,
  and on this EPYC (128-bit FP units, 32-byte loads split in two) that is about
  three load slots per five MACs — ~3.3 MAC/cycle, which is where the kernel
  sits. A dense f32 FMA still reaches 1.5–2× the GEMV's MAC rate, and a pure
  read of the same packed bytes is 23–36× faster. A 7B-shaped layer compiled by
  `velocec` runs at 30–32 GMAC/s (4.7–4.9 tok/s for 32 layers). A correct
  bandwidth roofline on this VM is 9.5–15.8 tok/s, so the PRD's 15 tok/s target
  is only reachable *if the decode overhead is removed* — it is not a hardware
  wall, and the honest status is "FAIL today, real headroom".
* **End-to-end mid-size model** (`benchmarks/bench_midi_lm.py`): a genuine
  275.6M-ternary-parameter decoder (6 layers, d=1536, ffn=6144, V=32000,
  T=1 without KV cache) built with the production packing code, written as a
  153.5 MB `.veloce` file and run from a memory map. The module itself is
  rendered by `veloce/lmmodule.py` from a shape config, so the same command
  measures other sizes (`--config D=2048 --config F=5632 --config layers=7`,
  i.e. `make bench-midi-05b`); `examples/midi_lm.vel` is that renderer's
  default-config output and `test_default_render_matches_example` fails if the
  two drift. Measured **25.9 tok/s on
  one thread** (38.5 ms median, best of 5 timed runs) in the committed
  artefact (`measured_at` 2026-11-19T21:16Z); with the KV cache, 32 tokens in
  chunks of two run at **25.6 tok/s** (39.0 ms/token, 0.99x eager, rel err
  1.25e-6). Threads 2-4 are slower on this shared host;
  the logits match a NumPy reference to 1.4e-6 relative error and hash
  identically at every thread count. The model sustains ~7 GMAC/s, the same
  rate as the standalone kernel, i.e. it runs at the kernel's compute
  roofline; four threads are usually *slower* than two because the VM has no
  spare cores.

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
| `make test` | 37 end-to-end tests (`test_veloce.py`, incl. 4 FR-4.2 thread-consistency tests, numeric coverage for every low-bit kernel, chunked cached-decode mask checks (per-row and garbage-slot) and an `-ffast-math` SiLU saturation regression) + 49 front-end/codegen/packing tests (`test_compiler.py`, incl. 5 for the decoder-module renderer, 5 for the cached-module renderer and 2 guarding the committed benchmark artefact) — compiles and runs real C, checks vs. NumPy. Runs inside an auto-created `.venv` with NumPy installed |
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
