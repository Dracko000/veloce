# Veloce — backlog

Living list for the daily development runs. Work top-down; keep this file
honest. An item is done only when a test or a measured number proves it.

Rules for every run:

* Verify with `make test` (both suites) before claiming anything works.
* Never report a result you did not get from real tool output.
* Keep `docs/implementation-status.md`, `docs/language.md` and `README.md` in
  sync with the code in the same commit that changes behaviour.
* Commit only when the suites are green, one coherent change per commit.
* If a claim in the docs is wrong, fix the docs — do not weaken the test.

## Git rules (every run)

Remote: `origin` = https://github.com/Dracko000/veloce (public, branch `main`,
authenticated as Dracko000 via `gh`). There is no upstream and no fork.

* Commit **and push** every run that produces a verified change. Leaving work
  uncommitted is a failure of the run, not a neutral outcome.
* Push only when both suites are green and `git status` shows nothing you did
  not intend to include.
* Never commit: `.venv/`, `build/`, `__pycache__/`, `*.so`, `*.bin`, or the
  scratch `examples/codegen_*.vel` / `examples/midi_lm_05b.vel` modules (all
  ignored in `.gitignore` — do not `git add -f` them).
* Never commit secrets. Before pushing, scan what you are about to commit:
  `git ls-files -co --exclude-standard | grep -v '^BACKLOG.md$' | xargs -r grep -lniE "api[_-]?key|secret|ghp_|sk-[A-Za-z0-9]{16,}|password|PRIVATE KEY"`
  must print nothing. `BACKLOG.md` is excluded only because it quotes this
  pattern; never exclude anything else. If the scan prints a file, stop and
  report it instead of pushing.
* One coherent change per commit; message style `feat:` / `fix:` / `docs:` /
  `test:` with a body explaining *why*, as in the existing commits.
* Author identity is already configured per-command in the job prompt; use
  `git -c user.name="Dracko000" -c user.email="Dracko000@users.noreply.github.com" commit`.
* After pushing, confirm `git rev-parse HEAD` equals `git rev-parse origin/main`
  and include the commit hash in the report. Do not claim a push you did not
  verify.
* If a push fails (auth, network, non-fast-forward), report the exact error.
  Never force-push and never rewrite published history.

## Open

### 1. The 7B-class target is still failing

`benchmarks/results.json` (2026-11-20) projects **4.03 tok/s** for the
7B-class shape against a PRD target of 15 tok/s, with a **14.78 tok/s**
bandwidth ceiling from the same-run probe (19.53 GB/s at 4 threads). The GEMV
is decode-bound, not bandwidth-bound: 5.23 GB/s = **27% of the streaming peak**
after the phase-grouped kernel, i.e. ~3.3 MAC/cycle for 3 load slots per 5
MACs. Four threads now really do scale the kernel (9.19 → 26.13 GMAC/s, 2.84x)
since the chunk-grain fix below, so the remaining 3.7x is inside the kernel,
not in the team.

Measuring the per-query shapes (2026-11-21, 4096², one thread, three runs) puts
a number on what a token can cost: `M == 1` is 8.6 GMAC/s per token, two query
rows per weight pass 12.9-14.0, and the shipped four-query pass 13.5 (1.63x
over one row). Four output rows * four query rows is 32 accumulator registers,
twice what AVX2 has, so a pass cannot get below one byte fetch plus one LUT row
per twenty MACs -- that is the shape ceiling unless the decode itself changes.
The single-stream projection is the `M == 1` number, so none of this moves it:
the 3.7x has to come out of the one-row inner loop, not out of chunking.

Next levers, in the order they look worth measuring: (a) the per-token cost of
the non-matmul work (rmsnorm, silu, softmax, launch) has never been isolated on
a 7B-shaped layer. At the 275.6M shape the gap between the modelled matmul time
and the measured 15.7 ms/token on four threads says the kernel is what is left,
but two attempts at a two-shape fit (D=1536 vs D=2048, cached, 4 threads) were
inconclusive: on 2026-11-20 the same-run bandwidth probe read 3.0 GB/s where the
275M run read 19.5, and on 2026-11-21 one binary measured 4096² layered GEMV at
0.65, 0.83 and 1.75 ms/token on three consecutive runs. Instrument the compiled
layer directly instead of fitting end-to-end times.  (b) `M == 3` still takes
the per-row path; it needs a three-query pass, or one two-query pass plus one
row, to share the weight stream as `M == 4` now does.

### 2. Logit hashes depend on compiler code layout

Recompiling the same generated C against a runtime header whose only textual
change was dead chunk arithmetic (`noop` variant, identical semantics) moved
the 275.6M cached-decode hash from `dc987e4dba318be1` to `5e3465d5445b9ecc`
and the logit sum from 8484.671124 to 8484.672795 (relative difference ~2e-7,
well inside the 1.3e-6 verify tolerance). `-ffast-math` lets GCC re-contract
FMAs when inlining changes, so "bit-identical" holds across thread counts and
repeats *within one build*, which is what the tests check, but not across
builds. Either pin the contraction (`-ffp-contract=fast` is already implied;
an explicit `-mfma`/`-fno-tree-vectorize` knob would not help) or state the
per-build caveat in the docs. Evidence: three binaries from one source,
`grep CACHED` on each.

## Done (most recent first)

* **`data/corpus.txt` was a scrape of unrelated local project notes.** The
  corpus that the bigram and native-trainer demos read was documentation copied
  from other projects on this machine, and it carried a demo account password
  (`data/corpus.txt` line 1291, Tirtasari demo login) plus internal server
  notes -- exactly what the repo's own pre-push secret scan is meant to stop.
  The scan printed `data/corpus.txt`; the five commits that added it were still
  local, so the file was dropped from them with an index-filter rewrite before
  anything was published, and the corpus is now generated by
  `tools/make_corpus.py` (`make corpus`) from this repository's own docs:
  66 602 bytes from 8 files, nothing that is not already public here. Its
  numbers were re-measured, not carried over: bigram **4.2838 bit/byte**
  (ppl 19.48), native trainer **262 steps/s**, loss 5.6029 -> 3.0793, held-out
  ppl 27.68, weight hash `def72f261ff7fa89` across runs.

* **A 1B-class ternary decoder runs end to end.** 18 layers, d=2048, F=5632,
  990.4M ternary params + 65.5M fp16 embedding, **329.3 MB** weight file (198 MB
  of trits), KV cache 64 slots -- all compiled by the same `.vel` -> C path as
  the 275M shape (`make model-1b`). Measured single-token forward, 5 runs per
  thread count: **7.25 tok/s one thread** (137.9 ms median of medians), 12.42
  two, 11.48 four (that column spans 46.6-148.8 ms across runs -- the 4-vCPU box
  is shared). KV-cache decode: 7.22 / 13.09 / 11.0 tok/s, 0.84-0.88x the eager
  path at this size. Determinism PASS: one logit hash across every thread count
  (`cfb675cd4012f43a`), `nonfinite=0`, argmax 31195. Artefacts
  `benchmarks/veloce_1b_results.json`, `veloce_1b_cached.json`,
  `veloce_1b_repeat.json`; the weights are initialised, not trained, and
  `models/veloce-1b/README.md` says so.

* **Training: a hand-written C byte-LM trainer and a closed-form bigram.** No
  Python in the loop -- forward, backward, Adam and sampling in one binary
  (`runtime/train_native.c`), fixed step budget and no training-time RNG, so
  reruns are bit-identical (`def72f261ff7fa89`). Measured 57 792 params,
  context 16, 3 000 steps: **262 steps/s**. `runtime/bigram.c` is the control:
  a bigram MLE is one counting pass (0.31 ms, 0 gradient steps). The NumPy
  counterpart (`benchmarks/np_baseline_same_arch.py`) runs the same architecture
  at **229.8 steps/s**, so the native binary is 1.14x -- less than expected, and
  the loss curves differ because initialisation order differs. The differentiable
  graph (`veloce/train/graph.py`) has three finite-difference gradient checks
  (`tests/test_train.py`, worst rel 5.1e-4); emitting forward/backward through
  the DSL is still open (`docs/training-framework.md`, phases 2-3).


* **A four-token chunk now shares one weight pass across four query rows.**
  `M == 3/4` fell back to the per-row kernel, so a four-token chunk streamed
  the whole matrix once per token; the obvious fix -- four output rows * four
  query rows -- needs 32 vector accumulators and AVX2 has 16.
  `veloce_trit_dot2x4_avx2` runs two output rows with four query rows (16
  accumulators: 4 queries * 2 rows * 2 chains), fetches each packed 32-bit word
  and each LUT row once for all four, and keeps the one-row kernel's two
  accumulator chains and reduction order per (query row, neuron).
  `veloce_matmul_trit` dispatches `M == 4` onto a chunk driver whose work item
  is 2*period rows instead of 4*period. Measured 4096², one thread, three runs:
  **1.94 → 1.19 ms per token, 8.6 → 13.5 GMAC/s per token (1.63x)**; four
  threads best-of-10 0.65 → 0.50 ms/token (1.3x, host noisy). New
  `test_trit_matmul_four_rows_share_one_weight_pass` pins `M == 4` to four
  `M == 1` passes with `memcmp` at K = 1003/512/1024/1536 over whole tiles,
  plus a float64 reference; pointing the second row's fetch at the first row's
  bytes makes it fail (`four_rows_matches_gemv FAIL`). Suites 39 → 40 e2e.


* **The parallel split was one worker doing everything.** `chunk =
  ceil(n/lanes)` was clamped up to `VELOCE_PAR_GRAIN` (512 *elements*), so any
  loop with fewer than 512*lanes items went to worker 0 in one piece and the
  caller got nothing; a separate `bytes_per_item < 8192` term vetoed every
  1536-wide trit tile outright, so most of the 275.6M decoder never forked. A
  two-row 4096² GEMV with `VELOCE_THREADS=4` reported **1.06 CPUs utilised**
  and ran 1.5x *slower* than one thread; the committed decode artefact had 4
  threads at 15.4 tok/s against 22.3 on one. The grain is now a byte budget
  (L2/4, expressed in items, capped by the element floor) and the per-item size
  no longer vetoes a split, so a 1.9 MB loop of 6 KB tiles splits four ways.
  New `test_parallel_for_hands_every_lane_work` counts distinct `thrd_current`
  ids, asserts every item ran exactly once and that the 1536-wide tile shape is
  forkable; with the old formula it reports `lanes=1 covered=205` and fails,
  and the old `bytes_per_item` veto makes its MIDI assertions fail too. Measured
  in the regenerated artefacts (`measured_at` 2026-11-20T21:30Z, host load 1.5):
  trit GEMV 4096² **9.19 → 26.13 GMAC/s (2.84x)** one→four threads, 275.6M
  decoder **25.1 tok/s on one thread, 62.9 on four**, KV-cached decode
  **33.7 → 63.6 tok/s (1.89x)**, 7B projection 3.04 → **4.03 tok/s** against a
  14.78 tok/s ceiling. Determinism PASS, cached verify rel err 1.25e-6. Suites
  38 → 39 e2e.

* **Two query rows share one weight pass in the trit GEMV.** A chunked decode
  multiplies the same packed matrix by two x vectors, and the `M == 2` path
  simply ran the per-row kernel twice -- the whole matrix streamed twice, so
  chunking bought only 9% (chunk 2 was 24.3 vs 22.2 tok/s at chunk 1).
  `veloce_trit_dot4x2_avx2` now loads each packed byte and its 32-byte LUT
  row once for both query rows (same four-row phase groups), with the one-row
  accumulator chains and reduction order preserved, and
  `veloce_matmul_trit` dispatches `M == 2` onto it. Microbench (100 reps,
  1 thread): 4096² **7.59 → 10.71 GMAC/s per token** (2.21 → 1.57 ms),
  K=1536 5.4 → 6.6-9.4, K=1024 5.4 → 9.4. New
  `test_trit_matmul_two_rows_share_one_weight_pass` pins the result to two
  `M == 1` passes with `memcmp` (bit-identical), on K=1003 and K=1024, plus a
  float64 reference. `make bench-midi` (artefact regenerated, host load 2.56):
  cached decode **30.48 tok/s, 32.81 ms/token** (was 25.62), 1.37x the eager
  path in the same run, determinism PASS, verify rel err 1.51e-6. Suites
  37 -> 38 e2e.

* **The cached decoder now decodes a chunk with a 2-D, per-row mask.**
  `veloce/lmmodule.py` rendered `mask: i32[CT]` and decoded one token per
  call, so the per-row softmax kernel was unit-tested but never reached end to
  end -- and with a single row a broadcast mask would have been
  indistinguishable anyway. `cached_module(cfg, slots, chunk=2)` now takes
  `tok: i32[T]`, `mask: i32[T, CT]` and writes `ck[pos + r] = k[r]` /
  `cv[pos + r] = v[r]` for every row (the renderer unrolls the stores; the DSL
  index accepts `pos + r`). Row `r` hides every slot past `pos + r`, so the
  mask rows differ and reading row 0 for all rows changes the logits --
  forcing the broadcast kernel in `codegen.py` fails
  `test_cached_decode_attention_mask` at `max|err| = 8.07`. New
  `test_cached_decode_mask_hides_garbage_slots` starts every cache row at
  `+-1e30` and still matches the float64 reference bit for bit
  (`max|err| = 5.80` if the mask stops hiding unwritten slots), and
  `TestCachedModule` pins the rendering (`mask: i32[T, CT]`, one store per
  row) and the kernel choice (`veloce_softmax_masked_rows`, never the
  broadcast kernel). `make bench-midi` records the chunked decode:
  **25.62 tok/s, 39.03 ms/token** (32 tokens, 64 slots, 2/call) against
  22.21 tok/s for the old one-token path, determinism PASS, verify rel err
  1.25e-6. Suites 36 -> 37 e2e, 44 -> 49 compiler.

* **`softmax(x, mask)` with a 2-D mask was a silent miscompile.** The
  typechecker accepted `i32[T, C]` (checking only the width) while the runtime
  read `mask[i]` for `i < C`, so every row saw mask row 0: a per-head causal
  mask produced a plausible but wrong distribution with no diagnostic. The
  lowering now picks `veloce_softmax_masked_rows` for a 2-D mask and keeps the
  broadcast `veloce_softmax_masked` for a 1-D one, both over one shared row
  body. A short 2-D mask (row count not equal to the scores rows) and any
  non-i32 mask are now compile errors; `i64`/`i8`/`i2` used to be passed
  straight through as 32-bit words. New `test_masked_softmax_per_row_mask`
  uses a mask whose rows differ and poisons the hidden entries with
  `NaN`/`+-1e30`; forcing the broadcast kernel makes it fail, so the test has
  teeth. Suites 34 -> 36 e2e, 42 -> 44 compiler.

* **`benchmarks/midi_lm_results.json` now carries the cached block.** The
  committed artefact said `"cached": null` because `make bench-midi` never
  passed `--cached`; the numbers existed only in reports. The target now runs
  `--verify --cached`, and `TestBenchmarkArtefact` fails if the JSON loses the
  block or if the Makefile target stops asking for it. Measured on the
  regenerated artefact (2026-11-18): eager 25.68 tok/s, cached decode
  **22.21 tok/s (45.02 ms/token, 32 tokens, 64 slots)**, determinism PASS,
  cached verify rel err 1.51e-6 PASS, 0.86x the eager path.

* **Trit GEMV: four output rows per pass, grouped by byte phase.** Flat
  base-3 packing puts row j at byte `(j*K)/5`, slot `(j*K)%5`, so rows only
  share an `x` window when they are in the same phase. Padding every row to a
  byte boundary was prototyped and measured first: it works (1.2-1.6x) but
  needs a new `.veloce` layout, so it was dropped. The shipped kernel keeps
  the flat format and instead groups rows `j, j+p, j+2p, j+3p`
  (`p = 5/gcd(K,5)`), which all start at slot `(j*K)%5` and then advance one
  byte per five trits: one pair of 32-byte `x` loads now feeds four LUT loads
  and four FMAs instead of re-reading `x` for every output neuron. Measured
  in one process by `make bench-roofline`, median of three runs:
  **1.33x (K=4096), 1.30x (K=1536), 1.34x (K=1024)** over the
  one-neuron-at-a-time kernel; 1-thread GEMV 8.8-9.3 GMAC/s, 4 threads
  19.7-23.0 GMAC/s (2.2-2.5x), **22-31% of the streaming peak instead of 19%**.
  End to end `make bench-midi`: **23.6 tok/s, 42.3 ms/token** (was 22.3 tok/s,
  45-50 ms), determinism PASS, verify rel err 1.54e-6. New test
  `test_trit_gemv_phase_grouped_rows` (K=1003 -> p=5, N=20 = one tile) checks
  the grouped path against a float64 reference and pins the batched path to it
  bit for bit; with the phase stride broken on purpose it fails at
  `maxerr=5.361e+01` (mutation-verified), so the test has teeth. Remaining
  wall, now narrower: one byte load + one 32-byte LUT load per five MACs is
  ~3 load slots per 5 MACs on this 128-bit-wide EPYC, i.e. ~3.3 MAC/cycle --
  not memory bandwidth.

* Corrected the bandwidth claims and fixed the benchmark that produced them.
  `bench_membw` was a scalar 1-thread loop compared against a 4-thread SIMD
  kernel, so "99% of streaming peak" compared four cores to one; it also
  divided by `T4_MS` unconditionally. Probe now runs at the same thread count
  with a real read loop, and the ratio is taken at the matching thread count.
  Honest number: GEMV = 19% of the real 18.7 GB/s peak, i.e. decode-bound.
  New `make bench-roofline` quantifies it: kernel/ceiling = 0.36-0.46x while
  reading 4x fewer bytes than a dense FMA.


* Benchmark artefacts regenerated and stamped with `measured_at` (UTC):
  `benchmarks/results.json` (2026-11-16T01:36Z) and
  `benchmarks/midi_lm_results.json` (2026-11-16T01:37Z), so a number can no
  longer be mistaken for a different host state. Current honest picture: 2.80x
  parallel speedup, 90% smaller than fp16, zero heap allocation, 3 ms compile,
  and the 7B-class 15 tok/s target still FAILING at 3.27 tok/s projected with
  a measured 2.68 tok/s hardware ceiling.

* `softmax(x, mask)` builtin: an `i32` mask excludes hidden entries from the
  row max and normaliser, so a hidden entry contributes exactly zero whatever
  it holds (tested with `NaN` and `±1e30` in the hidden columns). Replaces the
  `-1e30` score bias everywhere, including `veloce/lmmodule.py` and the cached
  benchmark driver. Measured after the switch: 32 tokens through a 64-slot
  cache at 27.84 tok/s, determinism PASS, verify rel err 1.16e-6. Suites now
  33 e2e + 40 compiler.

* **The "two-layer cached decode diverges at ~1e30" blocker was not a compiler
  bug.** The parked test's C driver built the `forward_cached` argument list by
  hand -- every KV cache first, then every weight -- while the generated
  signature interleaves them per layer (`ck1, cv1, nw1, wq1, ..., ck2, ...`).
  For `L == 1` both orders coincide, which is exactly why one layer looked
  exact and two layers exploded: every argument past `cv1` was shifted by two,
  so layer 2 read layer 1's weights and a `f32[D]` array as a trit matrix.
  `test_cached_decode_attention_mask` is un-parked (no other quarantine
  added), builds its argument list from the parsed signature, and runs 1, 2 and
  3 layers. Runtime suite 28 -> 31 tests, compiler suite 37 -> 40.
  Real measurement at the default 275M-trit shape (`--cached --cache-slots 64
  --tokens 32 --verify`): 22.34 tok/s, 44.76 ms/token, determinism PASS (one
  distinct logit hash), `max|veloce-numpy| = 1.458e-04` (relative 1.16e-06)
  PASS, 0.75x the eager path.

* **`matmul_i2g` / `matmul_fp8g` group-scaled variants** (real 2-bit and FP8
  checkpoints ship per-group scales). Both reuse the established contract --
  `f32[N, K/G]`, group count must divide K -- through a single typechecker
  rule (`_group_scaled_matmul`) and a single lowering path shared with
  `matmul_i4g`. Runtime: `veloce_i2_dot_scaled` / `veloce_fp8_dot_scaled` plus
  gemv/mat row kernels. Numeric test against a float64 reference
  (`test_group_scaled_i2_fp8_matmul_numeric`), ragged-group rejection for all
  three kernels, and codegen assertions on the derived group size.
  Also fixed a real defect the new test exposed: `weights.unpack_fp8` did
  `exp - 7` on a NumPy `uint8`, which wrapped to 249 and decoded every byte to
  ~1e77 instead of a finite value.

* `bench_midi_lm.py` now generates its decoder module from a shape config
  (`--config KEY=VAL`, `--model-dir`, `--prepare-only`) via the new
  `veloce/lmmodule.py` renderer, instead of hardcoding one model size. The
  five orphan `.vel` example modules are gone (four `codegen_*.vel` and
  `midi_lm_05b.vel`), and `examples/midi_lm.vel` is now generated output
  guarded by `test_default_render_matches_example`. `make bench-midi-05b`
  (D=2048/F=5632/7 layers) runs and verifies at rel err 1.9e-6. Compiler
  suite 32 -> 37 tests.
* AVX2 over-read fixed in the trit/i4/i2 dot kernels: the wide block loop
  could read up to three floats past `x[K-1]`, which is out of bounds on the
  caller's buffer (proved with ASAN before the fix, clean after). Each loop
  now requires enough remaining elements for its widest load.

* Orphan example files resolved by generating them instead of keeping them:
  `veloce/lmmodule.py` renders the decoder module from a shape config, and
  `bench_midi_lm.py` grew `--config KEY=VAL` / `--model-dir` (plus
  `--prepare-only`), so one command measures both shapes. The five untracked
  orphans (`codegen_lm/multi/many/multi10.vel`, `midi_lm_05b.vel`) are
  deleted. `examples/midi_lm.vel` is now the renderer's default output, and
  `TestDecoderModule.test_default_render_matches_example` fails if it drifts;
  `make bench-midi-05b` builds d_model=2048/F=5632/7 layers from the same
  code path (measured: verify max|err| 3.4e-4, relative 1.9e-6, PASS).
  Compiler suite 32 -> 37 tests.

* Documentation brought in line with the code: `i2`/`fp8` kernels and their
  measured error, the group/row-scaled variants and the "group count must
  divide K" contract, `rw` parameters and row assignment, the KV-cache example
  in `README.md`/`Makefile` (`make kv`), and the test counts (28 + 32).

* `i2` (4 values/byte) and `fp8` E4M3: runtime kernels + AVX2 paths, compiler
  lowering, packing, `.veloce` read/write, numeric tests. Measured max |err|
  5.6e-6 (i2) and 4.3e-6 (fp8); i2 footprint 1756 B vs 3511 B for i4.
* KV-cache row stores: `m[i] = vector` was a silent miscompile
  (`(float)(ptr)`); now a `veloce_copyf` row copy, validated by shape.
  `examples/run_kv_decode.py` decodes 5 tokens, max |err| 7.7e-7.
* `matmul_i4g` group-size bug: codegen used floor division while the runtime
  used ceil, giving a 35.8 error for K=100 over 7 groups. Now the group count
  must divide K, rejected at compile time otherwise.
* Test harness `cf()` helper: `f"{v:.9g}f"` emitted `-11f`, an invalid integer
  suffix, which made the suite fail to compile before any of the above.
* Numeric coverage for `matmul_i4g` and `matmul_trit_s` (previously none).
