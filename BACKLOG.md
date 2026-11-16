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

Nothing queued. The next run should pick the highest-value item itself and add
it here first, with the evidence that justifies it.

Candidates, in rough priority order:

1. **Close the 7B-class speed gap honestly.** `make bench` fails the PRD's
   15 tok/s target (3.27 projected, hardware ceiling 2.68 on this 4-vCPU VM),
   so the target is unreachable by any implementation here. Either state that
   in `PRD.md`/`README.md` as a measured hardware bound, or find a real win:
   the measured gap between the model and the streaming-read probe is still
   12-50%, which suggests the kernel is not yet at the bandwidth roofline.
2. **`benchmarks/midi_lm_results.json` has no cached block.** The cached
   numbers (27.84 tok/s, 32 tokens, 64 slots) are only in the report and the
   log, not in the committed artefact. Add them so the JSON tells the whole
   story.
3. **Attention masking for 2-D masks** is validated but only exercised with a
   1-D mask; a per-row mask (e.g. per-head causal) deserves a test.

## Done (most recent first)

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
