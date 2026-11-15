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

### 1. Documentation is stale (highest priority, blocks nothing)

Code landed for `i2`, `fp8`, KV-cache row stores and `matmul_i4g`/`matmul_trit_s`
coverage, but the docs still say otherwise:

* `docs/implementation-status.md` FR-1.1 says "`i2`/`fp8` are recognised dtype
  names only" — both now have packed kernels, `_s` variants and `.veloce`
  round-trip.
* `docs/model-capabilities.md` says "Prefix caching / KV cache ❌ (v1.0 is T=1,
  no KV cache)" — row stores work and `examples/kv_decode.vel` proves it.
* `docs/language.md` builtin table is missing `matmul_i2`, `matmul_i2_s`,
  `matmul_fp8`, `matmul_fp8_s`, `matmul_i4g`, `matmul_trit_s`; it also does not
  document `rw` parameters or row assignment (`m[i] = vector`).
* `README.md` quick-start does not mention `examples/run_kv_decode.py`.
* `Makefile` has no target for the KV-cache example.

### 2. Orphan example files — decide, do not just commit

These are referenced by nothing (no test, no Makefile target, no doc):

* `examples/codegen_lm.vel`, `codegen_multi.vel`, `codegen_many.vel`,
  `codegen_multi10.vel` — four ad-hoc next-char decoders differing only in
  shape and prompt length.
* `examples/midi_lm_05b.vel` — a 0.5B-shaped variant of `examples/midi_lm.vel`
  (D=2048/F=5632 vs D=1536/F=6144) that `benchmarks/bench_midi_lm.py` cannot
  produce, because it hardcodes `MODEL` and always compiles
  `examples/midi_lm.vel`.

Preferred resolution: make `bench_midi_lm.py` accept `--config`/`--model-dir`
so one command generates both shapes, then delete the orphan `.vel` files.
Fallback: delete them and note it in the commit message.

### 3. KV cache in the real benchmark

`benchmarks/bench_midi_lm.py` is still T=1 with no cache, so it recomputes the
whole prefix per token and its tok/s number is not comparable to a real decoder.
Add a cached path (cache tensors + `rw` parameters + a host loop) and report
both numbers side by side.

### 4. `matmul_i2` / `matmul_fp8` group-scaled variants

Only row-scaled (`_s`) exist. Real 2-bit and FP8 checkpoints ship per-group
scales (`matmul_i2g`, `matmul_fp8g`), like `matmul_i4g`. The group-size
contract is already established: the group count must divide K exactly.

### 5. Attention masking

`softmax` has no additive mask, so a KV cache can only attend over every slot.
Unwritten slots are zeros and steal probability mass. Needs either a `mask`
argument on `softmax` or an `-inf` bias vector added to the scores.

### 6. Stale benchmark artefacts

`benchmarks/results.json` holds numbers from an older host state; re-run
`make bench` and commit the fresh file, or state the date in the file.

## Done (most recent first)

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
