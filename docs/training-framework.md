# Training framework (design + current state)

Goal: **one model definition, two backends** — a NumPy reference for gradients and
Veloce-compiled C for speed. No PyTorch, no autodiff library; the graph is ours and the
compiler is ours.

```
        model builder (Python)                     the one source of truth
                 │
     ┌───────────┴────────────┐
     ▼                        ▼
 NumPy forward +          Veloce DSL emitter
 analytic grads           (forward + backward source)
 (reference, tests)              │
                                 ▼
                        veloce.compile_file → C11
                                 │
                                 ▼
                     shared lib (ctypes) or standalone trainer
                                 │
                                 ▼
                  Adam in the driver · ternary export (.veloce)
```

## What exists today

- `veloce/train/graph.py` — tensor graph with ops:
  `matmul, add, mul, transpose, rmsnorm, silu, softmax, mean_rows, embedding,
  cross_entropy, sum_sq` — each with a NumPy forward and an analytic reverse-mode rule.
- `tests/test_train.py` — three gradient checks against central finite differences:

| test | what it covers | worst relative error |
|---|---|---|
| `test_mlp_single_block` | embedding → mean-pool → rmsnorm → matmul → silu → matmul → CE | 2.4e-4 |
| `test_softmax_transpose_mul` | transpose, mul, softmax, CE | 1.7e-4 |
| `test_rmsnorm_rows` | rmsnorm over rows, weighted loss | 5.1e-4 |

Run: `python tests/test_train.py` (needs `numpy` from `.venv`).

## What is not built yet

1. **DSL emitter** — graph → `.vel` source for `forward` and `backward`. The op set above
   maps onto DSL builtins (`matmul`, `rmsnorm`, `silu`, `softmax`, `transpose`,
   `embedding`), so the emitter is a printer plus a schedule of `let` bindings; the
   backward needs transposed matmuls and the saved `inv` / `sig` / `p` tensors as extra
   inputs.
2. **Driver** — compile the emitted module to a shared object with a fixed ABI
   (`void fwd(float* out, const float* p, const int* ids)`), call it from Python with
   NumPy buffers, and run Adam there. Alternative: emit a `train.c` with the loop inside.
3. **Causal attention + mask** — `softmax_mask` exists in the DSL; the graph needs a
   `mask` op and its gradient rule (`p * (g - sum(g·p))` under the same mask).
4. **Data path** — BPE tokenizer (32k), shard writer, sequence sampler.
5. **Ternarisation on export** — `weights.pack_trits` already exists; the trainer must
   keep a float latent master copy (as `train_tiny_lm.py` does with straight-through).

## Phasing

| phase | deliverable | proof |
|---|---|---|
| 1 (done) | graph + analytic gradients | 3 gradient checks vs finite differences |
| 2 | DSL emitter for this op set | emitted module compiles; C forward matches NumPy to 1e-6 |
| 3 | backward emission + ctypes driver | compiled backward matches NumPy gradients; one Adam step lowers loss |
| 4 | causal LM block (mask, positions) | loss decreases on real text, not synthetic |
| 5 | tokenizer + shards + ternary export | trained tiny model runs through `veloce` inference |
| 6 | scale | 1B remains an inference target until GPU budget exists |

## Design decisions worth keeping

- **Three files of state per tensor**: value, grad, and the saved forward intermediates
  needed by the rule. No tape of closures — explicit attrs make the emitter trivial.
- **Ops are closed set.** Adding an op means adding forward + rule + emitter line + a
  gradient check. The test file is the contract.
- **Float latent, ternary export.** Training never uses trit weights; it ternarises at
  export, exactly like `examples/train_tiny_lm.py`.
- **The gradient check is mandatory.** A wrong rule is invisible in the loss curve for
  hundreds of steps.
