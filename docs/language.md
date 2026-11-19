# The Veloce Language (v1.0)

Veloce is a statically typed, ahead-of-time language for describing LLM
inference graphs. Tensor code is declared, not looped over.

## Module

```veloce
module my_module

dim T = 4          # bind symbolic dimensions to concrete sizes (required)
dim D = 16
dim F = 32
target "x86_64_avx512"   # optional; recorded in the MODULE entity
cache 64                 # optional; cache-line bytes, recorded in the MODULE
```

Every symbolic dimension used in a type must be bound by a `dim` at the top of
the module, because the compiler sizes all buffers statically.

## Execution model

Veloce programs are implicitly parallel: the compiled kernels partition
independent output elements (rows of a matmul / normalisation, elements of an
elementwise op) across the machine's CPU cores. There are no user-visible
threads, and every output element is accumulated by exactly one worker in a
fixed order, so results are bit-identical for any number of threads.

Threading is selected at build time by the compiler driver:

| Build | Backend | Dependency |
|-------|---------|------------|
| `velocec build ...` with OpenMP-capable `cc` | OpenMP (`-fopenmp`) | toolchain runtime |
| `velocec build ...` without OpenMP | C11 `<threads.h>` pool | libc only |
| `-DVELOCE_POOL=0` | serial | none |

Compile-time knobs: `VELOCE_THREADS=n` pins worker count, `VELOCE_PAR_MIN`
(default 1024) sets the minimum tensor size before threading, `VELOCE_PAR_GRAIN`
sets the minimum task size, `VELOCE_POOL_SPIN` sets how long an idle worker
spins before sleeping.

## Types

| Type | Meaning |
|------|---------|
| `f32` / `fp32` | 32-bit float (the compute dtype for all activations) |
| `fp16` | 16-bit float, storage |
| `trit` | 1.58-bit ternary `{-1, 0, +1}`, packed 5-per-byte, storage |
| `i4` | 4-bit signed integer, packed 2-per-byte, storage (`matmul_i4`) |
| `i8`/`u8`/`i2` | small integers (storage; `i2` has no packed kernel yet) |
| `i32`/`i64` | scalars |
| `T[N]` | 1-D tensor |
| `T[R, C]` | 2-D tensor |

Tensors may be 1-D or 2-D. Dimension entries are integer literals or symbolic
names bound by `dim`.

## Functions

```veloce
fn name(a: f32[T, D], b: trit[D, D]) -> f32[T, D] { ... }
fn pair(x: f32[D]) -> f32[D], f32[D] { ... }   # multiple returns
extern fn host_helper(x: f32[D]) -> f32[D]     # provided by the host (C ABI)
```

Parameters and returns are tensors or scalars. A tensor parameter may be a
`trit`/`i4`/`i2`/`fp8`/`fp16` *storage* tensor (weights); it is passed as a
packed pointer.

A parameter prefixed with `rw` is a **caller-owned mutable buffer**:

```veloce
fn append(rw cache: f32[CT, D], pos: i32, row: f32[1, D]) -> f32[1, D] {
    cache[pos] = row        # writes through to the caller's memory
    return row
}
```

The buffer is never allocated by the generated code and never copied in or
out, so a host loop can keep it alive across calls — that is how a KV cache is
built (`examples/kv_decode.vel`). Only `rw` parameters and `let` bindings can
be written to. The index may be an expression (`cache[pos + 1] = k[1]`), which
is what lets one call write a whole chunk of rows.

A cached decoder that consumes `T` tokens per call passes the attention mask as
`i32[T, CT]` — one row per query row, so row `r` at position `pos + r` may see
`pos + r` while row `r-1` may not. That is the per-row `softmax` form below;
a 1-D mask would give every row the same visible slots.

## Statements

```veloce
let x = expr            # mutable binding (reassignable)
const y = expr          # immutable binding
let a, b = fn(...)      # multiple binding from a multi-return call
x = expr                # assignment (target must be `let`)
v[i] = expr             # element write (1-D), or whole-row write on a 2-D tensor
m[i, j] = expr          # element write on a 2-D tensor
return expr
return a, b
if cond { ... } else { ... }
while cond { ... }
break
continue
```

`for` loops do not exist. Use declarative tensor operations instead.

## Expressions

* Arithmetic: `+ - * / // %`, unary `-`, `!`, comparisons, `&&`, `||`.
* Casts: `f32(expr)`, `i32(expr)`.
* Indexing: `v[i]` (element of 1-D, or row of 2-D), `m[i, j]` (element of 2-D).
* Matrix multiply: `a @ b` (float weights) — synonymous with `matmul(a, b)`.

### Tensor builtins (declarative)

| Builtin | Signature | Notes |
|---------|-----------|-------|
| `matmul(a, b)` / `a @ b` | `[M,K] @ [K,N] -> [M,N]` | float weights |
| `matmul_trit(x, w)` | `[M,K], trit[N,K] -> [M,N]` | **add/sub only**; `w` is `[out, in]` |
| `matmul_i4(x, w)` | `[M,K], i4[N,K] -> [M,N]` | 4-bit signed weights, 2 per byte; `w` is `[out, in]` |
| `matmul_i2(x, w)` | `[M,K], i2[N,K] -> [M,N]` | 2-bit signed weights, 4 per byte, values in `[-2, 1]` |
| `matmul_fp8(x, w)` | `[M,K], fp8[N,K] -> [M,N]` | E4M3 weights, 1 byte each, decoded to f32 in the dot |
| `matmul_i4g(x, w, s)` | `[M,K], i4[N,K], f32[N,G] -> [M,N]` | per-group scales; `G` must divide `K` |
| `matmul_i2g(x, w, s)` | `[M,K], i2[N,K], f32[N,G] -> [M,N]` | per-group scales; `G` must divide `K` |
| `matmul_fp8g(x, w, s)` | `[M,K], fp8[N,K], f32[N,G] -> [M,N]` | per-group scales; `G` must divide `K` |
| `matmul_trit_s(x, w, s)` | `[M,K], trit[N,K], f32[N] -> [M,N]` | one scale per output row |
| `matmul_i2_s(x, w, s)` | `[M,K], i2[N,K], f32[N] -> [M,N]` | one scale per output row |
| `matmul_fp8_s(x, w, s)` | `[M,K], fp8[N,K], f32[N] -> [M,N]` | one scale per output row |
| `rmsnorm(x, w)` | `[T,D], [D] -> [T,D]` | row-wise RMS norm with gain |
| `silu(x)` / `relu(x)` | `[..] -> [..]` | activation |
| `softmax(x)` | `[T,N] -> [T,N]` | row-wise softmax |
| `softmax(x, mask)` | `[T,N], i32[N] -> [T,N]` | masked row softmax: `mask[i] == 0` hides column `i` (probability exactly 0, excluded from the max and normaliser) |
| `softmax(x, mask)` | `[T,N], i32[T,N] -> [T,N]` | per-row mask: mask row `r` applies to scores row `r` (per-head causal), so different rows may hide different columns. The mask must be `i32`; other integer dtypes are rejected rather than reinterpreted |
| `transpose(x)` | `[R,C] -> [C,R]` | |
| `rope(x)` | `[T,D] -> [T,D]` | rotary position embedding |
| `embedding(table, idx)` | `[V,D], i32 -> [1,D]` | single token lookup |
| `embedding_seq(table, ids)` | `[V,D], i32[T] -> [T,D]` | batched token lookup |
| `meanpool(x)` | `[T,D] -> [1,D]` | mean over sequence axis |
| `argmax(x)` | `[..] -> i32` | index of max |
| `zeros(n)` / `zeros(r,c)` | -> tensor | zeroed activation |

Elementwise `+ - * /` work between same-shape tensors, tensor⊕scalar, and
2-D⊕1-D (bias broadcast across rows).

## Semantics & guarantees

* **Deterministic** — identical inputs produce bit-identical outputs.
* **Zero heap allocation** — every activation lives in a per-function `static`
  arena sized at compile time; buffers are reused across non-overlapping
  lifetimes.
* **C ABI** — each function compiles to `veloce_<name>(...)` with C types, so it
  links into C/C++/Rust/Go/Python `ctypes` hosts directly.
