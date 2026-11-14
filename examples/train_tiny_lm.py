#!/usr/bin/env python3
"""Train, ternarize (1.58-bit), and deploy a tiny language model with Veloce.

This is the PRD's "LLM development" workflow end to end:

  1. Define the model in Veloce (`tiny_lm_cls.vel`) — a mean-pooled classifier.
  2. Train it in NumPy with quantization-aware training (straight-through
     sign estimator), so the ternary weights are learned, not bolted on.
  3. Export packed weights to the `.veloce` format (trit = 5 values / byte,
     64-byte aligned for mmap).
  4. Compile the Veloce model to C, link a driver that *mmaps* the weight
     file zero-copy, and confirm the deployed binary's predictions match the
     NumPy reference.

Run:  python3 examples/train_tiny_lm.py
"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import veloce  # noqa: E402
from veloce import weights  # noqa: E402

T, V, D, K, F = 16, 32, 64, 4, 128
TRIG = 4          # trigger tokens per class  (K*TRIG = 16 <= V)
NOISE_LO = K * TRIG   # tokens 16..31 are "noise"
EPOCHS = 240
LR = 0.02
BATCH = 32
SEED = 0


# ---------------------------------------------------------------------------
# dataset: topic classification — a doc's class is decided by which class's
# trigger tokens it contains.
# ---------------------------------------------------------------------------
def make_dataset(n, rng):
    ids = np.empty((n, T), dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    noise_pool = np.arange(NOISE_LO, V)
    for i in range(n):
        k = rng.integers(0, K)
        trig = np.arange(k * TRIG, k * TRIG + TRIG)
        n_trig = rng.integers(5, 9)
        n_noise = T - n_trig
        row = np.concatenate([
            rng.choice(trig, size=n_trig, replace=True),
            rng.choice(noise_pool, size=n_noise, replace=True),
        ])
        rng.shuffle(row)
        ids[i] = row
        labels[i] = k
    return ids, labels


def rmsnorm_fwd(x, w):
    m = np.mean(x * x, axis=-1, keepdims=True) + 1e-5
    inv = 1.0 / np.sqrt(m)
    return x * inv * w, inv


def rmsnorm_bwd(dy, x, w, inv, D):
    # per-row gradients
    d_inv = np.sum(dy * x * w, axis=-1, keepdims=True)          # [B,1]
    dx = dy * inv * w - (d_inv * inv ** 3 / D) * x
    dnw = np.sum(dy * x * inv, axis=-1)                          # [B,D] -> sum over batch later
    return dx, dnw


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward(params, ids):
    emb, nw, wq, wo, w1, w2, cls = params
    E = emb[ids]                          # [B,T,D]
    x = E.mean(axis=1)                    # [B,D]
    xn, inv1 = rmsnorm_fwd(x, nw)
    h = xn @ wq.T
    h2 = h @ wo.T
    x1 = x + h2
    xn2, inv2 = rmsnorm_fwd(x1, nw)
    f1 = xn2 @ w1.T
    a = silu(f1)
    f2 = a @ w2.T
    x2 = x1 + f2
    logits = x2 @ cls.T
    cache = (E, x, xn, inv1, h, h2, x1, xn2, inv2, f1, a, f2)
    return logits, cache


def softmax_ce(logits, labels):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    B = logits.shape[0]
    loss = -np.log(p[np.arange(B), labels] + 1e-12).mean()
    dlogits = p
    dlogits[np.arange(B), labels] -= 1.0
    dlogits /= B
    return loss, dlogits, p


def trit(w):
    return np.sign(w)


def train():
    rng = np.random.default_rng(SEED)
    tr_ids, tr_y = make_dataset(4000, rng)
    va_ids, va_y = make_dataset(800, rng)

    def init(*shape):
        return rng.standard_normal(shape) * (1.0 / np.sqrt(shape[-1]))

    # latent (float) weights; forward always uses sign() for the trit ones.
    params = [
        rng.standard_normal((V, D)) * 0.1,   # emb
        np.ones(D),                          # nw
        init(D, D), init(D, D),              # wq, wo
        init(F, D), init(D, F),              # w1, w2
        init(K, D),                          # cls
    ]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0

    for ep in range(EPOCHS):
        order = rng.permutation(len(tr_ids))
        tot = 0.0
        for s in range(0, len(tr_ids), BATCH):
            idx = order[s:s + BATCH]
            ids, y = tr_ids[idx], tr_y[idx]
            emb, nw, Wq, Wo, W1, W2, Cl = params
            wq, wo, w1, w2, cls = trit(Wq), trit(Wo), trit(W1), trit(W2), trit(Cl)
            logits, cache = forward((emb, nw, wq, wo, w1, w2, cls), ids)
            loss, dlogits, p = softmax_ce(logits, y)
            tot += loss * len(idx)

            E, x, xn, inv1, h, h2, x1, xn2, inv2, f1, a, f2 = cache
            B = ids.shape[0]
            x2 = x1 + f2
            dcls = dlogits.T @ x2                       # [K,D]
            dx2 = dlogits @ cls                         # [B,D]
            dx1 = dx2.copy()
            df2 = dx2
            dw2 = df2.T @ a
            da = df2 @ w2
            sg = sigmoid(f1)
            df1 = da * (sg * (1.0 + f1 * (1.0 - sg)))
            dw1 = df1.T @ xn2
            dxn2 = df1 @ w1
            dxn2r, dnw2b = rmsnorm_bwd(dxn2, x1, nw, inv2, D)
            dx1 += dxn2r
            dnw = dnw2b.sum(axis=0)
            dx = dx1.copy()
            dh2 = dx1
            dwo = dh2.T @ h
            dh = dh2 @ wo
            dwq = dh.T @ xn
            dxn = dh @ wq
            dxr, dnw2 = rmsnorm_bwd(dxn, x, nw, inv1, D)
            dx += dxr
            dnw += dnw2.sum(axis=0)
            # x = mean(E)  -> dE = dx/T
            dE = np.broadcast_to((dx / T)[:, None, :], (B, T, D)).copy()
            demb = np.zeros_like(emb)
            np.add.at(demb, ids, dE)

            grads = [demb, dnw, dwq, dwo, dw1, dw2, dcls]
            step += 1
            for i, g in enumerate(grads):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * (g * g)
                mh = m[i] / (1 - b1 ** step)
                vh = v[i] / (1 - b2 ** step)
                params[i] -= LR * mh / (np.sqrt(vh) + eps)

        if ep % 40 == 0 or ep == EPOCHS - 1:
            acc = evaluate(params, va_ids, va_y)
            print(f"  epoch {ep:3d}  loss {tot/len(tr_ids):.4f}  val_acc {acc*100:.1f}%")

    return params, va_ids, va_y


def cache_x2(cache):
    E, x, xn, inv1, h, h2, x1, xn2, inv2, f1, a, f2 = cache
    return x1 + f2


def predict(params, ids, use_trit=True):
    emb, nw, Wq, Wo, W1, W2, Cl = params
    if use_trit:
        wq, wo, w1, w2, cls = trit(Wq), trit(Wo), trit(W1), trit(W2), trit(Cl)
    else:
        wq, wo, w1, w2, cls = Wq, Wo, W1, W2, Cl
    logits, _ = forward((emb, nw, wq, wo, w1, w2, cls), ids)
    return logits.argmax(axis=1)


def evaluate(params, ids, y, use_trit=True):
    return float((predict(params, ids, use_trit=use_trit) == y).mean())


def main():
    print("Training tiny ternary LM (NumPy, QAT with straight-through sign)...")
    params, va_ids, va_y = train()
    emb, nw, Wq, Wo, W1, W2, Cl = params
    float_acc = evaluate(params, va_ids, va_y) * 100
    tern_acc = evaluate(params, va_ids, va_y, use_trit=True) * 100
    print(f"\nfloat model  val_acc: {float_acc:.1f}%")
    print(f"ternary (1.58-bit) val_acc: {tern_acc:.1f}%")

    # ---- export packed weights (trit where the model uses matmul_trit) ----
    tensors = [
        ("emb", emb.astype(np.float32), "f32"),
        ("nw", nw.astype(np.float32), "f32"),
        ("wq", trit(Wq).astype(np.int8), "trit"),
        ("wo", trit(Wo).astype(np.int8), "trit"),
        ("w1", trit(W1).astype(np.int8), "trit"),
        ("w2", trit(W2).astype(np.int8), "trit"),
        ("cls", trit(Cl).astype(np.int8), "trit"),
    ]
    wpath = os.path.join(HERE, "tiny_lm.veloce")
    total_params = sum(int(np.prod(t[1].shape)) for t in tensors)
    trit_params = sum(int(np.prod(t[1].shape)) for t in tensors if t[2] == "trit")
    from veloce.capabilities import make_model_card
    card = make_model_card(
        "tiny-lm-classifier-qat",
        model_type="toy ternary classifier (mean-pooled)",
        params={"total": total_params, "ternary": trit_params},
        training={
            "status": "trained (QAT)",
            "method": "NumPy straight-through ternary estimator, "
                      f"{EPOCHS} epochs, lr={LR}",
            "task": "synthetic topic classification (32-token vocabulary)",
            "val_accuracy": round(tern_acc / 100.0, 6),
        },
        capabilities={
            # A 27k-parameter toy classifier cannot support any of the four
            # pillars; all stay 'declared' -- this model makes no capability
            # claim.  Keep this dict empty to inherit the conservative default.
        },
    )
    nbytes = weights.save_veloce(wpath, tensors, model_card=card)
    print(f"\nwrote {wpath} ({nbytes} bytes)")
    print(f"  params: {total_params}  of which ternary: {trit_params} "
          f"({trit_params/total_params*100:.0f}%)")

    # ---- build a zero-copy mmap deployment -----------------------------
    dep = deploy(wpath, va_ids, va_y, params)
    print(f"\nDeployed binary (mmap'd {os.path.basename(wpath)}): "
          f"{dep['agree']}/{dep['n']} predictions match NumPy reference")
    if dep["agree"] != dep["n"]:
        raise SystemExit("FAIL: deployed binary disagrees with reference")
    print("PASS: train -> ternarize -> pack -> compile -> mmap-deploy works end to end")


def deploy(wpath, test_ids, test_y, params):
    # Compile the Veloce model to C (inference = zero runtime allocation).
    res = veloce.compile_file(os.path.join(HERE, "tiny_lm_cls.vel"))
    cpath = "/tmp/tiny_lm.gen.c"
    with open(cpath, "w") as f:
        f.write(res.c_source)
    print("\n" + res.report())

    # Read header offsets so the C driver can mmap each tensor by offset.
    vf = weights.VeloceFile(wpath)
    off = {n: vf.entries[n]["offset"] for n in vf.names()}
    vf.close()

    # Pick 32 test docs; compare deployed vs NumPy on each.
    n = 32
    ids = test_ids[:n]
    ref = predict(params, ids).tolist()

    defines = "\n".join(f'#define OFF_{k.upper()} {off[k]}' for k in off)
    ids_c = "static int32_t IDS[%d][%d] = {%s};" % (
        n, T, ",".join("{" + ",".join(str(int(v)) for v in row) + "}" for row in ids))

    driver = f"""#include "veloce_runtime.h"
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
{defines}
{ids_c}
extern int32_t veloce_predict_class(int32_t*, float*, float*,
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*);
int main(int argc, char** argv) {{
    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t* base = mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    float* emb = (float*)(base + OFF_EMB);
    float* nw  = (float*)(base + OFF_NW);
    uint8_t* wq = base + OFF_WQ, *wo = base + OFF_WO,
             *w1 = base + OFF_W1, *w2 = base + OFF_W2, *cls = base + OFF_CLS;
    for (int i = 0; i < {n}; i++)
        printf("%d\\n", (int)veloce_predict_class(IDS[i], emb, nw, wq, wo, w1, w2, cls));
    munmap(base, st.st_size); close(fd);
    return 0;
}}
"""
    dpath = "/tmp/tiny_lm.deploy.c"
    with open(dpath, "w") as f:
        f.write(driver)
    exe = "/tmp/tiny_lm.deploy"
    cc = os.environ.get("CC", "cc")
    r = subprocess.run([cc, "-std=c11", "-O3", "-march=native", "-ffast-math",
                        "-I", os.path.join(ROOT, "runtime"), dpath, cpath, "-lm",
                        "-o", exe], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("deploy compile failed")
    r = subprocess.run([exe, wpath], capture_output=True, text=True)
    got = [int(x) for x in r.stdout.split()]
    agree = sum(1 for a, b in zip(got, ref) if a == b)
    return {"agree": agree, "n": n}


if __name__ == "__main__":
    main()
