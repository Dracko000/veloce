"""Garis dasar NumPy untuk arsitektur yang SAMA dengan runtime/train_native.c."""
import time
import numpy as np

V, W, D, H, B = 256, 16, 64, 128, 64
rng = np.random.default_rng(0)
data = np.frombuffer(open('data/corpus.txt','rb').read(), dtype=np.uint8)
cut = int(len(data)*0.9)
emb = (rng.standard_normal((V,D))*0.05).astype(np.float32)
nw  = np.ones(D, dtype=np.float32)
w1  = (rng.standard_normal((H,D))/np.sqrt(D)).astype(np.float32)
b1  = np.zeros(H, dtype=np.float32)
w2  = (rng.standard_normal((V,H))/np.sqrt(H)).astype(np.float32)
b2  = np.zeros(V, dtype=np.float32)
P=[emb,nw,w1,b1,w2,b2]
M=[np.zeros_like(p) for p in P]; S=[np.zeros_like(p) for p in P]
lr,b1_,b2_,eps=3e-3,0.9,0.999,1e-8
steps=300
t0=time.perf_counter()
for s in range(steps):
    idx = ((np.arange(B)+s*B)*2654435761 % (cut-W-1)).astype(np.int64)
    ctx = data[idx[:,None]+np.arange(W)]           # [B,W]
    tgt = data[idx+W]
    h0 = emb[ctx].mean(axis=1)                      # [B,D]
    inv = 1/np.sqrt((h0*h0).mean(axis=1,keepdims=True)+1e-5)
    y = h0*inv*nw
    z1 = y@w1.T + b1
    a1 = z1/(1+np.exp(-z1))
    logits = a1@w2.T + b2
    mx = logits.max(axis=1,keepdims=True)
    e = np.exp(logits-mx); p = e/e.sum(axis=1,keepdims=True)
    loss = -np.log(p[np.arange(B),tgt]+1e-12).mean()
    dlog = p.copy(); dlog[np.arange(B),tgt]-=1; dlog/=B
    g_w2 = dlog.T@a1; g_b2 = dlog.sum(0)
    da1 = dlog@w2
    sg = 1/(1+np.exp(-z1)); dz1 = da1*(sg*(1+z1*(1-sg)))
    g_w1 = dz1.T@y; g_b1 = dz1.sum(0)
    dy = dz1@w1
    dot = (dy*h0*nw).sum(axis=1,keepdims=True)
    dh0 = dy*inv*nw - (dot*inv**3/D)*h0
    g_nw = (dy*h0*inv).sum(axis=0)
    g_emb = np.zeros_like(emb); np.add.at(g_emb, ctx.reshape(-1),
            np.repeat(dh0/W, W, axis=0))
    grads=[g_emb,g_nw,g_w1,g_b1,g_w2,g_b2]
    for i,g in enumerate(grads):
        M[i]=b1_*M[i]+(1-b1_)*g; S[i]=b2_*S[i]+(1-b2_)*g*g
        mh=M[i]/(1-b1_**(s+1)); vh=S[i]/(1-b2_**(s+1))
        P[i]-=lr*mh/(np.sqrt(vh)+eps)
el=time.perf_counter()-t0
print(f"numpy langkah/s: {steps/el:.1f} · ms/langkah {el*1e3/steps:.2f} · sampel/s {steps*B/el:.0f} · loss akhir {float(loss):.4f}")
