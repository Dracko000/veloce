/* train_native.c — latih LM byte-level sepenuhnya di C.
 *
 * Tidak ada Python di jalur training: tidak ada loop NumPy, tidak ada overhead
 * interpreter per langkah, tidak ada serialisasi. Program ini membaca korpus,
 * menjalankan forward/backward/Adam sendiri, dan menulis bobot.
 *
 * Anggaran langkah DITETAPKAN (bukan "sampai konvergen") dan tanpa RNG saat
 * training: sampel ke-i ditentukan oleh rumus, init dari seed konstan. Karena
 * itu dua jalan menghasilkan bobot bit-identik — training jadi komputasi pasti,
 * bukan proses yang "kadang begini kadang begitu".
 *
 * Bangun: cc -O3 -march=native -o build/train_native runtime/train_native.c -lm
 * Pakai : ./build/train_native data/corpus.txt [steps]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define V 256
#define W 16
#define D 64
#define H 128
#define B 64
#define LR 3e-3f
#define B1 0.9f
#define B2 0.999f
#define EPS 1e-8f
#define SPLIT 0.9

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---- RNG deterministik (xorshift64*) ---- */
static uint64_t rs = 88172645463325252ULL;
static float frand(float scale) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    double u = (double)(rs >> 11) / 9007199254740992.0;   /* [0,1) */
    return (float)((u * 2.0 - 1.0) * scale);
}

typedef struct { float *e; } Model;
typedef struct {
    float *emb;   /* [V][D]  */
    float *nw;    /* [D]     */
    float *w1;    /* [H][D]  */
    float *b1;    /* [H]     */
    float *w2;    /* [V][H]  */
    float *b2;    /* [V]     */
    /* momen Adam */
    float *m_emb, *m_nw, *m_w1, *m_b1, *m_w2, *m_b2;
    float *v_emb, *v_nw, *v_w1, *v_b1, *v_w2, *v_b2;
} Params;

static float *alloc(size_t n) {
    float *p = calloc(n, sizeof(float));
    if (!p) { fprintf(stderr, "OOM %zu\n", n); exit(1); }
    return p;
}

static void init_params(Params *p) {
    p->emb = alloc((size_t)V * D);  for (size_t i = 0; i < (size_t)V * D; i++) p->emb[i] = frand(0.05f);
    p->nw = alloc(D);               for (int i = 0; i < D; i++) p->nw[i] = 1.0f;
    p->w1 = alloc((size_t)H * D);   for (size_t i = 0; i < (size_t)H * D; i++) p->w1[i] = frand(1.0f / sqrtf(D));
    p->b1 = alloc(H);
    p->w2 = alloc((size_t)V * H);   for (size_t i = 0; i < (size_t)V * H; i++) p->w2[i] = frand(1.0f / sqrtf(H));
    p->b2 = alloc(V);
    p->m_emb = alloc((size_t)V * D); p->v_emb = alloc((size_t)V * D);
    p->m_nw = alloc(D);  p->v_nw = alloc(D);
    p->m_w1 = alloc((size_t)H * D); p->v_w1 = alloc((size_t)H * D);
    p->m_b1 = alloc(H);  p->v_b1 = alloc(H);
    p->m_w2 = alloc((size_t)V * H); p->v_w2 = alloc((size_t)V * H);
    p->m_b2 = alloc(V);  p->v_b2 = alloc(V);
}

/* ---- satu langkah: forward + backward + Adam, batch B ---- */
static float train_step(Params *P, const unsigned char *data, long ntrain, long step) {
    static float g_emb[V * D], g_nw[D], g_w1[H * D], g_b1[H], g_w2[V * H], g_b2[V];
    memset(g_emb, 0, sizeof g_emb);
    memset(g_nw, 0, sizeof g_nw);
    memset(g_w1, 0, sizeof g_w1);
    memset(g_b1, 0, sizeof g_b1);
    memset(g_w2, 0, sizeof g_w2);
    memset(g_b2, 0, sizeof g_b2);
    double loss = 0.0;

    for (int b = 0; b < B; b++) {
        /* sampel deterministik: posisi = (step*B + b) * konstanta mod rentang */
        long span = ntrain - W - 1;
        long pos = (long)(((uint64_t)(step * B + b) * 2654435761u) % (uint64_t)span);
        const unsigned char *ctx = data + pos;
        int target = data[pos + W];

        /* forward */
        float h0[D] = {0};
        for (int t = 0; t < W; t++) {
            int id = ctx[t];
            for (int d = 0; d < D; d++) h0[d] += P->emb[id * D + d];
        }
        for (int d = 0; d < D; d++) h0[d] /= (float)W;

        float ms = 0.0f;
        for (int d = 0; d < D; d++) ms += h0[d] * h0[d];
        ms = ms / (float)D + 1e-5f;
        float inv = 1.0f / sqrtf(ms);
        float y[D];
        for (int d = 0; d < D; d++) y[d] = h0[d] * inv * P->nw[d];

        float z1[H];
        for (int h = 0; h < H; h++) {
            float s = P->b1[h];
            const float *row = P->w1 + (size_t)h * D;
            for (int d = 0; d < D; d++) s += row[d] * y[d];
            z1[h] = s;
        }
        float a1[H];
        for (int h = 0; h < H; h++) a1[h] = z1[h] / (1.0f + expf(-z1[h]));

        float logits[V];
        for (int v = 0; v < V; v++) {
            float s = P->b2[v];
            const float *row = P->w2 + (size_t)v * H;
            for (int h = 0; h < H; h++) s += row[h] * a1[h];
            logits[v] = s;
        }
        float mx = logits[0];
        for (int v = 1; v < V; v++) if (logits[v] > mx) mx = logits[v];
        float sum = 0.0f;
        for (int v = 0; v < V; v++) { logits[v] = expf(logits[v] - mx); sum += logits[v]; }
        for (int v = 0; v < V; v++) logits[v] /= sum;
        loss += -log((double)logits[target] + 1e-12);

        /* backward */
        float dlog[V];
        for (int v = 0; v < V; v++) dlog[v] = logits[v];
        dlog[target] -= 1.0f;

        float da1[H] = {0};
        for (int v = 0; v < V; v++) {
            float g = dlog[v] / (float)B;
            float *row = g_w2 + (size_t)v * H;
            for (int h = 0; h < H; h++) {
                row[h] += g * a1[h];
                da1[h] += P->w2[(size_t)v * H + h] * g;
            }
            g_b2[v] += g;
        }
        float dz1[H];
        for (int h = 0; h < H; h++) {
            float s = 1.0f / (1.0f + expf(-z1[h]));
            dz1[h] = da1[h] * (s * (1.0f + z1[h] * (1.0f - s)));
            g_b1[h] += dz1[h];
            float *row = g_w1 + (size_t)h * D;
            for (int d = 0; d < D; d++) row[d] += dz1[h] * y[d];
        }
        float dy[D] = {0};
        for (int h = 0; h < H; h++) {
            const float *row = P->w1 + (size_t)h * D;
            for (int d = 0; d < D; d++) dy[d] += row[d] * dz1[h];
        }
        /* rmsnorm backward (per baris) */
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += dy[d] * h0[d] * P->nw[d];
        float dh0[D];
        for (int d = 0; d < D; d++) {
            dh0[d] = dy[d] * inv * P->nw[d] - (dot * inv * inv * inv / (float)D) * h0[d];
            g_nw[d] += dy[d] * h0[d] * inv;
        }
        for (int t = 0; t < W; t++) {
            int id = ctx[t];
            float *row = g_emb + (size_t)id * D;
            for (int d = 0; d < D; d++) row[d] += dh0[d] / (float)W;
        }
    }

    /* Adam, satu langkah, dengan bias correction */
    float bc1 = 1.0f - powf(B1, (float)(step + 1));
    float bc2 = 1.0f - powf(B2, (float)(step + 1));
    float lr_t = LR * sqrtf(bc2) / bc1;
#define ADAM(p, g, m, v, n)                                                       \
    for (size_t i = 0; i < (size_t)(n); i++) {                                    \
        (m)[i] = B1 * (m)[i] + (1 - B1) * (g)[i];                                  \
        (v)[i] = B2 * (v)[i] + (1 - B2) * (g)[i] * (g)[i];                         \
        (p)[i] -= lr_t * (m)[i] / (sqrtf((v)[i]) + EPS);                           \
    }
    ADAM(P->emb, g_emb, P->m_emb, P->v_emb, (size_t)V * D)
    ADAM(P->nw, g_nw, P->m_nw, P->v_nw, D)
    ADAM(P->w1, g_w1, P->m_w1, P->v_w1, (size_t)H * D)
    ADAM(P->b1, g_b1, P->m_b1, P->v_b1, H)
    ADAM(P->w2, g_w2, P->m_w2, P->v_w2, (size_t)V * H)
    ADAM(P->b2, g_b2, P->m_b2, P->v_b2, V)
#undef ADAM
    return (float)(loss / B);
}

static double eval_loss(Params *P, const unsigned char *data, long from, long to) {
    double tot = 0.0;
    long count = 0;
    for (long pos = from; pos + W < to; pos += 7) {   /* langkah 7: cukup untuk estimasi */
        const unsigned char *ctx = data + pos;
        int target = data[pos + W];
        float h0[D] = {0};
        for (int t = 0; t < W; t++) {
            int id = ctx[t];
            for (int d = 0; d < D; d++) h0[d] += P->emb[id * D + d];
        }
        for (int d = 0; d < D; d++) h0[d] /= (float)W;
        float ms = 0.0f;
        for (int d = 0; d < D; d++) ms += h0[d] * h0[d];
        float inv = 1.0f / sqrtf(ms / (float)D + 1e-5f);
        float y[D];
        for (int d = 0; d < D; d++) y[d] = h0[d] * inv * P->nw[d];
        float z1[H];
        for (int h = 0; h < H; h++) {
            float s = P->b1[h];
            for (int d = 0; d < D; d++) s += P->w1[(size_t)h * D + d] * y[d];
            z1[h] = s;
        }
        float a1[H];
        for (int h = 0; h < H; h++) a1[h] = z1[h] / (1.0f + expf(-z1[h]));
        float mx = -1e30f, sum = 0.0f;
        float logits[V];
        for (int v = 0; v < V; v++) {
            float s = P->b2[v];
            for (int h = 0; h < H; h++) s += P->w2[(size_t)v * H + h] * a1[h];
            logits[v] = s;
            if (s > mx) mx = s;
        }
        for (int v = 0; v < V; v++) { logits[v] = expf(logits[v] - mx); sum += logits[v]; }
        tot += -log((double)(logits[target] / sum) + 1e-12);
        count++;
    }
    return tot / (double)count;
}

static uint64_t fnv_params(Params *P) {
    uint64_t h = 1469598103934665603ULL;
    const float *blocks[] = {P->emb, P->nw, P->w1, P->b1, P->w2, P->b2};
    const size_t lens[] = {(size_t)V * D, D, (size_t)H * D, H, (size_t)V * H, V};
    for (int k = 0; k < 6; k++)
        for (size_t i = 0; i < lens[k]; i++) {
            uint32_t bits;
            memcpy(&bits, &blocks[k][i], 4);
            for (int b = 0; b < 4; b++) { h ^= (bits >> (8 * b)) & 0xFF; h *= 1099511628211ULL; }
        }
    return h;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "data/corpus.txt";
    long steps = argc > 2 ? atol(argv[2]) : 3000;
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *data = malloc((size_t)n);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) { perror("read"); return 1; }
    fclose(f);
    long cut = (long)((double)n * SPLIT);

    Params P;
    init_params(&P);

    printf("=== training native (C, tanpa Python) ===\n");
    printf("model   : byte-LM  emb %dx%d, konteks %d, hidden %d\n", V, D, W, H);
    printf("parameter: %zu (emb %d, w1 %d, w2 %d)\n",
           (size_t)V * D + D + (size_t)H * D + H + (size_t)V * H + V, V * D, H * D, V * H);
    printf("korpus  : %ld byte (latih %ld, uji %ld)\n", n, cut, n - cut);
    printf("langkah : %ld (tetap), batch %d, lr %.0e\n", steps, B, LR);

    double t0 = now_s();
    float l0 = 0.0f;
    for (long s = 0; s < steps; s++) {
        float l = train_step(&P, data, cut, s);
        if (s == 0) l0 = l;
        if (s % 250 == 0 || s == steps - 1)
            printf("  step %5ld  loss %.4f  (%.1f langkah/s)\n", s, l,
                   (double)(s + 1) / (now_s() - t0));
    }
    double el = now_s() - t0;
    double ev = eval_loss(&P, data, cut, n);

    printf("waktu   : %.2f s · %.0f langkah/s · %.2f ms/langkah\n", el, (double)steps / el,
           el * 1e3 / (double)steps);
    printf("loss    : awal %.4f -> akhir %.4f · uji %.4f (ppl %.2f)\n", l0,
           eval_loss(&P, data, 0, cut), ev, exp(ev));
    printf("hash    : %016llx (bobot akhir, bit-identik antar-run)\n",
           (unsigned long long)fnv_params(&P));

    /* simpan: float latent + trit terpaket 5 per byte (format Veloce) */
    FILE *o = fopen("build/native.f32", "wb");
    if (o) {
        fwrite(P.emb, 4, (size_t)V * D, o);
        fwrite(P.nw, 4, D, o);
        fwrite(P.w1, 4, (size_t)H * D, o);
        fwrite(P.b1, 4, H, o);
        fwrite(P.w2, 4, (size_t)V * H, o);
        fwrite(P.b2, 4, V, o);
        fclose(o);
    }
    size_t total = (size_t)V * D + (size_t)H * D + (size_t)V * H;
    static const unsigned char pow3[5] = {1, 3, 9, 27, 81};
    unsigned char *packed = calloc((total + 4) / 5, 1);
    float *src[] = {P.emb, P.w1, P.w2};
    size_t lens[] = {(size_t)V * D, (size_t)H * D, (size_t)V * H};
    size_t k = 0;
    for (int m = 0; m < 3; m++)
        for (size_t i = 0; i < lens[m]; i++, k++) {
            int t = src[m][i] > 0.1f ? 1 : (src[m][i] < -0.1f ? -1 : 0);
            packed[k / 5] += (unsigned char)((t + 1) * pow3[k % 5]);   /* 5 trit per byte, basis 3 */
        }
    o = fopen("build/native.pack", "wb");
    if (o) { fwrite(packed, 1, (total + 4) / 5, o); fclose(o); }
    printf("bobot   : build/native.f32 (%zu float) · build/native.pack (%zu byte)\n",
           (size_t)V * D + D + (size_t)H * D + H + (size_t)V * H + V, (total + 4) / 5);
    return 0;
}
