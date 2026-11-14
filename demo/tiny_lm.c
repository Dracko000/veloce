/* Tiny-LM demo driver for the generated Veloce module examples/tiny_lm.vel.
 *
 * Builds deterministic pseudo-random weights (embedding, RMSNorm gain, and two
 * PACKED ternary weight matrices), then runs the compiled inference functions
 * and prints a few next-token predictions.
 *
 * Build:  make demo        (see Makefile at repo root)
 */
#include "veloce_runtime.h"
#include <stdio.h>

#define V 16
#define D 16
#define F 32

/* prototypes emitted by velocec (see examples/tiny_lm.vel) */
int32_t veloce_next_token(float *emb, float *nw, uint8_t *w1, uint8_t *w2, int32_t tok);
float  *veloce_logits_of(float *emb, float *nw, uint8_t *w1, uint8_t *w2, int32_t tok);

static uint32_t lcg = 123456789u;
static float frand(void) {                 /* uniform in [-1, 1) */
    lcg = lcg * 1664525u + 1013904223u;
    return ((lcg >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}
static int8_t trand(void) {                /* ternary {-1,0,1} */
    lcg = lcg * 1664525u + 1013904223u;
    return (int8_t)((lcg >> 16) % 3) - 1;
}

int main(void) {
    float emb[V * D];
    float nw[D];
    int8_t w1_t[F * D], w2_t[V * F];
    static uint8_t w1[(F * D + 4) / 5];
    static uint8_t w2[(V * F + 4) / 5];

    for (int i = 0; i < V * D; i++) emb[i] = frand();
    for (int i = 0; i < D; i++) nw[i] = 0.5f + 0.5f * frand();
    for (int i = 0; i < F * D; i++) w1_t[i] = trand();
    for (int i = 0; i < V * F; i++) w2_t[i] = trand();

    /* Pack ternary weights into base-3 bytes — exactly the on-disk layout the
     * compiler emits for .veloce weight blobs (zero-copy ready). */
    veloce_pack_trits(w1, w1_t, F * D);
    veloce_pack_trits(w2, w2_t, V * F);

    size_t w1_bytes = (F * D + 4) / 5;
    size_t w2_bytes = (V * F + 4) / 5;
    size_t fp16_would = (F * D + V * F) * 2;      /* bytes if stored fp16 */
    printf("ternary weights: %zu B packed  (vs %zu B fp16, %.1f%% smaller)\n",
           w1_bytes + w2_bytes, fp16_would,
           100.0 * (1.0 - (double)(w1_bytes + w2_bytes) / (double)fp16_would));

    printf("\nnext-token predictions:\n");
    for (int tok = 0; tok < 6; tok++) {
        int32_t pred = veloce_next_token(emb, nw, w1, w2, tok);
        float *logits = veloce_logits_of(emb, nw, w1, w2, tok);
        printf("  token %2d -> %2d   (logits:", tok, pred);
        for (int i = 0; i < V; i++) printf(" %+.2f", logits[i]);
        printf(")\n");
    }
    return 0;
}
