/* bigram.c — latih model bigram tanpa satu pun iterasi gradient.
 *
 * Untuk model bigram, maximum-likelihood estimate punya bentuk tertutup:
 *     P(next | prev) = count(prev, next) / count(prev)
 * Jadi "training" = satu lintasan menghitung, bukan ribuan langkah turun-gradien.
 * Ini contoh jujur kapan training bisa benar-benar tanpa loop optimisasi.
 *
 * Bangun:  cc -O2 -march=native -o build/bigram runtime/bigram.c
 * Pakai:   ./build/bigram data/corpus.txt
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define V 256
#define SPLIT 0.9

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "data/corpus.txt";
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { perror("read"); return 1; }
    fclose(f);

    long cut = (long)((double)n * SPLIT);

    /* ---- satu lintasan: hitung pasangan byte (bentuk tertutup MLE) ---- */
    double t0 = now_s();
    static uint64_t cnt[V][V];
    static uint64_t row[V];
    for (long i = 0; i + 1 < cut; i++) {
        cnt[buf[i]][buf[i + 1]]++;
        row[buf[i]]++;
    }
    double t_count = now_s() - t0;

    /* ---- evaluasi pada 10% terakhir: cross-entropy & perplexity ---- */
    double ce = 0.0;
    long seen = 0;
    for (long i = cut; i + 1 < n; i++) {
        unsigned a = buf[i], b = buf[i + 1];
        /* add-1 smoothing supaya tidak ada log(0) */
        double p = ((double)cnt[a][b] + 1.0) / ((double)row[a] + (double)V);
        ce += -log(p);
        seen++;
    }
    double ppl = exp(ce / (double)seen);

    printf("korpus        : %ld byte (latih %ld, uji %ld)\n", n, cut, n - cut);
    printf("hitung pasangan: %.3f ms (%.1f MB/s)\n", t_count * 1e3,
           (double)cut / t_count / 1e6);
    printf("cross-entropy : %.4f bit/byte (natural log: %.4f)\n", ce / (double)seen / log(2.0),
           ce / (double)seen);
    printf("perplexity    : %.2f\n", ppl);
    printf("iterasi gradien: 0\n");

    /* ---- simpan model sebagai probabilitas float (bentuk tertutup) ---- */
    static float prob[V][V];
    for (int a = 0; a < V; a++) {
        double denom = (double)row[a] + (double)V;
        for (int b = 0; b < V; b++) prob[a][b] = (float)(((double)cnt[a][b] + 1.0) / denom);
    }
    const char *out = "build/bigram.model";
    FILE *o = fopen(out, "wb");
    if (o) {
        fwrite(prob, sizeof(float), V * V, o);
        fclose(o);
        printf("model         : %s (%d float, %zu byte)\n", out, V * V, sizeof(prob));
    }

    /* prediksi teratas setelah beberapa konteks */
    const char *ctx = "a e i o t ";
    for (const char *c = ctx; *c; c++) {
        if (*c == ' ') continue;
        unsigned a = (unsigned char)*c;
        int best[3] = {0, 0, 0};
        for (int b = 0; b < V; b++)
            for (int k = 0; k < 3; k++)
                if (prob[a][b] > prob[a][best[k]]) {
                    for (int m = 2; m > k; m--) best[m] = best[m - 1];
                    best[k] = b;
                    break;
                }
        printf("  setelah '%c' -> ", *c);
        for (int k = 0; k < 3; k++) {
            char ch = (char)best[k];
            printf("%s'%c'", k ? " " : "", (ch >= 32 && ch < 127) ? ch : '?');
        }
        printf("\n");
    }
    free(buf);
    return 0;
}
