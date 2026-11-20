/* Veloce runtime — zero-allocation, static-arena tensor kernels.
 *
 * Everything here is `static inline` so a generated translation unit stays a
 * single self-contained object with no external runtime dependency (PRD NFR:
 * "single standalone compiler binary ... zero external runtime dependencies").
 *
 * Storage dtypes
 * --------------
 *   trit  : weights stored PACKED, 5 base-3 trits per byte. Kernels decode the
 *           trit {-1,0,+1} inline, so an mmap'd packed weight binary is used
 *           zero-copy (FR-1.2 / FR-3.3).
 *   fp16  : uint16_t IEEE-half storage; decoded on the fly.
 *   f32   : float activations (the compute dtype).
 *
 * Parallelism (FR-4.2)
 * --------------------
 * Work-splitting in the hot kernels goes through a tiny internal thread pool
 * (`veloce_parallel_for`). Build the generated object with `-fopenmp` to use
 * OpenMP; without it the runtime falls back to C11 <threads.h>, and where
 * neither is available every kernel runs serially. All modes are bit-exact:
 * each output element is accumulated in a fixed order by one worker.
 *
 * Tuning knobs (compile-time, all optional):
 *   VELOCE_THREADS       worker count (default: online CPUs)
 *   VELOCE_PAR_MIN       min output elements before threading (default 1024)
 *   VELOCE_PAR_GRAIN     min output elements per worker task  (default 512)
 *   VELOCE_POOL          thread-pool backend: auto|omp|threads|none
 */
#ifndef VELOCE_RUNTIME_H
#define VELOCE_RUNTIME_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdatomic.h>

/* ------------------------------------------------------------------ */
/* Parallel execution policy (FR-4.2)                                 */
/* ------------------------------------------------------------------ */
#ifndef VELOCE_THREADS
#define VELOCE_THREADS 0              /* 0 = decide at runtime */
#endif
#ifndef VELOCE_PAR_MIN
#define VELOCE_PAR_MIN 1024           /* small problems stay serial */
#endif
#ifndef VELOCE_PAR_GRAIN
#define VELOCE_PAR_GRAIN 512          /* min elements per parallel task */
#endif
#ifndef VELOCE_POOL_SPIN
#define VELOCE_POOL_SPIN 8000         /* spin iterations before sleeping */
#endif

#if defined(OPENMP) || defined(_OPENMP)
#  define VELOCE_HAVE_OMP 1
#  include <omp.h>
#else
#  define VELOCE_HAVE_OMP 0
#endif

#if !defined(__STDC_NO_THREADS__) && (defined(__unix__) || defined(__APPLE__)) \
    && !defined(__wasi__) && !defined(__EMSCRIPTEN__)
#  define VELOCE_HAVE_THREADS 1
#  include <threads.h>
#else
#  define VELOCE_HAVE_THREADS 0
#endif

#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>
#endif

#ifndef VELOCE_POOL
#  if VELOCE_HAVE_OMP
#    define VELOCE_POOL 2             /* omp */
#  elif VELOCE_HAVE_THREADS
#    define VELOCE_POOL 1             /* threads.h */
#  else
#    define VELOCE_POOL 0             /* serial */
#  endif
#endif

static inline size_t veloce_cpu_count(void) {
#if VELOCE_HAVE_OMP
    return (size_t)omp_get_max_threads();
#elif defined(__unix__) || defined(__APPLE__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (size_t)n : 1;
#else
    return 1;
#endif
}

/* Worker count: a compile-time pin (`-DVELOCE_THREADS=n`) or the
 * `VELOCE_THREADS` environment variable override; otherwise one worker per
 * online CPU. Automatic under all normal builds — no user-managed threads. */
static inline size_t veloce_threads(void) {
#if VELOCE_POOL == 0
    return 1;
#elif VELOCE_THREADS > 0
    return (size_t)(VELOCE_THREADS);   /* compile-time pin wins */
#else
    const char *env = getenv("VELOCE_THREADS");
    if (env && *env) {
        int n = atoi(env);
        if (n > 0) return (size_t)n;
    }
    static size_t cached = 0;
    if (!cached) cached = veloce_cpu_count();
    return cached;
#endif
}

/* Cache topology (FR-4.2): used to decide whether a tensor is big enough to
 * pay for cross-core work splitting. Values are advisory; fallbacks are used
 * on platforms without sysconf cache queries. */
static inline size_t veloce_cache_bytes(int level) {
#if defined(__linux__)
    int name = level == 1 ? _SC_LEVEL1_DCACHE_SIZE
             : level == 2 ? _SC_LEVEL2_CACHE_SIZE
                          : _SC_LEVEL3_CACHE_SIZE;
    long n = sysconf(name);
    if (n > 0) return (size_t)n;
#endif
    return level == 1 ? (size_t)32 * 1024
         : level == 2 ? (size_t)256 * 1024
                      : (size_t)8 * 1024 * 1024;
}

/* A tensor is worth threading only when it does not fit comfortably in L2
 * (below that a single core streams it faster than a team can be launched). */
static inline int veloce_worth_parallel(size_t n, size_t bytes_per_item) {
    if (veloce_threads() <= 1 || n < 2) return 0;
    /* Enough independent work to amortise a fork-join?  The working set must
     * exceed half of L2.  The per-item size must not veto the split on its
     * own: a 1536-wide trit tile is 6 KB of packed weights, but 308 of them
     * are 1.9 MB, exactly the size that pays for a fork.  How many ways it
     * splits is decided by the chunk grain below, which is a byte budget. */
    size_t total = n * bytes_per_item;
    size_t l2 = veloce_cache_bytes(2);
    return total >= l2 / 2;
}

/* Smallest chunk worth handing to a worker: a byte budget (a quarter of L2)
 * expressed in items, never larger than the element floor VELOCE_PAR_GRAIN.
 * An element is not a unit of work -- one item can be a 4-byte float or a
 * 16 KB packed-weight tile -- so a floor expressed only in elements silently
 * serialised whole kernels: `chunk = max(ceil(n/chunks), 512)` gives worker 0
 * every item of any loop with n < 512*chunks and leaves the caller empty
 * (measured: a 4096^2 two-row GEMV reported 1.06 CPUs utilised with
 * VELOCE_THREADS=4, and 4 threads were 1.5x *slower* than one). */
static inline size_t veloce_chunk_grain(size_t bytes_per_item) {
    size_t grain = (size_t)VELOCE_PAR_GRAIN;
    if (bytes_per_item) {
        const size_t budget = veloce_cache_bytes(2) / 4;
        size_t by_bytes = (budget + bytes_per_item - 1) / bytes_per_item;
        if (by_bytes < 1) by_bytes = 1;
        if (by_bytes < grain) grain = by_bytes;
    }
    if (grain < 1) grain = 1;
    return grain;
}

/* Iterate `n` independent output elements. `body(ctx, i)` must be
 * deterministic per element; workers never share an accumulator. */
typedef void (*veloce_body_fn)(void *ctx, size_t i);

#if VELOCE_POOL == 2
static inline void veloce_parallel_for_sized(size_t n, size_t bytes_per_item,
                                             veloce_body_fn body, void *ctx) {
    if (!veloce_worth_parallel(n, bytes_per_item) || omp_in_parallel())
        { for (size_t i = 0; i < n; i++) body(ctx, i); return; }
    #pragma omp parallel for schedule(static) num_threads((int)veloce_threads())
    for (size_t i = 0; i < n; i++) body(ctx, i);
}
static inline void veloce_parallel_for(size_t n, veloce_body_fn body, void *ctx) {
    veloce_parallel_for_sized(n, (size_t)sizeof(float), body, ctx);
}
#elif VELOCE_POOL == 1 /* ---- C11 threads.h pool ---- */
#define VELOCE_MAX_WORKERS 64

typedef struct veloce_pool veloce_pool_t;
typedef struct { veloce_pool_t *pool; size_t id; } veloce_worker_t;

struct veloce_pool {
    mtx_t lock; cnd_t wake; cnd_t done;
    veloce_body_fn fn; void *ctx;
    size_t n, chunk;
    atomic_size_t generation;      /* bumped by the caller for every fork */
    atomic_size_t remaining;       /* workers that have not finished yet */
    atomic_int quitting;
    size_t started;                /* workers parked and ready */
    size_t nthreads;
    thrd_t threads[VELOCE_MAX_WORKERS];
    veloce_worker_t workers[VELOCE_MAX_WORKERS];
};

/* Pause hint for the spin phase of the fork-join handshake. */
static inline void veloce_cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}

/* Threads created by this pool; used to detect re-entrant parallel calls. */
static _Thread_local int veloce_in_parallel = 0;

static inline size_t veloce_pool_chunks(const veloce_pool_t *p) {
    return p->nthreads + 1;               /* workers + calling thread */
}

static inline int veloce_pool_worker(void *arg) {
    veloce_worker_t *w = (veloce_worker_t *)arg;
    veloce_pool_t *p = w->pool;
    size_t seen = atomic_load_explicit(&p->generation, memory_order_acquire);
    /* Startup barrier: park only once the pool knows this worker is live, so
     * a fork can never be published before every worker has a baseline. */
    mtx_lock(&p->lock);
    p->started++;
    cnd_broadcast(&p->done);
    mtx_unlock(&p->lock);
    for (;;) {
        /* Spin before sleeping.  A transformer layer issues ~30 tiny GEMVs per
         * token; if every fork costs a futex round-trip the pool spends more
         * time waking up than working (measured: 4 threads ~= 1 thread). */
        int spins = 0;
        while (atomic_load_explicit(&p->generation, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&p->quitting, memory_order_relaxed)) return 0;
            if (++spins <= VELOCE_POOL_SPIN) {
                veloce_cpu_relax();
                continue;
            }
            mtx_lock(&p->lock);
            while (atomic_load_explicit(&p->generation, memory_order_acquire) == seen &&
                   !atomic_load_explicit(&p->quitting, memory_order_relaxed))
                cnd_wait(&p->wake, &p->lock);
            mtx_unlock(&p->lock);
        }
        if (atomic_load_explicit(&p->quitting, memory_order_relaxed)) return 0;
        seen = atomic_load_explicit(&p->generation, memory_order_acquire);

        mtx_lock(&p->lock);
        veloce_body_fn fn = p->fn; void *ctx = p->ctx;
        size_t begin = w->id * p->chunk;
        size_t end = begin + p->chunk;
        if (begin > p->n) begin = p->n;
        if (end > p->n) end = p->n;
        mtx_unlock(&p->lock);
        veloce_in_parallel++;
        for (size_t i = begin; i < end; i++) fn(ctx, i);
        veloce_in_parallel--;
        if (atomic_fetch_sub_explicit(&p->remaining, 1, memory_order_acq_rel) == 1) {
            mtx_lock(&p->lock);
            cnd_signal(&p->done);
            mtx_unlock(&p->lock);
        }
    }
}

static veloce_pool_t veloce_global_pool;
static once_flag veloce_pool_once = ONCE_FLAG_INIT;
static void veloce_pool_init(void) {
    veloce_pool_t *p = &veloce_global_pool;
    mtx_init(&p->lock, mtx_plain);
    cnd_init(&p->wake);
    cnd_init(&p->done);
    size_t want = veloce_threads();
    if (want > VELOCE_MAX_WORKERS) want = VELOCE_MAX_WORKERS;
    size_t made = 0;
    for (size_t i = 0; i + 1 < want; i++) {
        p->workers[i].pool = p;
        p->workers[i].id = i;
        if (thrd_create(&p->threads[i], veloce_pool_worker, &p->workers[i]) != thrd_success)
            break;
        made++;
    }
    p->nthreads = made;
    /* Wait until every worker has parked: a fork must not be published before
     * a worker has captured its baseline generation. */
    mtx_lock(&p->lock);
    while (p->started < p->nthreads) cnd_wait(&p->done, &p->lock);
    mtx_unlock(&p->lock);
}

static inline void veloce_parallel_for_sized(size_t n, size_t bytes_per_item,
                                             veloce_body_fn body, void *ctx) {
    if (!veloce_worth_parallel(n, bytes_per_item) || veloce_in_parallel) {
        for (size_t i = 0; i < n; i++) body(ctx, i);
        return;
    }
    call_once(&veloce_pool_once, veloce_pool_init);
    veloce_pool_t *p = &veloce_global_pool;
    if (p->nthreads == 0) {
        for (size_t i = 0; i < n; i++) body(ctx, i);
        return;
    }
    size_t chunks = veloce_pool_chunks(p);
    /* Split into at most as many chunks as the grain budget allows, then as
     * many as there are lanes: every lane gets real work or stays out of the
     * fork, never "worker 0 takes everything". */
    size_t grain = veloce_chunk_grain(bytes_per_item);
    size_t max_chunks = n / grain;
    if (max_chunks < 1) max_chunks = 1;
    if (max_chunks > chunks) max_chunks = chunks;
    size_t chunk = (n + max_chunks - 1) / max_chunks;

    mtx_lock(&p->lock);
    p->fn = body; p->ctx = ctx; p->n = n; p->chunk = chunk;
    atomic_store_explicit(&p->remaining, p->nthreads, memory_order_relaxed);
    atomic_store_explicit(&p->generation,
                          atomic_load_explicit(&p->generation, memory_order_relaxed) + 1,
                          memory_order_release);
    cnd_broadcast(&p->wake);
    mtx_unlock(&p->lock);

    /* the calling thread takes the final chunk */
    size_t begin = p->nthreads * chunk;
    if (begin > n) begin = n;
    veloce_in_parallel++;
    for (size_t i = begin; i < n; i++) body(ctx, i);
    veloce_in_parallel--;

    /* Spin for the stragglers before sleeping: a futex round-trip on the
     * caller side costs more than the tail of most transformer GEMVs. */
    int spins = 0;
    while (atomic_load_explicit(&p->remaining, memory_order_acquire) != 0) {
        if (++spins <= VELOCE_POOL_SPIN) { veloce_cpu_relax(); continue; }
        mtx_lock(&p->lock);
        if (atomic_load_explicit(&p->remaining, memory_order_acquire) != 0)
            cnd_wait(&p->done, &p->lock);
        mtx_unlock(&p->lock);
    }
}
static inline void veloce_parallel_for(size_t n, veloce_body_fn body, void *ctx) {
    veloce_parallel_for_sized(n, (size_t)sizeof(float), body, ctx);
}
#else /* ---- serial ---- */
static inline void veloce_parallel_for_sized(size_t n, size_t bytes_per_item,
                                             veloce_body_fn body, void *ctx) {
    (void)bytes_per_item;
    for (size_t i = 0; i < n; i++) body(ctx, i);
}
static inline void veloce_parallel_for(size_t n, veloce_body_fn body, void *ctx) {
    for (size_t i = 0; i < n; i++) body(ctx, i);
}
#endif

/* ------------------------------------------------------------------ */
/* Trit bit-packing (FR-1.2): 5 trits per byte, base-3 encoding.       */
/*   digit 0 -> -1, 1 -> 0, 2 -> +1                                    */
/* ------------------------------------------------------------------ */
static inline int8_t veloce_unpack_trit(uint8_t word, int slot) {
    static const int8_t lut[3] = { -1, 0, 1 };
    int divisor = 1;
    for (int i = 0; i < slot; i++) divisor *= 3;
    return lut[(word / divisor) % 3];
}

static inline uint8_t veloce_pack_trit5(const int8_t *vals) {
    int w = 0, p = 1;
    for (int i = 0; i < 5; i++) { w += (vals[i] + 1) * p; p *= 3; }
    return (uint8_t)w;
}

/* 256-entry float decode table: one packed byte -> five floats in {-1,0,+1}.
 * Padded to 8 floats / 32 bytes so an AVX2 FMA can consume one byte's worth
 * of weights per load (5 useful lanes, 3 zero lanes). */
#define VELOCE_TRIT_F(b) \
    { (float)(((b) % 3) - 1), (float)((((b) / 3) % 3) - 1), \
      (float)((((b) / 9) % 3) - 1), (float)((((b) / 27) % 3) - 1), \
      (float)((((b) / 81) % 3) - 1), 0.0f, 0.0f, 0.0f }
#define VELOCE_TRIT_F16(b) \
    VELOCE_TRIT_F(b), VELOCE_TRIT_F(b+1), VELOCE_TRIT_F(b+2), VELOCE_TRIT_F(b+3), \
    VELOCE_TRIT_F(b+4), VELOCE_TRIT_F(b+5), VELOCE_TRIT_F(b+6), VELOCE_TRIT_F(b+7), \
    VELOCE_TRIT_F(b+8), VELOCE_TRIT_F(b+9), VELOCE_TRIT_F(b+10), VELOCE_TRIT_F(b+11), \
    VELOCE_TRIT_F(b+12), VELOCE_TRIT_F(b+13), VELOCE_TRIT_F(b+14), VELOCE_TRIT_F(b+15)
#define VELOCE_TRIT_F64(b) \
    VELOCE_TRIT_F16(b), VELOCE_TRIT_F16(b+16), VELOCE_TRIT_F16(b+32), VELOCE_TRIT_F16(b+48)
static const float VELOCE_TRIT_LUT_F32[256][8] __attribute__((aligned(32))) = {
    VELOCE_TRIT_F64(0), VELOCE_TRIT_F64(64), VELOCE_TRIT_F64(128), VELOCE_TRIT_F64(192)
};

/* Number of bytes a row of `k` trits occupies. */
static inline size_t veloce_trit_rowbytes(size_t k) { return (k + 4) / 5; }

/* 256-entry decode table: one packed byte -> its five trits (-1/0/+1).
 * A byte can only encode 3^5 = 243 distinct vectors; this removes the
 * per-element divide/modulo loop that dominated the old trit kernel. */
static const int8_t VELOCE_TRIT_LUT[256][8] = {
{-1,-1,-1,-1,-1, 0, 0, 0},
{ 0,-1,-1,-1,-1, 0, 0, 0},
{ 1,-1,-1,-1,-1, 0, 0, 0},
{-1, 0,-1,-1,-1, 0, 0, 0},
{ 0, 0,-1,-1,-1, 0, 0, 0},
{ 1, 0,-1,-1,-1, 0, 0, 0},
{-1, 1,-1,-1,-1, 0, 0, 0},
{ 0, 1,-1,-1,-1, 0, 0, 0},
{ 1, 1,-1,-1,-1, 0, 0, 0},
{-1,-1, 0,-1,-1, 0, 0, 0},
{ 0,-1, 0,-1,-1, 0, 0, 0},
{ 1,-1, 0,-1,-1, 0, 0, 0},
{-1, 0, 0,-1,-1, 0, 0, 0},
{ 0, 0, 0,-1,-1, 0, 0, 0},
{ 1, 0, 0,-1,-1, 0, 0, 0},
{-1, 1, 0,-1,-1, 0, 0, 0},
{ 0, 1, 0,-1,-1, 0, 0, 0},
{ 1, 1, 0,-1,-1, 0, 0, 0},
{-1,-1, 1,-1,-1, 0, 0, 0},
{ 0,-1, 1,-1,-1, 0, 0, 0},
{ 1,-1, 1,-1,-1, 0, 0, 0},
{-1, 0, 1,-1,-1, 0, 0, 0},
{ 0, 0, 1,-1,-1, 0, 0, 0},
{ 1, 0, 1,-1,-1, 0, 0, 0},
{-1, 1, 1,-1,-1, 0, 0, 0},
{ 0, 1, 1,-1,-1, 0, 0, 0},
{ 1, 1, 1,-1,-1, 0, 0, 0},
{-1,-1,-1, 0,-1, 0, 0, 0},
{ 0,-1,-1, 0,-1, 0, 0, 0},
{ 1,-1,-1, 0,-1, 0, 0, 0},
{-1, 0,-1, 0,-1, 0, 0, 0},
{ 0, 0,-1, 0,-1, 0, 0, 0},
{ 1, 0,-1, 0,-1, 0, 0, 0},
{-1, 1,-1, 0,-1, 0, 0, 0},
{ 0, 1,-1, 0,-1, 0, 0, 0},
{ 1, 1,-1, 0,-1, 0, 0, 0},
{-1,-1, 0, 0,-1, 0, 0, 0},
{ 0,-1, 0, 0,-1, 0, 0, 0},
{ 1,-1, 0, 0,-1, 0, 0, 0},
{-1, 0, 0, 0,-1, 0, 0, 0},
{ 0, 0, 0, 0,-1, 0, 0, 0},
{ 1, 0, 0, 0,-1, 0, 0, 0},
{-1, 1, 0, 0,-1, 0, 0, 0},
{ 0, 1, 0, 0,-1, 0, 0, 0},
{ 1, 1, 0, 0,-1, 0, 0, 0},
{-1,-1, 1, 0,-1, 0, 0, 0},
{ 0,-1, 1, 0,-1, 0, 0, 0},
{ 1,-1, 1, 0,-1, 0, 0, 0},
{-1, 0, 1, 0,-1, 0, 0, 0},
{ 0, 0, 1, 0,-1, 0, 0, 0},
{ 1, 0, 1, 0,-1, 0, 0, 0},
{-1, 1, 1, 0,-1, 0, 0, 0},
{ 0, 1, 1, 0,-1, 0, 0, 0},
{ 1, 1, 1, 0,-1, 0, 0, 0},
{-1,-1,-1, 1,-1, 0, 0, 0},
{ 0,-1,-1, 1,-1, 0, 0, 0},
{ 1,-1,-1, 1,-1, 0, 0, 0},
{-1, 0,-1, 1,-1, 0, 0, 0},
{ 0, 0,-1, 1,-1, 0, 0, 0},
{ 1, 0,-1, 1,-1, 0, 0, 0},
{-1, 1,-1, 1,-1, 0, 0, 0},
{ 0, 1,-1, 1,-1, 0, 0, 0},
{ 1, 1,-1, 1,-1, 0, 0, 0},
{-1,-1, 0, 1,-1, 0, 0, 0},
{ 0,-1, 0, 1,-1, 0, 0, 0},
{ 1,-1, 0, 1,-1, 0, 0, 0},
{-1, 0, 0, 1,-1, 0, 0, 0},
{ 0, 0, 0, 1,-1, 0, 0, 0},
{ 1, 0, 0, 1,-1, 0, 0, 0},
{-1, 1, 0, 1,-1, 0, 0, 0},
{ 0, 1, 0, 1,-1, 0, 0, 0},
{ 1, 1, 0, 1,-1, 0, 0, 0},
{-1,-1, 1, 1,-1, 0, 0, 0},
{ 0,-1, 1, 1,-1, 0, 0, 0},
{ 1,-1, 1, 1,-1, 0, 0, 0},
{-1, 0, 1, 1,-1, 0, 0, 0},
{ 0, 0, 1, 1,-1, 0, 0, 0},
{ 1, 0, 1, 1,-1, 0, 0, 0},
{-1, 1, 1, 1,-1, 0, 0, 0},
{ 0, 1, 1, 1,-1, 0, 0, 0},
{ 1, 1, 1, 1,-1, 0, 0, 0},
{-1,-1,-1,-1, 0, 0, 0, 0},
{ 0,-1,-1,-1, 0, 0, 0, 0},
{ 1,-1,-1,-1, 0, 0, 0, 0},
{-1, 0,-1,-1, 0, 0, 0, 0},
{ 0, 0,-1,-1, 0, 0, 0, 0},
{ 1, 0,-1,-1, 0, 0, 0, 0},
{-1, 1,-1,-1, 0, 0, 0, 0},
{ 0, 1,-1,-1, 0, 0, 0, 0},
{ 1, 1,-1,-1, 0, 0, 0, 0},
{-1,-1, 0,-1, 0, 0, 0, 0},
{ 0,-1, 0,-1, 0, 0, 0, 0},
{ 1,-1, 0,-1, 0, 0, 0, 0},
{-1, 0, 0,-1, 0, 0, 0, 0},
{ 0, 0, 0,-1, 0, 0, 0, 0},
{ 1, 0, 0,-1, 0, 0, 0, 0},
{-1, 1, 0,-1, 0, 0, 0, 0},
{ 0, 1, 0,-1, 0, 0, 0, 0},
{ 1, 1, 0,-1, 0, 0, 0, 0},
{-1,-1, 1,-1, 0, 0, 0, 0},
{ 0,-1, 1,-1, 0, 0, 0, 0},
{ 1,-1, 1,-1, 0, 0, 0, 0},
{-1, 0, 1,-1, 0, 0, 0, 0},
{ 0, 0, 1,-1, 0, 0, 0, 0},
{ 1, 0, 1,-1, 0, 0, 0, 0},
{-1, 1, 1,-1, 0, 0, 0, 0},
{ 0, 1, 1,-1, 0, 0, 0, 0},
{ 1, 1, 1,-1, 0, 0, 0, 0},
{-1,-1,-1, 0, 0, 0, 0, 0},
{ 0,-1,-1, 0, 0, 0, 0, 0},
{ 1,-1,-1, 0, 0, 0, 0, 0},
{-1, 0,-1, 0, 0, 0, 0, 0},
{ 0, 0,-1, 0, 0, 0, 0, 0},
{ 1, 0,-1, 0, 0, 0, 0, 0},
{-1, 1,-1, 0, 0, 0, 0, 0},
{ 0, 1,-1, 0, 0, 0, 0, 0},
{ 1, 1,-1, 0, 0, 0, 0, 0},
{-1,-1, 0, 0, 0, 0, 0, 0},
{ 0,-1, 0, 0, 0, 0, 0, 0},
{ 1,-1, 0, 0, 0, 0, 0, 0},
{-1, 0, 0, 0, 0, 0, 0, 0},
{ 0, 0, 0, 0, 0, 0, 0, 0},
{ 1, 0, 0, 0, 0, 0, 0, 0},
{-1, 1, 0, 0, 0, 0, 0, 0},
{ 0, 1, 0, 0, 0, 0, 0, 0},
{ 1, 1, 0, 0, 0, 0, 0, 0},
{-1,-1, 1, 0, 0, 0, 0, 0},
{ 0,-1, 1, 0, 0, 0, 0, 0},
{ 1,-1, 1, 0, 0, 0, 0, 0},
{-1, 0, 1, 0, 0, 0, 0, 0},
{ 0, 0, 1, 0, 0, 0, 0, 0},
{ 1, 0, 1, 0, 0, 0, 0, 0},
{-1, 1, 1, 0, 0, 0, 0, 0},
{ 0, 1, 1, 0, 0, 0, 0, 0},
{ 1, 1, 1, 0, 0, 0, 0, 0},
{-1,-1,-1, 1, 0, 0, 0, 0},
{ 0,-1,-1, 1, 0, 0, 0, 0},
{ 1,-1,-1, 1, 0, 0, 0, 0},
{-1, 0,-1, 1, 0, 0, 0, 0},
{ 0, 0,-1, 1, 0, 0, 0, 0},
{ 1, 0,-1, 1, 0, 0, 0, 0},
{-1, 1,-1, 1, 0, 0, 0, 0},
{ 0, 1,-1, 1, 0, 0, 0, 0},
{ 1, 1,-1, 1, 0, 0, 0, 0},
{-1,-1, 0, 1, 0, 0, 0, 0},
{ 0,-1, 0, 1, 0, 0, 0, 0},
{ 1,-1, 0, 1, 0, 0, 0, 0},
{-1, 0, 0, 1, 0, 0, 0, 0},
{ 0, 0, 0, 1, 0, 0, 0, 0},
{ 1, 0, 0, 1, 0, 0, 0, 0},
{-1, 1, 0, 1, 0, 0, 0, 0},
{ 0, 1, 0, 1, 0, 0, 0, 0},
{ 1, 1, 0, 1, 0, 0, 0, 0},
{-1,-1, 1, 1, 0, 0, 0, 0},
{ 0,-1, 1, 1, 0, 0, 0, 0},
{ 1,-1, 1, 1, 0, 0, 0, 0},
{-1, 0, 1, 1, 0, 0, 0, 0},
{ 0, 0, 1, 1, 0, 0, 0, 0},
{ 1, 0, 1, 1, 0, 0, 0, 0},
{-1, 1, 1, 1, 0, 0, 0, 0},
{ 0, 1, 1, 1, 0, 0, 0, 0},
{ 1, 1, 1, 1, 0, 0, 0, 0},
{-1,-1,-1,-1, 1, 0, 0, 0},
{ 0,-1,-1,-1, 1, 0, 0, 0},
{ 1,-1,-1,-1, 1, 0, 0, 0},
{-1, 0,-1,-1, 1, 0, 0, 0},
{ 0, 0,-1,-1, 1, 0, 0, 0},
{ 1, 0,-1,-1, 1, 0, 0, 0},
{-1, 1,-1,-1, 1, 0, 0, 0},
{ 0, 1,-1,-1, 1, 0, 0, 0},
{ 1, 1,-1,-1, 1, 0, 0, 0},
{-1,-1, 0,-1, 1, 0, 0, 0},
{ 0,-1, 0,-1, 1, 0, 0, 0},
{ 1,-1, 0,-1, 1, 0, 0, 0},
{-1, 0, 0,-1, 1, 0, 0, 0},
{ 0, 0, 0,-1, 1, 0, 0, 0},
{ 1, 0, 0,-1, 1, 0, 0, 0},
{-1, 1, 0,-1, 1, 0, 0, 0},
{ 0, 1, 0,-1, 1, 0, 0, 0},
{ 1, 1, 0,-1, 1, 0, 0, 0},
{-1,-1, 1,-1, 1, 0, 0, 0},
{ 0,-1, 1,-1, 1, 0, 0, 0},
{ 1,-1, 1,-1, 1, 0, 0, 0},
{-1, 0, 1,-1, 1, 0, 0, 0},
{ 0, 0, 1,-1, 1, 0, 0, 0},
{ 1, 0, 1,-1, 1, 0, 0, 0},
{-1, 1, 1,-1, 1, 0, 0, 0},
{ 0, 1, 1,-1, 1, 0, 0, 0},
{ 1, 1, 1,-1, 1, 0, 0, 0},
{-1,-1,-1, 0, 1, 0, 0, 0},
{ 0,-1,-1, 0, 1, 0, 0, 0},
{ 1,-1,-1, 0, 1, 0, 0, 0},
{-1, 0,-1, 0, 1, 0, 0, 0},
{ 0, 0,-1, 0, 1, 0, 0, 0},
{ 1, 0,-1, 0, 1, 0, 0, 0},
{-1, 1,-1, 0, 1, 0, 0, 0},
{ 0, 1,-1, 0, 1, 0, 0, 0},
{ 1, 1,-1, 0, 1, 0, 0, 0},
{-1,-1, 0, 0, 1, 0, 0, 0},
{ 0,-1, 0, 0, 1, 0, 0, 0},
{ 1,-1, 0, 0, 1, 0, 0, 0},
{-1, 0, 0, 0, 1, 0, 0, 0},
{ 0, 0, 0, 0, 1, 0, 0, 0},
{ 1, 0, 0, 0, 1, 0, 0, 0},
{-1, 1, 0, 0, 1, 0, 0, 0},
{ 0, 1, 0, 0, 1, 0, 0, 0},
{ 1, 1, 0, 0, 1, 0, 0, 0},
{-1,-1, 1, 0, 1, 0, 0, 0},
{ 0,-1, 1, 0, 1, 0, 0, 0},
{ 1,-1, 1, 0, 1, 0, 0, 0},
{-1, 0, 1, 0, 1, 0, 0, 0},
{ 0, 0, 1, 0, 1, 0, 0, 0},
{ 1, 0, 1, 0, 1, 0, 0, 0},
{-1, 1, 1, 0, 1, 0, 0, 0},
{ 0, 1, 1, 0, 1, 0, 0, 0},
{ 1, 1, 1, 0, 1, 0, 0, 0},
{-1,-1,-1, 1, 1, 0, 0, 0},
{ 0,-1,-1, 1, 1, 0, 0, 0},
{ 1,-1,-1, 1, 1, 0, 0, 0},
{-1, 0,-1, 1, 1, 0, 0, 0},
{ 0, 0,-1, 1, 1, 0, 0, 0},
{ 1, 0,-1, 1, 1, 0, 0, 0},
{-1, 1,-1, 1, 1, 0, 0, 0},
{ 0, 1,-1, 1, 1, 0, 0, 0},
{ 1, 1,-1, 1, 1, 0, 0, 0},
{-1,-1, 0, 1, 1, 0, 0, 0},
{ 0,-1, 0, 1, 1, 0, 0, 0},
{ 1,-1, 0, 1, 1, 0, 0, 0},
{-1, 0, 0, 1, 1, 0, 0, 0},
{ 0, 0, 0, 1, 1, 0, 0, 0},
{ 1, 0, 0, 1, 1, 0, 0, 0},
{-1, 1, 0, 1, 1, 0, 0, 0},
{ 0, 1, 0, 1, 1, 0, 0, 0},
{ 1, 1, 0, 1, 1, 0, 0, 0},
{-1,-1, 1, 1, 1, 0, 0, 0},
{ 0,-1, 1, 1, 1, 0, 0, 0},
{ 1,-1, 1, 1, 1, 0, 0, 0},
{-1, 0, 1, 1, 1, 0, 0, 0},
{ 0, 0, 1, 1, 1, 0, 0, 0},
{ 1, 0, 1, 1, 1, 0, 0, 0},
{-1, 1, 1, 1, 1, 0, 0, 0},
{ 0, 1, 1, 1, 1, 0, 0, 0},
{ 1, 1, 1, 1, 1, 0, 0, 0},
{-1,-1,-1,-1,-1, 0, 0, 0},
{ 0,-1,-1,-1,-1, 0, 0, 0},
{ 1,-1,-1,-1,-1, 0, 0, 0},
{-1, 0,-1,-1,-1, 0, 0, 0},
{ 0, 0,-1,-1,-1, 0, 0, 0},
{ 1, 0,-1,-1,-1, 0, 0, 0},
{-1, 1,-1,-1,-1, 0, 0, 0},
{ 0, 1,-1,-1,-1, 0, 0, 0},
{ 1, 1,-1,-1,-1, 0, 0, 0},
{-1,-1, 0,-1,-1, 0, 0, 0},
{ 0,-1, 0,-1,-1, 0, 0, 0},
{ 1,-1, 0,-1,-1, 0, 0, 0},
{-1, 0, 0,-1,-1, 0, 0, 0}
};

/* Decode trit at (row, k) from a FLAT packed [rows, K] buffer.
 * The packing is unpadded: element (row,k) lives at flat index row*K + k. */
static inline int8_t veloce_trit_at(const uint8_t *w, size_t row, size_t k, size_t K) {
    size_t e = row * K + k;
    return VELOCE_TRIT_LUT[w[e / 5]][e % 5];
}

/* Expand a packed trit buffer of n elements into an int8 buffer (tools). */
static inline void veloce_expand_trit(int8_t *dst, const uint8_t *packed, size_t n) {
    size_t i = 0;
    for (size_t b = 0; b < (n + 4) / 5; b++) {
        const int8_t *e = VELOCE_TRIT_LUT[packed[b]];
        for (int q = 0; q < 5 && i < n; q++) dst[i++] = e[q];
    }
}

/* Pack n int8 trits into a byte buffer. */
static inline void veloce_pack_trits(uint8_t *packed, const int8_t *trits, size_t n) {
    size_t nbytes = veloce_trit_rowbytes(n);
    for (size_t b = 0; b < nbytes; b++) {
        int8_t vals[5] = {0,0,0,0,0};
        for (int k = 0; k < 5; k++) { size_t idx = b * 5 + k; vals[k] = idx < n ? trits[idx] : 0; }
        packed[b] = veloce_pack_trit5(vals);
    }
}

/* ------------------------------------------------------------------ */
/* fp16 decode                                                        */
/* ------------------------------------------------------------------ */
static inline float veloce_half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out; memcpy(&out, &f, 4); return out;
}

/* ------------------------------------------------------------------ */
/* Elementwise kernels (contiguous, f32)                              */
/* ------------------------------------------------------------------ */
static inline void veloce_copyf(float *d, const float *s, size_t n) {
    if (d != s) memmove(d, s, n * sizeof(float));
}
static inline void veloce_zerof(float *d, size_t n) { memset(d, 0, n * sizeof(float)); }

#define VELOCE_EW2(NAME, EXPR)                                              \
    static inline void NAME(float *d, const float *a, const float *b, size_t n) { \
        for (size_t i = 0; i < n; i++) { float x = a[i], y = b[i]; d[i] = (EXPR); } \
    }
VELOCE_EW2(veloce_add, x + y)
VELOCE_EW2(veloce_sub, x - y)
VELOCE_EW2(veloce_mul, x * y)
VELOCE_EW2(veloce_div, x / y)
#undef VELOCE_EW2

#define VELOCE_EWS(NAME, EXPR)                                              \
    static inline void NAME(float *d, const float *a, float s, size_t n) {  \
        for (size_t i = 0; i < n; i++) { float x = a[i]; d[i] = (EXPR); }   \
    }
VELOCE_EWS(veloce_sadd, x + s)
VELOCE_EWS(veloce_ssub, x - s)
VELOCE_EWS(veloce_smul, x * s)
VELOCE_EWS(veloce_sdiv, x / s)
VELOCE_EWS(veloce_rsub, s - x)
#undef VELOCE_EWS

/* Row-broadcast: out[r,c] = a[r,c] (op) b[c]   (bias [C] onto [R,C]) */
#define VELOCE_ROWB(NAME, EXPR)                                                 \
    static inline void NAME(float *d, const float *a, const float *b, size_t cols) { \
        for (size_t i = 0; i < cols; i++) { float x = a[i], y = b[i]; d[i] = (EXPR); } \
    }
#define VELOCE_ROWB_R(NAME, EXPR)                                               \
    static inline void NAME(float *d, const float *a, const float *b, size_t cols, \
                            size_t rows) {                                       \
        for (size_t r = 0; r < rows; r++)                                        \
            for (size_t c = 0; c < cols; c++) {                                  \
                float x = a[r * cols + c], y = b[c]; d[r * cols + c] = (EXPR);   \
            }                                                                    \
    }
#define VELOCE_LROWB(NAME, EXPR)                                                \
    static inline void NAME(float *d, const float *a, const float *b, size_t cols, \
                            size_t rows) {                                       \
        for (size_t r = 0; r < rows; r++)                                        \
            for (size_t c = 0; c < cols; c++) {                                  \
                float x = a[r * cols + c], y = b[c]; d[r * cols + c] = (EXPR);   \
            }                                                                    \
    }
/* right bias: d = a (op) b ; here a and b are full/vector, need rows count */
static inline void veloce_rowadd(float *d, const float *a, const float *b,
                                 size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = a[i] + b[i % cols];
}
static inline void veloce_rowsub(float *d, const float *a, const float *b,
                                 size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = a[i] - b[i % cols];
}
static inline void veloce_rowmul(float *d, const float *a, const float *b,
                                 size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = a[i] * b[i % cols];
}
static inline void veloce_rowdiv(float *d, const float *a, const float *b,
                                 size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = a[i] / b[i % cols];
}
/* left bias: d = b[c] (op) a[r,c] */
static inline void veloce_lrowadd(float *d, const float *a, const float *b,
                                  size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = b[i % cols] + a[i];
}
static inline void veloce_lrowsub(float *d, const float *a, const float *b,
                                  size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = b[i % cols] - a[i];
}
static inline void veloce_lrowmul(float *d, const float *a, const float *b,
                                  size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = b[i % cols] * a[i];
}
static inline void veloce_lrowdiv(float *d, const float *a, const float *b,
                                  size_t cols, size_t rows) {
    for (size_t i = 0; i < rows * cols; i++) d[i] = b[i % cols] / a[i];
}

/* ------------------------------------------------------------------ */
/* Matmul: float weights   x:[M,K]  w:[K,N]  out:[M,N]                */
/* ------------------------------------------------------------------ */
/* i-k-j loop order: for each output row, broadcast A[i,k] over the contiguous
 * row B[k,:]. Keeps B sequential (the naive j-major/k-inner loop strides B
 * with N*4 bytes and runs ~40x slower). Each C row is produced by one worker;
 * the per-element accumulation order is unchanged. */
typedef struct { float *out; const float *x, *w; size_t M, K, N; } veloce_mm_f32_ctx;
static inline void veloce_mm_f32_row(void *vctx, size_t i) {
    const veloce_mm_f32_ctx *c = (const veloce_mm_f32_ctx *)vctx;
    float *restrict cr = c->out + i * c->N;
    memset(cr, 0, c->N * sizeof(float));
    for (size_t k = 0; k < c->K; k++) {
        const float a = c->x[i * c->K + k];
        const float *restrict br = c->w + k * c->N;
        for (size_t j = 0; j < c->N; j++) cr[j] += a * br[j];
    }
}
static inline void veloce_matmul_f32(float *out, const float *x, const float *w,
                                     size_t M, size_t K, size_t N) {
    veloce_mm_f32_ctx c = { out, x, w, M, K, N };
    /* one work item = one output row; it streams K*N weight floats */
    veloce_parallel_for_sized(M, K * N * sizeof(float), veloce_mm_f32_row, &c);
}

/* Matmul with fp16 weights:  w:[K,N] uint16 half storage. */
typedef struct { float *out; const float *x; const uint16_t *w; size_t M, K, N; } veloce_mm_f16_ctx;
static inline void veloce_mm_f16_row(void *vctx, size_t i) {
    const veloce_mm_f16_ctx *c = (const veloce_mm_f16_ctx *)vctx;
    float *restrict cr = c->out + i * c->N;
    memset(cr, 0, c->N * sizeof(float));
    for (size_t k = 0; k < c->K; k++) {
        const float a = c->x[i * c->K + k];
        const uint16_t *restrict br = c->w + k * c->N;
        for (size_t j = 0; j < c->N; j++) cr[j] += a * veloce_half_to_float(br[j]);
    }
}
static inline void veloce_matmul_f16w(float *out, const float *x, const uint16_t *w,
                                      size_t M, size_t K, size_t N) {
    veloce_mm_f16_ctx c = { out, x, w, M, K, N };
    veloce_parallel_for_sized(M, K * N * sizeof(uint16_t), veloce_mm_f16_row, &c);
}

/* Matmul-free trit matmul (FR-1.3).
 *   x: [M,K] f32,  w: [N,K] packed base-3 trits,  out: [M,N] f32
 * The trit weight buffer is packed FLAT: element (n,k) at index n*K+k, byte
 * (n*K+k)/5, slot (n*K+k)%5 (matches veloce_pack_trits). Each output is
 * sum_k sign(w[n,k]) * x[m,k]; sign in {-1,0,+1} so this is pure add /
 * subtract / skip — no floating-point multiply, and the packed weights are
 * read in place (zero-copy).
 *
 * Fast path: one table lookup decodes a whole byte (5 trits), the 5 values
 * are expanded into a small stack block and consumed by a vectorizable
 * multiply-accumulate loop. Output elements are independent, so the kernel
 * parallelises across rows (M) or, for GEMV (M == 1), across outputs (N);
 * each element's accumulation order is unchanged, so results are bit-exact
 * in every thread count. */
#define VELOCE_TRIT_BLOCK 32   /* packed bytes per decode block (160 trits) */

/* Dot product of x[0..K) against the trit row that starts at flat trit index
 * `base` in a FLAT base-3 packed buffer (5 trits/byte, rows are not padded to
 * byte boundaries). Decodes one byte -> five trits via the LUT, in blocks of
 * VELOCE_TRIT_BLOCK bytes, and never reads past the last byte that holds a
 * real element (the blob is exactly ceil(N*K/5) bytes). */
/* Portable float-LUT decode: five independent accumulators per byte. */
static inline float veloce_trit_dot_lut(const float *x, const uint8_t *w,
                                        size_t base, size_t K) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f;
    size_t q = 0, e = base;
    size_t skip = e % 5;
    if (skip) {
        const float *v = VELOCE_TRIT_LUT_F32[w[e / 5]];
        size_t n = 5 - skip; if (n > K) n = K;
        for (size_t s = 0; s < n; s++) a0 += x[q++] * v[skip + s];
        e += n;
    }
    size_t full = (K - q) / 5;
    for (size_t b = 0; b < full; b++, q += 5, e += 5) {
        const float *v = VELOCE_TRIT_LUT_F32[w[e / 5]];
        const float *xp = x + q;
        a0 += xp[0] * v[0]; a1 += xp[1] * v[1]; a2 += xp[2] * v[2];
        a3 += xp[3] * v[3]; a4 += xp[4] * v[4];
    }
    float acc = ((a0 + a1) + (a2 + a3)) + a4;
    for (; q < K; q++, e++) {
        const float *v = VELOCE_TRIT_LUT_F32[w[e / 5]];
        acc += x[q] * v[e % 5];
    }
    return acc;
}

#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
#include <immintrin.h>
/* AVX2 trit dot: four packed bytes (20 trits) per iteration. Each byte's
 * five floats are a single aligned 8-float LUT load + one FMA. Because every
 * byte's trits are consecutive, the five loads are unaligned but contiguous. */
static inline float veloce_trit_dot_avx2(const float *x, const uint8_t *w,
                                         size_t base, size_t K) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    float tail = 0.0f;
    size_t q = 0, e = base;
    size_t skip = e % 5;
    if (skip) {
        const float *v = VELOCE_TRIT_LUT_F32[w[e / 5]];
        size_t n = 5 - skip; if (n > K) n = K;
        for (size_t s = 0; s < n; s++) tail += x[q++] * v[skip + s];
        e += n;
    }
    size_t full = (K - q) / 5;          /* whole bytes left */
    size_t bi = e / 5;                  /* byte index of first whole byte */
    size_t b = 0;
    /* A four-byte block loads 8 floats at q+15, i.e. it touches x[..q+22],
     * so it may only run while 23 elements remain.  Without the bound the
     * last block reads up to three floats past x[K-1]: out of bounds on the
     * caller's buffer, and 0.0f * a signalling pattern can yield NaN. The
     * leftover bytes fall through to the scalar loops below, which read
     * exactly the elements they consume. */
    for (; b + 4 <= full && q + 23 <= K; b += 4, bi += 4, q += 20, e += 20) {
        const float *v0 = VELOCE_TRIT_LUT_F32[w[bi]];
        const float *v1 = VELOCE_TRIT_LUT_F32[w[bi + 1]];
        const float *v2 = VELOCE_TRIT_LUT_F32[w[bi + 2]];
        const float *v3 = VELOCE_TRIT_LUT_F32[w[bi + 3]];
        __m256 x0 = _mm256_loadu_ps(x + q);
        __m256 x1 = _mm256_loadu_ps(x + q + 5);
        __m256 x2 = _mm256_loadu_ps(x + q + 10);
        __m256 x3 = _mm256_loadu_ps(x + q + 15);
        acc0 = _mm256_fmadd_ps(x0, _mm256_load_ps(v0), acc0);
        acc1 = _mm256_fmadd_ps(x1, _mm256_load_ps(v1), acc1);
        acc2 = _mm256_fmadd_ps(x2, _mm256_load_ps(v2), acc2);
        acc3 = _mm256_fmadd_ps(x3, _mm256_load_ps(v3), acc3);
    }
    for (; b < full; b++, bi++, q += 5, e += 5) {
        const float *v = VELOCE_TRIT_LUT_F32[w[bi]];
        const float *xp = x + q;
        tail += xp[0] * v[0] + xp[1] * v[1] + xp[2] * v[2] + xp[3] * v[3] + xp[4] * v[4];
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4); s4 = _mm_hadd_ps(s4, s4);
    float acc = tail + _mm_cvtss_f32(s4);
    for (; q < K; q++, e++) {
        const float *v = VELOCE_TRIT_LUT_F32[w[e / 5]];
        acc += x[q] * v[e % 5];
    }
    return acc;
}
#define VELOCE_HAVE_TRIT_AVX2 1
#endif

static inline float veloce_trit_dot(const float *x, const uint8_t *w, size_t base, size_t K) {
#if defined(VELOCE_HAVE_TRIT_AVX2)
    return veloce_trit_dot_avx2(x, w, base, K);
#else
    return veloce_trit_dot_lut(x, w, base, K);
#endif
}

typedef struct { float *out; const float *x; const uint8_t *w; size_t M, K, N; } veloce_mm_trit_ctx;

/* ---- Four output rows per pass -------------------------------------
 * The packed matrix is flat: element (j,k) lives at trit index j*K+k, i.e.
 * byte (j*K+k)/5, slot (j*K+k)%5.  Row j therefore starts at slot (j*K)%5, and
 * rows can only share x loads when they are in the same phase.  Rows
 * j, j+p, j+2p, j+3p with p = 5/gcd(K,5) all start at that same slot and then
 * advance exactly one byte per five trits, so they read the same x window at
 * every step: one pair of 32-byte x loads now feeds four LUT loads and four
 * FMAs.  Before this, every output neuron re-read the whole x vector (a
 * 32-byte load for 5 useful values, K/5 times) -- the reason the kernel decoded
 * at ~19% of this machine's FMA rate.
 *
 * Four rows in flight also give the out-of-order engine four independent
 * accumulator chains, which is what hides the byte-load -> LUT-load dependency
 * the one-row-at-a-time loop serialised on.
 *
 * No row's arithmetic changes: each row keeps two accumulator chains and a
 * fixed reduction order, so a row's value never depends on which other rows
 * share its pass, nor on the thread count. */
#define VELOCE_TRIT_ROWS 4

/* Byte phase repeats every `p` rows of a flat [N, K] trit matrix. */
static inline size_t veloce_trit_phase(size_t K) { return (K % 5 == 0) ? 1 : 5; }

#if defined(VELOCE_HAVE_TRIT_AVX2)
/* Four rows starting at trit index base, base+stride5*5, base+2*stride5*5,
 * base+3*stride5*5 -- all in the same byte phase. */
static inline void veloce_trit_dot4_avx2(const float *restrict x, const uint8_t *restrict w,
                                         size_t base, size_t stride5, size_t K,
                                         float *restrict out) {
    __m256 acc[VELOCE_TRIT_ROWS][2];
    for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
        acc[r][0] = _mm256_setzero_ps();
        acc[r][1] = _mm256_setzero_ps();
        out[r] = 0.0f;
    }
    const size_t skip = base % 5;       /* slot of the group's first trit */
    size_t bi = base / 5;               /* byte that holds it */
    size_t q = 0;
    if (skip) {
        /* Finish the partial first byte (at most four trits) in scalar. */
        const size_t n = (5 - skip < K) ? 5 - skip : K;
        for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
            const float *v = VELOCE_TRIT_LUT_F32[w[bi + (size_t)r * stride5]];
            float t = 0.0f;
            for (size_t s = 0; s < n; s++) t += x[s] * v[skip + s];
            out[r] = t;
        }
        q = n;
        bi += 1;                        /* first whole byte of every row */
    }
    const size_t full = (K - q) / 5;    /* whole packed bytes per row */
    size_t b = 0;
    /* A four-byte block reads x[..q+22] (eight floats from each of q, q+5,
     * q+10, q+15), so it may only run while 23 elements remain -- the same
     * bound as the single-row kernel, which keeps every read in bounds. */
    for (; b + 4 <= full && q + 23 <= K; b += 4, q += 20) {
        const __m256 x0 = _mm256_loadu_ps(x + q);
        const __m256 x1 = _mm256_loadu_ps(x + q + 5);
        const __m256 x2 = _mm256_loadu_ps(x + q + 10);
        const __m256 x3 = _mm256_loadu_ps(x + q + 15);
        for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
            const uint8_t *wr = w + bi + (size_t)r * stride5 + b;
            acc[r][0] = _mm256_fmadd_ps(x0, _mm256_load_ps(VELOCE_TRIT_LUT_F32[wr[0]]), acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(x1, _mm256_load_ps(VELOCE_TRIT_LUT_F32[wr[1]]), acc[r][1]);
            acc[r][0] = _mm256_fmadd_ps(x2, _mm256_load_ps(VELOCE_TRIT_LUT_F32[wr[2]]), acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(x3, _mm256_load_ps(VELOCE_TRIT_LUT_F32[wr[3]]), acc[r][1]);
        }
    }
    for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
        const __m256 s = _mm256_add_ps(acc[r][0], acc[r][1]);
        const __m128 lo = _mm256_castps256_ps128(s);
        const __m128 hi = _mm256_extractf128_ps(s, 1);
        __m128 t = _mm_add_ps(lo, hi);
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        float a = out[r] + _mm_cvtss_f32(t);
        const uint8_t *wr = w + bi + (size_t)r * stride5;
        size_t ee = b, qq = q;
        for (; ee < full; ee++, qq += 5) {
            const float *v = VELOCE_TRIT_LUT_F32[wr[ee]];
            a += x[qq] * v[0] + x[qq + 1] * v[1] + x[qq + 2] * v[2]
               + x[qq + 3] * v[3] + x[qq + 4] * v[4];
        }
        if (qq < K) {
            const float *v = VELOCE_TRIT_LUT_F32[wr[ee]];   /* the row's last, partial byte */
            for (size_t k = qq, slot = 0; k < K; k++, slot++) a += x[k] * v[slot];
        }
        out[r] = a;
    }
}
#endif

/* ---- Two query rows, one weight pass -------------------------------
 * A decode that consumes two tokens per call multiplies the *same* packed
 * matrix by two different x vectors.  Running the one-row kernel twice
 * streams the whole matrix twice; here each packed byte and its 32-byte LUT
 * row are loaded once and used by both query rows, so the weight traffic per
 * token halves.  Every (query row, neuron) keeps the two-accumulator chain
 * and the reduction order of the one-row kernel, so the values are
 * bit-identical to two one-row passes (there is a test for exactly that).
 */
#if defined(VELOCE_HAVE_TRIT_AVX2)
static inline void veloce_trit_dot4x2_avx2(const float *restrict xa,
                                           const float *restrict xb,
                                           const uint8_t *restrict w,
                                           size_t base, size_t stride5, size_t K,
                                           float *restrict out /* [2][4] */) {
    __m256 acc[2][VELOCE_TRIT_ROWS][2];
    float init[2][VELOCE_TRIT_ROWS];
    const float *xs[2] = { xa, xb };
    for (int m = 0; m < 2; m++)
        for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
            acc[m][r][0] = _mm256_setzero_ps();
            acc[m][r][1] = _mm256_setzero_ps();
            init[m][r] = 0.0f;
        }
    const size_t skip = base % 5;       /* slot of the group's first trit */
    size_t bi = base / 5;
    size_t q = 0;
    if (skip) {
        const size_t n = (5 - skip < K) ? 5 - skip : K;
        for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
            const float *v = VELOCE_TRIT_LUT_F32[w[bi + (size_t)r * stride5]];
            for (int m = 0; m < 2; m++) {
                const float *x = xs[m];
                float t = 0.0f;
                for (size_t s = 0; s < n; s++) t += x[s] * v[skip + s];
                init[m][r] = t;
            }
        }
        q = n;
        bi += 1;
    }
    const size_t full = (K - q) / 5;
    size_t b = 0;
    for (; b + 4 <= full && q + 23 <= K; b += 4, q += 20) {
        for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
            const uint8_t *wr = w + bi + (size_t)r * stride5 + b;
            const float *v0 = VELOCE_TRIT_LUT_F32[wr[0]];
            const float *v1 = VELOCE_TRIT_LUT_F32[wr[1]];
            const float *v2 = VELOCE_TRIT_LUT_F32[wr[2]];
            const float *v3 = VELOCE_TRIT_LUT_F32[wr[3]];
            for (int m = 0; m < 2; m++) {
                const float *x = xs[m] + q;
                acc[m][r][0] = _mm256_fmadd_ps(_mm256_loadu_ps(x), _mm256_load_ps(v0), acc[m][r][0]);
                acc[m][r][1] = _mm256_fmadd_ps(_mm256_loadu_ps(x + 5), _mm256_load_ps(v1), acc[m][r][1]);
                acc[m][r][0] = _mm256_fmadd_ps(_mm256_loadu_ps(x + 10), _mm256_load_ps(v2), acc[m][r][0]);
                acc[m][r][1] = _mm256_fmadd_ps(_mm256_loadu_ps(x + 15), _mm256_load_ps(v3), acc[m][r][1]);
            }
        }
    }
    for (int m = 0; m < 2; m++) {
        const float *xr = xs[m];
        for (int r = 0; r < VELOCE_TRIT_ROWS; r++) {
            const __m256 s = _mm256_add_ps(acc[m][r][0], acc[m][r][1]);
            const __m128 lo = _mm256_castps256_ps128(s);
            const __m128 hi = _mm256_extractf128_ps(s, 1);
            __m128 t = _mm_add_ps(lo, hi);
            t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
            float a = init[m][r] + _mm_cvtss_f32(t);
            const uint8_t *wr = w + bi + (size_t)r * stride5;
            size_t ee = b, qq = q;
            for (; ee < full; ee++, qq += 5) {
                const float *v = VELOCE_TRIT_LUT_F32[wr[ee]];
                a += xr[qq] * v[0] + xr[qq + 1] * v[1] + xr[qq + 2] * v[2]
                   + xr[qq + 3] * v[3] + xr[qq + 4] * v[4];
            }
            if (qq < K) {
                const float *v = VELOCE_TRIT_LUT_F32[wr[ee]];
                for (size_t k = qq, slot = 0; k < K; k++, slot++)
                    a += xr[k] * v[slot];
            }
            out[m * VELOCE_TRIT_ROWS + r] = a;
        }
    }
}
#endif

static inline void veloce_trit_dot4(const float *x, const uint8_t *w, size_t base,
                                    size_t stride5, size_t K, float *out) {
#if defined(VELOCE_HAVE_TRIT_AVX2)
    veloce_trit_dot4_avx2(x, w, base, stride5, K, out);
#else
    for (size_t r = 0; r < VELOCE_TRIT_ROWS; r++)
        out[r] = veloce_trit_dot(x, w, base + r * stride5 * 5, K);
#endif
}

static inline void veloce_trit_dot4x2(const float *xa, const float *xb,
                                      const uint8_t *w, size_t base,
                                      size_t stride5, size_t K, float *out) {
#if defined(VELOCE_HAVE_TRIT_AVX2)
    veloce_trit_dot4x2_avx2(xa, xb, w, base, stride5, K, out);
#else
    veloce_trit_dot4(xa, w, base, stride5, K, out);
    veloce_trit_dot4(xb, w, base, stride5, K, out + VELOCE_TRIT_ROWS);
#endif
}

/* One GEMV work item: a tile of VELOCE_TRIT_ROWS*phase neurons, run as `phase`
 * groups of four rows that share a byte phase. */
static inline void veloce_mm_trit_gemv_chunk(void *vctx, size_t t) {
    const veloce_mm_trit_ctx *c = (const veloce_mm_trit_ctx *)vctx;
    const size_t period = veloce_trit_phase(c->K);
    const size_t tile = VELOCE_TRIT_ROWS * period;
    const size_t stride5 = period * c->K / 5;
    const size_t j0 = t * tile;
    if (j0 + tile <= c->N) {
        for (size_t ph = 0; ph < period; ph++) {
            float part[VELOCE_TRIT_ROWS];
            veloce_trit_dot4(c->x, c->w, (j0 + ph) * c->K, stride5, c->K, part);
            for (size_t r = 0; r < VELOCE_TRIT_ROWS; r++)
                c->out[j0 + ph + r * period] = part[r];
        }
        return;
    }
    for (size_t j = j0; j < c->N; j++)
        c->out[j] = veloce_trit_dot(c->x, c->w, j * c->K, c->K);
}
static inline void veloce_mm_trit_row(void *vctx, size_t i) {
    const veloce_mm_trit_ctx *c = (const veloce_mm_trit_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    const size_t period = veloce_trit_phase(c->K);
    const size_t tile = VELOCE_TRIT_ROWS * period;
    const size_t stride5 = period * c->K / 5;
    float *orow = c->out + i * c->N;
    size_t j0 = 0;
    for (; j0 + tile <= c->N; j0 += tile)
        for (size_t ph = 0; ph < period; ph++) {
            float part[VELOCE_TRIT_ROWS];
            veloce_trit_dot4(xr, c->w, (j0 + ph) * c->K, stride5, c->K, part);
            for (size_t r = 0; r < VELOCE_TRIT_ROWS; r++)
                orow[j0 + ph + r * period] = part[r];
        }
    for (; j0 < c->N; j0++)
        orow[j0] = veloce_trit_dot(xr, c->w, j0 * c->K, c->K);
}
/* Same work split as the one-row GEMV, but both query rows for a tile are
 * computed while its packed bytes are in registers. */
typedef struct { float *out; const float *xa, *xb; const uint8_t *w;
                 size_t K, N; } veloce_mm_trit2_ctx;

static inline void veloce_mm_trit2_chunk(void *vctx, size_t t) {
    const veloce_mm_trit2_ctx *c = (const veloce_mm_trit2_ctx *)vctx;
    const size_t period = veloce_trit_phase(c->K);
    const size_t tile = VELOCE_TRIT_ROWS * period;
    const size_t stride5 = period * c->K / 5;
    const size_t j0 = t * tile;
    if (j0 + tile <= c->N) {
        for (size_t ph = 0; ph < period; ph++) {
            float part[2 * VELOCE_TRIT_ROWS];
            veloce_trit_dot4x2(c->xa, c->xb, c->w, (j0 + ph) * c->K, stride5,
                               c->K, part);
            for (size_t r = 0; r < VELOCE_TRIT_ROWS; r++)
                for (int m = 0; m < 2; m++)
                    c->out[(size_t)m * c->N + j0 + ph + r * period] =
                        part[m * VELOCE_TRIT_ROWS + r];
        }
        return;
    }
    for (size_t j = j0; j < c->N; j++) {
        c->out[j] = veloce_trit_dot(c->xa, c->w, j * c->K, c->K);
        c->out[c->N + j] = veloce_trit_dot(c->xb, c->w, j * c->K, c->K);
    }
}

static inline void veloce_matmul_trit(float *out, const float *x, const uint8_t *w,
                                      size_t M, size_t K, size_t N) {
    veloce_mm_trit_ctx c = { out, x, w, M, K, N };
    const size_t row_bytes = veloce_trit_rowbytes(K);
    const size_t tile = VELOCE_TRIT_ROWS * veloce_trit_phase(K);
    if (M == 1) {
        /* GEMV: one work item = one tile of output neurons in the same byte
         * phase; it streams tile*row_bytes of packed weights. */
        veloce_parallel_for_sized((N + tile - 1) / tile, tile * row_bytes,
                                  veloce_mm_trit_gemv_chunk, &c);
    } else if (M == 2) {
        /* Two query rows per weight load: the packed bytes are decoded once
         * for both tokens, so a two-token chunk streams the matrix once. */
        veloce_mm_trit2_ctx c2 = { out, x, x + K, w, K, N };
        veloce_parallel_for_sized((N + tile - 1) / tile, tile * row_bytes,
                                  veloce_mm_trit2_chunk, &c2);
    } else {
        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_trit_row, &c);
    }
}

/* ------------------------------------------------------------------ */
/* Packed i4 weights (FR-1.1): 2 signed nibbles per byte.              */
/*   low nibble  = element 2k, high nibble = element 2k+1               */
/*   value = sign-extended 4-bit int in [-8, 7]                        */
/* ------------------------------------------------------------------ */
static inline uint8_t veloce_pack_i4(const int8_t *vals, size_t n) {
    (void)n;
    return (uint8_t)((vals[0] & 0x0F) | ((vals[1] & 0x0F) << 4));
}

/* 256-entry decode table -> 8 floats (-8..7), 32-byte aligned. Two useful
 * floats per byte (low nibble, high nibble); six zero pad lanes. */
#define VELOCE_I4_F(b) \
    { (float)(((b) & 0x0F) < 8 ? ((b) & 0x0F) : ((b) & 0x0F) - 16), \
      (float)((((b) >> 4) & 0x0F) < 8 ? (((b) >> 4) & 0x0F) : (((b) >> 4) & 0x0F) - 16), \
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }
#define VELOCE_I4_F16(b) \
    VELOCE_I4_F(b), VELOCE_I4_F(b+1), VELOCE_I4_F(b+2), VELOCE_I4_F(b+3), \
    VELOCE_I4_F(b+4), VELOCE_I4_F(b+5), VELOCE_I4_F(b+6), VELOCE_I4_F(b+7), \
    VELOCE_I4_F(b+8), VELOCE_I4_F(b+9), VELOCE_I4_F(b+10), VELOCE_I4_F(b+11), \
    VELOCE_I4_F(b+12), VELOCE_I4_F(b+13), VELOCE_I4_F(b+14), VELOCE_I4_F(b+15)
#define VELOCE_I4_F64(b) \
    VELOCE_I4_F16(b), VELOCE_I4_F16(b+16), VELOCE_I4_F16(b+32), VELOCE_I4_F16(b+48)
static const float VELOCE_I4_LUT[256][8] __attribute__((aligned(32))) = {
    VELOCE_I4_F64(0), VELOCE_I4_F64(64), VELOCE_I4_F64(128), VELOCE_I4_F64(192)
};

/* Number of bytes a row of `k` i4 values occupies (2 per byte, padded). */
static inline size_t veloce_i4_rowbytes(size_t k) { return (k + 1) / 2; }

/* Portable i4 dot: two nibbles per byte. */
static inline float veloce_i4_dot_lut(const float *x, const uint8_t *w,
                                      size_t base, size_t K) {
    float a0 = 0.f, a1 = 0.f;
    size_t q = 0, e = base;
    size_t skip = e % 2;
    if (skip && q < K) {
        const float *v = VELOCE_I4_LUT[w[e / 2]];
        a0 += x[q++] * v[1];
        e++;
    }
    size_t full = (K - q) / 2;
    for (size_t b = 0; b < full; b++, q += 2, e += 2) {
        const float *v = VELOCE_I4_LUT[w[e / 2]];
        a0 += x[q] * v[0];
        a1 += x[q + 1] * v[1];
    }
    float acc = a0 + a1;
    for (; q < K; q++, e++) {
        const float *v = VELOCE_I4_LUT[w[e / 2]];
        acc += x[q] * v[e % 2];
    }
    return acc;
}

#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
/* AVX2 i4 dot: 16 packed bytes (32 values) per iteration, 8-wide FMA. */
static inline float veloce_i4_dot_avx2(const float *x, const uint8_t *w,
                                       size_t base, size_t K) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    float tail = 0.0f;
    size_t q = 0, e = base;
    size_t skip = e % 2;
    if (skip && q < K) {
        const float *v = VELOCE_I4_LUT[w[e / 2]];
        tail += x[q++] * v[1];
        e++;
    }
    size_t full = (K - q) / 2;          /* whole bytes left */
    size_t bi = e / 2;
    size_t b = 0;
    /* Eight bytes cover 16 elements but load 8 floats at q+14, touching
     * x[..q+21]: require 22 elements left so the block stays in bounds. */
    for (; b + 8 <= full && q + 22 <= K; b += 8, bi += 8, q += 16, e += 16) {
        const float *v0 = VELOCE_I4_LUT[w[bi]];
        const float *v1 = VELOCE_I4_LUT[w[bi + 1]];
        const float *v2 = VELOCE_I4_LUT[w[bi + 2]];
        const float *v3 = VELOCE_I4_LUT[w[bi + 3]];
        const float *v4 = VELOCE_I4_LUT[w[bi + 4]];
        const float *v5 = VELOCE_I4_LUT[w[bi + 5]];
        const float *v6 = VELOCE_I4_LUT[w[bi + 6]];
        const float *v7 = VELOCE_I4_LUT[w[bi + 7]];
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q),      _mm256_load_ps(v0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 2),  _mm256_load_ps(v1), acc1);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 4),  _mm256_load_ps(v2), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 6),  _mm256_load_ps(v3), acc1);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 8),  _mm256_load_ps(v4), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 10), _mm256_load_ps(v5), acc1);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 12), _mm256_load_ps(v6), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 14), _mm256_load_ps(v7), acc1);
    }
    for (; b < full; b++, bi++, q += 2, e += 2) {
        const float *v = VELOCE_I4_LUT[w[bi]];
        tail += x[q] * v[0] + x[q + 1] * v[1];
    }
    __m256 s = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4); s4 = _mm_hadd_ps(s4, s4);
    float acc = tail + _mm_cvtss_f32(s4);
    for (; q < K; q++, e++) {
        const float *v = VELOCE_I4_LUT[w[e / 2]];
        acc += x[q] * v[e % 2];
    }
    return acc;
}
#endif

static inline float veloce_i4_dot(const float *x, const uint8_t *w,
                                  size_t base, size_t K) {
#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
    return veloce_i4_dot_avx2(x, w, base, K);
#else
    return veloce_i4_dot_lut(x, w, base, K);
#endif
}

typedef struct { float *out; const float *x; const uint8_t *w; size_t M, K, N; } veloce_mm_i4_ctx;
static inline void veloce_mm_i4_gemv_row(void *vctx, size_t j) {
    const veloce_mm_i4_ctx *c = (const veloce_mm_i4_ctx *)vctx;
    c->out[j] = veloce_i4_dot(c->x, c->w, j * c->K, c->K);
}
static inline void veloce_mm_i4_mat_row(void *vctx, size_t i) {
    const veloce_mm_i4_ctx *c = (const veloce_mm_i4_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = veloce_i4_dot(xr, c->w, j * c->K, c->K);
}
static inline void veloce_matmul_i4(float *out, const float *x, const uint8_t *w,
                                    size_t M, size_t K, size_t N) {
    veloce_mm_i4_ctx c = { out, x, w, M, K, N };
    const size_t row_bytes = veloce_i4_rowbytes(K);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_i4_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_i4_mat_row, &c);
}

/* ------------------------------------------------------------------ */
/* Group-scaled i4 matmul (real-model import path).                     */
/*   w: [N,K] packed i4, scales: [N, K/G] f32.                           */
/*   out[m,n] = sum_g scales[n,g] * sum_{k in g} x[m,k] * i4(w[n,k])     */
/* ------------------------------------------------------------------ */
static inline float veloce_i4_dot_scaled(const float *x, const uint8_t *w,
                                         const float *scales, size_t base,
                                         size_t K, size_t G) {
    float acc = 0.0f;
    size_t groups = G ? K / G : 0;
    for (size_t g = 0; g < groups; g++) {
        float part = veloce_i4_dot_lut(x + g * G, w, base + g * G, G);
        acc += part * scales[g];
    }
    size_t done = groups * G;
    if (done < K) {
        float part = veloce_i4_dot_lut(x + done, w, base + done, K - done);
        acc += part * scales[groups];
    }
    return acc;
}
typedef struct { float *out; const float *x; const uint8_t *w; const float *s;
                 size_t M, K, N, G; } veloce_mm_i4g_ctx;
static inline void veloce_mm_i4g_gemv_row(void *vctx, size_t j) {
    const veloce_mm_i4g_ctx *c = (const veloce_mm_i4g_ctx *)vctx;
    c->out[j] = veloce_i4_dot_scaled(c->x, c->w, c->s + j * ((c->K + c->G - 1) / c->G),
                                     j * c->K, c->K, c->G);
}
static inline void veloce_mm_i4g_mat_row(void *vctx, size_t i) {
    const veloce_mm_i4g_ctx *c = (const veloce_mm_i4g_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    size_t ng = (c->K + c->G - 1) / c->G;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = veloce_i4_dot_scaled(xr, c->w, c->s + j * ng,
                                                    j * c->K, c->K, c->G);
}
static inline void veloce_matmul_i4g(float *out, const float *x, const uint8_t *w,
                                     const float *scales, size_t M, size_t K,
                                     size_t N, size_t G) {
    veloce_mm_i4g_ctx c = { out, x, w, scales, M, K, N, G };
    const size_t row_bytes = veloce_i4_rowbytes(K) + ((K + G - 1) / G) * sizeof(float);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_i4g_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_i4g_mat_row, &c);
}

/* Row-scaled trit matmul: w [N,K] packed trits, scales [N] f32. */
static inline float veloce_trit_dot_scaled(const float *x, const uint8_t *w,
                                           float scale, size_t base, size_t K) {
    return scale * veloce_trit_dot(x, w, base, K);
}
typedef struct { float *out; const float *x; const uint8_t *w; const float *s;
                 size_t M, K, N; } veloce_mm_trits_ctx;
static inline void veloce_mm_trits_gemv_row(void *vctx, size_t t) {
    const veloce_mm_trits_ctx *c = (const veloce_mm_trits_ctx *)vctx;
    const size_t period = veloce_trit_phase(c->K);
    const size_t tile = VELOCE_TRIT_ROWS * period;
    const size_t stride5 = period * c->K / 5;
    const size_t j0 = t * tile;
    if (j0 + tile <= c->N) {
        for (size_t ph = 0; ph < period; ph++) {
            float part[VELOCE_TRIT_ROWS];
            veloce_trit_dot4(c->x, c->w, (j0 + ph) * c->K, stride5, c->K, part);
            for (size_t r = 0; r < VELOCE_TRIT_ROWS; r++) {
                const size_t j = j0 + ph + r * period;
                c->out[j] = c->s[j] * part[r];
            }
        }
        return;
    }
    for (size_t j = j0; j < c->N; j++)
        c->out[j] = veloce_trit_dot_scaled(c->x, c->w, c->s[j], j * c->K, c->K);
}
static inline void veloce_mm_trits_mat_row(void *vctx, size_t i) {
    const veloce_mm_trits_ctx *c = (const veloce_mm_trits_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    const size_t period = veloce_trit_phase(c->K);
    const size_t tile = VELOCE_TRIT_ROWS * period;
    const size_t stride5 = period * c->K / 5;
    float *orow = c->out + i * c->N;
    size_t j0 = 0;
    for (; j0 + tile <= c->N; j0 += tile)
        for (size_t ph = 0; ph < period; ph++) {
            float part[VELOCE_TRIT_ROWS];
            veloce_trit_dot4(xr, c->w, (j0 + ph) * c->K, stride5, c->K, part);
            for (size_t r = 0; r < VELOCE_TRIT_ROWS; r++) {
                const size_t j = j0 + ph + r * period;
                orow[j] = c->s[j] * part[r];
            }
        }
    for (; j0 < c->N; j0++)
        orow[j0] = veloce_trit_dot_scaled(xr, c->w, c->s[j0], j0 * c->K, c->K);
}
static inline void veloce_matmul_trit_s(float *out, const float *x, const uint8_t *w,
                                        const float *scales, size_t M, size_t K,
                                        size_t N) {
    veloce_mm_trits_ctx c = { out, x, w, scales, M, K, N };
    const size_t row_bytes = veloce_trit_rowbytes(K) + sizeof(float);
    const size_t tile = VELOCE_TRIT_ROWS * veloce_trit_phase(K);
    if (M == 1) veloce_parallel_for_sized((N + tile - 1) / tile, tile * row_bytes,
                                          veloce_mm_trits_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_trits_mat_row, &c);
}

/* ------------------------------------------------------------------ */
/* Packed i2 weights (FR-1.1): 4 signed 2-bit values per byte.         */
/*   slot k (low -> high) = element 4b + k                             */
/*   value = sign-extended 2-bit int in [-2, 1]                        */
/* The narrowest PRD dtype: 4 values/byte, so a 7B model's weights     */
/* cost 1.75 GB against 1.32 GB for trit and 13.2 GB for fp16.         */
/* ------------------------------------------------------------------ */
static inline size_t veloce_i2_rowbytes(size_t k) { return (k + 3) / 4; }

static inline uint8_t veloce_pack_i2(const int8_t *vals, size_t n) {
    (void)n;
    uint8_t b = 0;
    for (int s = 0; s < 4; s++) b |= (uint8_t)((vals[s] & 0x03) << (2 * s));
    return b;
}

/* 256-entry float decode table: one packed byte -> four floats.
 * Padded to 8 floats / 32 bytes so an AVX2 FMA can consume a byte per load. */
#define VELOCE_I2_F(b) \
    { (float)(((b) & 0x03) < 2 ? ((b) & 0x03) : ((b) & 0x03) - 4), \
      (float)((((b) >> 2) & 0x03) < 2 ? (((b) >> 2) & 0x03) : (((b) >> 2) & 0x03) - 4), \
      (float)((((b) >> 4) & 0x03) < 2 ? (((b) >> 4) & 0x03) : (((b) >> 4) & 0x03) - 4), \
      (float)((((b) >> 6) & 0x03) < 2 ? (((b) >> 6) & 0x03) : (((b) >> 6) & 0x03) - 4), \
      0.0f, 0.0f, 0.0f, 0.0f }
#define VELOCE_I2_F16(b) \
    VELOCE_I2_F(b), VELOCE_I2_F(b+1), VELOCE_I2_F(b+2), VELOCE_I2_F(b+3), \
    VELOCE_I2_F(b+4), VELOCE_I2_F(b+5), VELOCE_I2_F(b+6), VELOCE_I2_F(b+7), \
    VELOCE_I2_F(b+8), VELOCE_I2_F(b+9), VELOCE_I2_F(b+10), VELOCE_I2_F(b+11), \
    VELOCE_I2_F(b+12), VELOCE_I2_F(b+13), VELOCE_I2_F(b+14), VELOCE_I2_F(b+15)
#define VELOCE_I2_F64(b) \
    VELOCE_I2_F16(b), VELOCE_I2_F16(b+16), VELOCE_I2_F16(b+32), VELOCE_I2_F16(b+48)
static const float VELOCE_I2_LUT[256][8] __attribute__((aligned(32))) = {
    VELOCE_I2_F64(0), VELOCE_I2_F64(64), VELOCE_I2_F64(128), VELOCE_I2_F64(192)
};

/* Portable i2 dot: four signed 2-bit values per byte. */
static inline float veloce_i2_dot_lut(const float *x, const uint8_t *w,
                                      size_t base, size_t K) {
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    size_t q = 0, e = base;
    size_t skip = e % 4;
    if (skip) {
        const float *v = VELOCE_I2_LUT[w[e / 4]];
        size_t n = 4 - skip; if (n > K) n = K;
        for (size_t s = 0; s < n; s++) a0 += x[q++] * v[skip + s];
        e += n;
    }
    size_t full = (K - q) / 4;
    for (size_t b = 0; b < full; b++, q += 4, e += 4) {
        const float *v = VELOCE_I2_LUT[w[e / 4]];
        const float *xp = x + q;
        a0 += xp[0] * v[0]; a1 += xp[1] * v[1];
        a2 += xp[2] * v[2]; a3 += xp[3] * v[3];
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (; q < K; q++, e++) {
        const float *v = VELOCE_I2_LUT[w[e / 4]];
        acc += x[q] * v[e % 4];
    }
    return acc;
}

#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
/* AVX2 i2 dot: four packed bytes (16 values) per iteration. */
static inline float veloce_i2_dot_avx2(const float *x, const uint8_t *w,
                                       size_t base, size_t K) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    float tail = 0.0f;
    size_t q = 0, e = base;
    size_t skip = e % 4;
    if (skip) {
        const float *v = VELOCE_I2_LUT[w[e / 4]];
        size_t n = 4 - skip; if (n > K) n = K;
        for (size_t s = 0; s < n; s++) tail += x[q++] * v[skip + s];
        e += n;
    }
    size_t full = (K - q) / 4;
    size_t bi = e / 4, b = 0;
    /* Four bytes cover 16 elements but load 8 floats at q+12, touching
     * x[..q+19]: require 20 elements left so the block stays in bounds. */
    for (; b + 4 <= full && q + 20 <= K; b += 4, bi += 4, q += 16, e += 16) {
        const float *v0 = VELOCE_I2_LUT[w[bi]];
        const float *v1 = VELOCE_I2_LUT[w[bi + 1]];
        const float *v2 = VELOCE_I2_LUT[w[bi + 2]];
        const float *v3 = VELOCE_I2_LUT[w[bi + 3]];
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q),     _mm256_load_ps(v0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 4), _mm256_load_ps(v1), acc1);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 8), _mm256_load_ps(v2), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + q + 12), _mm256_load_ps(v3), acc1);
    }
    for (; b < full; b++, bi++, q += 4, e += 4) {
        const float *v = VELOCE_I2_LUT[w[bi]];
        tail += x[q] * v[0] + x[q + 1] * v[1] + x[q + 2] * v[2] + x[q + 3] * v[3];
    }
    __m256 s = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(s);
    __m128 hi = _mm256_extractf128_ps(s, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4); s4 = _mm_hadd_ps(s4, s4);
    float acc = tail + _mm_cvtss_f32(s4);
    for (; q < K; q++, e++) {
        const float *v = VELOCE_I2_LUT[w[e / 4]];
        acc += x[q] * v[e % 4];
    }
    return acc;
}
#endif

static inline float veloce_i2_dot(const float *x, const uint8_t *w,
                                  size_t base, size_t K) {
#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
    return veloce_i2_dot_avx2(x, w, base, K);
#else
    return veloce_i2_dot_lut(x, w, base, K);
#endif
}

typedef struct { float *out; const float *x; const uint8_t *w; size_t M, K, N; } veloce_mm_i2_ctx;
static inline void veloce_mm_i2_gemv_row(void *vctx, size_t j) {
    const veloce_mm_i2_ctx *c = (const veloce_mm_i2_ctx *)vctx;
    c->out[j] = veloce_i2_dot(c->x, c->w, j * c->K, c->K);
}
static inline void veloce_mm_i2_mat_row(void *vctx, size_t i) {
    const veloce_mm_i2_ctx *c = (const veloce_mm_i2_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = veloce_i2_dot(xr, c->w, j * c->K, c->K);
}
/* x: [M,K] f32, w: [N,K] packed i2 (4/byte, FLAT), out: [M,N] f32. */
static inline void veloce_matmul_i2(float *out, const float *x, const uint8_t *w,
                                    size_t M, size_t K, size_t N) {
    veloce_mm_i2_ctx c = { out, x, w, M, K, N };
    const size_t row_bytes = veloce_i2_rowbytes(K);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_i2_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_i2_mat_row, &c);
}

/* Row-scaled i2 matmul: w [N,K] packed i2, scales [N] f32. */
typedef struct { float *out; const float *x; const uint8_t *w; const float *s;
                 size_t M, K, N; } veloce_mm_i2s_ctx;
static inline void veloce_mm_i2s_gemv_row(void *vctx, size_t j) {
    const veloce_mm_i2s_ctx *c = (const veloce_mm_i2s_ctx *)vctx;
    c->out[j] = c->s[j] * veloce_i2_dot(c->x, c->w, j * c->K, c->K);
}
static inline void veloce_mm_i2s_mat_row(void *vctx, size_t i) {
    const veloce_mm_i2s_ctx *c = (const veloce_mm_i2s_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = c->s[j] * veloce_i2_dot(xr, c->w, j * c->K, c->K);
}
static inline void veloce_matmul_i2_s(float *out, const float *x, const uint8_t *w,
                                      const float *scales, size_t M, size_t K, size_t N) {
    veloce_mm_i2s_ctx c = { out, x, w, scales, M, K, N };
    const size_t row_bytes = veloce_i2_rowbytes(K) + sizeof(float);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_i2s_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_i2s_mat_row, &c);
}

/* ------------------------------------------------------------------ */
/* Group-scaled i2 matmul (real 2-bit checkpoints).                    */
/*   w: [N,K] packed i2, scales: [N, K/G] f32.                         */
/*   out[m,n] = sum_g scales[n,g] * sum_{k in g} x[m,k] * i2(w[n,k])   */
/* Same contract as `matmul_i4g`: the caller passes the group SIZE and */
/* the group count must divide K exactly (enforced by the typechecker),*/
/* so `K / G` here is the exact number of groups per row.              */
/* ------------------------------------------------------------------ */
static inline float veloce_i2_dot_scaled(const float *x, const uint8_t *w,
                                         const float *scales, size_t base,
                                         size_t K, size_t G) {
    float acc = 0.0f;
    const size_t groups = G ? K / G : 0;
    for (size_t g = 0; g < groups; g++) {
        float part = veloce_i2_dot(x + g * G, w, base + g * G, G);
        acc += part * scales[g];
    }
    /* Ragged tail: only reachable when K is not a multiple of G, which the
     * typechecker rejects for literal dims.  Symbolic dims that resolve
     * badly still produce the mathematically consistent answer here. */
    size_t done = groups * G;
    if (done < K) {
        float part = veloce_i2_dot(x + done, w, base + done, K - done);
        acc += part * scales[groups];
    }
    return acc;
}
typedef struct { float *out; const float *x; const uint8_t *w; const float *s;
                 size_t M, K, N, G; } veloce_mm_i2g_ctx;
static inline void veloce_mm_i2g_gemv_row(void *vctx, size_t j) {
    const veloce_mm_i2g_ctx *c = (const veloce_mm_i2g_ctx *)vctx;
    c->out[j] = veloce_i2_dot_scaled(c->x, c->w, c->s + j * (c->K / c->G),
                                     j * c->K, c->K, c->G);
}
static inline void veloce_mm_i2g_mat_row(void *vctx, size_t i) {
    const veloce_mm_i2g_ctx *c = (const veloce_mm_i2g_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    const size_t ng = c->K / c->G;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = veloce_i2_dot_scaled(xr, c->w, c->s + j * ng,
                                                    j * c->K, c->K, c->G);
}
static inline void veloce_matmul_i2g(float *out, const float *x, const uint8_t *w,
                                     const float *scales, size_t M, size_t K,
                                     size_t N, size_t G) {
    veloce_mm_i2g_ctx c = { out, x, w, scales, M, K, N, G };
    const size_t row_bytes = veloce_i2_rowbytes(K) + (K / G) * sizeof(float);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_i2g_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_i2g_mat_row, &c);
}

/* ------------------------------------------------------------------ */
/* fp8 weights (FR-1.1): E4M3 (bias 7), one byte per value.            */
/*   sign | 4-bit exponent | 3-bit mantissa                            */
/*   finite range +/-448, smallest normal 2^-6, subnormals 2^-9 * m    */
/*   E4M3 has no infinity; 0x7F/0xFF are NaN, which decode to 0 so a   */
/*   corrupt byte cannot poison an entire activation row.              */
/* Decoded to f32 on the fly, exactly like fp16 storage: the packed    */
/* layout is the on-disk layout, so an E4M3 checkpoint maps zero-copy. */
/* ------------------------------------------------------------------ */
static inline float veloce_e4m3_to_float(uint8_t b) {
    const unsigned sign = (unsigned)(b >> 7);
    const unsigned exp = (unsigned)((b >> 3) & 0x0F);
    const unsigned man = (unsigned)(b & 0x07);
    float v;
    if (exp == 0) {
        /* subnormal: 2^-6 * (man/8) */
        v = (float)man * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else if (exp == 15 && man == 7) {
        v = 0.0f;                      /* NaN in E4M3 (both signs) */
    } else {
        /* normal: 2^(exp-7) * (1 + man/8); exp==15 & man<7 is finite */
        float p = 1.0f;
        int e = (int)exp - 7;
        /* avoid ldexpf (libm dependency in the hot loop) */
        p = (e >= 0) ? (float)(1 << e) : (1.0f / (float)(1 << (-e)));
        v = p * (1.0f + (float)man * (1.0f / 8.0f));
    }
    return sign ? -v : v;
}

/* 256-entry fp8 decode table, padded to 8 floats like the other LUTs.
 * It is filled once on first use: `veloce_e4m3_to_float` is not a constant
 * expression, so a static initialiser cannot be used (and the table must
 * stay in BSS anyway to keep the "zero runtime allocation" rule -- it is a
 * fixed-size static, not heap). */
static float VELOCE_FP8_LUT[256][8] __attribute__((aligned(32)));
static int VELOCE_FP8_LUT_READY = 0;
static inline const float *veloce_fp8_lut_init(void) {
    if (!VELOCE_FP8_LUT_READY) {
        for (int b = 0; b < 256; b++) {
            const float v = veloce_e4m3_to_float((uint8_t)b);
            for (int s = 0; s < 8; s++) VELOCE_FP8_LUT[b][s] = v;
        }
        VELOCE_FP8_LUT_READY = 1;
    }
    return &VELOCE_FP8_LUT[0][0];
}
#define VELOCE_F8(b) (veloce_fp8_lut_init(), VELOCE_FP8_LUT[b][0])

/* Number of bytes a row of `k` fp8 values occupies (1 per byte). */
static inline size_t veloce_fp8_rowbytes(size_t k) { return k; }

static inline float veloce_fp8_dot_lut(const float *x, const uint8_t *w,
                                       size_t base, size_t K) {
    float a0 = 0.f, a1 = 0.f;
    const uint8_t *wr = w + base;
    size_t q = 0;
    for (; q + 2 <= K; q += 2) {
        a0 += x[q] * VELOCE_F8(wr[q]);
        a1 += x[q + 1] * VELOCE_F8(wr[q + 1]);
    }
    float acc = a0 + a1;
    for (; q < K; q++) acc += x[q] * VELOCE_F8(wr[q]);
    return acc;
}

#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
/* AVX2 fp8 dot: 8 weights per iteration.  AVX2 has no E4M3 decode, so the
 * bytes go through the LUT and are assembled into a vector with set_ps.  The
 * win over the scalar loop is the fused multiply-add and the single
 * horizontal reduction, not the decode. */
static inline float veloce_fp8_dot_avx2(const float *x, const uint8_t *w,
                                        size_t base, size_t K) {
    __m256 acc = _mm256_setzero_ps();
    const uint8_t *wr = w + base;
    size_t q = 0;
    for (; q + 8 <= K; q += 8) {
        const __m256 wv = _mm256_set_ps(
            VELOCE_F8(wr[q + 7]), VELOCE_F8(wr[q + 6]),
            VELOCE_F8(wr[q + 5]), VELOCE_F8(wr[q + 4]),
            VELOCE_F8(wr[q + 3]), VELOCE_F8(wr[q + 2]),
            VELOCE_F8(wr[q + 1]), VELOCE_F8(wr[q]));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + q), wv, acc);
    }
    float tail = 0.0f;
    for (; q < K; q++) tail += x[q] * VELOCE_F8(wr[q]);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_hadd_ps(s4, s4); s4 = _mm_hadd_ps(s4, s4);
    return tail + _mm_cvtss_f32(s4);
}
#endif

static inline float veloce_fp8_dot(const float *x, const uint8_t *w,
                                   size_t base, size_t K) {
#if defined(__AVX2__) && !defined(VELOCE_NO_AVX2)
    return veloce_fp8_dot_avx2(x, w, base, K);
#else
    return veloce_fp8_dot_lut(x, w, base, K);
#endif
}

typedef struct { float *out; const float *x; const uint8_t *w; size_t M, K, N; } veloce_mm_fp8_ctx;
static inline void veloce_mm_fp8_gemv_row(void *vctx, size_t j) {
    const veloce_mm_fp8_ctx *c = (const veloce_mm_fp8_ctx *)vctx;
    c->out[j] = veloce_fp8_dot(c->x, c->w, j * c->K, c->K);
}
static inline void veloce_mm_fp8_mat_row(void *vctx, size_t i) {
    const veloce_mm_fp8_ctx *c = (const veloce_mm_fp8_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = veloce_fp8_dot(xr, c->w, j * c->K, c->K);
}
/* x: [M,K] f32, w: [N,K] fp8 E4M3 (1/byte, FLAT), out: [M,N] f32. */
static inline void veloce_matmul_fp8(float *out, const float *x, const uint8_t *w,
                                     size_t M, size_t K, size_t N) {
    /* Build the decode table before the fork-join so concurrent workers never
     * race on it (it is a fixed static, so this is not a heap allocation). */
    (void)veloce_fp8_lut_init();
    veloce_mm_fp8_ctx c = { out, x, w, M, K, N };
    const size_t row_bytes = veloce_fp8_rowbytes(K);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_fp8_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_fp8_mat_row, &c);
}

/* Row-scaled fp8 matmul: w [N,K] fp8, scales [N] f32. */
typedef struct { float *out; const float *x; const uint8_t *w; const float *s;
                 size_t M, K, N; } veloce_mm_fp8s_ctx;
static inline void veloce_mm_fp8s_gemv_row(void *vctx, size_t j) {
    const veloce_mm_fp8s_ctx *c = (const veloce_mm_fp8s_ctx *)vctx;
    c->out[j] = c->s[j] * veloce_fp8_dot(c->x, c->w, j * c->K, c->K);
}
static inline void veloce_mm_fp8s_mat_row(void *vctx, size_t i) {
    const veloce_mm_fp8s_ctx *c = (const veloce_mm_fp8s_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = c->s[j] * veloce_fp8_dot(xr, c->w, j * c->K, c->K);
}
static inline void veloce_matmul_fp8_s(float *out, const float *x, const uint8_t *w,
                                       const float *scales, size_t M, size_t K, size_t N) {
    (void)veloce_fp8_lut_init();
    veloce_mm_fp8s_ctx c = { out, x, w, scales, M, K, N };
    const size_t row_bytes = veloce_fp8_rowbytes(K) + sizeof(float);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_fp8s_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_fp8s_mat_row, &c);
}

/* ------------------------------------------------------------------ */
/* Group-scaled fp8 matmul: w [N,K] E4M3, scales [N, K/G] f32.         */
/* Same contract as `matmul_i4g` / `matmul_i2g`: the group count must  */
/* divide K exactly, so `K / G` is the exact number of groups.         */
/* ------------------------------------------------------------------ */
static inline float veloce_fp8_dot_scaled(const float *x, const uint8_t *w,
                                          const float *scales, size_t base,
                                          size_t K, size_t G) {
    float acc = 0.0f;
    const size_t groups = G ? K / G : 0;
    for (size_t g = 0; g < groups; g++) {
        float part = veloce_fp8_dot(x + g * G, w, base + g * G, G);
        acc += part * scales[g];
    }
    size_t done = groups * G;
    if (done < K) {
        float part = veloce_fp8_dot(x + done, w, base + done, K - done);
        acc += part * scales[groups];
    }
    return acc;
}
typedef struct { float *out; const float *x; const uint8_t *w; const float *s;
                 size_t M, K, N, G; } veloce_mm_fp8g_ctx;
static inline void veloce_mm_fp8g_gemv_row(void *vctx, size_t j) {
    const veloce_mm_fp8g_ctx *c = (const veloce_mm_fp8g_ctx *)vctx;
    c->out[j] = veloce_fp8_dot_scaled(c->x, c->w, c->s + j * (c->K / c->G),
                                      j * c->K, c->K, c->G);
}
static inline void veloce_mm_fp8g_mat_row(void *vctx, size_t i) {
    const veloce_mm_fp8g_ctx *c = (const veloce_mm_fp8g_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    const size_t ng = c->K / c->G;
    for (size_t j = 0; j < c->N; j++)
        c->out[i * c->N + j] = veloce_fp8_dot_scaled(xr, c->w, c->s + j * ng,
                                                     j * c->K, c->K, c->G);
}
static inline void veloce_matmul_fp8g(float *out, const float *x, const uint8_t *w,
                                      const float *scales, size_t M, size_t K,
                                      size_t N, size_t G) {
    (void)veloce_fp8_lut_init();
    veloce_mm_fp8g_ctx c = { out, x, w, scales, M, K, N, G };
    const size_t row_bytes = veloce_fp8_rowbytes(K) + (K / G) * sizeof(float);
    if (M == 1) veloce_parallel_for_sized(N, row_bytes, veloce_mm_fp8g_gemv_row, &c);
    else        veloce_parallel_for_sized(M, N * row_bytes, veloce_mm_fp8g_mat_row, &c);
}

/* fp16-weight matmul in HF [N,K] (out,in) layout: out[m,n] = dot(x[m,:], w[n,:]). */
typedef struct { float *out; const float *x; const uint16_t *w; size_t M, K, N; } veloce_mm_f16t_ctx;
static inline void veloce_mm_f16t_row(void *vctx, size_t i) {
    const veloce_mm_f16t_ctx *c = (const veloce_mm_f16t_ctx *)vctx;
    const float *xr = c->x + i * c->K;
    for (size_t j = 0; j < c->N; j++) {
        const uint16_t *wr = c->w + j * c->K;
        float acc = 0.0f;
        for (size_t k = 0; k < c->K; k++)
            acc += xr[k] * veloce_half_to_float(wr[k]);
        c->out[i * c->N + j] = acc;
    }
}
/* GEMV work item = one OUTPUT neuron j (M == 1). */
static inline void veloce_mm_f16t_gemv(void *vctx, size_t j) {
    const veloce_mm_f16t_ctx *c = (const veloce_mm_f16t_ctx *)vctx;
    const uint16_t *wr = c->w + j * c->K;
    float acc = 0.0f;
    for (size_t k = 0; k < c->K; k++) acc += c->x[k] * veloce_half_to_float(wr[k]);
    c->out[j] = acc;
}
static inline void veloce_matmul_f16wt(float *out, const float *x, const uint16_t *w,
                                       size_t M, size_t K, size_t N) {
    veloce_mm_f16t_ctx c = { out, x, w, M, K, N };
    if (M == 1) veloce_parallel_for_sized(N, K * sizeof(uint16_t), veloce_mm_f16t_gemv, &c);
    else        veloce_parallel_for_sized(M, N * K * sizeof(uint16_t), veloce_mm_f16t_row, &c);
}

/* Position-offset RoPE applied independently per attention head.
 * x: [rows, dim] with dim = n_heads * head_dim; row r is at absolute
 * position pos + r.  Half-split (LLaMA/NeoX) rotation. */
typedef struct { float *out; const float *x; size_t rows, dim, heads, head_dim, pos; float theta; }
        veloce_ropeh_ctx;
static inline void veloce_rope_heads_row(void *vctx, size_t r) {
    const veloce_ropeh_ctx *c = (const veloce_ropeh_ctx *)vctx;
    const float *xr = c->x + r * c->dim;
    float *orow = c->out + r * c->dim;
    const size_t half = c->head_dim / 2;
    for (size_t h = 0; h < c->heads; h++) {
        const float *xh = xr + h * c->head_dim;
        float *oh = orow + h * c->head_dim;
        for (size_t i = 0; i < half; i++) {
            float freq = powf(c->theta, -2.0f * (float)i / (float)c->head_dim);
            float ang = (float)(c->pos + r) * freq;
            float co = cosf(ang), si = sinf(ang);
            float a = xh[i], b = xh[i + half];
            oh[i] = a * co - b * si;
            oh[i + half] = b * co + a * si;
        }
    }
}
static inline void veloce_rope_heads(float *out, const float *x, size_t rows,
                                     size_t dim, size_t heads, size_t pos, float theta) {
    veloce_ropeh_ctx c = { out, x, rows, dim, heads, dim / heads, pos, theta };
    veloce_parallel_for_sized(rows, dim * 2 * sizeof(float), veloce_rope_heads_row, &c);
}

/* ------------------------------------------------------------------ */
/* Normalisation / activations / attention primitives                 */
/* All row-independent kernels below are parallelised across rows;    */
/* each row keeps its original serial accumulation order (bit-exact). */
/* ------------------------------------------------------------------ */
typedef struct { float *out; const float *x; const float *w; size_t total, dim; } veloce_rms_ctx;
static inline void veloce_rmsnorm_row(void *vctx, size_t r) {
    const veloce_rms_ctx *c = (const veloce_rms_ctx *)vctx;
    const float *xr = c->x + r * c->dim;
    float ss = 0.0f;
    for (size_t i = 0; i < c->dim; i++) ss += xr[i] * xr[i];
    float inv = 1.0f / sqrtf(ss / (float)c->dim + 1e-5f);
    for (size_t i = 0; i < c->dim; i++)
        c->out[r * c->dim + i] = xr[i] * inv * (c->w ? c->w[i] : 1.0f);
}
static inline void veloce_rmsnorm_core(float *out, const float *x,
                                       const float *w, size_t total, size_t dim) {
    veloce_rms_ctx c = { out, x, w, total, dim };
    veloce_parallel_for_sized(total / dim, dim * 2 * sizeof(float), veloce_rmsnorm_row, &c);
}
static inline void veloce_rmsnorm(float *out, const float *x, const float *w,
                                  size_t total, size_t dim) {
    veloce_rmsnorm_core(out, x, w, total, dim);
}
typedef struct { float *out; const float *x; const uint16_t *w; size_t total, dim; } veloce_rmsh_ctx;
static inline void veloce_rmsnorm_h_row(void *vctx, size_t r) {
    const veloce_rmsh_ctx *c = (const veloce_rmsh_ctx *)vctx;
    const float *xr = c->x + r * c->dim;
    float ss = 0.0f;
    for (size_t i = 0; i < c->dim; i++) ss += xr[i] * xr[i];
    float inv = 1.0f / sqrtf(ss / (float)c->dim + 1e-5f);
    for (size_t i = 0; i < c->dim; i++)
        c->out[r * c->dim + i] = xr[i] * inv * veloce_half_to_float(c->w[i]);
}
static inline void veloce_rmsnorm_h(float *out, const float *x, const uint16_t *w,
                                    size_t total, size_t dim) {
    veloce_rmsh_ctx c = { out, x, w, total, dim };
    veloce_parallel_for_sized(total / dim, dim * (sizeof(float) + sizeof(uint16_t)),
                              veloce_rmsnorm_h_row, &c);
}

typedef struct { float *out; const float *x; } veloce_ew_ctx;
static inline void veloce_silu_i(void *vctx, size_t i) {
    const veloce_ew_ctx *c = (const veloce_ew_ctx *)vctx;
    float v = c->x[i];
    /* The exponent is always <= 0, so expf() cannot overflow to +inf.
     * The naive v / (1 + expf(-v)) returns NaN for v < -88 under
     * -ffast-math: GCC vectorises the divide into vrcpps(), and
     * vrcpps(inf) == 0 while inf * 0 == NaN.  This form stays finite for
     * every input, including +-inf (silu(-inf) = -0, silu(+inf) = +inf). */
    if (v >= 0.0f) {
        c->out[i] = v / (1.0f + expf(-v));
    } else {
        float e = expf(v);                       /* in (0, 1] */
        c->out[i] = (v * e) / (1.0f + e);
    }
}
static inline void veloce_silu(float *out, const float *x, size_t n) {
    veloce_ew_ctx c = { out, x };
    veloce_parallel_for_sized(n, 2 * sizeof(float), veloce_silu_i, &c);
}
static inline void veloce_relu_i(void *vctx, size_t i) {
    const veloce_ew_ctx *c = (const veloce_ew_ctx *)vctx;
    c->out[i] = c->x[i] > 0.0f ? c->x[i] : 0.0f;
}
static inline void veloce_relu(float *out, const float *x, size_t n) {
    veloce_ew_ctx c = { out, x };
    veloce_parallel_for_sized(n, 2 * sizeof(float), veloce_relu_i, &c);
}
/* Row-wise softmax over the last axis: x is [total/cols, cols]. */
typedef struct { float *out; const float *x; size_t cols; } veloce_softmax_ctx;
static inline void veloce_softmax_row(void *vctx, size_t r) {
    const veloce_softmax_ctx *c = (const veloce_softmax_ctx *)vctx;
    const float *xr = c->x + r * c->cols;
    float *orow = c->out + r * c->cols;
    float mx = xr[0];
    for (size_t i = 1; i < c->cols; i++) if (xr[i] > mx) mx = xr[i];
    float sum = 0.0f;
    for (size_t i = 0; i < c->cols; i++) { float e = expf(xr[i] - mx); orow[i] = e; sum += e; }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < c->cols; i++) orow[i] *= inv;
}
static inline void veloce_softmax(float *out, const float *x, size_t total, size_t cols) {
    veloce_softmax_ctx c = { out, x, cols };
    veloce_parallel_for_sized(total / cols, cols * 2 * sizeof(float), veloce_softmax_row, &c);
}

/* Masked softmax: an integer mask hides entries.
 *   mask[i] != 0  -> entry i participates
 *   mask[i] == 0  -> entry i gets probability exactly 0
 * The max and the sum are taken over visible entries only, so a hidden entry
 * can never reach the result no matter what it contains - there is no sentinel
 * magnitude to overflow, which is what the old `score + -1e30` convention
 * relied on.  A row with nothing visible produces all zeros (not NaN), since
 * a fully-masked row has no distribution to normalise.
 * Accumulation stays in index order per row, so results are bit-identical at
 * every thread count.
 *
 * Two entry points, one row body:
 *   veloce_softmax_masked      1-D mask `i32[C]`, broadcast to every row
 *   veloce_softmax_masked_rows 2-D mask `i32[T, C]`, one mask row per scores
 *                              row (per-head causal masks)
 * The 2-D form exists because a broadcast read of a per-row mask applies mask
 * row 0 to every row: the builtin used to accept a 2-D mask and read it as
 * 1-D, producing plausible but wrong probabilities instead of an error.
 */
static inline void veloce_softmax_mask_apply(float *orow, const float *xr,
                                             const int32_t *mr, size_t cols) {
    float mx = -INFINITY;
    int any = 0;
    for (size_t i = 0; i < cols; i++) {
        if (mr[i]) {
            if (xr[i] > mx) mx = xr[i];
            any = 1;
        }
    }
    if (!any) {
        for (size_t i = 0; i < cols; i++) orow[i] = 0.0f;
        return;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < cols; i++) {
        if (mr[i]) {
            float e = expf(xr[i] - mx);
            orow[i] = e;
            sum += e;
        } else {
            orow[i] = 0.0f;
        }
    }
    const float inv = 1.0f / sum;
    for (size_t i = 0; i < cols; i++) orow[i] *= inv;
}
typedef struct { float *out; const float *x; const int32_t *mask; size_t cols; int per_row; }
        veloce_softmax_mask_ctx;
static inline void veloce_softmax_mask_row(void *vctx, size_t r) {
    const veloce_softmax_mask_ctx *c = (const veloce_softmax_mask_ctx *)vctx;
    const float *xr = c->x + r * c->cols;
    const int32_t *mr = c->per_row ? c->mask + r * c->cols : c->mask;
    veloce_softmax_mask_apply(c->out + r * c->cols, xr, mr, c->cols);
}
static inline void veloce_softmax_masked(float *out, const float *x, const int32_t *mask,
                                         size_t total, size_t cols) {
    veloce_softmax_mask_ctx c = { out, x, mask, cols, 0 };
    veloce_parallel_for_sized(total / cols, cols * 2 * sizeof(float),
                              veloce_softmax_mask_row, &c);
}
static inline void veloce_softmax_masked_rows(float *out, const float *x, const int32_t *mask,
                                              size_t total, size_t cols) {
    veloce_softmax_mask_ctx c = { out, x, mask, cols, 1 };
    veloce_parallel_for_sized(total / cols, cols * 2 * sizeof(float),
                              veloce_softmax_mask_row, &c);
}

/* Matrix transpose: in [R,C] -> out [C,R]. Parallel across input rows. */
typedef struct { float *out; const float *in; size_t R, C; } veloce_transpose_ctx;
static inline void veloce_transpose_row(void *vctx, size_t r) {
    const veloce_transpose_ctx *c = (const veloce_transpose_ctx *)vctx;
    for (size_t col = 0; col < c->C; col++)
        c->out[col * c->R + r] = c->in[r * c->C + col];
}
static inline void veloce_transpose(float *out, const float *in, size_t R, size_t C) {
    veloce_transpose_ctx c = { out, in, R, C };
    veloce_parallel_for_sized(R, C * 2 * sizeof(float), veloce_transpose_row, &c);
}

/* Rotary position embedding (NeoX/LLaMA half-split layout). */
typedef struct { float *out; const float *x; size_t rows, dim; } veloce_rope_ctx;
static inline void veloce_rope_row(void *vctx, size_t r) {
    const veloce_rope_ctx *c = (const veloce_rope_ctx *)vctx;
    const float *xr = c->x + r * c->dim;
    float *orow = c->out + r * c->dim;
    size_t half = c->dim / 2;
    for (size_t i = 0; i < half; i++) {
        float freq = powf(10000.0f, -2.0f * (float)i / (float)c->dim);
        float ang = (float)r * freq;
        float co = cosf(ang), si = sinf(ang);
        float a = xr[i], b = xr[i + half];
        orow[i] = a * co - b * si;
        orow[i + half] = b * co + a * si;
    }
}
static inline void veloce_rope(float *out, const float *x, size_t rows, size_t dim) {
    veloce_rope_ctx c = { out, x, rows, dim };
    veloce_parallel_for_sized(rows, dim * 2 * sizeof(float), veloce_rope_row, &c);
}

/* Token embedding lookup: out[1,dim] <- table[idx]. */
static inline void veloce_embedding(float *out, const float *table, int idx, size_t dim) {
    memcpy(out, table + (size_t)idx * dim, dim * sizeof(float));
}
static inline void veloce_embedding_h(float *out, const uint16_t *table, int idx, size_t dim) {
    for (size_t i = 0; i < dim; i++)
        out[i] = veloce_half_to_float(table[(size_t)idx * dim + i]);
}

/* Sequence embedding: out[t,:] <- table[ids[t],:] for t in [0,T). */
typedef struct { float *out; const float *table; const int32_t *ids; size_t dim; } veloce_embseq_ctx;
static inline void veloce_embedding_seq_row(void *vctx, size_t t) {
    const veloce_embseq_ctx *c = (const veloce_embseq_ctx *)vctx;
    memcpy(c->out + t * c->dim, c->table + (size_t)c->ids[t] * c->dim,
           c->dim * sizeof(float));
}
static inline void veloce_embedding_seq(float *out, const float *table,
                                        const int32_t *ids, size_t T, size_t dim) {
    veloce_embseq_ctx c = { out, table, ids, dim };
    veloce_parallel_for_sized(T, dim * sizeof(float), veloce_embedding_seq_row, &c);
}

/* Sequence embedding from an fp16 table: out[t,:] <- half(table[ids[t],:]). */
typedef struct { float *out; const uint16_t *table; const int32_t *ids; size_t dim; }
        veloce_embseqh_ctx;
static inline void veloce_embedding_seq_h_row(void *vctx, size_t t) {
    const veloce_embseqh_ctx *c = (const veloce_embseqh_ctx *)vctx;
    const uint16_t *src = c->table + (size_t)c->ids[t] * c->dim;
    float *dst = c->out + t * c->dim;
    for (size_t i = 0; i < c->dim; i++) dst[i] = veloce_half_to_float(src[i]);
}
static inline void veloce_embedding_seq_h(float *out, const uint16_t *table,
                                          const int32_t *ids, size_t T, size_t dim) {
    veloce_embseqh_ctx c = { out, table, ids, dim };
    veloce_parallel_for_sized(T, dim * sizeof(float), veloce_embedding_seq_h_row, &c);
}

/* Mean-pool a [T,D] tensor over the sequence axis -> [1,D]. */
typedef struct { float *out; const float *x; size_t T, dim; } veloce_meanpool_ctx;
static inline void veloce_meanpool_col(void *vctx, size_t d) {
    const veloce_meanpool_ctx *c = (const veloce_meanpool_ctx *)vctx;
    float s = 0.0f;
    for (size_t t = 0; t < c->T; t++) s += c->x[t * c->dim + d];
    c->out[d] = s / (float)c->T;
}
static inline void veloce_meanpool(float *out, const float *x, size_t T, size_t dim) {
    veloce_meanpool_ctx c = { out, x, T, dim };
    veloce_parallel_for_sized(dim, T * sizeof(float), veloce_meanpool_col, &c);
}

/* argmax over a float vector */
static inline int32_t veloce_argmax(const float *x, size_t n) {
    int32_t best = 0;
    for (size_t i = 1; i < n; i++) if (x[i] > x[best]) best = (int32_t)i;
    return best;
}

/* ------------------------------------------------------------------ */
/* Zero-copy model loading (FR-3.3).                                  */
/* A .veloce weight file's packed layout matches the compiler's        */
/* expected in-memory layout, so it can be mmap'd directly and handed  */
/* to the kernels with no deserialisation or copy.                     */
/* ------------------------------------------------------------------ */
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct { const void *data; size_t size; int fd; } veloce_map;

static inline veloce_map veloce_mmap_ro(const char *path) {
    veloce_map m = { NULL, 0, -1 };
    int fd = open(path, O_RDONLY);
    if (fd < 0) return m;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return m; }
    size_t sz = (size_t)st.st_size;
    void *p = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { close(fd); return m; }
    m.data = p; m.size = sz; m.fd = fd;
    return m;
}
static inline void veloce_munmap(veloce_map *m) {
    if (m->data && m->data != MAP_FAILED) munmap((void *)m->data, m->size);
    if (m->fd >= 0) close(m->fd);
    m->data = NULL; m->size = 0; m->fd = -1;
}
#endif /* unix */

#endif /* VELOCE_RUNTIME_H */
