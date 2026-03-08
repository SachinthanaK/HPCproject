#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* ===========================================================================
 * BigInt: LITTLE-ENDIAN (digits[0] = least significant digit)
 * =========================================================================== */
typedef struct {
    int *digits;
    int  size;
} BigInt;

static void freeBigInt(BigInt *b) {
    free(b->digits);
    b->digits = NULL;
    b->size   = 0;
}

static int effectiveSize(const BigInt *b) {
    int s = b->size;
    while (s > 1 && b->digits[s - 1] == 0) s--;
    return s;
}

void initializeBigInt(BigInt *b, const char *str) {
    b->size   = (int)strlen(str);
    b->digits = (int *)malloc(b->size * sizeof(int));
    for (int i = 0; i < b->size; i++)
        b->digits[i] = str[b->size - 1 - i] - '0';
}

void randomBigInt(BigInt *b, int ndigits, unsigned int *seed) {
    char *str = (char *)malloc(ndigits + 1);
    str[0] = '1' + (rand_r(seed) % 9);
    for (int i = 1; i < ndigits; i++)
        str[i] = '0' + (rand_r(seed) % 10);
    str[ndigits] = '\0';
    initializeBigInt(b, str);
    free(str);
}

/* ===========================================================================
 * Arithmetic primitives
 * =========================================================================== */

static void addBigInts(const BigInt *a, const BigInt *b, BigInt *result) {
    int maxSize = a->size > b->size ? a->size : b->size;
    result->digits = (int *)calloc(maxSize + 1, sizeof(int));
    result->size   = maxSize + 1;
    int carry = 0;
    for (int i = 0; i < maxSize || carry; i++) {
        int sum = carry;
        if (i < a->size) sum += a->digits[i];
        if (i < b->size) sum += b->digits[i];
        result->digits[i] = sum % 10;
        carry = sum / 10;
    }
}

static void subtractBigInts(const BigInt *a, const BigInt *b, BigInt *result) {
    result->digits = (int *)calloc(a->size, sizeof(int));
    result->size   = a->size;
    int borrow = 0;
    for (int i = 0; i < a->size; i++) {
        int diff = a->digits[i] - (i < b->size ? b->digits[i] : 0) - borrow;
        if (diff < 0) { diff += 10; borrow = 1; } else borrow = 0;
        result->digits[i] = diff;
    }
}

static void shiftLeft(BigInt *b, int shift) {
    if (shift == 0) return;
    int newSize    = b->size + shift;
    int *newDigits = (int *)calloc(newSize, sizeof(int));
    memcpy(newDigits + shift, b->digits, b->size * sizeof(int));
    free(b->digits);
    b->digits = newDigits;
    b->size   = newSize;
}

static void splitBigInt(const BigInt *b, BigInt *high, BigInt *low, int half) {
    low->size   = half;
    low->digits = (int *)malloc(half * sizeof(int));
    memcpy(low->digits, b->digits, half * sizeof(int));

    high->size = b->size - half;
    if (high->size <= 0) {
        high->size   = 1;
        high->digits = (int *)calloc(1, sizeof(int));
    } else {
        high->digits = (int *)malloc(high->size * sizeof(int));
        memcpy(high->digits, b->digits + half, high->size * sizeof(int));
    }
}

/* ===========================================================================
 * SERIAL Karatsuba
 * Base case falls back to single-digit multiply below THRESHOLD.
 * =========================================================================== */
#define KARATSUBA_THRESHOLD 32

static void karatsubaCore(const BigInt *a, const BigInt *b, BigInt *result) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    int n  = na > nb ? na : nb;

    /* Base case: single or very small */
    if (n <= KARATSUBA_THRESHOLD) {
        int maxSize    = a->size + b->size;
        result->digits = (int *)calloc(maxSize, sizeof(int));
        result->size   = maxSize;
        for (int i = 0; i < a->size; i++)
            for (int j = 0; j < b->size; j++) {
                int p = a->digits[i] * b->digits[j] + result->digits[i+j];
                result->digits[i+j]   = p % 10;
                result->digits[i+j+1] += p / 10;
            }
        return;
    }

    int half = n / 2;

    BigInt aPad = { (int *)calloc(n, sizeof(int)), n };
    BigInt bPad = { (int *)calloc(n, sizeof(int)), n };
    memcpy(aPad.digits, a->digits, na * sizeof(int));
    memcpy(bPad.digits, b->digits, nb * sizeof(int));

    BigInt a1, a0, b1, b0;
    splitBigInt(&aPad, &a1, &a0, half);
    splitBigInt(&bPad, &b1, &b0, half);
    freeBigInt(&aPad);
    freeBigInt(&bPad);

    /* z2 = a1*b1,  z0 = a0*b0 */
    BigInt z2, z0;
    karatsubaCore(&a1, &b1, &z2);
    karatsubaCore(&a0, &b0, &z0);

    /* z1 = (a1+a0)*(b1+b0) - z2 - z0 */
    BigInt a1a0, b1b0, prod, tmp, z1;
    addBigInts(&a1, &a0, &a1a0);
    addBigInts(&b1, &b0, &b1b0);
    karatsubaCore(&a1a0, &b1b0, &prod);
    freeBigInt(&a1a0); freeBigInt(&b1b0);
    freeBigInt(&a0);   freeBigInt(&a1);
    freeBigInt(&b0);   freeBigInt(&b1);

    subtractBigInts(&prod, &z2, &tmp); freeBigInt(&prod);
    subtractBigInts(&tmp,  &z0, &z1);  freeBigInt(&tmp);

    shiftLeft(&z2, 2 * half);
    shiftLeft(&z1, half);

    BigInt sum1, sum2;
    addBigInts(&z2, &z1, &sum1);      freeBigInt(&z2); freeBigInt(&z1);
    addBigInts(&sum1, &z0, &sum2);    freeBigInt(&sum1); freeBigInt(&z0);
    *result = sum2;
}

void karatsubaSerial(const BigInt *a, const BigInt *b, BigInt *result) {
    karatsubaCore(a, b, result);
}

/* ===========================================================================
 * PARALLEL Karatsuba with OpenMP tasks
 *
 * The three recursive calls  z2 = a1*b1,  z0 = a0*b0,  z1 = (a1+a0)*(b1+b0)
 * are completely independent --- perfect for task parallelism.
 *
 * OMP_TASK_THRESHOLD: below this digit count we stop creating new tasks and
 * fall back to serial, avoiding task-spawn overhead on tiny sub-problems.
 * =========================================================================== */
#define OMP_TASK_THRESHOLD 500

static void karatsubaParallelInner(const BigInt *a, const BigInt *b,
                                   BigInt *result) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    int n  = na > nb ? na : nb;

    /* Below task threshold: run serially --- no more task overhead */
    if (n <= OMP_TASK_THRESHOLD) {
        karatsubaCore(a, b, result);
        return;
    }

    int half = n / 2;

    BigInt aPad = { (int *)calloc(n, sizeof(int)), n };
    BigInt bPad = { (int *)calloc(n, sizeof(int)), n };
    memcpy(aPad.digits, a->digits, na * sizeof(int));
    memcpy(bPad.digits, b->digits, nb * sizeof(int));

    BigInt a1, a0, b1, b0;
    splitBigInt(&aPad, &a1, &a0, half);
    splitBigInt(&bPad, &b1, &b0, half);
    freeBigInt(&aPad);
    freeBigInt(&bPad);

    /* Pre-compute sums needed for z1 --- no dependency on z2/z0 */
    BigInt a1a0, b1b0;
    addBigInts(&a1, &a0, &a1a0);
    addBigInts(&b1, &b0, &b1b0);

    BigInt z2, z0, z1prod;

    /* Deep-copy inputs for tasks: firstprivate only copies the struct shell
     * (pointer + size), NOT the heap array behind it.  If the spawning thread
     * frees a0/a1/b0/b1 before a task reads them we get use-after-free.
     * Giving each task its own heap copy avoids the race entirely. */
    BigInt a1c = { (int *)malloc(a1.size * sizeof(int)), a1.size };
    BigInt b1c = { (int *)malloc(b1.size * sizeof(int)), b1.size };
    BigInt a0c = { (int *)malloc(a0.size * sizeof(int)), a0.size };
    BigInt b0c = { (int *)malloc(b0.size * sizeof(int)), b0.size };
    memcpy(a1c.digits, a1.digits, a1.size * sizeof(int));
    memcpy(b1c.digits, b1.digits, b1.size * sizeof(int));
    memcpy(a0c.digits, a0.digits, a0.size * sizeof(int));
    memcpy(b0c.digits, b0.digits, b0.size * sizeof(int));

    /* Spawn z2 = a1*b1 and z0 = a0*b0 as independent parallel tasks.
     * Each task owns its copies and frees them on completion. */
    #pragma omp task shared(z2) firstprivate(a1c, b1c)
    {
        karatsubaParallelInner(&a1c, &b1c, &z2);
        freeBigInt(&a1c);
        freeBigInt(&b1c);
    }

    #pragma omp task shared(z0) firstprivate(a0c, b0c)
    {
        karatsubaParallelInner(&a0c, &b0c, &z0);
        freeBigInt(&a0c);
        freeBigInt(&b0c);
    }

    /* Current thread computes z1 while the two tasks run in parallel */
    karatsubaParallelInner(&a1a0, &b1b0, &z1prod);
    freeBigInt(&a1a0); freeBigInt(&b1b0);
    freeBigInt(&a0);   freeBigInt(&a1);
    freeBigInt(&b0);   freeBigInt(&b1);

    /* Wait for both tasks before combining */
    #pragma omp taskwait

    BigInt tmp, z1;
    subtractBigInts(&z1prod, &z2, &tmp); freeBigInt(&z1prod);
    subtractBigInts(&tmp,    &z0, &z1);  freeBigInt(&tmp);

    shiftLeft(&z2, 2 * half);
    shiftLeft(&z1, half);

    BigInt sum1, sum2;
    addBigInts(&z2, &z1, &sum1);      freeBigInt(&z2); freeBigInt(&z1);
    addBigInts(&sum1, &z0, &sum2);    freeBigInt(&sum1); freeBigInt(&z0);
    *result = sum2;
}

void karatsubaParallel(const BigInt *a, const BigInt *b, BigInt *result) {
    #pragma omp parallel
    #pragma omp single
    karatsubaParallelInner(a, b, result);
}

/* ===========================================================================
 * Correctness check
 * =========================================================================== */
static int bigIntsEqual(const BigInt *a, const BigInt *b) {
    int na = effectiveSize(a), nb = effectiveSize(b);
    if (na != nb) return 0;
    for (int i = 0; i < na; i++)
        if (a->digits[i] != b->digits[i]) return 0;
    return 1;
}

static void printBigIntShort(const BigInt *b) {
    int top = effectiveSize(b) - 1;
    if (top + 1 <= 20) {
        for (int i = top; i >= 0; i--) printf("%d", b->digits[i]);
    } else {
        for (int i = top; i > top - 10; i--) printf("%d", b->digits[i]);
        printf("...[%d digits]...", top + 1);
        for (int i = 9; i >= 0; i--) printf("%d", b->digits[i]);
    }
    printf("\n");
}

/* ===========================================================================
 * main
 * =========================================================================== */
int main(void) {
    printf("=================================================================\n");
    printf("  Karatsuba: Serial vs OpenMP Parallel\n");
    printf("  Threads          : %d\n", omp_get_max_threads());
    printf("  Base threshold   : %d digits  (switch to schoolbook)\n",
           KARATSUBA_THRESHOLD);
    printf("  Task threshold   : %d digits  (stop spawning OMP tasks)\n",
           OMP_TASK_THRESHOLD);
    printf("=================================================================\n\n");

    /* -----------------------------------------------------------------------
     * Correctness: serial result must equal parallel result
     * --------------------------------------------------------------------- */
    printf("--- Correctness (10 random pairs, varying sizes) ---\n");
    unsigned int seed = 42;
    int allOk = 1;
    int check_sizes[] = { 7, 9, 50, 100, 500, 1000, 5000, 10000, 50000, 100000 };
    for (int t = 0; t < 10; t++) {
        BigInt a, b, ser, par;
        randomBigInt(&a, check_sizes[t], &seed);
        randomBigInt(&b, check_sizes[t], &seed);
        karatsubaSerial(&a, &b, &ser);
        karatsubaParallel(&a, &b, &par);
        int ok = bigIntsEqual(&ser, &par);
        if (!ok) allOk = 0;
        printf("  %6d digits: Serial==Parallel: %s\n",
               check_sizes[t], ok ? "PASS ✓" : "FAIL ✗");
        freeBigInt(&a); freeBigInt(&b);
        freeBigInt(&ser); freeBigInt(&par);
    }
    printf("  Overall: %s\n\n", allOk ? "ALL PASSED ✓" : "SOME FAILED ✗");

    /* -----------------------------------------------------------------------
     * Sample output (just to show numbers look right)
     * --------------------------------------------------------------------- */
    printf("--- Sample output (123456789 x 987654321) ---\n");
    BigInt sa, sb, sser, spar;
    initializeBigInt(&sa, "123456789");
    initializeBigInt(&sb, "987654321");
    karatsubaSerial(&sa, &sb, &sser);
    karatsubaParallel(&sa, &sb, &spar);
    printf("  Serial  : "); printBigIntShort(&sser);
    printf("  Parallel: "); printBigIntShort(&spar);
    printf("  Match   : %s\n\n", bigIntsEqual(&sser, &spar) ? "YES ✓" : "NO ✗");
    freeBigInt(&sa); freeBigInt(&sb);
    freeBigInt(&sser); freeBigInt(&spar);

    /* -----------------------------------------------------------------------
     * Benchmark
     * --------------------------------------------------------------------- */
    printf("--- Benchmark (wall-clock time per single multiplication) ---\n\n");
    printf("  %-10s  %-6s  %-14s  %-14s  %-10s\n",
           "Digits", "Reps", "Serial (s)", "Parallel (s)", "Speedup");
    printf("  %-10s  %-6s  %-14s  %-14s  %-10s\n",
           "----------", "------", "--------------",
           "--------------", "----------");

    struct { int digits; int reps; } cases[] = {
        {       9, 100000 },
        {      50,  50000 },
        {     100,  10000 },
        {     500,   2000 },
        {    1000,    500 },
        {    5000,     20 },
        {   10000,     10 },
        {   50000,      5 },
        {  100000,      3 },
        {  500000,      1 },
        { 1000000,      1 },
    };

    for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); i++) {
        int nd   = cases[i].digits;
        int reps = cases[i].reps;

        printf("  %-10d  %-6d  ", nd, reps);
        fflush(stdout);

        unsigned int s = 77777;
        BigInt a, b;
        randomBigInt(&a, nd, &s);
        randomBigInt(&b, nd, &s);

        BigInt res = {NULL, 0};

        /* Serial */
        double t0 = omp_get_wtime();
        for (int r = 0; r < reps; r++) {
            if (res.digits) freeBigInt(&res);
            karatsubaSerial(&a, &b, &res);
        }
        double ser_time = (omp_get_wtime() - t0) / reps;
        if (res.digits) freeBigInt(&res);

        /* Parallel */
        t0 = omp_get_wtime();
        for (int r = 0; r < reps; r++) {
            if (res.digits) freeBigInt(&res);
            karatsubaParallel(&a, &b, &res);
        }
        double par_time = (omp_get_wtime() - t0) / reps;
        if (res.digits) freeBigInt(&res);

        double speedup = ser_time / par_time;
        printf("%-14.6f  %-14.6f  %.2fx\n", ser_time, par_time, speedup);
        fflush(stdout);

        freeBigInt(&a);
        freeBigInt(&b);
    }

    printf("\n  Speedup > 1.0x means OpenMP is faster than serial.\n");
    printf("  Speedup < 1.0x means task overhead outweighs parallelism gain.\n");
    printf("  (More threads = higher speedup at large digit counts)\n");
    return 0;
}