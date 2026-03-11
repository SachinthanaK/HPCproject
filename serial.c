
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------------------
 * BigInt: digits stored LITTLE-ENDIAN (digits[0] = least significant digit)
 * --------------------------------------------------------------------------- */
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

/* Build a BigInt from a decimal string */
void initializeBigInt(BigInt *b, const char *str) {
    b->size   = (int)strlen(str);
    b->digits = (int *)malloc(b->size * sizeof(int));
    for (int i = 0; i < b->size; i++)
        b->digits[i] = str[b->size - 1 - i] - '0';
}

/* Build a BigInt from a random n-digit decimal number */
void randomBigInt(BigInt *b, int ndigits, unsigned int *seed) {
    char *str = (char *)malloc(ndigits + 1);
    str[0] = '1' + (rand_r(seed) % 9);   /* no leading zero */
    for (int i = 1; i < ndigits; i++)
        str[i] = '0' + (rand_r(seed) % 10);
    str[ndigits] = '\0';
    initializeBigInt(b, str);
    free(str);
}

void printBigIntShort(const BigInt *b, int maxDigits) {
    int i = b->size - 1;
    while (i > 0 && b->digits[i] == 0) i--;
    int total = i + 1;
    if (total <= maxDigits * 2) {
        for (; i >= 0; i--) printf("%d", b->digits[i]);
    } else {
        /* print first maxDigits ... last maxDigits */
        for (int j = i; j > i - maxDigits; j--) printf("%d", b->digits[j]);
        printf("...[%d digits total]...", total);
        for (int j = maxDigits - 1; j >= 0; j--) printf("%d", b->digits[j]);
    }
    printf("\n");
}

/* ---- Arithmetic ---------------------------------------------------------- */

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

/* ---- Grade-school O(n²) -------------------------------------------------- */
void multiplyBigInts(const BigInt *a, const BigInt *b, BigInt *result) {
    int maxSize    = a->size + b->size;
    result->digits = (int *)calloc(maxSize, sizeof(int));
    result->size   = maxSize;
    for (int i = 0; i < a->size; i++) {
        for (int j = 0; j < b->size; j++) {
            int product = a->digits[i] * b->digits[j] + result->digits[i + j];
            result->digits[i + j]     = product % 10;
            result->digits[i + j + 1] += product / 10;
        }
    }
}

/* ---- Karatsuba O(n^1.585) ------------------------------------------------
 *
 * HYBRID: switch to grade-school below KARATSUBA_THRESHOLD digits.
 * This eliminates the recursion overhead that made Karatsuba slow on small
 * inputs, and is exactly what production big-integer libraries do.
 * Tune KARATSUBA_THRESHOLD to find the crossover point on your CPU.
 * --------------------------------------------------------------------------- */
#define KARATSUBA_THRESHOLD 32

void karatsubaMultiply(const BigInt *a, const BigInt *b, BigInt *result) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    int n  = na > nb ? na : nb;

    /* Base case: fall back to grade-school for small numbers */
    if (n <= KARATSUBA_THRESHOLD) {
        multiplyBigInts(a, b, result);
        return;
    }

    int half = n / 2;

    /* Pad both operands to exactly n digits */
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
    karatsubaMultiply(&a1, &b1, &z2);
    karatsubaMultiply(&a0, &b0, &z0);

    /* z1 = (a1+a0)*(b1+b0) - z2 - z0 */
    BigInt a1a0, b1b0, prod, tmp, z1;
    addBigInts(&a1, &a0, &a1a0);
    addBigInts(&b1, &b0, &b1b0);
    karatsubaMultiply(&a1a0, &b1b0, &prod);
    freeBigInt(&a1a0); freeBigInt(&b1b0);
    freeBigInt(&a0);   freeBigInt(&a1);
    freeBigInt(&b0);   freeBigInt(&b1);

    subtractBigInts(&prod, &z2, &tmp); freeBigInt(&prod);
    subtractBigInts(&tmp,  &z0, &z1);  freeBigInt(&tmp);

    /* result = z2*B^(2*half) + z1*B^half + z0 */
    shiftLeft(&z2, 2 * half);
    shiftLeft(&z1, half);

    BigInt sum1, sum2;
    addBigInts(&z2, &z1, &sum1);      freeBigInt(&z2); freeBigInt(&z1);
    addBigInts(&sum1, &z0, &sum2);    freeBigInt(&sum1); freeBigInt(&z0);
    *result = sum2;
}

/* ---- Equality check ------------------------------------------------------ */
static int bigIntsEqual(const BigInt *a, const BigInt *b) {
    int na = effectiveSize(a), nb = effectiveSize(b);
    if (na != nb) return 0;
    for (int i = 0; i < na; i++)
        if (a->digits[i] != b->digits[i]) return 0;
    return 1;
}

/* ---- Benchmark helper ---------------------------------------------------- */
typedef struct {
    double gs_sec;       /* grade-school time  (-1 = skipped)  */
    double kar_sec;      /* karatsuba time                      */
    int    correct;      /* 1 if results match (or GS skipped)  */
    int    result_digits;
} BenchResult;

static double now_sec(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

BenchResult runBenchmark(int ndigits, int gs_reps, int kar_reps, int skip_gs) {
    BenchResult r = { -1.0, 0.0, 1, 0 };
    unsigned int seed = 12345;

    BigInt a, b;
    randomBigInt(&a, ndigits, &seed);
    randomBigInt(&b, ndigits, &seed);

    BigInt gs_result  = { NULL, 0 };
    BigInt kar_result = { NULL, 0 };

    /* --- Grade-school --- */
    if (!skip_gs) {
        double t0 = now_sec();
        for (int i = 0; i < gs_reps; i++) {
            if (gs_result.digits) freeBigInt(&gs_result);
            multiplyBigInts(&a, &b, &gs_result);
        }
        r.gs_sec = now_sec() - t0;
    }

    /* --- Karatsuba --- */
    double t0 = now_sec();
    for (int i = 0; i < kar_reps; i++) {
        if (kar_result.digits) freeBigInt(&kar_result);
        karatsubaMultiply(&a, &b, &kar_result);
    }
    r.kar_sec = now_sec() - t0;

    /* --- Correctness (only when both ran) --- */
    if (!skip_gs && gs_result.digits && kar_result.digits)
        r.correct = bigIntsEqual(&gs_result, &kar_result);

    r.result_digits = effectiveSize(&kar_result);

    if (gs_result.digits)  freeBigInt(&gs_result);
    if (kar_result.digits) freeBigInt(&kar_result);
    freeBigInt(&a);
    freeBigInt(&b);
    return r;
}

/* ---- main ---------------------------------------------------------------- */
int main(void) {
    printf("============================================================\n");
    printf("  Big-Integer Multiplication Benchmark\n");
    printf("  Karatsuba threshold: %d digits\n", KARATSUBA_THRESHOLD);
    printf("============================================================\n\n");

    /* Quick correctness check first */
    printf("--- Correctness check (9-digit numbers, 5 random pairs) ---\n");
    unsigned int seed = 99;
    int allOk = 1;
    for (int t = 0; t < 5; t++) {
        BigInt a, b, gs, kar;
        randomBigInt(&a, 9, &seed);
        randomBigInt(&b, 9, &seed);
        multiplyBigInts(&a, &b, &gs);
        karatsubaMultiply(&a, &b, &kar);
        int ok = bigIntsEqual(&gs, &kar);
        if (!ok) allOk = 0;
        printf("  Test %d: %s\n", t + 1, ok ? "PASS ✓" : "FAIL ✗");
        freeBigInt(&a); freeBigInt(&b); freeBigInt(&gs); freeBigInt(&kar);
    }
    printf("  All correctness tests: %s\n\n", allOk ? "PASSED ✓" : "FAILED ✗");

    /* Scaling benchmark table */
    printf("%-12s  %-8s  %-14s  %-14s  %-10s  %s\n",
           "Digits", "Reps", "GradeSchool(s)", "Karatsuba(s)", "Speedup", "Correct");
    printf("%-12s  %-8s  %-14s  %-14s  %-10s  %s\n",
           "------------", "--------", "--------------",
           "--------------", "----------", "-------");

    struct { int digits; int gs_reps; int kar_reps; int skip_gs; } cases[] = {
        {       9,  100000,  100000, 0 },
        {      50,   10000,   10000, 0 },
        {     100,    1000,    1000, 0 },
        {     500,      50,     100, 0 },
        {    1000,      10,      50, 0 },
        {    5000,       1,      10, 0 },   /* GS too slow — skip */
        {   10000,       1,       5, 0 },   /* GS too slow — skip */
        {  100000,       1,       3, 0 },
        { 1000000,       1,       1, 0 },
    };

    for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); i++) {
        int    nd      = cases[i].digits;
        int    gs_r    = cases[i].gs_reps;
        int    kar_r   = cases[i].kar_reps;
        int    skip_gs = cases[i].skip_gs;

        printf("%-12d  ", nd);
        fflush(stdout);

        BenchResult r = runBenchmark(nd, gs_r, kar_r, skip_gs);

        /* Normalise to per-run times */
        double gs_per  = skip_gs ? -1.0 : r.gs_sec  / gs_r;
        double kar_per = r.kar_sec / kar_r;

        if (skip_gs) {
            printf("%-8d  %-14s  %-14.6f  %-10s  %s\n",
                   kar_r, "skipped", kar_per, "N/A",
                   "N/A (GS skipped)");
        } else {
            double speedup = gs_per / kar_per;
            printf("%-8d  %-14.6f  %-14.6f  %-10.2fx  %s\n",
                   gs_r, gs_per, kar_per, speedup,
                   r.correct ? "YES ✓" : "NO ✗");
        }
        fflush(stdout);
    }

    printf("\nNote: times are per single multiplication.\n");
    printf("Negative speedup (<1x) means Karatsuba is slower (small-number overhead).\n");
    return 0;
}

