#include <cuda_runtime.h>

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* BigInt digits are LITTLE-ENDIAN:
 * digits[0] is the least significant decimal digit.
 */
typedef struct {
    int *digits;
    int  size;
} BigInt;

#define KARATSUBA_CUDA_THRESHOLD 8192

static void die(const char *msg) {
    fprintf(stderr, "Fatal: %s\n", msg);
    exit(1);
}

static void checkCuda(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

static double nowSeconds(void) {
    using clock = std::chrono::high_resolution_clock;
    static const clock::time_point start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

static void freeBigInt(BigInt *b) {
    free(b->digits);
    b->digits = NULL;
    b->size = 0;
}

static int effectiveSize(const BigInt *b) {
    int s = b->size;
    while (s > 1 && b->digits[s - 1] == 0) s--;
    return s <= 0 ? 1 : s;
}

static void trimBigInt(BigInt *b) {
    int s = effectiveSize(b);
    if (s == b->size) return;

    int *tmp = (int *)realloc(b->digits, (size_t)s * sizeof(int));
    if (!tmp && s > 0) die("realloc failed");
    b->digits = tmp;
    b->size = s;
}

static void initZero(BigInt *b) {
    b->digits = (int *)calloc(1, sizeof(int));
    if (!b->digits) die("calloc failed");
    b->size = 1;
}

static void initializeBigInt(BigInt *b, const char *str) {
    int n = (int)strlen(str);
    if (n <= 0) die("empty input string");

    b->digits = (int *)malloc((size_t)n * sizeof(int));
    if (!b->digits) die("malloc failed");
    b->size = n;

    for (int i = 0; i < n; i++) {
        char c = str[n - 1 - i];
        if (c < '0' || c > '9') die("invalid decimal digit");
        b->digits[i] = c - '0';
    }
    trimBigInt(b);
}

static void randomBigInt(BigInt *b, int ndigits) {
    char *str = (char *)malloc((size_t)ndigits + 1);
    if (!str) die("malloc failed");

    str[0] = (char)('1' + (rand() % 9));
    for (int i = 1; i < ndigits; i++)
        str[i] = (char)('0' + (rand() % 10));
    str[ndigits] = '\0';

    initializeBigInt(b, str);
    free(str);
}

static void addBigInts(const BigInt *a, const BigInt *b, BigInt *result) {
    int maxSize = a->size > b->size ? a->size : b->size;
    result->digits = (int *)calloc((size_t)maxSize + 1, sizeof(int));
    if (!result->digits) die("calloc failed");
    result->size = maxSize + 1;

    int carry = 0;
    for (int i = 0; i < maxSize || carry; i++) {
        int sum = carry;
        if (i < a->size) sum += a->digits[i];
        if (i < b->size) sum += b->digits[i];
        result->digits[i] = sum % 10;
        carry = sum / 10;
    }
    trimBigInt(result);
}

/* Assumes a >= b. This is true for Karatsuba's prod - z2 - z0 step. */
static void subtractBigInts(const BigInt *a, const BigInt *b, BigInt *result) {
    result->digits = (int *)calloc((size_t)a->size, sizeof(int));
    if (!result->digits) die("calloc failed");
    result->size = a->size;

    int borrow = 0;
    for (int i = 0; i < a->size; i++) {
        int diff = a->digits[i] - (i < b->size ? b->digits[i] : 0) - borrow;
        if (diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        result->digits[i] = diff;
    }
    trimBigInt(result);
}

static void shiftLeft(BigInt *b, int shift) {
    if (shift <= 0) return;

    int newSize = b->size + shift;
    int *newDigits = (int *)calloc((size_t)newSize, sizeof(int));
    if (!newDigits) die("calloc failed");

    memcpy(newDigits + shift, b->digits, (size_t)b->size * sizeof(int));
    free(b->digits);
    b->digits = newDigits;
    b->size = newSize;
    trimBigInt(b);
}

static void padToSize(const BigInt *src, int n, BigInt *dst) {
    dst->digits = (int *)calloc((size_t)n, sizeof(int));
    if (!dst->digits) die("calloc failed");
    dst->size = n;

    int copySize = effectiveSize(src);
    if (copySize > n) copySize = n;
    memcpy(dst->digits, src->digits, (size_t)copySize * sizeof(int));
}

static void splitBigInt(const BigInt *b, BigInt *high, BigInt *low, int half) {
    low->size = half > 0 ? half : 1;
    low->digits = (int *)calloc((size_t)low->size, sizeof(int));
    if (!low->digits) die("calloc failed");

    for (int i = 0; i < half && i < b->size; i++)
        low->digits[i] = b->digits[i];
    trimBigInt(low);

    int highSize = b->size - half;
    if (highSize <= 0) {
        initZero(high);
        return;
    }

    high->digits = (int *)malloc((size_t)highSize * sizeof(int));
    if (!high->digits) die("malloc failed");
    high->size = highSize;
    memcpy(high->digits, b->digits + half, (size_t)highSize * sizeof(int));
    trimBigInt(high);
}

static int bigIntsEqual(const BigInt *a, const BigInt *b) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    if (na != nb) return 0;

    for (int i = 0; i < na; i++)
        if (a->digits[i] != b->digits[i]) return 0;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* CPU serial grade-school multiplication                                    */
/* ------------------------------------------------------------------------- */
static void multiplyBigIntsSerial(const BigInt *a, const BigInt *b, BigInt *result) {
    int rawSize = effectiveSize(a) + effectiveSize(b);
    unsigned long long *raw =
        (unsigned long long *)calloc((size_t)rawSize, sizeof(unsigned long long));
    if (!raw) die("calloc failed");

    for (int i = 0; i < effectiveSize(a); i++) {
        for (int j = 0; j < effectiveSize(b); j++) {
            raw[i + j] += (unsigned long long)a->digits[i] * (unsigned long long)b->digits[j];
        }
    }

    result->digits = (int *)calloc((size_t)rawSize + 1, sizeof(int));
    if (!result->digits) die("calloc failed");
    result->size = rawSize + 1;

    unsigned long long carry = 0;
    for (int i = 0; i < rawSize; i++) {
        unsigned long long value = raw[i] + carry;
        result->digits[i] = (int)(value % 10ULL);
        carry = value / 10ULL;
    }
    result->digits[rawSize] = (int)carry;

    trimBigInt(result);
    free(raw);
}

/* ------------------------------------------------------------------------- */
/* CUDA grade-school diagonal multiplication                                 */
/* ------------------------------------------------------------------------- */
__global__ static void gpuMultiplyKernel(const int *A,
                                         int na,
                                         const int *B,
                                         int nb,
                                         unsigned long long *C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int nc = na + nb - 1;
    if (k >= nc) return;

    int start = k - (nb - 1);
    if (start < 0) start = 0;

    int end = k;
    if (end > na - 1) end = na - 1;

    unsigned long long sum = 0;
    for (int i = start; i <= end; i++) {
        int j = k - i;
        sum += (unsigned long long)A[i] * (unsigned long long)B[j];
    }
    C[k] = sum;
}

static void multiplyBigIntsCudaGradeSchool(const BigInt *a,
                                           const BigInt *b,
                                           BigInt *result) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    int rawSize = na + nb;
    int nc = na + nb - 1;

    int *d_A = NULL;
    int *d_B = NULL;
    unsigned long long *d_C = NULL;
    unsigned long long *h_C =
        (unsigned long long *)calloc((size_t)rawSize, sizeof(unsigned long long));
    if (!h_C) die("calloc failed");

    checkCuda(cudaMalloc((void **)&d_A, (size_t)na * sizeof(int)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, (size_t)nb * sizeof(int)), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C, (size_t)rawSize * sizeof(unsigned long long)),
              "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, a->digits, (size_t)na * sizeof(int), cudaMemcpyHostToDevice),
              "copy A to device");
    checkCuda(cudaMemcpy(d_B, b->digits, (size_t)nb * sizeof(int), cudaMemcpyHostToDevice),
              "copy B to device");
    checkCuda(cudaMemset(d_C, 0, (size_t)rawSize * sizeof(unsigned long long)),
              "clear device result");

    int threadsPerBlock = 256;
    int blocksPerGrid = (nc + threadsPerBlock - 1) / threadsPerBlock;
    gpuMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, na, d_B, nb, d_C);
    checkCuda(cudaGetLastError(), "gpuMultiplyKernel launch");
    checkCuda(cudaMemcpy(h_C, d_C, (size_t)rawSize * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost),
              "copy result to host");

    result->digits = (int *)calloc((size_t)rawSize + 1, sizeof(int));
    if (!result->digits) die("calloc failed");
    result->size = rawSize + 1;

    unsigned long long carry = 0;
    for (int i = 0; i < rawSize; i++) {
        unsigned long long value = h_C[i] + carry;
        result->digits[i] = (int)(value % 10ULL);
        carry = value / 10ULL;
    }
    result->digits[rawSize] = (int)carry;
    trimBigInt(result);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);
}

/* ------------------------------------------------------------------------- */
/* CUDA-backed Karatsuba multiplication                                      */
/* ------------------------------------------------------------------------- */
static void multiplyBigIntsCudaKaratsuba(const BigInt *a,
                                         const BigInt *b,
                                         BigInt *result) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    int n = na > nb ? na : nb;

    if (n <= KARATSUBA_CUDA_THRESHOLD) {
        multiplyBigIntsCudaGradeSchool(a, b, result);
        return;
    }

    int half = n / 2;

    BigInt aPad, bPad;
    padToSize(a, n, &aPad);
    padToSize(b, n, &bPad);

    BigInt a1, a0, b1, b0;
    splitBigInt(&aPad, &a1, &a0, half);
    splitBigInt(&bPad, &b1, &b0, half);
    freeBigInt(&aPad);
    freeBigInt(&bPad);

    BigInt z2 = {NULL, 0};
    BigInt z0 = {NULL, 0};
    BigInt prod = {NULL, 0};
    BigInt s1 = {NULL, 0};
    BigInt s2 = {NULL, 0};

    multiplyBigIntsCudaKaratsuba(&a1, &b1, &z2);
    multiplyBigIntsCudaKaratsuba(&a0, &b0, &z0);

    addBigInts(&a1, &a0, &s1);
    addBigInts(&b1, &b0, &s2);
    multiplyBigIntsCudaKaratsuba(&s1, &s2, &prod);

    freeBigInt(&a1);
    freeBigInt(&a0);
    freeBigInt(&b1);
    freeBigInt(&b0);
    freeBigInt(&s1);
    freeBigInt(&s2);

    BigInt tmp = {NULL, 0};
    BigInt z1 = {NULL, 0};
    BigInt sum1 = {NULL, 0};
    BigInt sum2 = {NULL, 0};

    subtractBigInts(&prod, &z2, &tmp);
    subtractBigInts(&tmp, &z0, &z1);
    freeBigInt(&prod);
    freeBigInt(&tmp);

    shiftLeft(&z2, 2 * half);
    shiftLeft(&z1, half);

    addBigInts(&z2, &z1, &sum1);
    addBigInts(&sum1, &z0, &sum2);

    freeBigInt(&z2);
    freeBigInt(&z1);
    freeBigInt(&z0);
    freeBigInt(&sum1);

    *result = sum2;
}

/* ------------------------------------------------------------------------- */
/* Benchmark                                                                 */
/* ------------------------------------------------------------------------- */
typedef struct {
    double serial_sec;
    double cuda_gs_sec;
    double cuda_kar_sec;
} BenchResult;

typedef struct {
    double seconds;
    int result_digits;
} TimedRun;

static double timeMethod(void (*method)(const BigInt *, const BigInt *, BigInt *),
                         const BigInt *a,
                         const BigInt *b,
                         int reps) {
    BigInt result = {NULL, 0};
    double t0 = nowSeconds();
    for (int r = 0; r < reps; r++) {
        if (result.digits) freeBigInt(&result);
        method(a, b, &result);
    }
    checkCuda(cudaDeviceSynchronize(), "benchmark synchronize");
    double elapsed = nowSeconds() - t0;
    if (result.digits) freeBigInt(&result);
    return elapsed / reps;
}

static TimedRun timeMethodWithDigits(void (*method)(const BigInt *, const BigInt *, BigInt *),
                                     const BigInt *a,
                                     const BigInt *b,
                                     int reps) {
    BigInt result = {NULL, 0};
    double t0 = nowSeconds();
    for (int r = 0; r < reps; r++) {
        if (result.digits) freeBigInt(&result);
        method(a, b, &result);
    }
    checkCuda(cudaDeviceSynchronize(), "large benchmark synchronize");
    double elapsed = nowSeconds() - t0;

    TimedRun run;
    run.seconds = elapsed / reps;
    run.result_digits = effectiveSize(&result);

    if (result.digits) freeBigInt(&result);
    return run;
}

int main(int argc, char **argv) {
    printf("=================================================================\n");
    printf("  Big-Integer Multiplication: Serial vs CUDA Grade vs CUDA Karatsuba\n");
    printf("=================================================================\n\n");

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        printf("No CUDA-capable GPU found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    printf("  Using GPU Device        : %s\n", prop.name);
    printf("  Global memory           : %.2f GB\n",
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Max Threads/Block       : %d\n", prop.maxThreadsPerBlock);
    printf("  Karatsuba CUDA threshold: %d digits\n", KARATSUBA_CUDA_THRESHOLD);
    printf("=================================================================\n\n");

    if (argc >= 2) {
        int ndigits = atoi(argv[1]);
        int reps = (argc >= 3) ? atoi(argv[2]) : 1;
        if (ndigits <= 0 || reps <= 0) {
            fprintf(stderr, "Usage: %s [digits reps]\n", argv[0]);
            return 1;
        }

        printf("--- Large CUDA-only Benchmark ---\n");
        printf("Digits per operand : %d\n", ndigits);
        printf("Repetitions        : %d\n", reps);
        printf("Serial baseline    : skipped for large-input mode\n\n");

        srand(12345);
        BigInt a = {NULL, 0}, b = {NULL, 0};
        randomBigInt(&a, ndigits);
        randomBigInt(&b, ndigits);

        TimedRun cudaGs = timeMethodWithDigits(multiplyBigIntsCudaGradeSchool, &a, &b, reps);
        TimedRun cudaKar = timeMethodWithDigits(multiplyBigIntsCudaKaratsuba, &a, &b, reps);

        printf("  %-14s  %-16s  %-16s\n", "Method", "Time(s)", "ResultDigits");
        printf("  %-14s  %-16s  %-16s\n", "--------------", "----------------", "----------------");
        printf("  %-14s  %-16.6f  %-16d\n",
               "CUDA-GS", cudaGs.seconds, cudaGs.result_digits);
        printf("  %-14s  %-16.6f  %-16d\n",
               "CUDA-Karatsuba", cudaKar.seconds, cudaKar.result_digits);
        printf("\nNote: correctness is not checked against serial in large-input mode.\n");

        freeBigInt(&a);
        freeBigInt(&b);
        return 0;
    }

    printf("--- Correctness Check (random pairs, varying sizes) ---\n");
    srand(12345);
    int check_sizes[] = {7, 9, 50, 100, 500, 1000, 2000, 5000, 10000};
    int allOk = 1;

    for (int t = 0; t < (int)(sizeof check_sizes / sizeof check_sizes[0]); t++) {
        BigInt a = {NULL, 0}, b = {NULL, 0};
        BigInt serial = {NULL, 0}, cudaGs = {NULL, 0}, cudaKar = {NULL, 0};

        randomBigInt(&a, check_sizes[t]);
        randomBigInt(&b, check_sizes[t]);

        multiplyBigIntsSerial(&a, &b, &serial);
        multiplyBigIntsCudaGradeSchool(&a, &b, &cudaGs);
        multiplyBigIntsCudaKaratsuba(&a, &b, &cudaKar);

        int gsOk = bigIntsEqual(&serial, &cudaGs);
        int karOk = bigIntsEqual(&serial, &cudaKar);
        if (!gsOk || !karOk) allOk = 0;

        printf("  %6d digits: Serial==CUDA-Grade: %-4s  Serial==CUDA-Karatsuba: %s\n",
               check_sizes[t],
               gsOk ? "PASS" : "FAIL",
               karOk ? "PASS" : "FAIL");

        freeBigInt(&a);
        freeBigInt(&b);
        freeBigInt(&serial);
        freeBigInt(&cudaGs);
        freeBigInt(&cudaKar);
    }
    printf("  Overall: %s\n\n", allOk ? "ALL PASSED" : "SOME FAILED");

    printf("--- Benchmark (wall-clock time per single multiplication) ---\n\n");
    printf("  %-10s  %-6s  %-14s  %-14s  %-14s  %-12s  %-12s\n",
           "Digits", "Reps", "Serial(s)", "CUDA-GS(s)", "CUDA-Kar(s)",
           "GS Speedup", "Kar Speedup");
    printf("  %-10s  %-6s  %-14s  %-14s  %-14s  %-12s  %-12s\n",
           "----------", "------", "--------------", "--------------",
           "--------------", "------------", "------------");

    struct {
        int digits;
        int reps;
    } cases[] = {
        {       9, 1000 },
        {      50, 1000 },
        {     100,  500 },
        {     500,  200 },
        {    1000,  100 },
        {    5000,   10 },
        {   10000,    5 },
        {   20000,    2 },
    };

    for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); i++) {
        int nd = cases[i].digits;
        int reps = cases[i].reps;

        BigInt a = {NULL, 0}, b = {NULL, 0};
        randomBigInt(&a, nd);
        randomBigInt(&b, nd);

        double serialTime = timeMethod(multiplyBigIntsSerial, &a, &b, reps);
        double cudaGsTime = timeMethod(multiplyBigIntsCudaGradeSchool, &a, &b, reps);
        double cudaKarTime = timeMethod(multiplyBigIntsCudaKaratsuba, &a, &b, reps);

        char gsSpeed[32];
        char karSpeed[32];
        snprintf(gsSpeed, sizeof gsSpeed, "%.2fx", serialTime / cudaGsTime);
        snprintf(karSpeed, sizeof karSpeed, "%.2fx", serialTime / cudaKarTime);

        printf("  %-10d  %-6d  %-14.6f  %-14.6f  %-14.6f  %-12s  %-12s\n",
               nd,
               reps,
               serialTime,
               cudaGsTime,
               cudaKarTime,
               gsSpeed,
               karSpeed);
        fflush(stdout);

        freeBigInt(&a);
        freeBigInt(&b);
    }

    return 0;
}
