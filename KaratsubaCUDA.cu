/*
 * CUDA Big-Integer Multiplication
 *
 * This file adds a CUDA/GPU comparison point for the existing Karatsuba
 * project. It uses the same BigInt idea as the C files: decimal digits stored
 * little-endian, where digits[0] is the least significant digit.
 *
 * Compile:
 *   nvcc -O2 -std=c++11 KaratsubaCUDA.cu -o KaratsubaCUDA.exe
 *
 * Run:
 *   .\KaratsubaCUDA.exe 10000 5
 *
 * Arguments:
 *   argv[1] = number of digits per operand
 *   argv[2] = repetitions for timing
 */

#include <cuda_runtime.h>

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *digits;
    int size;
} BigInt;

static void checkCuda(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

static void freeBigInt(BigInt *b) {
    free(b->digits);
    b->digits = NULL;
    b->size = 0;
}

static int effectiveSize(const BigInt *b) {
    int s = b->size;
    while (s > 1 && b->digits[s - 1] == 0) s--;
    return s;
}

static void trimBigInt(BigInt *b) {
    int s = effectiveSize(b);
    if (s == b->size) return;

    int *tmp = (int *)realloc(b->digits, (size_t)s * sizeof(int));
    if (!tmp && s > 0) {
        fprintf(stderr, "realloc failed\n");
        exit(1);
    }
    b->digits = tmp;
    b->size = s;
}

static unsigned int nextRandom(unsigned int *seed) {
    *seed = (*seed * 1664525u) + 1013904223u;
    return *seed;
}

static void initializeBigInt(BigInt *b, const char *str) {
    int n = (int)strlen(str);
    if (n <= 0) {
        fprintf(stderr, "empty number string\n");
        exit(1);
    }

    b->size = n;
    b->digits = (int *)malloc((size_t)n * sizeof(int));
    if (!b->digits) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        char c = str[n - 1 - i];
        if (c < '0' || c > '9') {
            fprintf(stderr, "invalid digit in input\n");
            exit(1);
        }
        b->digits[i] = c - '0';
    }
    trimBigInt(b);
}

static void randomBigInt(BigInt *b, int ndigits, unsigned int *seed) {
    if (ndigits <= 0) {
        fprintf(stderr, "digit count must be positive\n");
        exit(1);
    }

    char *str = (char *)malloc((size_t)ndigits + 1);
    if (!str) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    str[0] = (char)('1' + (nextRandom(seed) % 9));
    for (int i = 1; i < ndigits; i++) {
        str[i] = (char)('0' + (nextRandom(seed) % 10));
    }
    str[ndigits] = '\0';

    initializeBigInt(b, str);
    free(str);
}

static void printBigIntShort(const BigInt *b, int maxDigits) {
    int top = effectiveSize(b) - 1;
    int total = top + 1;

    if (total <= 2 * maxDigits) {
        for (int i = top; i >= 0; i--) printf("%d", b->digits[i]);
    } else {
        for (int i = top; i > top - maxDigits; i--) printf("%d", b->digits[i]);
        printf("...[%d digits total]...", total);
        for (int i = maxDigits - 1; i >= 0; i--) printf("%d", b->digits[i]);
    }
    printf("\n");
}

static int bigIntsEqual(const BigInt *a, const BigInt *b) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    if (na != nb) return 0;

    for (int i = 0; i < na; i++) {
        if (a->digits[i] != b->digits[i]) return 0;
    }
    return 1;
}

/*
 * CPU reference. This is grade-school multiplication written with a wide
 * temporary array so carry propagation is clean and easy to verify.
 */
static void multiplyBigIntsSerial(const BigInt *a, const BigInt *b, BigInt *result) {
    int rawSize = a->size + b->size;
    unsigned long long *raw =
        (unsigned long long *)calloc((size_t)rawSize, sizeof(unsigned long long));
    if (!raw) {
        fprintf(stderr, "calloc failed\n");
        exit(1);
    }

    for (int i = 0; i < a->size; i++) {
        for (int j = 0; j < b->size; j++) {
            raw[i + j] += (unsigned long long)a->digits[i] * (unsigned long long)b->digits[j];
        }
    }

    result->digits = (int *)calloc((size_t)rawSize + 32, sizeof(int));
    if (!result->digits) {
        fprintf(stderr, "calloc failed\n");
        exit(1);
    }

    unsigned long long carry = 0;
    int out = 0;
    for (int k = 0; k < rawSize; k++) {
        unsigned long long value = raw[k] + carry;
        result->digits[out++] = (int)(value % 10u);
        carry = value / 10u;
    }
    while (carry > 0) {
        result->digits[out++] = (int)(carry % 10u);
        carry /= 10u;
    }

    result->size = out;
    trimBigInt(result);
    free(raw);
}

/*
 * One CUDA thread computes one output diagonal:
 *
 * raw[k] = sum(a[i] * b[j]) for every i + j == k
 *
 * This creates the same convolution as grade-school multiplication. Carry
 * propagation is done afterward on the CPU because it is cheap compared with
 * the O(n^2) multiply work and keeps the CUDA code simple for coursework use.
 */
__global__ static void digitConvolutionKernel(const int *a,
                                              int na,
                                              const int *b,
                                              int nb,
                                              unsigned long long *raw,
                                              int rawSize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= rawSize) return;

    int start = k - (nb - 1);
    if (start < 0) start = 0;

    int end = k;
    if (end > na - 1) end = na - 1;

    unsigned long long sum = 0;
    for (int i = start; i <= end; i++) {
        int j = k - i;
        if (j >= 0 && j < nb) {
            sum += (unsigned long long)a[i] * (unsigned long long)b[j];
        }
    }

    raw[k] = sum;
}

static void multiplyBigIntsCUDA(const BigInt *a, const BigInt *b, BigInt *result) {
    int *d_a = NULL;
    int *d_b = NULL;
    unsigned long long *d_raw = NULL;

    int rawSize = a->size + b->size;
    size_t aBytes = (size_t)a->size * sizeof(int);
    size_t bBytes = (size_t)b->size * sizeof(int);
    size_t rawBytes = (size_t)rawSize * sizeof(unsigned long long);

    checkCuda(cudaMalloc((void **)&d_a, aBytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc((void **)&d_b, bBytes), "cudaMalloc d_b");
    checkCuda(cudaMalloc((void **)&d_raw, rawBytes), "cudaMalloc d_raw");

    checkCuda(cudaMemcpy(d_a, a->digits, aBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy a host->device");
    checkCuda(cudaMemcpy(d_b, b->digits, bBytes, cudaMemcpyHostToDevice),
              "cudaMemcpy b host->device");

    int threadsPerBlock = 256;
    int blocks = (rawSize + threadsPerBlock - 1) / threadsPerBlock;
    digitConvolutionKernel<<<blocks, threadsPerBlock>>>(d_a, a->size, d_b, b->size,
                                                        d_raw, rawSize);
    checkCuda(cudaGetLastError(), "digitConvolutionKernel launch");
    checkCuda(cudaDeviceSynchronize(), "digitConvolutionKernel synchronize");

    unsigned long long *raw =
        (unsigned long long *)malloc((size_t)rawSize * sizeof(unsigned long long));
    if (!raw) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }

    checkCuda(cudaMemcpy(raw, d_raw, rawBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy raw device->host");

    result->digits = (int *)calloc((size_t)rawSize + 32, sizeof(int));
    if (!result->digits) {
        fprintf(stderr, "calloc failed\n");
        exit(1);
    }

    unsigned long long carry = 0;
    int out = 0;
    for (int k = 0; k < rawSize; k++) {
        unsigned long long value = raw[k] + carry;
        result->digits[out++] = (int)(value % 10u);
        carry = value / 10u;
    }
    while (carry > 0) {
        result->digits[out++] = (int)(carry % 10u);
        carry /= 10u;
    }

    result->size = out;
    trimBigInt(result);

    free(raw);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_raw);
}

int main(int argc, char **argv) {
    int ndigits = 10000;
    int reps = 5;

    if (argc >= 2) ndigits = atoi(argv[1]);
    if (argc >= 3) reps = atoi(argv[2]);

    if (ndigits <= 0 || reps <= 0) {
        fprintf(stderr, "Usage: %s <digits> <reps>\n", argv[0]);
        return 1;
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA-capable device found\n");
        return 1;
    }

    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    printf("============================================================\n");
    printf(" CUDA Big-Integer Multiplication Benchmark\n");
    printf("============================================================\n");
    printf("GPU       : %s\n", prop.name);
    printf("Digits    : %d per operand\n", ndigits);
    printf("Reps      : %d\n", reps);
    printf("Algorithm : GPU grade-school convolution + CPU carry\n");
    printf("============================================================\n\n");

    BigInt sa = {NULL, 0}, sb = {NULL, 0};
    BigInt sref = {NULL, 0}, scuda = {NULL, 0};

    initializeBigInt(&sa, "123456789");
    initializeBigInt(&sb, "987654321");
    multiplyBigIntsSerial(&sa, &sb, &sref);
    multiplyBigIntsCUDA(&sa, &sb, &scuda);

    printf("Sample: 123456789 x 987654321\n");
    printf("CUDA result : ");
    printBigIntShort(&scuda, 20);
    printf("Match       : %s\n\n", bigIntsEqual(&sref, &scuda) ? "YES" : "NO");

    freeBigInt(&sa);
    freeBigInt(&sb);
    freeBigInt(&sref);
    freeBigInt(&scuda);

    unsigned int seed = 12345u;
    BigInt a = {NULL, 0}, b = {NULL, 0};
    randomBigInt(&a, ndigits, &seed);
    randomBigInt(&b, ndigits, &seed);

    BigInt result = {NULL, 0};
    multiplyBigIntsCUDA(&a, &b, &result);
    freeBigInt(&result);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; r++) {
        if (result.digits) freeBigInt(&result);
        multiplyBigIntsCUDA(&a, &b, &result);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double cudaSeconds =
        std::chrono::duration<double>(t1 - t0).count() / (double)reps;

    printf("CUDA average time per multiplication: %.6f s\n", cudaSeconds);
    printf("Result digits                      : %d\n", effectiveSize(&result));

    if (ndigits <= 5000) {
        BigInt ref = {NULL, 0};

        auto s0 = std::chrono::high_resolution_clock::now();
        multiplyBigIntsSerial(&a, &b, &ref);
        auto s1 = std::chrono::high_resolution_clock::now();

        double serialSeconds = std::chrono::duration<double>(s1 - s0).count();
        printf("Serial reference time              : %.6f s\n", serialSeconds);
        printf("CUDA == Serial                     : %s\n",
               bigIntsEqual(&result, &ref) ? "YES" : "NO");
        printf("Speedup vs serial reference        : %.2fx\n",
               serialSeconds / cudaSeconds);

        freeBigInt(&ref);
    } else {
        printf("Serial reference skipped           : digit count > 5000\n");
    }

    freeBigInt(&a);
    freeBigInt(&b);
    freeBigInt(&result);

    return 0;
}
