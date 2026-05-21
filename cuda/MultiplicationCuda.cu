#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// BigInt structure: digits stored LITTLE-ENDIAN (digits[0] = least significant digit)
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
    while (s > 0 && b->digits[s - 1] == 0) s--;
    return s == 0 ? 1 : s;
}

void initializeBigInt(BigInt *b, const char *str) {
    b->size   = (int)strlen(str);
    b->digits = (int *)malloc(b->size * sizeof(int));
    if (!b->digits) {
        fprintf(stderr, "Fatal: malloc failed in initializeBigInt\n");
        exit(1);
    }
    for (int i = 0; i < b->size; i++)
        b->digits[i] = str[b->size - 1 - i] - '0';
}

void randomBigInt(BigInt *b, int ndigits, unsigned int *seed) {
    char *str = (char *)malloc(ndigits + 1);
    if (!str) {
        fprintf(stderr, "Fatal: malloc failed in randomBigInt\n");
        exit(1);
    }
    str[0] = '1' + (rand() % 9); // Use standard rand() for simplicity in CUDA host code
    for (int i = 1; i < ndigits; i++)
        str[i] = '0' + (rand() % 10);
    str[ndigits] = '\0';
    initializeBigInt(b, str);
    free(str);
}

// Grade-school O(n²) serial implementation for correctness check
void multiplyBigIntsSerial(const BigInt *a, const BigInt *b, BigInt *result) {
    int maxSize    = a->size + b->size;
    result->digits = (int *)calloc(maxSize, sizeof(int));
    if (!result->digits) {
        fprintf(stderr, "Fatal: calloc failed in multiplyBigIntsSerial\n");
        exit(1);
    }
    result->size   = maxSize;
    for (int i = 0; i < a->size; i++) {
        for (int j = 0; j < b->size; j++) {
            int product = a->digits[i] * b->digits[j] + result->digits[i + j];
            result->digits[i + j]     = product % 10;
            result->digits[i + j + 1] += product / 10;
        }
    }
}

/* ---------------------------------------------------------------------------
 * CUDA GPU Kernels and Functions
 * --------------------------------------------------------------------------- */

// CUDA Kernel: Computes the diagonal sums of the product.
// C[k] = sum_{i+j=k} A[i] * B[j]
__global__ void gpuMultiplyKernel(const int *A, int na, const int *B, int nb, unsigned long long *C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int nc = na + nb - 1;
    if (k >= nc) return;

    unsigned long long sum = 0;
    int start = (k - nb + 1 > 0) ? (k - nb + 1) : 0;
    int end = (k < na - 1) ? k : (na - 1);
    
    for (int i = start; i <= end; i++) {
        sum += (unsigned long long)A[i] * B[k - i];
    }
    C[k] = sum;
}

// Host wrapper for CUDA multiplication
void cudaMultiply(const BigInt *a, const BigInt *b, BigInt *result) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    int nc = na + nb - 1; // Size of product before final carry propagation
    int outSize = na + nb;

    // Allocate host memory for raw GPU output
    unsigned long long *h_C_raw = (unsigned long long *)calloc(outSize, sizeof(unsigned long long));
    if (!h_C_raw) {
        fprintf(stderr, "Fatal: calloc failed for host raw buffer\n");
        exit(1);
    }

    // Allocate device memory
    int *d_A, *d_B;
    unsigned long long *d_C;
    cudaMalloc((void **)&d_A, na * sizeof(int));
    cudaMalloc((void **)&d_B, nb * sizeof(int));
    cudaMalloc((void **)&d_C, outSize * sizeof(unsigned long long));

    // Copy operands to device
    cudaMemcpy(d_A, a->digits, na * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b->digits, nb * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, outSize * sizeof(unsigned long long));

    // Launch CUDA kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (nc + threadsPerBlock - 1) / threadsPerBlock;
    gpuMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, na, d_B, nb, d_C);

    // Copy raw sums back to host
    cudaMemcpy(h_C_raw, d_C, outSize * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Carry propagation on the host (sequential O(N))
    result->digits = (int *)calloc(outSize, sizeof(int));
    if (!result->digits) {
        fprintf(stderr, "Fatal: calloc failed for CUDA result digits\n");
        exit(1);
    }
    result->size = outSize;

    unsigned long long carry = 0;
    for (int i = 0; i < outSize; i++) {
        unsigned long long sum = h_C_raw[i] + carry;
        result->digits[i] = (int)(sum % 10);
        carry = sum / 10;
    }

    free(h_C_raw);
}

/* ---------------------------------------------------------------------------
 * Helpers and Benchmarking
 * --------------------------------------------------------------------------- */

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

int main(void) {
    printf("=================================================================\n");
    printf("  Big-Integer Multiplication: Serial vs CUDA GPU\n");
    printf("=================================================================\n\n");

    // Print CUDA device properties
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA-capable GPU found!\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  Using GPU Device : %s\n", prop.name);
    printf("  Global memory    : %.2f GB\n", (double)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("=================================================================\n\n");

    /* Correctness check */
    printf("--- Correctness Check (10 random pairs, varying sizes) ---\n");
    srand(12345);
    int allOk = 1;
    int check_sizes[] = { 7, 9, 50, 100, 500, 1000, 2000, 5000, 10000 };
    for (int t = 0; t < 9; t++) {
        BigInt a, b, ser, gpu;
        randomBigInt(&a, check_sizes[t], NULL);
        randomBigInt(&b, check_sizes[t], NULL);
        
        multiplyBigIntsSerial(&a, &b, &ser);
        cudaMultiply(&a, &b, &gpu);
        
        int ok = bigIntsEqual(&ser, &gpu);
        if (!ok) allOk = 0;
        printf("  %6d digits: Serial==CUDA: %s\n",
               check_sizes[t], ok ? "PASS ✓" : "FAIL ✗");
               
        freeBigInt(&a); freeBigInt(&b);
        freeBigInt(&ser); freeBigInt(&gpu);
    }
    printf("  Overall: %s\n\n", allOk ? "ALL PASSED ✓" : "SOME FAILED ✗");

    /* Scaling benchmark */
    printf("--- Benchmark (wall-clock time per single multiplication) ---\n\n");
    printf("  %-10s  %-6s  %-14s  %-14s  %-10s\n",
           "Digits", "Reps", "Serial (s)", "CUDA (s)", "Speedup");
    printf("  %-10s  %-6s  %-14s  %-14s  %-10s\n",
           "----------", "------", "--------------",
           "--------------", "----------");

    struct { int digits; int reps; } cases[] = {
        {       9,  1000 },
        {      50,  1000 },
        {     100,   500 },
        {     500,   200 },
        {    1000,   100 },
        {    5000,    10 },
        {   10000,     5 },
        {   20000,     2 },
    };

    for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); i++) {
        int nd   = cases[i].digits;
        int reps = cases[i].reps;

        printf("  %-10d  %-6d  ", nd, reps);
        fflush(stdout);

        BigInt a, b;
        randomBigInt(&a, nd, NULL);
        randomBigInt(&b, nd, NULL);

        BigInt res = {NULL, 0};

        /* Serial */
        clock_t t0 = clock();
        for (int r = 0; r < reps; r++) {
            if (res.digits) freeBigInt(&res);
            multiplyBigIntsSerial(&a, &b, &res);
        }
        double ser_time = (double)(clock() - t0) / CLOCKS_PER_SEC / reps;
        if (res.digits) freeBigInt(&res);

        /* CUDA */
        t0 = clock();
        for (int r = 0; r < reps; r++) {
            if (res.digits) freeBigInt(&res);
            cudaMultiply(&a, &b, &res);
        }
        double cuda_time = (double)(clock() - t0) / CLOCKS_PER_SEC / reps;
        if (res.digits) freeBigInt(&res);

        double speedup = ser_time / cuda_time;
        printf("%-14.6f  %-14.6f  %.2fx\n", ser_time, cuda_time, speedup);
        fflush(stdout);

        freeBigInt(&a);
        freeBigInt(&b);
    }

    return 0;
}
