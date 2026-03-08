#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

typedef struct {
    int *digits;
    int size;
} BigInt;

#define KARATSUBA_THRESHOLD   32
#define OMP_TASK_THRESHOLD   256
#define OMP_MAX_DEPTH          4
#define MPI_THRESHOLD        512

/* ------------------------------------------------------------------------- */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------- */
static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static void initZero(BigInt *b) {
    b->size = 1;
    b->digits = (int *)calloc(1, sizeof(int));
    if (!b->digits) die("calloc failed");
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
    int *tmp = (int *)realloc(b->digits, s * sizeof(int));
    if (!tmp && s > 0) die("realloc failed");
    b->digits = tmp;
    b->size = s;
}

static void copyBigInt(const BigInt *src, BigInt *dst) {
    dst->size = src->size;
    dst->digits = (int *)malloc((size_t)dst->size * sizeof(int));
    if (!dst->digits) die("malloc failed");
    memcpy(dst->digits, src->digits, (size_t)dst->size * sizeof(int));
}

static void initializeBigInt(BigInt *b, const char *str) {
    int n = (int)strlen(str);
    if (n <= 0) die("empty string");

    b->size = n;
    b->digits = (int *)malloc((size_t)n * sizeof(int));
    if (!b->digits) die("malloc failed");

    for (int i = 0; i < n; i++) {
        if (str[n - 1 - i] < '0' || str[n - 1 - i] > '9')
            die("invalid digit");
        b->digits[i] = str[n - 1 - i] - '0';
    }
    trimBigInt(b);
}

static void randomBigInt(BigInt *b, int ndigits, unsigned int *seed) {
    char *str = (char *)malloc((size_t)ndigits + 1);
    if (!str) die("malloc failed");

    str[0] = '1' + (rand_r(seed) % 9);
    for (int i = 1; i < ndigits; i++)
        str[i] = '0' + (rand_r(seed) % 10);
    str[ndigits] = '\0';

    initializeBigInt(b, str);
    free(str);
}

static void printBigIntShort(const BigInt *b, int maxDigits) {
    int i = effectiveSize(b) - 1;
    int total = i + 1;

    if (total <= 2 * maxDigits) {
        for (; i >= 0; i--) printf("%d", b->digits[i]);
    } else {
        for (int j = i; j > i - maxDigits; j--) printf("%d", b->digits[j]);
        printf("...[%d digits total]...", total);
        for (int j = maxDigits - 1; j >= 0; j--) printf("%d", b->digits[j]);
    }
    printf("\n");
}

/* ------------------------------------------------------------------------- */
/* Arithmetic                                                                 */
/* ------------------------------------------------------------------------- */
static void addBigInts(const BigInt *a, const BigInt *b, BigInt *result) {
    int maxSize = (a->size > b->size) ? a->size : b->size;
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

/* assumes a >= b */
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

static void splitBigInt(const BigInt *b, BigInt *high, BigInt *low, int half) {
    low->size = (half > 0) ? half : 1;
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

    high->size = highSize;
    high->digits = (int *)malloc((size_t)highSize * sizeof(int));
    if (!high->digits) die("malloc failed");
    memcpy(high->digits, b->digits + half, (size_t)highSize * sizeof(int));
    trimBigInt(high);
}

static void padToSize(const BigInt *src, int n, BigInt *dst) {
    dst->size = n;
    dst->digits = (int *)calloc((size_t)n, sizeof(int));
    if (!dst->digits) die("calloc failed");
    int m = effectiveSize(src);
    memcpy(dst->digits, src->digits, (size_t)m * sizeof(int));
}

/* ------------------------------------------------------------------------- */
/* Grade-school multiplication                                                */
/* ------------------------------------------------------------------------- */
static void multiplyBigInts(const BigInt *a, const BigInt *b, BigInt *result) {
    int maxSize = a->size + b->size;
    result->digits = (int *)calloc((size_t)maxSize, sizeof(int));
    if (!result->digits) die("calloc failed");
    result->size = maxSize;

    for (int i = 0; i < a->size; i++) {
        int carry = 0;
        for (int j = 0; j < b->size; j++) {
            int idx = i + j;
            int prod = a->digits[i] * b->digits[j] + result->digits[idx] + carry;
            result->digits[idx] = prod % 10;
            carry = prod / 10;
        }

        int idx = i + b->size;
        while (carry > 0) {
            int sum = result->digits[idx] + carry;
            result->digits[idx] = sum % 10;
            carry = sum / 10;
            idx++;
        }
    }
    trimBigInt(result);
}

/* ------------------------------------------------------------------------- */
/* OpenMP Karatsuba                                                           */
/* ------------------------------------------------------------------------- */
static void karatsubaOMPRec(const BigInt *a, const BigInt *b, BigInt *result, int depth) {
    int na = effectiveSize(a);
    int nb = effectiveSize(b);
    int n = (na > nb) ? na : nb;

    if (n <= KARATSUBA_THRESHOLD) {
        multiplyBigInts(a, b, result);
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

    BigInt z2 = {NULL, 0}, z0 = {NULL, 0}, prod = {NULL, 0};
    int spawn = (n >= OMP_TASK_THRESHOLD && depth < OMP_MAX_DEPTH);

    if (spawn) {
        #pragma omp task shared(z2)
        karatsubaOMPRec(&a1, &b1, &z2, depth + 1);

        #pragma omp task shared(z0)
        karatsubaOMPRec(&a0, &b0, &z0, depth + 1);

        #pragma omp task shared(prod)
        {
            BigInt s1 = {NULL, 0}, s2 = {NULL, 0};
            addBigInts(&a1, &a0, &s1);
            addBigInts(&b1, &b0, &s2);
            karatsubaOMPRec(&s1, &s2, &prod, depth + 1);
            freeBigInt(&s1);
            freeBigInt(&s2);
        }

        #pragma omp taskwait
    } else {
        karatsubaOMPRec(&a1, &b1, &z2, depth + 1);
        karatsubaOMPRec(&a0, &b0, &z0, depth + 1);

        BigInt s1 = {NULL, 0}, s2 = {NULL, 0};
        addBigInts(&a1, &a0, &s1);
        addBigInts(&b1, &b0, &s2);
        karatsubaOMPRec(&s1, &s2, &prod, depth + 1);
        freeBigInt(&s1);
        freeBigInt(&s2);
    }

    freeBigInt(&a1); freeBigInt(&a0);
    freeBigInt(&b1); freeBigInt(&b0);

    BigInt tmp = {NULL, 0}, z1 = {NULL, 0}, sum1 = {NULL, 0}, sum2 = {NULL, 0};
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

static void karatsubaOMP(const BigInt *a, const BigInt *b, BigInt *result) {
    #pragma omp parallel
    {
        #pragma omp single
        karatsubaOMPRec(a, b, result, 0);
    }
}

/* ------------------------------------------------------------------------- */
/* MPI BigInt transfer                                                        */
/* ------------------------------------------------------------------------- */
static void mpiSendBigInt(const BigInt *b, int dest, int tag, MPI_Comm comm) {
    int sz = effectiveSize(b);
    MPI_Send(&sz, 1, MPI_INT, dest, tag, comm);
    MPI_Send(b->digits, sz, MPI_INT, dest, tag + 1, comm);
}

static void mpiRecvBigInt(BigInt *b, int src, int tag, MPI_Comm comm) {
    int sz = 0;
    MPI_Recv(&sz, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    b->size = sz;
    b->digits = (int *)malloc((size_t)sz * sizeof(int));
    if (!b->digits) die("malloc failed");
    MPI_Recv(b->digits, sz, MPI_INT, src, tag + 1, comm, MPI_STATUS_IGNORE);
    trimBigInt(b);
}

static void mpiBroadcastBigInt(BigInt *b, int root, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int sz = 0;
    if (rank == root) sz = effectiveSize(b);
    MPI_Bcast(&sz, 1, MPI_INT, root, comm);

    if (rank != root) {
        b->size = sz;
        b->digits = (int *)malloc((size_t)sz * sizeof(int));
        if (!b->digits) die("malloc failed");
    }
    MPI_Bcast(b->digits, sz, MPI_INT, root, comm);
    b->size = sz;
    trimBigInt(b);
}

/* ------------------------------------------------------------------------- */
/* Hybrid MPI + OpenMP top-level Karatsuba                                    */
/* ------------------------------------------------------------------------- */
static void hybridKaratsubaMPIOMP(const BigInt *a, const BigInt *b, BigInt *result, MPI_Comm comm) {
    int rank, world;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world);

    int n = 0;
    if (rank == 0) {
        int na = effectiveSize(a);
        int nb = effectiveSize(b);
        n = (na > nb) ? na : nb;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, comm);

    if (n < MPI_THRESHOLD || world < 3) {
        if (rank == 0) karatsubaOMP(a, b, result);
        return;
    }

    BigInt A = {NULL, 0}, B = {NULL, 0};
    if (rank == 0) {
        copyBigInt(a, &A);
        copyBigInt(b, &B);
    }

    mpiBroadcastBigInt(&A, 0, comm);
    mpiBroadcastBigInt(&B, 0, comm);

    int half = n / 2;

    BigInt aPad, bPad;
    padToSize(&A, n, &aPad);
    padToSize(&B, n, &bPad);

    freeBigInt(&A);
    freeBigInt(&B);

    BigInt a1, a0, b1, b0;
    splitBigInt(&aPad, &a1, &a0, half);
    splitBigInt(&bPad, &b1, &b0, half);

    freeBigInt(&aPad);
    freeBigInt(&bPad);

    int role = rank % 3;
    BigInt local = {NULL, 0};

    if (role == 0) {
        karatsubaOMP(&a1, &b1, &local);
    } else if (role == 1) {
        karatsubaOMP(&a0, &b0, &local);
    } else {
        BigInt s1 = {NULL, 0}, s2 = {NULL, 0};
        addBigInts(&a1, &a0, &s1);
        addBigInts(&b1, &b0, &s2);
        karatsubaOMP(&s1, &s2, &local);
        freeBigInt(&s1);
        freeBigInt(&s2);
    }

    freeBigInt(&a1); freeBigInt(&a0);
    freeBigInt(&b1); freeBigInt(&b0);

    if (rank == 0) {
        BigInt z2 = local, z0 = {NULL, 0}, prod = {NULL, 0};

        int src_z0 = -1, src_prod = -1;
        for (int r = 1; r < world; r++) {
            if (src_z0 == -1 && r % 3 == 1) src_z0 = r;
            if (src_prod == -1 && r % 3 == 2) src_prod = r;
        }

        if (src_z0 == -1 || src_prod == -1)
            die("Need at least one process for z0 and prod roles");

        mpiRecvBigInt(&z0, src_z0, 100, comm);
        mpiRecvBigInt(&prod, src_prod, 200, comm);

        BigInt tmp = {NULL, 0}, z1 = {NULL, 0}, sum1 = {NULL, 0}, sum2 = {NULL, 0};

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
    } else {
        if (role == 1) {
            mpiSendBigInt(&local, 0, 100, comm);
        } else if (role == 2) {
            mpiSendBigInt(&local, 0, 200, comm);
        }
        freeBigInt(&local);
    }
}

/* ------------------------------------------------------------------------- */
/* Benchmark                                                                  */
/* ------------------------------------------------------------------------- */
typedef struct {
    double seconds;
    int result_digits;
} BenchResult;

static BenchResult runOneBenchmark(int ndigits, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    BigInt a = {NULL, 0}, b = {NULL, 0}, result = {NULL, 0};
    unsigned int seed = 12345u + (unsigned int)ndigits;

    if (rank == 0) {
        randomBigInt(&a, ndigits, &seed);
        randomBigInt(&b, ndigits, &seed);
    }

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    hybridKaratsubaMPIOMP(&a, &b, &result, comm);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();

    BenchResult br;
    br.seconds = t1 - t0;
    br.result_digits = (rank == 0) ? effectiveSize(&result) : 0;

    if (rank == 0) {
        freeBigInt(&a);
        freeBigInt(&b);
        freeBigInt(&result);
    }

    return br;
}

/* ------------------------------------------------------------------------- */
/* Main                                                                       */
/* ------------------------------------------------------------------------- */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (argc < 4) {
        if (rank == 0) {
            printf("Usage:\n");
            printf("  mpirun -np 3 ./hybrid_range <start> <end> <mode>\n\n");
            printf("Modes:\n");
            printf("  linear     -> start, start+1, ... , end\n");
            printf("  geometric  -> start, start*2, start*4, ... up to end\n\n");
            printf("Examples:\n");
            printf("  mpirun -np 3 ./hybrid_range 1 1000000 geometric\n");
            printf("  mpirun -np 3 ./hybrid_range 1 1000 linear\n");
        }
        MPI_Finalize();
        return 0;
    }

    int startDigits = atoi(argv[1]);
    int endDigits   = atoi(argv[2]);
    const char *mode = argv[3];

    if (startDigits <= 0 || endDigits <= 0 || startDigits > endDigits) {
        if (rank == 0) fprintf(stderr, "Invalid range\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("============================================================\n");
        printf(" Hybrid MPI + OpenMP Karatsuba Range Benchmark\n");
        printf("============================================================\n");
        printf("MPI ranks                : %d\n", world);
        printf("OpenMP threads per rank  : %d\n", omp_get_max_threads());
        printf("KARATSUBA_THRESHOLD      : %d\n", KARATSUBA_THRESHOLD);
        printf("OMP_TASK_THRESHOLD       : %d\n", OMP_TASK_THRESHOLD);
        printf("MPI_THRESHOLD            : %d\n", MPI_THRESHOLD);
        printf("Range                    : %d to %d digits\n", startDigits, endDigits);
        printf("Mode                     : %s\n", mode);
        printf("============================================================\n\n");

        printf("%-14s %-16s %-16s\n", "Digits", "Time(s)", "ResultDigits");
        printf("%-14s %-16s %-16s\n", "--------------", "----------------", "----------------");
    }

    if (strcmp(mode, "linear") == 0) {
        for (int n = startDigits; n <= endDigits; n++) {
            BenchResult br = runOneBenchmark(n, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("%-14d %-16.6f %-16d\n", n, br.seconds, br.result_digits);
                fflush(stdout);
            }
        }
    } else if (strcmp(mode, "geometric") == 0) {
        long long n = startDigits;
        while (n <= endDigits) {
            BenchResult br = runOneBenchmark((int)n, MPI_COMM_WORLD);
            if (rank == 0) {
                printf("%-14lld %-16.6f %-16d\n", n, br.seconds, br.result_digits);
                fflush(stdout);
            }

            if (n == endDigits) break;
            long long next = n * 2LL;
            if (next > endDigits) next = endDigits;
            if (next == n) break;
            n = next;
        }
    } else {
        if (rank == 0) fprintf(stderr, "Unknown mode. Use linear or geometric.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("\nDone.\n");
        printf("Note: 1..1000000 in linear mode can take an extremely long time.\n");
    }

    MPI_Finalize();
    return 0;
}