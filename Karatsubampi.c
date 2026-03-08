#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* ---------------------------------------------------------------------------
 * BigInt: LITTLE-ENDIAN (digits[0] = least-significant)
 * --------------------------------------------------------------------------- */
typedef struct { int *digits; int size; } BigInt;

static void freeBigInt(BigInt *b) {
    free(b->digits); b->digits = NULL; b->size = 0;
}
static int effectiveSize(const BigInt *b) {
    int s = b->size;
    while (s > 1 && b->digits[s-1] == 0) s--;
    return s;
}
static BigInt cloneBigInt(const BigInt *src) {
    BigInt dst;
    dst.size   = src->size;
    dst.digits = (int *)malloc(src->size * sizeof(int));
    memcpy(dst.digits, src->digits, src->size * sizeof(int));
    return dst;
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
static void printBigIntShort(const BigInt *b) {
    int top = effectiveSize(b) - 1;
    if (top + 1 <= 20) {
        for (int i = top; i >= 0; i--) printf("%d", b->digits[i]);
    } else {
        for (int i = top; i > top-10; i--) printf("%d", b->digits[i]);
        printf("...[%d digits]...", top+1);
        for (int i = 9; i >= 0; i--) printf("%d", b->digits[i]);
    }
    printf("\n");
}

/* ===========================================================================
 * Arithmetic primitives
 * =========================================================================== */
static void addBigInts(const BigInt *a, const BigInt *b, BigInt *r) {
    int mx = a->size > b->size ? a->size : b->size;
    r->digits = (int *)calloc(mx+1, sizeof(int));
    r->size   = mx+1;
    int carry = 0;
    for (int i = 0; i < mx || carry; i++) {
        int s = carry;
        if (i < a->size) s += a->digits[i];
        if (i < b->size) s += b->digits[i];
        r->digits[i] = s % 10; carry = s / 10;
    }
}
static void subtractBigInts(const BigInt *a, const BigInt *b, BigInt *r) {
    r->digits = (int *)calloc(a->size, sizeof(int));
    r->size   = a->size;
    int borrow = 0;
    for (int i = 0; i < a->size; i++) {
        int d = a->digits[i] - (i < b->size ? b->digits[i] : 0) - borrow;
        if (d < 0) { d += 10; borrow = 1; } else borrow = 0;
        r->digits[i] = d;
    }
}
static void shiftLeft(BigInt *b, int shift) {
    if (!shift) return;
    int ns = b->size + shift;
    int *nd = (int *)calloc(ns, sizeof(int));
    memcpy(nd + shift, b->digits, b->size * sizeof(int));
    free(b->digits); b->digits = nd; b->size = ns;
}
static void splitBigInt(const BigInt *b, BigInt *hi, BigInt *lo, int half) {
    lo->size   = half;
    lo->digits = (int *)malloc(half * sizeof(int));
    memcpy(lo->digits, b->digits, half * sizeof(int));
    hi->size = b->size - half;
    if (hi->size <= 0) {
        hi->size = 1;
        hi->digits = (int *)calloc(1, sizeof(int));
    } else {
        hi->digits = (int *)malloc(hi->size * sizeof(int));
        memcpy(hi->digits, b->digits + half, hi->size * sizeof(int));
    }
}

/* ===========================================================================
 * Serial Karatsuba (leaf computation within each rank)
 * =========================================================================== */
#define KARATSUBA_THRESHOLD 32

static void karatsubaSerial(const BigInt *a, const BigInt *b, BigInt *result) {
    int na = effectiveSize(a), nb = effectiveSize(b);
    int n  = na > nb ? na : nb;

    if (n <= KARATSUBA_THRESHOLD) {
        int ms = a->size + b->size;
        result->digits = (int *)calloc(ms, sizeof(int));
        result->size   = ms;
        for (int i = 0; i < a->size; i++)
            for (int j = 0; j < b->size; j++) {
                int p = a->digits[i]*b->digits[j] + result->digits[i+j];
                result->digits[i+j]   = p % 10;
                result->digits[i+j+1] += p / 10;
            }
        return;
    }

    int half = n / 2;
    BigInt aPad = {(int*)calloc(n,sizeof(int)),n};
    BigInt bPad = {(int*)calloc(n,sizeof(int)),n};
    memcpy(aPad.digits, a->digits, na*sizeof(int));
    memcpy(bPad.digits, b->digits, nb*sizeof(int));

    BigInt a1,a0,b1,b0;
    splitBigInt(&aPad,&a1,&a0,half); freeBigInt(&aPad);
    splitBigInt(&bPad,&b1,&b0,half); freeBigInt(&bPad);

    BigInt z2,z0;
    karatsubaSerial(&a1,&b1,&z2);
    karatsubaSerial(&a0,&b0,&z0);

    BigInt s1,s2,prod,tmp,z1;
    addBigInts(&a1,&a0,&s1); addBigInts(&b1,&b0,&s2);
    karatsubaSerial(&s1,&s2,&prod);
    freeBigInt(&s1); freeBigInt(&s2);
    freeBigInt(&a0); freeBigInt(&a1);
    freeBigInt(&b0); freeBigInt(&b1);

    subtractBigInts(&prod,&z2,&tmp); freeBigInt(&prod);
    subtractBigInts(&tmp, &z0,&z1);  freeBigInt(&tmp);

    shiftLeft(&z2, 2*half); shiftLeft(&z1, half);

    BigInt sum1,sum2;
    addBigInts(&z2,&z1,&sum1);   freeBigInt(&z2); freeBigInt(&z1);
    addBigInts(&sum1,&z0,&sum2); freeBigInt(&sum1); freeBigInt(&z0);
    *result = sum2;
}

/* ===========================================================================
 * MPI helpers
 * =========================================================================== */
static void mpiSendBigInt(const BigInt *b, int dst, int tag, MPI_Comm comm) {
    MPI_Send(&b->size,  1,       MPI_INT, dst, tag,   comm);
    MPI_Send(b->digits, b->size, MPI_INT, dst, tag+1, comm);
}
static void mpiRecvBigInt(BigInt *b, int src, int tag, MPI_Comm comm) {
    MPI_Recv(&b->size, 1, MPI_INT, src, tag, comm, MPI_STATUS_IGNORE);
    b->digits = (int *)malloc(b->size * sizeof(int));
    MPI_Recv(b->digits, b->size, MPI_INT, src, tag+1, comm, MPI_STATUS_IGNORE);
}
/* Broadcast from group leader (rank 0 in comm) to all members.
   Non-leaders enter with {NULL,0}; this function allocates and fills them. */
static void mpiBcastBigInt(BigInt *b, MPI_Comm comm) {
    int myrank; MPI_Comm_rank(comm, &myrank);
    MPI_Bcast(&b->size, 1, MPI_INT, 0, comm);
    if (myrank != 0) b->digits = (int *)malloc(b->size * sizeof(int));
    MPI_Bcast(b->digits, b->size, MPI_INT, 0, comm);
}

/* ===========================================================================
 * MPI_SERIAL_THRESHOLD
 * =========================================================================== */
#define MPI_SERIAL_THRESHOLD 500

/* ===========================================================================
 * karatsubaMPI  — collective: every rank in `comm` must call this.
 *
 * `a`, `b`   : valid only on rank 0 of `comm`; others pass {NULL,0}.
 * `result`   : written only on rank 0 of `comm` on return.
 *
 * The function NEVER frees or modifies the caller's `a` / `b`.
 * It works entirely on internal copies.
 * =========================================================================== */
static void karatsubaMPI(const BigInt *a, const BigInt *b,
                          BigInt *result, MPI_Comm comm) {
    int nproc, myrank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &myrank);

    /* Make local copies — caller's buffers are never touched */
    BigInt la = {NULL,0}, lb = {NULL,0};
    if (myrank == 0) { la = cloneBigInt(a); lb = cloneBigInt(b); }

    /* Broadcast operands to all ranks in this group */
    mpiBcastBigInt(&la, comm);
    mpiBcastBigInt(&lb, comm);

    int na = effectiveSize(&la);
    int nb = effectiveSize(&lb);
    int n  = na > nb ? na : nb;

    /* Leaf: single rank or small input → serial, no more communication */
    if (nproc < 3 || n <= MPI_SERIAL_THRESHOLD) {
        if (myrank == 0) karatsubaSerial(&la, &lb, result);
        freeBigInt(&la); freeBigInt(&lb);
        return;
    }

    /* Split */
    int half = n / 2;
    BigInt aPad = {(int*)calloc(n,sizeof(int)),n};
    BigInt bPad = {(int*)calloc(n,sizeof(int)),n};
    memcpy(aPad.digits, la.digits, na*sizeof(int));
    memcpy(bPad.digits, lb.digits, nb*sizeof(int));
    freeBigInt(&la); freeBigInt(&lb);

    BigInt a1,a0,b1,b0;
    splitBigInt(&aPad,&a1,&a0,half); freeBigInt(&aPad);
    splitBigInt(&bPad,&b1,&b0,half); freeBigInt(&bPad);

    BigInt a1a0,b1b0;
    addBigInts(&a1,&a0,&a1a0);
    addBigInts(&b1,&b0,&b1b0);

    /* Partition ranks into three groups */
    int base  = nproc / 3, extra = nproc % 3;
    int s0    = base + (extra > 0 ? 1 : 0);
    int s1    = base + (extra > 1 ? 1 : 0);
    int color = (myrank < s0) ? 0 : (myrank < s0+s1 ? 1 : 2);

    MPI_Comm sub_comm;
    MPI_Comm_split(comm, color, myrank, &sub_comm);
    int sub_rank; MPI_Comm_rank(sub_comm, &sub_rank);

    /* Global leader sends sub-problem operands to group 1 and 2 leaders */
    BigInt recv_a = {NULL,0}, recv_b = {NULL,0};

    if (myrank == 0) {
        mpiSendBigInt(&a0,   s0,      10, comm);
        mpiSendBigInt(&b0,   s0,      12, comm);
        mpiSendBigInt(&a1a0, s0+s1,   20, comm);
        mpiSendBigInt(&b1b0, s0+s1,   22, comm);
    }
    if (myrank == s0   && sub_rank == 0)
        { mpiRecvBigInt(&recv_a, 0, 10, comm); mpiRecvBigInt(&recv_b, 0, 12, comm); }
    if (myrank == s0+s1 && sub_rank == 0)
        { mpiRecvBigInt(&recv_a, 0, 20, comm); mpiRecvBigInt(&recv_b, 0, 22, comm); }

    /* Each group recurses; non-leaders pass empty operands (broadcast fills them) */
    BigInt empty_a = {NULL,0}, empty_b = {NULL,0};
    BigInt sub_result = {NULL,0};

    if (color == 0) {
        const BigInt *pa = (sub_rank==0) ? &a1    : &empty_a;
        const BigInt *pb = (sub_rank==0) ? &b1    : &empty_b;
        karatsubaMPI(pa, pb, &sub_result, sub_comm);
    } else if (color == 1) {
        const BigInt *pa = (sub_rank==0) ? &recv_a : &empty_a;
        const BigInt *pb = (sub_rank==0) ? &recv_b : &empty_b;
        karatsubaMPI(pa, pb, &sub_result, sub_comm);
    } else {
        const BigInt *pa = (sub_rank==0) ? &recv_a : &empty_a;
        const BigInt *pb = (sub_rank==0) ? &recv_b : &empty_b;
        karatsubaMPI(pa, pb, &sub_result, sub_comm);
    }

    MPI_Comm_free(&sub_comm);

    /* Group leaders send results to global leader */
    if (myrank == s0    && sub_rank == 0)
        { mpiSendBigInt(&sub_result, 0, 30, comm); freeBigInt(&sub_result); }
    if (myrank == s0+s1 && sub_rank == 0)
        { mpiSendBigInt(&sub_result, 0, 40, comm); freeBigInt(&sub_result); }

    /* Global leader assembles final result */
    if (myrank == 0) {
        BigInt z2 = sub_result;   /* group 0 result already here */
        BigInt z0={NULL,0}, z1prod={NULL,0};
        mpiRecvBigInt(&z0,    s0,    30, comm);
        mpiRecvBigInt(&z1prod,s0+s1, 40, comm);

        BigInt tmp, z1;
        subtractBigInts(&z1prod,&z2,&tmp); freeBigInt(&z1prod);
        subtractBigInts(&tmp,   &z0,&z1);  freeBigInt(&tmp);

        shiftLeft(&z2, 2*half); shiftLeft(&z1, half);

        BigInt sum1,sum2;
        addBigInts(&z2,&z1,&sum1);   freeBigInt(&z2); freeBigInt(&z1);
        addBigInts(&sum1,&z0,&sum2); freeBigInt(&sum1); freeBigInt(&z0);
        *result = sum2;
    }

    /* Free all locally held pieces */
    freeBigInt(&a1);    freeBigInt(&b1);
    freeBigInt(&a0);    freeBigInt(&b0);
    freeBigInt(&a1a0);  freeBigInt(&b1b0);
    freeBigInt(&recv_a);freeBigInt(&recv_b);
}

/* ===========================================================================
 * Correctness helper
 * =========================================================================== */
static int bigIntsEqual(const BigInt *a, const BigInt *b) {
    int na = effectiveSize(a), nb = effectiveSize(b);
    if (na != nb) return 0;
    for (int i = 0; i < na; i++)
        if (a->digits[i] != b->digits[i]) return 0;
    return 1;
}

/* ===========================================================================
 * main
 * =========================================================================== */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) {
        printf("=================================================================\n");
        printf("  Karatsuba Big-Integer Multiplication — MPI Version\n");
        printf("  MPI ranks        : %d\n", world_size);
        printf("  Serial threshold : %d digits  (switch to schoolbook)\n",
               KARATSUBA_THRESHOLD);
        printf("  MPI threshold    : %d digits  (stop spawning sub-comms)\n",
               MPI_SERIAL_THRESHOLD);
        printf("=================================================================\n\n");
    }

    /* --- Sample ------------------------------------------------------------ */
    if (world_rank == 0) printf("--- Sample: 123456789 x 987654321 ---\n");
    {
        BigInt sa={NULL,0}, sb={NULL,0};
        if (world_rank==0) { initializeBigInt(&sa,"123456789");
                             initializeBigInt(&sb,"987654321"); }
        BigInt smpi={NULL,0};
        karatsubaMPI(&sa,&sb,&smpi,MPI_COMM_WORLD);
        if (world_rank==0) {
            printf("  MPI result  : "); printBigIntShort(&smpi);
            BigInt sser={NULL,0};
            karatsubaSerial(&sa,&sb,&sser);
            printf("  Serial ref  : "); printBigIntShort(&sser);
            printf("  Match       : %s\n\n",
                   bigIntsEqual(&smpi,&sser)?"YES ✓":"NO ✗");
            freeBigInt(&sser); freeBigInt(&smpi);
            freeBigInt(&sa);   freeBigInt(&sb);
        }
    }

    /* --- Correctness ------------------------------------------------------- */
    if (world_rank==0)
        printf("--- Correctness (10 random pairs, varying sizes) ---\n");
    int check_sizes[] = {7,9,50,100,500,1000,5000,10000,50000,100000};
    int allOk = 1;
    for (int t = 0; t < 10; t++) {
        BigInt ca={NULL,0}, cb={NULL,0};
        unsigned int seed = 42 + t;
        if (world_rank==0) { randomBigInt(&ca,check_sizes[t],&seed);
                             randomBigInt(&cb,check_sizes[t],&seed); }
        BigInt cmpi={NULL,0};
        karatsubaMPI(&ca,&cb,&cmpi,MPI_COMM_WORLD);
        if (world_rank==0) {
            BigInt cser={NULL,0};
            karatsubaSerial(&ca,&cb,&cser);
            int ok = bigIntsEqual(&cmpi,&cser);
            if (!ok) allOk=0;
            printf("  %6d digits: Serial==MPI: %s\n",
                   check_sizes[t], ok?"PASS ✓":"FAIL ✗");
            freeBigInt(&cser); freeBigInt(&cmpi);
            freeBigInt(&ca);   freeBigInt(&cb);
        }
    }
    if (world_rank==0)
        printf("  Overall: %s\n\n", allOk?"ALL PASSED ✓":"SOME FAILED ✗");

    /* --- Benchmark --------------------------------------------------------- */
    if (world_rank==0) {
        printf("--- Benchmark (wall-clock time per single multiplication) ---\n\n");
        printf("  %-10s  %-6s  %-14s  %-14s  %-10s\n",
               "Digits","Reps","Serial (s)","MPI (s)","Speedup");
        printf("  %-10s  %-6s  %-14s  %-14s  %-10s\n",
               "----------","------","--------------","--------------","----------");
    }

    struct { int digits; int reps; } cases[] = {
        {       9, 10000 },
        {      50,  5000 },
        {     100,  2000 },
        {     500,   500 },
        {    1000,   200 },
        {    5000,    20 },
        {   10000,    10 },
        {   50000,     5 },
        {  100000,     3 },
        {  500000,     1 },
        { 1000000,     1 },
    };

    for (int i = 0; i < (int)(sizeof cases / sizeof cases[0]); i++) {
        int nd   = cases[i].digits;
        int reps = cases[i].reps;

        /* Generate operands — never modified by karatsubaMPI */
        BigInt ba={NULL,0}, bb={NULL,0};
        unsigned int s = 77777 + nd;
        if (world_rank==0) { randomBigInt(&ba,nd,&s); randomBigInt(&bb,nd,&s); }

        /* Serial timing: rank 0 only, BEFORE MPI benchmark loop */
        double ser_time = 0.0;
        if (world_rank==0) {
            BigInt bres={NULL,0};
            karatsubaSerial(&ba,&bb,&bres); freeBigInt(&bres); /* warm-up */
            double t0 = MPI_Wtime();
            for (int r=0; r<reps; r++) {
                if (bres.digits) freeBigInt(&bres);
                karatsubaSerial(&ba,&bb,&bres);
            }
            ser_time = (MPI_Wtime()-t0) / reps;
            if (bres.digits) freeBigInt(&bres);
        }

        /* MPI warm-up */
        { BigInt bres={NULL,0};
          karatsubaMPI(&ba,&bb,&bres,MPI_COMM_WORLD);
          if (world_rank==0 && bres.digits) freeBigInt(&bres); }

        /* MPI timed loop */
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int r=0; r<reps; r++) {
            BigInt bres={NULL,0};
            karatsubaMPI(&ba,&bb,&bres,MPI_COMM_WORLD);
            if (world_rank==0 && bres.digits) freeBigInt(&bres);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double mpi_time = (MPI_Wtime()-t0) / reps;

        if (world_rank==0) {
            freeBigInt(&ba); freeBigInt(&bb);
            double speedup = ser_time / mpi_time;
            printf("  %-10d  %-6d  %-14.6f  %-14.6f  %.2fx\n",
                   nd, reps, ser_time, mpi_time, speedup);
            fflush(stdout);
        }
    }

    if (world_rank==0) {
        printf("\n  Speedup > 1.0x  -> MPI is faster than serial.\n");
        printf("  Speedup < 1.0x  -> communication overhead dominates (small inputs).\n");
        printf("  (More ranks = higher speedup at large digit counts)\n");
    }

    MPI_Finalize();
    return 0;
}