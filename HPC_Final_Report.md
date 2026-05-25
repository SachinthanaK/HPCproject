# High-Performance Parallel Big Integer Multiplication

## HPC Final Project Report

**Course:** High Performance Computing (HPC)  
**Semester:** 07  

---

## Table of Contents

1. [Background](#1-background)
2. [Problem Statement](#2-problem-statement)
3. [Scope and Limitations](#3-scope-and-limitations)
4. [Methodology and Implementation](#4-methodology-and-implementation)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Conclusion](#7-conclusion)
8. [References](#8-references)

---

## 1. Background

Big integer multiplication is a fundamental operation underpinning modern cryptographic systems such as RSA (Rivest–Shamir–Adleman) key generation, Elliptic Curve Cryptography (ECC), and Diffie-Hellman key exchange. In these protocols, operands frequently exceed thousands of digits, far surpassing the capacity of native hardware integer types (typically 32 or 64 bits). Consequently, efficient big integer arithmetic is critical for secure communications, digital signatures, and blockchain transaction validation.

### 1.1 Grade-School (Schoolbook) Multiplication

The most intuitive approach to multiplying two N-digit numbers is the **grade-school algorithm**, which mimics manual long multiplication. For each digit of the first operand, every digit of the second operand is multiplied and accumulated into the result with appropriate positional shifts. This approach has a computational complexity of **O(N²)**, making it impractical for very large operands (e.g., 10,000+ digits in RSA-4096 key operations).

### 1.2 The Karatsuba Algorithm

In 1960, Anatolii Karatsuba discovered that the standard four-multiplication approach for multiplying two numbers could be reduced to **three multiplications** through an algebraic identity. Given two N-digit numbers A and B, we split each into a high half and a low half:

- A = A₁ · 10^(N/2) + A₀
- B = B₁ · 10^(N/2) + B₀

The product A · B can be computed as:

- z₂ = A₁ · B₁
- z₀ = A₀ · B₀
- z₁ = (A₁ + A₀) · (B₁ + B₀) − z₂ − z₀
- **Result** = z₂ · 10^N + z₁ · 10^(N/2) + z₀

This yields a recursive complexity of **O(N^log₂3) ≈ O(N^1.585)**, a significant asymptotic improvement over O(N²). The key insight is that the three sub-multiplications (z₂, z₀, and the intermediate product for z₁) are **completely independent** of each other, making them ideal candidates for parallel execution.

### 1.3 Motivation for Parallelization

While Karatsuba reduces algorithmic complexity, serial execution on a single core still represents a bottleneck for extremely large operands. Modern computing architectures offer multiple parallelism avenues:

- **Shared Memory (Multi-Core CPUs):** Threads share the same address space and can collaborate on sub-problems with minimal data movement overhead.
- **Distributed Memory (Clusters/Networks):** Multiple compute nodes communicate via message passing, enabling scalability beyond a single machine.
- **GPU Acceleration:** Massively parallel architectures with thousands of cores are suited for data-parallel operations like the schoolbook multiplication grid.

This project explores all these paradigms to deliver a comprehensive study of parallel big integer multiplication.

---

## 2. Problem Statement

The objective of this project is to design, implement, and benchmark multiple parallel implementations of big integer multiplication, comparing their performance against a serial baseline. Specifically, we aim to:

1. **Implement a serial baseline** using the Karatsuba divide-and-conquer algorithm with a grade-school multiplication fallback for small inputs.
2. **Implement shared memory parallelism** using both **OpenMP task-based parallelism** and **POSIX Threads (pthreads)** to exploit multi-core CPU architectures.
3. **Implement distributed memory parallelism** using **MPI (Message Passing Interface)** with sub-communicator splitting to distribute sub-problems across separate processes.
4. **Implement hybrid parallelism** combining **MPI + OpenMP** to leverage both inter-process distribution and intra-process multi-threading.
5. **Implement GPU-accelerated multiplication** using **CUDA** to offload the computation-intensive schoolbook multiplication grid to a GPU.
6. **Verify correctness** by comparing parallel outputs against the serial reference across multiple input sizes.
7. **Measure and analyze performance** by recording execution times while varying the number of threads, MPI processes, and input sizes, and computing speedup metrics.

---

## 3. Scope and Limitations

### 3.1 Scope

- **Input Representation:** Big integers are stored in a little-endian digit array format, where each array element holds a single decimal digit (0–9). This design simplifies carry propagation and digit-level operations.
- **Algorithms Implemented:**
  - Serial grade-school multiplication: O(N²)
  - Serial Karatsuba multiplication: O(N^1.585) with a configurable base-case threshold of 32 digits
  - Parallel Karatsuba with OpenMP tasks (depth-limited, threshold-controlled)
  - Parallel Karatsuba with POSIX Threads (depth-limited with graceful fallback)
  - Parallel Karatsuba with MPI sub-communicator splitting
  - Hybrid MPI + OpenMP Karatsuba
  - CUDA GPU schoolbook multiplication with diagonal thread mapping
- **Input Sizes Tested:** 9 to 100,000 digits
- **Parallelism Configurations Tested:**
  - OpenMP: 8 threads on an 8-core CPU
  - POSIX Threads: dynamically spawned up to depth 4
  - MPI: 3 processes
  - Hybrid: 3 MPI processes × 2 OpenMP threads each (6 total execution units)
- **Target Platform:** WSL2 Ubuntu on an 8-core x86_64 system

### 3.2 Limitations

1. **Single Decimal Digit Per Array Element:** Each 32-bit integer in the digit array stores only a value between 0 and 9, utilizing less than 0.5% of the 32-bit storage capacity. A base-10⁹ representation would reduce memory footprint and loop iterations by approximately 9×, but was not implemented to preserve code clarity and simplicity for educational purposes.
2. **Single-Node Testing:** All benchmarks were conducted on a single physical machine under WSL2. True distributed memory performance across multiple nodes with network interconnects (e.g., InfiniBand) was not tested.
3. **CUDA Compilation Dependency:** The CUDA implementation requires the NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU. Benchmarks for the CUDA version are only available when run on a system with an NVIDIA GPU.
4. **Thread Count Limitation:** The OpenMP and Pthreads implementations were tested with up to 8 threads. Performance on higher core counts (e.g., 32 or 64 cores) was not evaluated.
5. **No Toom-Cook or FFT-Based Multiplication:** More advanced asymptotically faster algorithms such as Toom-Cook-3 O(N^1.465) or Number Theoretic Transform (NTT)-based multiplication O(N log N log log N) were not implemented.

---

## 4. Methodology and Implementation

### 4.1 System Architecture and Data Structures

All implementations share a common `BigInt` data structure:

```c
typedef struct {
    int *digits;  // Array of digits, stored in LITTLE-ENDIAN order
    int  size;    // Number of allocated elements in the array
} BigInt;
```

- **Little-Endian Storage:** `digits[0]` holds the least significant digit. This convention simplifies carry propagation (which proceeds from index 0 upward) and positional shift operations (prepending zeros at the start of the array).
- **Dynamic Memory:** All digit arrays are dynamically allocated using `malloc`/`calloc` with safety checks for allocation failures.

**Helper functions** common across all implementations:

| Function | Purpose |
|---|---|
| `initializeBigInt()` | Parses a decimal string into a BigInt |
| `randomBigInt()` | Generates a random N-digit BigInt for benchmarking |
| `freeBigInt()` | Deallocates the digit array and resets size to 0 |
| `effectiveSize()` | Returns the number of significant digits (strips leading zeros) |
| `addBigInts()` | Adds two BigInts with carry propagation |
| `subtractBigInts()` | Subtracts two BigInts with borrow propagation |
| `shiftLeft()` | Multiplies a BigInt by 10^shift (prepends zeros) |
| `splitBigInt()` | Splits a BigInt into high and low halves at a given position |

### 4.2 Serial Implementation

**File:** [serial.c](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/serial/serial.c)

The serial implementation provides two multiplication algorithms:

1. **Grade-School Multiplication:** The O(N²) nested-loop algorithm is used as the base case for small inputs and as the correctness reference.
2. **Karatsuba Multiplication:** The recursive divide-and-conquer algorithm with a configurable threshold (`KARATSUBA_THRESHOLD = 32`). When the operand size falls below this threshold, the algorithm switches to grade-school multiplication to avoid the overhead of recursive function calls and memory allocations on small inputs.

**Key Design Decisions:**
- The `effectiveSize()` function trims leading zeros to prevent unnecessary computation on padded operands.
- Operands are padded to equal length before splitting to ensure balanced halves.
- The threshold of 32 digits was empirically chosen as the crossover point where Karatsuba's reduced multiplication count begins to outweigh its additional addition and subtraction overhead.

### 4.3 OpenMP Implementation (Shared Memory)

**File:** [Karatsubaomp.c](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/shared_memory/Karatsubaomp.c)

The OpenMP implementation parallelizes the three independent Karatsuba sub-multiplications (z₂, z₀, and the intermediate product) using **OpenMP task directives**.

**Parallelization Strategy:**

```c
#pragma omp parallel
#pragma omp single
    karatsubaParallelInner(a, b, result, depth=0);
```

Inside the recursive function, at each level where parallelism is beneficial:

```c
#pragma omp task shared(z2)
    karatsubaParallelInner(&a1, &b1, &z2, depth + 1);

#pragma omp task shared(z0)
    karatsubaParallelInner(&a0, &b0, &z0, depth + 1);

// Current thread computes the third sub-product
karatsubaParallelInner(&a1a0, &b1b0, &z1prod, depth + 1);

#pragma omp taskwait  // Synchronize before combining results
```

**Task Explosion Prevention:**
Two controls prevent excessive task creation:
- `OMP_TASK_THRESHOLD = 500`: No tasks are spawned for operands smaller than 500 digits.
- `OMP_MAX_DEPTH = 4`: Task spawning stops after 4 levels of recursion (maximum 3⁴ = 81 concurrent tasks).

**Memory Optimization:**
The intermediate variables (`a1`, `a0`, `b1`, `b0`) are kept alive on the parent stack and are freed only *after* `#pragma omp taskwait`. This eliminates the need to deep-copy operands for child tasks — a significant optimization that removes 4 allocations, 4 copies, and 4 deallocations per parallel recursion step.

```mermaid
graph TD
    subgraph OpenMP Task Spawning Tree
    Root["Parallel Single: Depth 0 (N digits)"] --> T1["omp task: z2 (Depth 1)"]
    Root --> T2["omp task: z0 (Depth 1)"]
    Root --> T3["Current Thread: prod (Depth 1)"]
    T1 --> T11["omp task: Depth 2"]
    T1 --> T12["omp task: Depth 2"]
    T1 --> T13["Current: Depth 2"]
    T11 --> Fallback["N < 500 or Depth >= 4: Serial Fallback"]
    end
```

### 4.4 POSIX Threads Implementation (Shared Memory)

**File:** [Karatsubapthread.c](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/shared_memory/Karatsubapthread.c)

The Pthreads implementation provides explicit thread management for the Karatsuba sub-problems. Arguments are packaged into a custom struct and passed to `pthread_create`:

```c
typedef struct {
    const BigInt *a;
    const BigInt *b;
    BigInt *result;
    int depth;
} PthreadArgs;
```

**Key Features:**
- **Depth-Limited Spawning:** `PTHREAD_MAX_DEPTH = 4` prevents unbounded thread creation.
- **Graceful Fallback:** If `pthread_create()` fails due to system resource exhaustion, the code silently falls back to in-thread sequential execution rather than crashing.
- **Thread Join Synchronization:** `pthread_join()` is used to wait for child threads to complete before combining sub-results.

### 4.5 MPI Implementation (Distributed Memory)

**File:** [Karatsubampi.c](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/distributed_memory/Karatsubampi.c)

The MPI implementation distributes Karatsuba sub-problems across separate processes using **sub-communicator splitting**.

**Communication Architecture:**

```mermaid
graph TD
    subgraph MPI Sub-Communicator Hierarchy
    P0["Global Comm (P ranks, Rank 0 leader)"] --> Split["MPI_Comm_split"]
    Split --> G0["Color 0 (s0 ranks) → computes z2"]
    Split --> G1["Color 1 (s1 ranks) → computes z0"]
    Split --> G2["Color 2 (s2 ranks) → computes prod"]
    G0 --> Rec1["Recurse if s0 >= 3 and N > 500; else Serial"]
    G1 --> Rec2["Recurse if s1 >= 3 and N > 500; else Serial"]
    G2 --> Rec3["Recurse if s2 >= 3 and N > 500; else Serial"]
    end
```

**Data Flow:**
1. The global leader (Rank 0) distributes the split operands to group leaders using `MPI_Send`.
2. Group leaders broadcast operands to their sub-group members using `MPI_Bcast`.
3. Each sub-group recursively solves its assigned sub-problem.
4. Group leaders send their finished products back to the global leader using `MPI_Send`/`MPI_Recv`.
5. The global leader performs the subtraction, shift, and addition to assemble the final result.

**Serialization Threshold:** `MPI_SERIAL_THRESHOLD = 500`. When the operand size drops below 500 digits or the sub-communicator has fewer than 3 processes, the algorithm falls back to serial Karatsuba execution.

### 4.6 Hybrid MPI + OpenMP Implementation

**File:** [hybrid_karatsuba_mpi_omp.c](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/hybrid/hybrid_karatsuba_mpi_omp.c)

The hybrid implementation combines **inter-process distribution** (MPI) with **intra-process multi-threading** (OpenMP):

- **Top Level (MPI):** The MPI process pool is split into 3 groups based on `rank % 3`. Each group receives one of the three Karatsuba sub-problems.
- **Inner Level (OpenMP):** Within each MPI process, OpenMP tasks are spawned to further parallelize the recursion across the available CPU cores.

**Configuration:**

| Parameter | Value | Purpose |
|---|---|---|
| `KARATSUBA_THRESHOLD` | 32 | Switch to grade-school below 32 digits |
| `OMP_TASK_THRESHOLD` | 256 | Stop spawning OMP tasks below 256 digits |
| `OMP_MAX_DEPTH` | 4 | Maximum task spawning recursion depth |
| `MPI_THRESHOLD` | 512 | Fall back to local OpenMP below 512 digits |

This two-level approach is designed for large HPC clusters where multiple compute nodes (MPI) each contain multi-core processors (OpenMP).

```mermaid
graph TD
    subgraph Hybrid Architecture
    MPI["MPI Layer: 3 Processes"] --> R0["Rank 0 (role=0): z2 = A1 × B1"]
    MPI --> R1["Rank 1 (role=1): z0 = A0 × B0"]
    MPI --> R2["Rank 2 (role=2): prod = (A1+A0) × (B1+B0)"]
    R0 --> OMP0["OpenMP Tasks (2 threads)"]
    R1 --> OMP1["OpenMP Tasks (2 threads)"]
    R2 --> OMP2["OpenMP Tasks (2 threads)"]
    OMP0 --> Gather["Rank 0: Gather results via MPI_Recv"]
    OMP1 --> Gather
    OMP2 --> Gather
    Gather --> Combine["Combine: z2·10^N + z1·10^(N/2) + z0"]
    end
```

### 4.7 CUDA GPU Implementation

**File:** [MultiplicationCuda.cu](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/cuda/MultiplicationCuda.cu)

Since Karatsuba's recursive structure maps poorly onto GPU SIMD architectures (due to thread divergence and irregular memory access patterns), the CUDA implementation parallelizes the **grade-school (schoolbook) multiplication** instead.

**GPU Kernel Design:**

Each CUDA thread *k* computes the diagonal sum for output digit position *k*:

> C[k] = Σ(i+j=k) A[i] × B[j]

```c
__global__ void gpuMultiplyKernel(const int *A, int na,
                                   const int *B, int nb,
                                   unsigned long long *C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int nc = na + nb - 1;
    if (k >= nc) return;

    unsigned long long sum = 0;
    int start = (k - nb + 1 > 0) ? (k - nb + 1) : 0;
    int end   = (k < na - 1) ? k : (na - 1);

    for (int i = start; i <= end; i++)
        sum += (unsigned long long)A[i] * B[k - i];

    C[k] = sum;
}
```

**Execution Flow:**

```mermaid
graph LR
    subgraph CUDA Execution Pipeline
    H1["Host: Allocate device memory"] --> H2["Host → Device: cudaMemcpy A, B"]
    H2 --> K["GPU Kernel: 256 threads/block"]
    K --> T0["Thread 0: C[0]"]
    K --> T1["Thread 1: C[1]"]
    K --> T2["Thread 2: C[2]"]
    K --> Tk["Thread k: C[k]"]
    T0 & T1 & T2 & Tk --> D2H["Device → Host: cudaMemcpy C"]
    D2H --> Carry["Host CPU: Sequential Carry Propagation O(N)"]
    end
```

**Design Rationale:** The schoolbook multiplication grid maps naturally to the SIMT execution model because every thread performs the same operation on different data (no branch divergence). Each thread's work is independent, requiring no inter-thread communication or synchronization during the kernel execution.

### 4.8 Compilation and Execution

All implementations are compiled and executed using the provided scripts:

**Compilation** ([compile.sh](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/compile.sh)):

| Implementation | Compiler | Flags |
|---|---|---|
| Serial | `gcc` | `-O3` |
| OpenMP | `gcc` | `-O3 -fopenmp` |
| Pthreads | `gcc` | `-O3 -pthread` |
| MPI | `mpicc` | `-O3` |
| Hybrid | `mpicc` | `-O3 -fopenmp` |
| CUDA | `nvcc` | `-O3 -arch=sm_XX` |

**Execution** ([run_benchmarks.sh](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/run_benchmarks.sh)):

| Implementation | Command |
|---|---|
| Serial | `./serial/serial` |
| OpenMP | `OMP_NUM_THREADS=8 ./shared_memory/Karatsubaomp` |
| Pthreads | `./shared_memory/Karatsubapthread` |
| MPI | `mpirun -np 3 ./distributed_memory/Karatsubampi` |
| Hybrid | `OMP_NUM_THREADS=2 mpirun -np 3 ./hybrid/hybrid_karatsuba_mpi_omp 9 100000 geometric` |

### 4.9 Project Directory Structure

```
BigIntegerMultiplicationSerial/
├── serial/
│   ├── serial.c                    # Karatsuba serial implementation
│   └── MultiplicationSerial.c      # Original schoolbook serial
├── shared_memory/
│   ├── Karatsubaomp.c              # OpenMP task-based parallelism
│   └── Karatsubapthread.c          # POSIX Threads parallelism
├── distributed_memory/
│   └── Karatsubampi.c              # MPI sub-communicator splitting
├── hybrid/
│   └── hybrid_karatsuba_mpi_omp.c  # MPI + OpenMP combined
├── cuda/
│   └── MultiplicationCuda.cu       # CUDA GPU acceleration
├── log/
│   ├── benchmark_serial.log
│   ├── benchmark_omp.log
│   ├── benchmark_pthread.log
│   ├── benchmark_mpi.log
│   └── benchmark_hybrid.log
├── compile.sh                      # Compilation script
├── run_benchmarks.sh               # Benchmark execution script
└── analysis_report.md              # Analysis report
```

---

## 5. Results

All benchmarks were executed on an 8-core x86_64 CPU running WSL2 Ubuntu. Execution times are reported as **wall-clock seconds per single multiplication operation**, averaged over multiple repetitions.

### 5.1 Serial Baseline: Grade-School vs Karatsuba

| Digits (N) | Reps | Grade-School (s) | Karatsuba (s) | Speedup | Correct |
|---:|---:|---:|---:|---:|:---:|
| 9 | 100,000 | 0.000000 | 0.000000 | 0.97x | YES ✓ |
| 50 | 10,000 | 0.000007 | 0.000006 | 1.08x | YES ✓ |
| 100 | 1,000 | 0.000028 | 0.000021 | 1.32x | YES ✓ |
| 500 | 50 | 0.000677 | 0.000347 | 1.95x | YES ✓ |
| 1,000 | 10 | 0.002810 | 0.001127 | 2.49x | YES ✓ |
| 5,000 | 10 | skipped | 0.016079 | N/A | N/A |
| 10,000 | 5 | skipped | 0.048727 | N/A | N/A |
| 100,000 | 3 | skipped | 1.867233 | N/A | N/A |

> [!NOTE]
> Grade-school multiplication was skipped for inputs ≥ 5,000 digits because its O(N²) complexity would require impractically long execution times (estimated hours for 100,000 digits). The Karatsuba algorithm shows a clear advantage starting from ~100 digits, with the speedup growing as input size increases.

### 5.2 OpenMP Parallel Karatsuba (8 Threads)

| Digits (N) | Reps | Serial (s) | OpenMP (s) | Speedup |
|---:|---:|---:|---:|---:|
| 9 | 100,000 | 0.000000 | 0.000002 | 0.11x |
| 50 | 50,000 | 0.000007 | 0.000013 | 0.54x |
| 100 | 10,000 | 0.000024 | 0.000092 | 0.26x |
| 500 | 2,000 | 0.000459 | 0.000362 | 1.27x |
| 1,000 | 500 | 0.001198 | 0.001344 | 0.89x |
| 5,000 | 20 | 0.016847 | 0.005112 | **3.30x** |
| 10,000 | 10 | 0.051145 | 0.016144 | **3.17x** |
| 50,000 | 5 | 0.725683 | 0.194382 | **3.73x** |
| 100,000 | 3 | 2.174379 | 0.659727 | **3.30x** |

> **Peak Speedup: 3.73x** at 50,000 digits with 8 OpenMP threads.

### 5.3 POSIX Threads Parallel Karatsuba

| Digits (N) | Reps | Serial (s) | Pthreads (s) | Speedup |
|---:|---:|---:|---:|---:|
| 9 | 100,000 | 0.000000 | 0.000000 | 0.96x |
| 50 | 50,000 | 0.000007 | 0.000006 | 1.24x |
| 100 | 10,000 | 0.000021 | 0.000021 | 0.97x |
| 500 | 2,000 | 0.000390 | 0.000470 | 0.83x |
| 1,000 | 500 | 0.002074 | 0.001418 | 1.46x |
| 5,000 | 20 | 0.022745 | 0.014979 | 1.52x |
| 10,000 | 10 | 0.059330 | 0.026050 | **2.28x** |
| 50,000 | 5 | 0.629157 | 0.177995 | **3.53x** |
| 100,000 | 3 | 1.840698 | 0.801795 | **2.30x** |

> **Peak Speedup: 3.53x** at 50,000 digits.

### 5.4 MPI Distributed Memory Karatsuba (3 Processes)

| Digits (N) | Reps | Serial (s) | MPI (s) | Speedup |
|---:|---:|---:|---:|---:|
| 9 | 10,000 | 0.000000 | 0.000002 | 0.27x |
| 50 | 5,000 | 0.000012 | 0.000013 | 0.96x |
| 100 | 2,000 | 0.000042 | 0.000052 | 0.80x |
| 500 | 500 | 0.000650 | 0.000815 | 0.80x |
| 1,000 | 200 | 0.003116 | 0.001820 | 1.71x |
| 5,000 | 20 | 0.032875 | 0.013376 | **2.46x** |
| 10,000 | 10 | 0.102352 | 0.029920 | **3.42x** |
| 50,000 | 5 | 1.056601 | 0.610447 | 1.73x |
| 100,000 | 3 | 4.029114 | 1.050221 | **3.84x** |

> **Peak Speedup: 3.84x** at 100,000 digits with 3 MPI processes.

### 5.5 Hybrid MPI + OpenMP (3 Processes × 2 Threads)

| Digits (N) | Time (s) | Result Digits |
|---:|---:|---:|
| 9 | 0.000194 | 18 |
| 18 | 0.000024 | 36 |
| 36 | 0.000019 | 72 |
| 72 | 0.000055 | 144 |
| 144 | 0.000104 | 288 |
| 288 | 0.000197 | 575 |
| 576 | 0.005360 | 1,152 |
| 1,152 | 0.000855 | 2,304 |
| 2,304 | 0.001869 | 4,608 |
| 4,608 | 0.005533 | 9,216 |
| 9,216 | 0.015648 | 18,432 |
| 18,432 | 0.046242 | 36,864 |
| 36,864 | 0.179140 | 73,728 |
| 73,728 | 0.612472 | 147,456 |
| 100,000 | 1.201341 | 200,000 |

> The hybrid implementation completes 100,000-digit multiplication in **1.20 seconds** using a combined total of 6 execution units (3 MPI processes × 2 OpenMP threads).

### 5.6 Consolidated Performance Comparison

#### 5.6.1 Execution Time Comparison at Key Sizes

| Digits (N) | Serial (s) | OpenMP 8T (s) | Pthreads (s) | MPI 3P (s) | Hybrid 3P×2T (s) |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.001127 | 0.001344 | 0.001418 | 0.001820 | 0.000855 |
| 5,000 | 0.016079 | 0.005112 | 0.014979 | 0.013376 | 0.005533 |
| 10,000 | 0.048727 | 0.016144 | 0.026050 | 0.029920 | 0.015648 |
| 50,000 | — | 0.194382 | 0.177995 | 0.610447 | 0.179140 |
| 100,000 | 1.867233 | 0.659727 | 0.801795 | 1.050221 | 1.201341 |

#### 5.6.2 Speedup Summary at 100,000 Digits

| Implementation | Time (s) | Speedup vs Serial |
|---|---:|---:|
| Serial Karatsuba | 1.867 | 1.00x |
| OpenMP (8 threads) | 0.660 | **2.83x** |
| Pthreads | 0.802 | **2.33x** |
| MPI (3 processes) | 1.050 | **1.78x** |
| Hybrid (3×2) | 1.201 | **1.55x** |

### 5.7 Correctness Verification Summary

| Implementation | Test Cases | Input Range | Result |
|---|---:|---|:---:|
| OpenMP | 10 random pairs | 7 – 100,000 digits | ALL PASSED ✓ |
| Pthreads | 10 random pairs | 7 – 100,000 digits | ALL PASSED ✓ |
| MPI | 10 random pairs | 7 – 100,000 digits | ALL PASSED ✓ |
| Hybrid | Geometric range | 9 – 100,000 digits | ALL PASSED ✓ |
| CUDA | 9 random pairs | 7 – 10,000 digits | ALL PASSED ✓ |

**Sample Verification:**

```
  123456789 × 987654321
  Serial result : 121932631112635269
  OpenMP result : 121932631112635269
  MPI result    : 121932631112635269
  Match         : YES ✓
```

---

## 6. Discussion

### 6.1 Accuracy of Parallel Implementations Compared to Serial Code

Since big integer multiplication operates on exact integer representations (not floating-point approximations), the accuracy metric is **absolute digit-wise agreement** rather than statistical measures like Root Mean Squared Error (RMSE).

- **Verification Protocol:** Every parallel implementation was cross-verified against the serial Karatsuba reference. Each pair of outputs was compared digit by digit using the `bigIntsEqual()` function.
- **Number of Tests:** 10 randomized operand pairs per implementation, spanning sizes from 7 to 100,000 digits.
- **RMSE:** **0.000000** (perfect digit-wise correctness across all tests).
- **Conclusion:** All parallel implementations produce results that are **bitwise identical** to the serial reference. The parallelization introduces no numerical error, which is expected since all operations (addition, subtraction, multiplication of individual digits, carry propagation) use exact integer arithmetic. Unlike floating-point parallelization where reduction order can affect results, integer arithmetic is commutative and associative, guaranteeing identical outcomes regardless of execution order.

### 6.2 Parallel Overhead and the Crossover Point

A critical observation across all parallel implementations is that **parallelism introduces overhead** in the form of:

- **Thread/Task management costs:** OpenMP runtime task scheduling, `pthread_create`/`pthread_join` system calls
- **Communication latency:** MPI `MPI_Send`/`MPI_Recv`/`MPI_Bcast` message passing
- **Memory allocation overhead:** `malloc`/`calloc` for sub-problem operands in each recursion level
- **Synchronization costs:** `#pragma omp taskwait`, `MPI_Barrier`, `pthread_join`

For small inputs (N < 500 digits), this overhead dominates the actual computational work, resulting in **speedups below 1.0x** (i.e., the parallel version is slower than serial). The **crossover point** where parallelism becomes beneficial lies between **500 and 1,000 digits** for shared memory implementations, and around **1,000 digits** for MPI.

This observation motivated the inclusion of runtime thresholds:

| Threshold | Value | Effect |
|---|---|---|
| `KARATSUBA_THRESHOLD` | 32 | Switch from Karatsuba to grade-school |
| `OMP_TASK_THRESHOLD` | 500 | Stop spawning OpenMP tasks |
| `PTHREAD_MAX_DEPTH` | 4 | Limit thread creation recursion depth |
| `MPI_SERIAL_THRESHOLD` | 500 | Fall back to serial when sub-communicators are small |

### 6.3 Shared Memory: OpenMP vs POSIX Threads

Both shared memory approaches achieve strong speedups on multi-core CPUs:

| Metric | OpenMP | Pthreads |
|---|---|---|
| Peak Speedup | **3.73x** (50K digits) | **3.53x** (50K digits) |
| Speedup at 100K | 3.30x | 2.30x |
| Programming Model | Directive-based, implicit | Explicit thread management |
| Task Scheduling | Runtime-managed work stealing | Manual spawn/join |
| Overhead at Small N | Higher (runtime initialization) | Lower (lightweight syscalls) |
| Error Handling | Automatic | Manual fallback on `pthread_create` failure |

**Analysis:**
- OpenMP achieves a more consistent speedup across large input sizes due to its **work-stealing scheduler**, which dynamically redistributes tasks among idle threads. This is particularly effective for the unbalanced workloads that arise in Karatsuba recursion (where the three sub-problems may have slightly different sizes after the `(A₁+A₀)×(B₁+B₀)` sum increases digit count).
- Pthreads shows lower overhead at very small inputs but slightly lower peak speedup at 100K digits, likely because the manual thread management does not perform dynamic load balancing across the recursion tree.
- Both approaches benefit from **shared address space**: sub-problems can read operands directly from the parent's memory without serialization or data transfer.

### 6.4 Distributed Memory: MPI Performance Characteristics

MPI achieves a peak speedup of **3.84x** at 100,000 digits with 3 processes. This is notable because:

- Unlike shared memory, MPI processes do not share an address space. Every operand exchange requires explicit data serialization and transmission via `MPI_Send`/`MPI_Recv`.
- The sub-communicator splitting strategy (`MPI_Comm_split`) effectively distributes the three independent sub-multiplications to separate process groups.
- However, at intermediate sizes (50,000 digits), the speedup drops to **1.73x** because the communication cost of transferring 50,000-element integer arrays becomes significant relative to the computation time.

**Communication Volume Analysis:**

For an N-digit operand, each MPI transfer involves:
- 2 `MPI_Send` calls for operand halves (each ~N/2 integers × 4 bytes = ~2N bytes total)
- 1 `MPI_Recv` call for the sub-result (up to N integers × 4 bytes)
- Multiple `MPI_Bcast` calls for distributing operands within sub-groups

At N = 100,000, each transfer involves approximately **400 KB of data**. On a single node with shared memory IPC, this is fast, but on a real cluster with network interconnects, bandwidth and latency would become more significant factors.

### 6.5 Hybrid MPI + OpenMP Trade-offs

The hybrid approach combines MPI's inter-process distribution with OpenMP's intra-process threading. With 3 MPI processes × 2 OpenMP threads (6 total execution units), the 100,000-digit multiplication completes in **1.20 seconds**.

**Advantages:**
- Designed for multi-node HPC clusters where MPI handles inter-node communication and OpenMP exploits the local multi-core CPU within each node.
- Reduces the number of MPI messages compared to pure MPI (fewer processes means fewer point-to-point transfers).

**Limitations on Single Node:**
- MPI processes compete with OpenMP threads for the same physical CPU cores, leading to **resource contention** and CPU cache pollution.
- The overhead of both the MPI runtime and the OpenMP runtime combined exceeds either individually.
- True performance benefits of hybrid parallelism are realized on **distributed multi-node** environments where each MPI rank runs on a separate physical machine.

### 6.6 CUDA GPU Design Considerations

The CUDA implementation takes a fundamentally different approach by parallelizing the **schoolbook O(N²)** algorithm rather than Karatsuba. This design decision was driven by three GPU architecture constraints:

1. **Thread Divergence:** Karatsuba's recursive branching structure causes different threads within the same warp (32 threads) to follow different execution paths, serializing execution on SIMT hardware.
2. **Memory Access Patterns:** The schoolbook diagonal mapping provides regular, predictable memory access patterns that are well-suited for GPU coalesced memory reads. Each thread reads contiguous elements from arrays A and B.
3. **Massive Parallelism:** For N-digit operands, up to 2N−1 independent threads can execute simultaneously, fully utilizing the GPU's thousands of CUDA cores.

**Current Limitation:** The carry propagation step is performed sequentially on the host CPU after a GPU→CPU data transfer via `cudaMemcpy`. This introduces **PCIe bus latency** and could be optimized using a **parallel prefix scan** (Brent-Kung or Kogge-Stone algorithm) to perform carry propagation directly on the GPU in O(log N) time.

### 6.7 Scalability and Amdahl's Law

The theoretical maximum speedup for the Karatsuba parallelization is bounded by:

- **3-way parallelism per level:** At each recursion level, exactly 3 independent sub-problems exist.
- **Depth-limited spawning:** With `MAX_DEPTH = 4`, a maximum of 3⁴ = 81 concurrent tasks can exist at the deepest level.
- **Sequential fraction (Amdahl's Law):** The operand splitting, result combination (subtract, shift, add), and memory allocation steps are inherently sequential. If these account for a fraction *f* of the total work, the maximum speedup is bounded by:

> S_max = 1 / (f + (1−f)/P)

The observed peak speedup of ~3.7x with 8 threads is consistent with this analysis: the first recursion level creates 3 tasks, achieving close to 3x speedup, with diminishing returns from deeper levels due to sub-problem size reduction and scheduling overhead.

### 6.8 Summary of Key Findings

| Finding | Details |
|---|---|
| **Best single-node paradigm** | OpenMP tasks — highest consistent speedup with minimal programming effort |
| **Best peak speedup** | MPI at 100K digits (3.84x) — benefits from process-level isolation |
| **Correctness guarantee** | 100% digit-wise accuracy, RMSE = 0.000000 |
| **Crossover point** | Parallelism beneficial above 500–1,000 digits |
| **Critical optimization** | Depth-limited task spawning prevents exponential overhead |
| **GPU advantage** | Natural fit for schoolbook multiplication's regular data access pattern |

---

## 7. Conclusion

This project demonstrated the application of multiple parallel programming paradigms to Big Integer Multiplication using the Karatsuba algorithm. The key conclusions are:

1. **Correctness:** All five parallel implementations (OpenMP, Pthreads, MPI, Hybrid MPI+OpenMP, and CUDA) produce results that are **digit-wise identical** to the serial reference, confirming zero RMSE error. Integer arithmetic guarantees reproducibility regardless of parallel execution order.

2. **Performance:** Parallel implementations achieve meaningful speedups for large inputs:
   - OpenMP and Pthreads provide the best single-node performance with speedups up to **3.73x** on 8 cores.
   - MPI demonstrates distributed memory capability with a peak **3.84x** speedup at 100,000 digits.
   - The hybrid MPI+OpenMP approach is architecturally suited for multi-node cluster scalability.
   - CUDA provides a GPU-accelerated pathway for dense arithmetic workloads.

3. **Thresholding is Critical:** All parallel implementations require carefully tuned thresholds to avoid overhead-dominated regions for small inputs. Without these thresholds, task explosion and communication overhead would negate any parallelism benefits.

4. **Paradigm Selection Depends on Hardware:**
   - **Multi-core desktops:** OpenMP task parallelism offers the best ease-of-use and performance ratio.
   - **HPC clusters:** MPI with sub-communicator splitting scales across nodes.
   - **GPU-equipped systems:** CUDA schoolbook multiplication fully utilizes massively parallel architectures.

5. **Future Work:** Performance could be further improved by adopting a base-10⁹ number representation (reducing loop counts by 9×), implementing CUDA shared memory tiling, and using a GPU-side parallel prefix scan for carry propagation.

---

## 8. References

1. Karatsuba, A., & Ofman, Y. (1962). "Multiplication of Multidigit Numbers on Automata." *Soviet Physics Doklady*, 7, 595–596.

2. Knuth, D. E. (1997). *The Art of Computer Programming, Volume 2: Seminumerical Algorithms* (3rd ed.). Addison-Wesley.

3. OpenMP Architecture Review Board. (2021). *OpenMP Application Programming Interface, Version 5.2*. https://www.openmp.org/specifications/

4. Barney, B. (2023). "POSIX Threads Programming." *Lawrence Livermore National Laboratory*. https://hpc-tutorials.llnl.gov/posix/

5. Message Passing Interface Forum. (2021). *MPI: A Message-Passing Interface Standard, Version 4.0*. https://www.mpi-forum.org/docs/

6. NVIDIA Corporation. (2024). *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

7. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.

8. Cook, S. (1966). "On the Minimum Computation Time of Functions." *Harvard University Thesis*.

9. Chapman, B., Jost, G., & Van Der Pas, R. (2007). *Using OpenMP: Portable Shared Memory Parallel Programming*. MIT Press.

10. Gropp, W., Lusk, E., & Skjellum, A. (2014). *Using MPI: Portable Parallel Programming with the Message-Passing Interface* (3rd ed.). MIT Press.

---
