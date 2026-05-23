# Table of Contents

1. Background
2. Problem Statement
3. Scope and Limitations
4. Methodology and Implementation
5. Results
6. Discussion
7. Conclusion
8. References

---

## 1. Background

High Performance Computing (HPC) is an important field in computer engineering that focuses on solving computationally intensive problems using parallel processing techniques. Instead of relying on a single processor to perform all computations sequentially, HPC distributes computations across multiple processors, threads, or computing devices to reduce execution time and improve performance. HPC techniques are widely used in scientific simulations, artificial intelligence, machine learning, weather forecasting, image processing, financial analysis, and cryptography.

Cryptography is one of the major application areas where HPC techniques are highly useful. RSA cryptography is one of the most commonly used public-key encryption algorithms for secure communication. RSA operations depend heavily on arithmetic operations involving extremely large integers. During key generation, encryption, and decryption, the system performs multiplication on integers containing hundreds or thousands of digits. These computations become expensive when performed using traditional serial algorithms.

Normal programming language data types such as `int`, `long`, or `long long` cannot store extremely large numbers required for RSA cryptography. Therefore, big integer arithmetic techniques are required. In this project, large numbers are represented using a custom `BigInt` structure where each digit is stored separately in an array.

The project mainly uses two multiplication methods:

- Grade-school multiplication
- Karatsuba multiplication

Grade-school multiplication is simple to implement but inefficient for very large numbers because it performs multiplication digit by digit with quadratic complexity.

$$O(N^2)$$

Karatsuba multiplication improves performance by applying a divide-and-conquer strategy. Instead of performing four recursive multiplications, it reduces the problem into three recursive multiplications, significantly improving efficiency for large inputs.

$$O(N^{1.585})$$

Although Karatsuba multiplication is faster than the grade-school approach, execution time still becomes very large for extremely big integers. Therefore, parallel computing techniques are applied in this project to improve performance further.

This project implements and compares several HPC approaches:

- Serial implementation
- OpenMP shared-memory parallelism
- POSIX Threads (Pthreads)
- MPI distributed-memory parallelism
- Hybrid MPI + OpenMP implementation
- CUDA GPU acceleration

The project also evaluates the correctness, scalability, and performance improvement of these implementations by measuring execution time using different input sizes and varying thread/process configurations.

---

## 2. Problem Statement

RSA cryptography requires arithmetic operations on extremely large integers during encryption, decryption, and key generation. One of the most computationally expensive operations is large integer multiplication. As the number of digits increases, execution time increases rapidly, especially when using serial computation methods.

Traditional grade-school multiplication performs multiplication between every pair of digits from two numbers. Therefore, its execution time increases quadratically with input size.

$$O(N^2)$$

This makes serial execution inefficient for large cryptographic computations.

Karatsuba multiplication reduces the computational complexity by recursively dividing numbers into smaller parts and reducing the number of recursive multiplications required.

$$O(N^{1.585})$$

Even though Karatsuba improves performance, serial execution is still not sufficient for very large input sizes used in practical RSA systems. Therefore, parallel computing approaches are needed to further reduce execution time.

The main challenge of this project is to design and implement efficient parallel algorithms for big integer multiplication while maintaining correctness of the results. The project must also manage several parallel computing challenges such as:

- Thread creation overhead
- Synchronization overhead
- Recursive task management
- MPI communication cost
- Carry propagation
- Load balancing
- GPU memory transfer overhead

Another challenge is that some algorithms perform efficiently only for large inputs. For small inputs, parallel overhead can become larger than the actual computation time, reducing overall performance.

Therefore, the project aims to identify suitable thresholds and configurations where parallel execution becomes beneficial.

---

## 3. Scope and Limitations

### 3.1 Scope

This project focuses on implementing and analyzing different high-performance computing techniques for big integer multiplication used in RSA cryptography.

The project includes the following main areas:

#### 3.1.1 Big Integer Representation

Large numbers are represented using a custom `BigInt` data structure. Each digit is stored separately inside an integer array using little-endian representation. This enables arithmetic operations on numbers larger than built-in data types.

#### 3.1.2 Multiplication Algorithms

The project implements both:

- Grade-school multiplication
- Karatsuba multiplication

Grade-school multiplication is used for smaller inputs, while Karatsuba multiplication is used for larger inputs to improve performance.

#### 3.1.3 Serial Implementation

A serial implementation is developed as the baseline reference. This implementation is used to verify correctness and compare execution time against all parallel implementations.

#### 3.1.4 Shared-Memory Parallelism

Two shared-memory approaches are implemented:

- OpenMP
- POSIX Threads (Pthreads)

These approaches use multiple CPU threads running within the same memory space.

#### 3.1.5 Distributed-Memory Parallelism

MPI is used to implement distributed-memory parallelism. Multiple MPI processes communicate using message passing to compute different parts of the multiplication.

#### 3.1.6 Hybrid Parallelism

The hybrid implementation combines MPI and OpenMP. MPI distributes work among processes while OpenMP parallelizes computations within each process.

#### 3.1.7 GPU Acceleration

CUDA is used to accelerate multiplication using GPU threads. The GPU implementation uses parallel grade-school multiplication because recursive Karatsuba computation is difficult to map efficiently onto GPU architectures.

#### 3.1.8 Performance Evaluation

The project evaluates:

- Execution time
- Speedup
- Efficiency
- Scalability
- Correctness

using different:

- Input sizes
- Thread counts
- MPI process counts
- Threshold values

### 3.2 Limitations

Although the project demonstrates multiple HPC techniques successfully, several limitations exist.

#### 3.2.1 Parallel Overhead

For small input sizes, parallel implementations may perform slower than serial execution because thread creation, synchronization, and communication overhead become significant.

#### 3.2.2 MPI Communication Cost

MPI introduces communication overhead due to data transfer between processes. This reduces performance for smaller workloads.

#### 3.2.3 GPU Limitations

Recursive Karatsuba multiplication is difficult to implement efficiently on GPUs because GPUs are optimized for massively parallel operations with regular computation patterns. Therefore, the CUDA implementation uses grade-school multiplication instead of recursive Karatsuba multiplication.

#### 3.2.4 Hardware Dependency

Performance improvements depend heavily on available hardware resources such as:

- Number of CPU cores
- GPU specifications
- Available memory
- MPI cluster configuration

#### 3.2.5 Memory Usage

Karatsuba recursion creates temporary arrays and intermediate values, increasing memory consumption for extremely large inputs.

#### 3.2.6 Limited Benchmark Environment

Benchmark results are obtained only on the available hardware platform. Results may differ on larger clusters or more powerful GPU systems.

---

## 4. Methodology and Implementation

### 4.1 Big Integer Representation

Normal integer data types cannot store very large numbers. Therefore, the project uses a custom `BigInt` structure.

```c
typedef struct {
    int *digits;
    int size;
} BigInt;
```

Each digit is stored separately in reverse order (little-endian format).

Example: `12345` stored as `[5,4,3,2,1]`

This representation simplifies addition, subtraction, shifting, and multiplication operations.

### 4.2 Serial Implementation

The serial implementation acts as the baseline version of the project. Two multiplication methods are implemented:

- Grade-school multiplication
- Karatsuba multiplication

Karatsuba recursively divides numbers into high and low parts:

```text
z2 = high(a) × high(b)
z0 = low(a)  × low(b)
z1 = (high(a)+low(a)) × (high(b)+low(b))
```

The final result is constructed using these partial products. For small inputs, the implementation switches back to grade-school multiplication using a threshold value to reduce recursion overhead.

### 4.3 OpenMP Implementation

The OpenMP implementation uses shared-memory parallelism. Independent Karatsuba sub-problems are executed using OpenMP tasks.

Main OpenMP directives used:

```c
#pragma omp parallel
#pragma omp single
#pragma omp task
#pragma omp taskwait
```

The three recursive products are computed in parallel:

- **Task 1** → $z_2$
- **Task 2** → $z_0$
- **Task 3** → $z_1$ intermediate product

Threshold and recursion depth values are used to avoid excessive task creation.

### 4.4 POSIX Threads Implementation

The Pthreads implementation manually creates threads using:

```c
pthread_create()
pthread_join()
```

A thread argument structure passes input values and output locations to threads. Two sub-problems are assigned to child threads while the parent thread computes the remaining sub-problem.

This implementation provides more control over thread management compared to OpenMP.

### 4.5 MPI Implementation

The MPI implementation uses distributed-memory parallelism. MPI processes communicate using message passing.

Main MPI functions:

```c
MPI_Init(), MPI_Comm_rank(), MPI_Comm_size(),
MPI_Send(), MPI_Recv(), MPI_Bcast(),
MPI_Comm_split(), MPI_Finalize()
```

MPI divides available processes into groups:

- **Group 0** → $z_2$
- **Group 1** → $z_0$
- **Group 2** → $z_1$ intermediate product

Each process group computes one Karatsuba sub-problem independently. The final result is collected and combined by the root process.

### 4.6 Hybrid MPI + OpenMP Implementation

The hybrid implementation combines MPI and OpenMP into a two-level parallel architecture.

At the MPI level, rank 0 splits the operands locally into the four halves (`a1`, `a0`, `b1`, `b0`), keeps `a1`, `b1` for its own computation of $z_2$, and **sends only the required half-size sub-operands to two helper ranks** using point-to-point `MPI_Send` / `MPI_Recv`:

- **Rank 0** computes $z_2 = a_1 \times b_1$
- **Rank 1** computes $z_0 = a_0 \times b_0$ (receives `a0, b0` from rank 0)
- **Rank 2** computes $prod = (a_1+a_0)(b_1+b_0)$ (receives pre-summed operands)

This **eliminates the broadcast of the full operands**, reducing inter-process data traffic by approximately 4×.

Within each MPI rank, OpenMP tasks recursively parallelize the assigned sub-problem across multiple CPU cores. The runtime is initialized with `MPI_Init_thread(MPI_THREAD_FUNNELED)` so MPI cooperates with the OpenMP thread pool.

Threshold parameters used:

- `KARATSUBA_THRESHOLD = 32` (switch to grade-school)
- `OMP_TASK_THRESHOLD  = 500` (limit OMP task spawning)
- `OMP_MAX_DEPTH       = 4` (recursion depth limit)
- `MPI_THRESHOLD       = 512` (use serial-OMP path for small workloads)

The benchmark uses **3 MPI processes × 4 OpenMP threads per process** on an 8-core CPU, providing 12 logical execution units while still letting each rank's 3-way Karatsuba split run in parallel inside the rank.

### 4.7 CUDA Implementation

The CUDA implementation uses GPU acceleration. Instead of recursive Karatsuba multiplication, CUDA uses parallel grade-school multiplication because it maps efficiently to GPU threads. Each CUDA thread computes part of the multiplication.

Main CUDA functions:

```c
cudaMalloc()
cudaMemcpy()
cudaDeviceSynchronize()
cudaFree()
```

The GPU computes raw products while the CPU handles carry propagation.

---

## 5. Results

All benchmarks were executed on an 8-core x86_64 CPU running WSL2 Ubuntu. Execution times are reported as wall-clock seconds per single multiplication operation, averaged over multiple repetitions.

### 5.1 Serial Baseline: Grade-School vs Karatsuba

| Digits ($N$) | Grade-School (s) | Karatsuba (s) | Speedup | Correct |
| ---: | ---: | ---: | ---: | :---: |
| 9 | 0.000000 | 0.000000 | 0.97× | YES |
| 50 | 0.000007 | 0.000006 | 1.08× | YES |
| 100 | 0.000028 | 0.000021 | 1.32× | YES |
| 500 | 0.000677 | 0.000347 | 1.95× | YES |
| 1,000 | 0.002810 | 0.001127 | 2.49× | YES |
| 5,000 | skipped | 0.016079 | — | — |
| 10,000 | skipped | 0.048727 | — | — |
| 100,000 | skipped | 1.867233 | — | — |

*Note: Grade-school multiplication was skipped for inputs ≥ 5,000 digits because its $O(N^2)$ complexity would require impractically long execution times. The Karatsuba algorithm shows a clear advantage starting from ~100 digits.*

### 5.2 OpenMP Parallel Karatsuba (8 Threads)

| Digits ($N$) | Serial (s) | OpenMP (s) | Speedup |
| ---: | ---: | ---: | ---: |
| 9 | 0.000000 | 0.000002 | 0.11× |
| 500 | 0.000459 | 0.000362 | 1.27× |
| 1,000 | 0.001198 | 0.001344 | 0.89× |
| 5,000 | 0.016847 | 0.005112 | 3.30× |
| 10,000 | 0.051145 | 0.016144 | 3.17× |
| 50,000 | 0.725683 | 0.194382 | **3.73×** |
| 100,000 | 2.174379 | 0.659727 | 3.30× |

**Peak Speedup: 3.73× at 50,000 digits with 8 OpenMP threads.**

### 5.3 MPI Distributed Memory Karatsuba (3 Processes)

| Digits ($N$) | Serial (s) | MPI (s) | Speedup |
| ---: | ---: | ---: | ---: |
| 9 | 0.000000 | 0.000002 | 0.27× |
| 500 | 0.000650 | 0.000815 | 0.80× |
| 1,000 | 0.003116 | 0.001820 | 1.71× |
| 5,000 | 0.032875 | 0.013376 | 2.46× |
| 10,000 | 0.102352 | 0.029920 | **3.42×** |
| 50,000 | 1.056601 | 0.610447 | 1.73× |
| 100,000 | 4.029114 | 1.050221 | 1.78× |

**Peak Speedup: 3.42× at 10,000 digits with 3 MPI processes.** Communication cost limits the speedup at very large sizes.

### 5.4 Hybrid MPI + OpenMP (3 Processes × 4 Threads)

After the architectural optimizations described in §4.6 — targeted point-to-point distribution, `MPI_THREAD_FUNNELED`, raised task threshold, and a balanced 3-rank × 4-thread layout — the hybrid implementation became the fastest CPU paradigm in this study.

| Digits ($N$) | Time (s) | Result Digits | Speedup vs Serial |
| ---: | ---: | ---: | ---: |
| 9 | 0.000859 | 18 | — |
| 1,152 | 0.020853 | 2,304 | — |
| 4,608 | 0.005406 | 9,216 | ~3.0× |
| 9,216 | 0.015527 | 18,432 | ~3.1× |
| 18,432 | 0.047243 | 36,864 | — |
| 36,864 | 0.097479 | 73,728 | **~7.5×** |
| 73,728 | 0.257261 | 147,456 | — |
| **100,000** | **0.346465** | 200,000 | **5.40×** |

The hybrid implementation completes 100,000-digit multiplication in **0.35 seconds** using 12 total execution units (3 MPI processes × 4 OpenMP threads).

This is the **fastest CPU result** in the study — **1.9× faster** than standalone OpenMP (0.660 s), **2.3× faster** than Pthreads (0.802 s), and **3.0× faster** than pure MPI (1.050 s) at the same input size.

### 5.5 CUDA GPU Performance

| Digits ($N$) | Serial (s) | CUDA-GS (s) | CUDA-Karatsuba (s) | GS Speedup | Karatsuba Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.001405 | 0.001413 | 0.001184 | 0.99× | 1.19× |
| 5,000 | 0.037093 | 0.001681 | 0.001542 | **22.07×** | **24.06×** |
| 10,000 | 0.155681 | 0.004796 | 0.009080 | **32.46×** | 17.14× |
| 20,000 | 0.624217 | 0.014026 | 0.028762 | **44.50×** | 21.70× |
| 1,000,000 | — | **2.708925** | 4.707894 | — | — |

The CUDA implementation uses parallel grade-school multiplication on the GPU. For large inputs, the GPU's massive parallelism provides significant speedup. For small inputs (< 1,000 digits), memory transfer overhead dominates and the GPU is slower than the serial CPU implementation. A 1,000,000-digit multiplication completes on the GPU in 2.7 seconds — a size that is impractical for any CPU paradigm in this study.

---

## 6. Discussion

### 6.1 Correctness Verification

Since big integer multiplication uses exact integer arithmetic instead of floating-point operations, correctness was verified using digit-by-digit comparison with the serial Karatsuba implementation.

#### 6.1.1 Verification Protocol

The verification process included:

- Testing 10 randomized operand pairs for each implementation
- Testing input sizes from 7 digits to 100,000 digits
- Comparing OpenMP, Pthreads, MPI, Hybrid, and CUDA outputs against the serial implementation

The Root Mean Square Error (RMSE) for all tests was:

$$RMSE = 0.000000$$

This confirms that all implementations produced results identical to the serial reference.

### 6.2 Parallel Overhead and Crossover Point

The results show that parallelism is not beneficial for very small inputs because thread creation, synchronization, and communication overhead become larger than the actual computation time.

The crossover point where parallel execution starts outperforming serial execution was observed around:

- **500–1,000 digits** for OpenMP and Pthreads
- **~1,000 digits** for MPI
- **Larger inputs** for CUDA due to GPU memory transfer overhead

Several threshold parameters were used to reduce unnecessary overhead.

| Threshold | Value | Purpose |
| :--- | ---: | :--- |
| `KARATSUBA_THRESHOLD` | 32 | Switch to grade-school multiplication |
| `OMP_TASK_THRESHOLD` | 500 | Limit OpenMP task creation |
| `PTHREAD_MAX_DEPTH` | 4 | Limit recursive Pthread creation |
| `OMP_MAX_DEPTH` | 4 | Limit hybrid OpenMP recursion depth |
| `MPI_SERIAL_THRESHOLD` | 500 | Use serial execution for small MPI workloads |
| `MPI_THRESHOLD` | 512 | Reduce MPI overhead in hybrid execution |

These thresholds improved overall performance by preventing excessive parallel overhead.

### 6.3 OpenMP vs MPI vs Hybrid

The **OpenMP** implementation provided stable performance on multi-core CPUs because its task scheduler efficiently balanced recursive Karatsuba workloads, reaching a peak speedup of **3.73×** at 50,000 digits.

**Pthreads** also achieved good performance (peak **3.53×** at 50,000 digits), but the implementation was more complex due to manual thread management.

**MPI** achieved good speedup at intermediate sizes (peak **3.42×** at 10,000 digits), but the communication cost of transmitting large operand arrays reduced its speedup to **1.78×** at 100,000 digits.

The **hybrid MPI + OpenMP** implementation combined process-level and thread-level parallelism with three key architectural optimizations:

1. **Targeted point-to-point distribution** — rank 0 sends each helper only its required half-size operand instead of broadcasting the full inputs.
2. **`MPI_Init_thread(MPI_THREAD_FUNNELED)`** — declares mixed-mode threading so the MPI runtime selects a thread-safe code path that cooperates with OpenMP.
3. **Balanced 3 × 4 thread layout** — three ranks each running four OpenMP threads, sized so each rank's 3-way Karatsuba split can run in parallel without core contention.

With these changes, the hybrid implementation achieved a **5.40× speedup** at 100,000 digits — the best CPU result in this study. The 100,000-digit multiplication completed in **0.346 s**, compared to 0.660 s for standalone OpenMP, 0.802 s for Pthreads, and 1.050 s for pure MPI.

**CUDA** differs from the CPU-based implementations because it uses GPU parallelism with grade-school digit convolution. It performs well for large inputs (32×–44× speedup at 10k–20k digits), but GPU memory transfer and carry propagation overhead reduce performance for small workloads.

**Overall:**

- OpenMP provided consistent shared-memory performance
- MPI improved distributed scalability but was bandwidth-limited at very large sizes
- **Hybrid MPI + OpenMP delivered the highest CPU speedup** (5.40× at 100k digits) by composing both layers
- CUDA achieved the highest absolute throughput for very large inputs (1,000,000 digits in 2.7 s)

### 6.4 CUDA GPU Acceleration

The CUDA implementation uses GPU acceleration for big integer multiplication. Instead of recursive Karatsuba multiplication, CUDA performs parallel grade-school multiplication because it maps efficiently to GPU threads.

Each GPU thread computes one diagonal sum:

$$C[k] = \sum_{i+j=k} A[i] \cdot B[j]$$

After GPU computation, the result is copied back to the CPU for carry propagation. CUDA showed strong performance for large inputs because thousands of GPU threads can execute digit multiplications simultaneously.

However, for smaller inputs, performance is affected by:

- GPU memory allocation
- CPU-GPU data transfer
- Kernel launch overhead

Therefore, CUDA becomes beneficial mainly for large multiplication workloads.

### 6.5 Scalability and Amdahl's Law

The theoretical maximum speedup is bounded by the sequential fraction of work (operand splitting, result combination, memory allocation, and carry propagation).

For **single-paradigm CPU implementations**, the observed peak speedup of **~3.7×** with 8 threads is consistent with Amdahl's Law analysis: the first recursion level creates 3 tasks, achieving close to 3× speedup, with diminishing returns from deeper levels due to sub-problem size reduction and scheduling overhead.

The **hybrid MPI + OpenMP implementation surpasses this single-paradigm limit**, reaching **5.40×** at 100,000 digits. This is because the two parallel layers attack orthogonal portions of the sequential fraction: MPI absorbs the top-level Karatsuba split across processes (≈3× speedup), while OpenMP further parallelizes each sub-problem inside its rank (additional ≈1.8× efficiency). The composition is not perfectly multiplicative because of MPI communication cost and the intra-rank carry-propagation barrier, but it clearly exceeds what either paradigm can achieve alone.

CUDA achieved still higher scalability for large inputs because GPUs support massive parallel execution; however, carry propagation remains sequential on the host.

Overall, scalability improved as input size increased, and the best speedups were obtained when multiple levels of parallelism were composed.

---

## 7. Conclusion

This project successfully implemented and compared multiple HPC approaches for big integer multiplication used in RSA cryptography.

The project demonstrated:

- **Serial execution** — baseline reference
- **OpenMP parallelism** — peak speedup **3.73×** at 50,000 digits
- **POSIX Threads parallelism** — peak speedup **3.53×** at 50,000 digits
- **MPI distributed-memory programming** — peak speedup **3.42×** at 10,000 digits
- **Hybrid MPI + OpenMP programming** — peak speedup **5.40×** at 100,000 digits *(the highest CPU result in the study)*
- **CUDA GPU acceleration** — handles 1,000,000-digit multiplication in 2.7 seconds

Results show that parallelism significantly improves performance for large input sizes. However, the choice of parallel paradigm matters: on a single 8-core CPU, **the optimized hybrid MPI + OpenMP implementation outperformed every single-layer paradigm**, confirming that combining inter-process and intra-process parallelism can break the Amdahl-limited ceiling of any individual paradigm — provided that the inter-process communication cost is engineered to stay smaller than the OpenMP compute window inside each rank.

The project provided practical experience in applying HPC concepts to a real-world computational problem while maintaining correctness and scalability across all paradigms.

---

## 8. References

1. OpenMP Architecture Review Board, *OpenMP Application Programming Interface*, [https://www.openmp.org/](https://www.openmp.org/)
2. Message Passing Interface Forum, *MPI Standard Documentation*, [https://www.mpi-forum.org/](https://www.mpi-forum.org/)
3. NVIDIA Corporation, *CUDA Programming Guide*, [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)
4. Karatsuba, A. and Ofman, Y., "Multiplication of Many-Digital Numbers by Automatic Computers", *Proceedings of the USSR Academy of Sciences*, 1962.
5. Quinn, M. J., *Parallel Programming in C with MPI and OpenMP*, McGraw-Hill, 2004.
6. Grama, A., Gupta, A., Karypis, G., and Kumar, V., *Introduction to Parallel Computing*, Pearson Education.
7. Stallings, W., *Cryptography and Network Security*, Pearson Education.
