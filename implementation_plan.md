# Implementation Plan - HPC Big Integer Multiplication

This plan addresses code quality improvements, performance enhancements, and broadening the scope of the High Performance Computing (HPC) Big Integer Multiplication project. It outlines how we can satisfy all guidelines (OpenMP, POSIX Threads, MPI, and CUDA) and prepare the benchmark data for the Analysis Report.

---

## Codebase Analysis & Best Practices Check

Based on our review of `serial.c`, `Karatsubaomp.c`, `Karatsubampi.c`, and `hybrid_karatsuba_mpi_omp.c`, we found the following areas for improvement:

1. **Unnecessary Memory Allocation & Copies in OpenMP (`Karatsubaomp.c`)**:
   - **Current Issue**: For every recursive division, the code allocates memory and clones digits for `a1c`, `b1c`, `a0c`, `b0c` to pass to OpenMP tasks, then frees them inside the tasks. This is done because the parent thread frees the original `a1`, `b1`, `a0`, `b0` before calling `#pragma omp taskwait`.
   - **Improvement**: If we defer freeing `a1`, `b1`, `a0`, `b0` in the parent thread to *after* the taskwait, they will remain valid. The tasks can read them directly without cloning. This will save 4 allocations, 4 copies, and 4 frees per parallel recursion step, significantly boosting performance.

2. **OpenMP Task Explosion**:
   - **Current Issue**: `Karatsubaomp.c` has a task spawning threshold (`OMP_TASK_THRESHOLD = 500` digits), but no depth limit. For very large inputs (e.g. 100,000+ digits), this spawns thousands of tasks, causing high runtime scheduling overhead.
   - **Improvement**: Implement a depth-limited task spawn model using `OMP_MAX_DEPTH` (similar to the hybrid code), preventing task explosion.

3. **Memory Allocation Safety**:
   - **Current Issue**: The codebase lacks checks for allocation failures (`malloc`/`calloc`/`realloc` returning `NULL`). When processing very large digits (e.g. 1,000,000+ digits), out-of-memory errors will trigger segmentation faults.
   - **Improvement**: Standardize allocation with helper functions or check return values and exit cleanly.

4. **Zero-Digit Safety in effectiveSize**:
   - **Current Issue**: If a `BigInt` size is 0, `effectiveSize` will check `b->digits[s - 1]` (index `-1`), causing undefined behavior.
   - **Improvement**: Add a check `while (s > 0 && b->digits[s - 1] == 0) s--;` and return `s == 0 ? 1 : s`.

---

## Proposed Improvements & Scope Expansion

We will expand the project scope to cover all requested technologies:

### 1. [MODIFY] [Karatsubaomp.c](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/Karatsubaomp.c)
- Optimize task spawning by deferring freeing parent variables to eliminate intermediate digit copies.
- Introduce `OMP_MAX_DEPTH` control.
- Add safety checks for `malloc` failures.

### 2. [NEW] [Karatsubapthread.c](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/Karatsubapthread.c)
- **Goal**: Add a direct **POSIX Threads (pthreads)** implementation of Karatsuba multiplication.
- **Approach**:
  - Implement a recursive Karatsuba function that spawns threads for the subproblems when below a certain depth limit (`PTHREAD_MAX_DEPTH`).
  - Pass arguments via a custom struct to each thread.
  - Call `pthread_create` and `pthread_join`.

### 3. [NEW] [MultiplicationCuda.cu](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/MultiplicationCuda.cu)
- **Goal**: Add a **CUDA** GPU implementation of Big Integer multiplication.
- **Approach**:
  - Because Karatsuba's recursive structure is poorly suited for GPU SIMD lock-step execution, we will parallelize the **Schoolbook (Grade-school) Multiplication** method. This allows us to map the $O(N^2)$ algorithm to a $O(N)$ execution time using $2N$ threads on the GPU.
  - **GPU Kernel**: Each thread $k$ (where $k$ corresponds to the output digit index) computes the diagonal sum: $C[k] = \sum_{i+j=k} A[i] \cdot B[j]$.
  - **Carry Propagation**: Since carry propagation is sequential and $O(N)$, we will copy the diagonal sums back to the host and perform the carry propagation on the CPU, or do a fast sequential pass on the GPU.
  - This satisfies the CUDA requirement and shows a classic GPU parallelization case.

### 4. [NEW] [compile.sh](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/compile.sh)
- A bash script to easily compile all five versions in WSL Ubuntu:
  - Serial: `gcc -O3 serial.c -o serial`
  - OpenMP: `gcc -O3 -fopenmp Karatsubaomp.c -o Karatsubaomp`
  - Pthreads: `gcc -O3 -pthread Karatsubapthread.c -o Karatsubapthread`
  - MPI: `mpicc -O3 Karatsubampi.c -o Karatsubampi`
  - Hybrid: `mpicc -O3 -fopenmp hybrid_karatsuba_mpi_omp.c -o hybrid_karatsuba_mpi_omp`

### 5. [NEW] [run_benchmarks.sh](file:///c:/Users/some1/Downloads/Sem%2007/HPC/project/BigIntegerMultiplicationSerial/run_benchmarks.sh)
- A script to run all implementations with various digit sizes and record the output timing.
- Will print results in a format suitable for the final analysis report.

---

## Verification Plan

### Automated Verification
- We will write a benchmark and validation harness in each file (as is currently done with `randomBigInt` and `bigIntsEqual`) that compares the parallel output against the serial output.
- We will execute the compilation and verify all binaries pass correctness checks on WSL Ubuntu.

### Manual Verification
- We will run the benchmarks for various numbers of threads (e.g. 1, 2, 4, 8) and MPI ranks to gather data for the timing analysis.
