# CUDA Benchmark

This folder contains the CUDA implementation:

```text
MultiplicationCuda.cu
```

The program benchmarks three methods:

```text
Serial CPU grade-school multiplication
CUDA grade-school multiplication
CUDA-backed Karatsuba multiplication
```

## Method Summary

`Serial CPU` is the baseline. It multiplies every digit pair on the CPU and performs carry propagation.

`CUDA grade-school` launches a CUDA kernel where each GPU thread computes one output diagonal:

```text
C[k] = sum(A[i] * B[j]) where i + j = k
```

`CUDA-backed Karatsuba` uses Karatsuba's divide-and-conquer split on the host:

```text
z2   = a1 * b1
z0   = a0 * b0
prod = (a1 + a0) * (b1 + b0)
z1   = prod - z2 - z0
```

Each Karatsuba leaf multiplication is offloaded to the CUDA grade-school kernel. The threshold is currently:

```text
KARATSUBA_CUDA_THRESHOLD = 8192 digits
```

This keeps the code understandable and avoids launching too many small CUDA kernels.

## Build In WSL

From WSL Ubuntu:

```bash
cd "/mnt/d/Semester/Semester 7/HPC/HPCproject"
nvcc -O3 cuda/MultiplicationCuda.cu -o cuda/MultiplicationCuda
```

## Run And Save Log

```bash
./cuda/MultiplicationCuda > log/benchmark_cuda.log 2>&1
cat log/benchmark_cuda.log
```

Run a large CUDA-only benchmark, for example 1,000,000 digits:

```bash
./cuda/MultiplicationCuda 1000000 1 > log/benchmark_cuda_1m.log 2>&1
cat log/benchmark_cuda_1m.log
```

Large-input mode skips the serial CPU baseline because serial grade-school multiplication is too slow at that scale.

## Windows Build Alternative

On Windows, CUDA compilation needs:

```text
NVIDIA CUDA Toolkit with nvcc
Microsoft Visual Studio C++ Build Tools with cl.exe
```

Then compile from **x64 Native Tools Command Prompt for VS**:

```cmd
cd /d "D:\Semester\Semester 7\HPC\HPCproject"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -O3 cuda\MultiplicationCuda.cu -o cuda\MultiplicationCuda.exe
```

Run:

```cmd
cuda\MultiplicationCuda.exe > log\benchmark_cuda.log 2>&1
```
