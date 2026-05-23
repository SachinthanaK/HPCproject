#!/bin/bash
# Runner script for HPC Big Integer Multiplication Benchmarks

echo "==========================================="
echo " Starting Benchmarks...                    "
echo "==========================================="

# Ensure scripts are executable
chmod +x compile.sh

# Compile first
./compile.sh

echo "Running Serial benchmark..."
./serial/serial > log/benchmark_serial.log 2>&1
echo "  ✓ Serial benchmark finished."

echo "Running OpenMP benchmark (threads: $(nproc))..."
OMP_NUM_THREADS=$(nproc) ./shared_memory/Karatsubaomp > log/benchmark_omp.log 2>&1
echo "  ✓ OpenMP benchmark finished."

echo "Running Pthreads benchmark..."
./shared_memory/Karatsubapthread > log/benchmark_pthread.log 2>&1
echo "  ✓ Pthreads benchmark finished."

echo "Running MPI benchmark (processes: 3)..."
mpirun --allow-run-as-root -np 3 ./distributed_memory/Karatsubampi > log/benchmark_mpi.log 2>&1
echo "  ✓ MPI benchmark finished."

echo "Running Hybrid (MPI+OpenMP) benchmark..."
# 3 MPI ranks x 4 OpenMP threads each. Each rank handles one of the 3 Karatsuba
# sub-problems (z2 / z0 / prod) and spawns OMP tasks internally for the deeper
# recursive splits. 4 threads/rank can actually run the 3-way split + spare.
export OMP_NUM_THREADS=4
mpirun --allow-run-as-root -np 3 ./hybrid/hybrid_karatsuba_mpi_omp 9 100000 geometric > log/benchmark_hybrid.log 2>&1
echo "  ✓ Hybrid benchmark finished."

echo "==========================================="
echo " Benchmarks completed! Logs saved to:"
echo "   - log/benchmark_serial.log"
echo "   - log/benchmark_omp.log"
echo "   - log/benchmark_pthread.log"
echo "   - log/benchmark_mpi.log"
echo "   - log/benchmark_hybrid.log"
echo "==========================================="
