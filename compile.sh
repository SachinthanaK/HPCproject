#!/bin/bash
# Compilation script for HPC Big Integer Multiplication

echo "==========================================="
echo " Compiling Big Integer Multiplication Code "
echo "==========================================="

# Compile Serial
echo "Compiling serial/serial.c -> serial/serial..."
gcc -O3 serial/serial.c -o serial/serial
if [ $? -eq 0 ]; then echo "  ✓ Serial compiled successfully."; else echo "  ✗ Serial compilation failed."; fi

# Compile OpenMP
echo "Compiling shared_memory/Karatsubaomp.c -> shared_memory/Karatsubaomp..."
gcc -O3 -fopenmp shared_memory/Karatsubaomp.c -o shared_memory/Karatsubaomp
if [ $? -eq 0 ]; then echo "  ✓ OpenMP compiled successfully."; else echo "  ✗ OpenMP compilation failed."; fi

# Compile Pthreads
echo "Compiling shared_memory/Karatsubapthread.c -> shared_memory/Karatsubapthread..."
gcc -O3 -pthread shared_memory/Karatsubapthread.c -o shared_memory/Karatsubapthread
if [ $? -eq 0 ]; then echo "  ✓ Pthreads compiled successfully."; else echo "  ✗ Pthreads compilation failed."; fi

# Compile MPI
echo "Compiling distributed_memory/Karatsubampi.c -> distributed_memory/Karatsubampi..."
mpicc -O3 distributed_memory/Karatsubampi.c -o distributed_memory/Karatsubampi
if [ $? -eq 0 ]; then echo "  ✓ MPI compiled successfully."; else echo "  ✗ MPI compilation failed."; fi

# Compile Hybrid (MPI + OpenMP)
echo "Compiling hybrid/hybrid_karatsuba_mpi_omp.c -> hybrid/hybrid_karatsuba_mpi_omp..."
mpicc -O3 -fopenmp hybrid/hybrid_karatsuba_mpi_omp.c -o hybrid/hybrid_karatsuba_mpi_omp
if [ $? -eq 0 ]; then echo "  ✓ Hybrid compiled successfully."; else echo "  ✗ Hybrid compilation failed."; fi

echo "==========================================="
echo " Compilation finished."
echo "==========================================="
