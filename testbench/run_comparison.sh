#!/bin/bash

# Check if the binary exists
if [ ! -f "./compare_gemm" ]; then
  echo "Building compare_gemm..."
  make
fi

# Default number of threads (use system's thread count if not specified)
NUM_THREADS=${1:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}
echo "Using $NUM_THREADS threads"

# Run with small matrix size
echo "---------------------------------------------"
echo "Running small matrix test (256 x 256 x 256)"
./compare_gemm 256 256 256 $NUM_THREADS

# Run with medium matrix size
echo "---------------------------------------------"
echo "Running medium matrix test (1024 x 1024 x 1024)"
./compare_gemm 1024 1024 1024 $NUM_THREADS

# Run with large matrix size
echo "---------------------------------------------"
echo "Running large matrix test (2048 x 2048 x 2048)"
./compare_gemm 2048 2048 2048 $NUM_THREADS

# Run with rectangular matrices
echo "---------------------------------------------"
echo "Running rectangular matrix test (4096 x 1024 x 2048)"
./compare_gemm 4096 1024 2048 $NUM_THREADS

echo "---------------------------------------------"
echo "All tests completed" 