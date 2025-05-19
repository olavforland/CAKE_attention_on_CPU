# Systolic Arrays for Attention Kernel (on CPU)

This repository contains code for benchmarking a fused softmax attention implementation using ARM NEON intrinsics. The implementation demonstrates significant speedups compared to a materialized baseline for small $N$ and $d$, but due to an unsatisfactory tiling approach, the results do not scale to larger matrices.

## Requirements

- C++ compiler with ARM NEON support
- Python 3.x with matplotlib, pandas, and numpy for plotting
- Make

## Directory Structure

```
.
├── figures/         # Benchmark result plots
├── include/         # Header files
├── results/         # CSV result files
├── scripts/         # Benchmark and plotting scripts
├── src/             # Implementation source code
└── testbench/       # Benchmark executable code
```

## Building

To build the benchmarking code:

```bash
source env.sh    # Set up environment variables
make             # Build the core library
cd testbench && make  # Build the benchmark executables
```

## Running Benchmarks

To run benchmarks across different sequence lengths (N) and hidden dimensions (d):

```bash
./scripts/generate_results.sh
```

This will:
1. Run the attention kernel with various matrix sizes
2. Record execution times for both materialized and fused implementations
3. Save results to a timestamped CSV file in the `results/` directory

You can specify a custom output path:

```bash
./scripts/generate_results.sh custom_results.csv
```

## Visualizing Results

To create plots from benchmark results:

```bash
python scripts/plot_results.py results/results_YYYYMMDD_HHMMSS.csv
```

This generates a log-log plot showing speedup factors across different sequence lengths and hidden dimensions, saving the resulting figure to `figures/speedup.pdf`.


## Fused vs. Materialized Implementation

The benchmarks compare two implementations:

1. **Materialized Baseline**: Computes attention by materializing the full N×N attention matrix.
2. **Fused Implementation**: Uses a systolic approach with 8×12 scratch tiles to avoid materializing the full attention matrix.

The fused implementation shows particularly strong speedups for small to medium dimensions, making it well-suited for applications requiring efficient attention computation on ARM platforms.

## Custom Benchmarking

To run a single benchmark with specific parameters:

```bash
./testbench/compare_gemm N d THREADS TRIALS
```


## Acknowledgements

This implementation is based on the CAKE (Constant-Attention-Kernel) matrix multiplication library for CPUs. The original CAKE repository can be found at [github.com/vnatesh/CAKE_on_CPU](https://github.com/vnatesh/CAKE_on_CPU). This work adapts the GEMM kernel of CAKE to implement a fused attention kernel for ARM platforms. 