#!/usr/bin/env bash
# Sweep problem sizes and collect timing results into CSV.
# Usage: ./generate_results.sh [output_file]

BIN_DIR="$(dirname "$0")/../testbench"
COMPARE_BIN="$BIN_DIR/compare_gemm"
RESULTS_DIR="$(dirname "$0")/../results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Default output file
OUTPUT_FILE="${1:-$RESULTS_DIR/results_$(date +"%Y%m%d_%H%M%S").csv}"

THREADS=1              # single-thread only
TRIALS=1
# Sequence lengths for sweep
N_LIST=(32 64 128 256 512)
# Hidden dimensions
D_LIST=(16 32 64 128 256)

# CSV header
echo "N,d,materialized_ms,fused_ms" > "$OUTPUT_FILE"

for d in "${D_LIST[@]}"; do
  for N in "${N_LIST[@]}"; do
    output=$("$COMPARE_BIN" "$N" "$d" "$THREADS" "$TRIALS" | tail -n 4)
    mat_time=$(echo "$output" | grep "Materialized time" | awk '{print $3}')
    fused_time=$(echo "$output" | grep "Fused time" | awk '{print $3}')
    echo "$N,$d,$mat_time,$fused_time" >> "$OUTPUT_FILE"
  done
done

echo "Results saved to $OUTPUT_FILE" 