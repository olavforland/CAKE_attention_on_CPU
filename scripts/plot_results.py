import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as mticker
import numpy as np
import os

if len(sys.argv) < 2:
    print("Usage: python plot_results.py results.csv")
    sys.exit(1)

csv_file = sys.argv[1]

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(figures_dir, exist_ok=True)

df = pd.read_csv(csv_file)

# Compute speed-up
df['speedup'] = df['materialized_ms'] / df['fused_ms']

for d in sorted(df['d'].unique()):
    sub = df[df['d'] == d]
    plt.plot(sub['N'], sub['speedup'], marker='o', label=f"d={d}")

plt.xscale('log', base=2)
plt.yscale('log', base=2)

# Use actual numbers (no 2^x) on log axes
xticks = sorted(df['N'].unique())
plt.xticks(xticks, [str(n) for n in xticks])

# Choose y-ticks as powers of two within range but label numerically
ymin, ymax = df['speedup'].min(), df['speedup'].max()
exponents = list(range(int(np.floor(np.log2(ymin))), int(np.ceil(np.log2(ymax)))+1))
yticks = [float(2**e) for e in exponents]
plt.yticks(yticks, [str(int(y)) if y == int(y) else f"{y:.1f}" for y in yticks])

plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())

plt.xlabel("Sequence length N (log₂)")
plt.ylabel("Speed-up vs materialised (log₂)")
plt.title("Fused Attention Kernel Speedup Compared to Materialised Baseline")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()

output_file = os.path.join(figures_dir, "speedup.pdf")
plt.savefig(output_file)
print(f"Saved figure to {output_file}") 