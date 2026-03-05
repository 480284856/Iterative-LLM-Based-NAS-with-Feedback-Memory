"""
Plot three accuracy curves for Qwen2.5-7B-Instruct (2000 iterations):
  1. Best So Far  – running maximum
  2. Accuracy     – error iterations replaced by previous round's accuracy
  3. Smoothed     – moving average of curve 2, window = 15
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ACC_TXT = HERE.parent.parent.parent / "experimental_result/cifar10/qwen2.5_7b_instruct_2000/qwen2.5_7b_instruct_2000_acc.txt"
OUT_PNG = HERE / "qwen_three_curves.png"
OUT_PDF = HERE / "qwen_three_curves.pdf"

# ── Load data ────────────────────────────────────────────────────────────────
acc = np.array([float(line.strip()) for line in ACC_TXT.read_text().splitlines() if line.strip()])
n = len(acc)
x = np.arange(1, n + 1)

# ── Compute curves ───────────────────────────────────────────────────────────
best_so_far = np.maximum.accumulate(acc)            # Curve 1

def moving_average(arr, window=15):
    """Centered moving average with edge-padding."""
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]

smoothed = moving_average(acc, window=15)           # Curve 3

# ── Plot ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(10, 4.5))

COLOR = "#4DBBD5"   # Qwen blue

ax.plot(x, acc,         color=COLOR,  alpha=0.30, linewidth=0.7, zorder=1, label="Accuracy")
ax.plot(x, smoothed,    color=COLOR,  alpha=0.85, linewidth=1.5, linestyle="--", zorder=2, label="Smoothed")
ax.plot(x, best_so_far, color=COLOR,  linewidth=2.2, zorder=3, label="Best So Far")

ax.set_xlabel("Iteration")
ax.set_ylabel("Accuracy")
ax.set_title("Qwen2.5-7B-Instruct")
ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
ax.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))

ymin = max(0.0, np.percentile(acc, 1) - 0.03)
ymax = min(1.0, np.max(acc) + 0.05)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
