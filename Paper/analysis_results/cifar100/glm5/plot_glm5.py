"""
Plot three accuracy curves for GLM-5 on CIFAR-100 (100 iterations):
  1. Best So Far  – running maximum
  2. Accuracy     – error iterations replaced by previous round's accuracy
  3. Smoothed     – moving average of curve 2, window = 5
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_LOG = HERE.parent.parent.parent / "experimental_result/cifar100/glm5_100/results.log"
OUT_PNG = HERE / "glm5_three_curves.png"
OUT_PDF = HERE / "glm5_three_curves.pdf"

RE_ACC = re.compile(r"iteration:\s*(\d+),\s*accuracy:\s*([\d.]+)")
RE_ERR = re.compile(r"iteration:\s*(\d+),\s*error:")

def extract_acc(log_path):
    raw = []
    last_acc = None
    with open(log_path) as f:
        for line in f:
            if m := RE_ACC.search(line):
                last_acc = float(m.group(2))
                raw.append(last_acc)
            elif RE_ERR.search(line):
                raw.append(last_acc)

    arr = np.array(raw, dtype=float)
    if np.any(np.isnan(arr)):
        first_valid_idx = np.where(~np.isnan(arr))[0]
        if len(first_valid_idx):
            arr[np.isnan(arr)] = arr[first_valid_idx[0]]
    return arr

acc = extract_acc(RESULTS_LOG)
n = len(acc)
x = np.arange(1, n + 1)
print(f"GLM-5 CIFAR-100: {n} iterations, first={acc[0]:.4f}, last={acc[-1]:.4f}, best={acc.max():.4f}")

best_so_far = np.maximum.accumulate(acc)

def moving_average(arr, window=5):
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]

smoothed = moving_average(acc, window=15)

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

COLOR = "#00A087"   # GLM green

ax.plot(x, acc,         color=COLOR,  alpha=0.30, linewidth=0.7, zorder=1, label="Accuracy")
ax.plot(x, smoothed,    color=COLOR,  alpha=0.85, linewidth=1.5, linestyle="--", zorder=2, label="Smoothed")
ax.plot(x, best_so_far, color=COLOR,  linewidth=2.2, zorder=3, label="Best So Far")

ax.set_xlabel("Iteration")
ax.set_ylabel("Accuracy")
ax.set_title("GLM-5")
ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
ax.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))

ymin = 0.0
ymax = min(1.0, np.max(acc) + 0.05)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
