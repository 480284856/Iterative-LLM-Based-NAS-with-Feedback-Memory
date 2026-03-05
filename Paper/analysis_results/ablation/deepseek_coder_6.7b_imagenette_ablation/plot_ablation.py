import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
BASE_LOG_FILE = HERE / "base_results.log"
NO_IMP_FILE  = HERE / "no_improver.log"

OUT_PNG = HERE / "deepseek_coder_6.7b_imagenette_ablation.png"
OUT_PDF = HERE / "deepseek_coder_6.7b_imagenette_ablation.pdf"

# ── Extraction Logic ─────────────────────────────────────────────────────────
RE_ACC = re.compile(r"iteration:\s*(\d+),\s*accuracy:\s*([\d.]+)")
RE_ERR = re.compile(r"iteration:\s*(\d+),\s*error:")

def extract_acc_from_log(log_path, max_iter=100):
    raw = []
    last_acc = np.nan
    with open(log_path) as f:
        for line in f:
            if m := RE_ACC.search(line):
                last_acc = float(m.group(2))
                raw.append(last_acc)
            elif RE_ERR.search(line):
                raw.append(last_acc)
            
            if len(raw) >= max_iter:
                break
                
    arr = np.array(raw, dtype=float)
    if np.any(np.isnan(arr)):
        first_valid_idx = np.where(~np.isnan(arr))[0]
        if len(first_valid_idx):
            # Fill preceding errors with 0.0 instead of the first valid accuracy
            arr[:first_valid_idx[0]] = 0.0
        else:
            arr = np.zeros_like(arr)
            
    if len(arr) < max_iter:
        last_val = arr[-1] if len(arr) > 0 else 0.0
        arr = np.pad(arr, (0, max_iter - len(arr)), mode='constant', constant_values=last_val)
        
    return arr

def extract_acc_from_txt(txt_path, max_iter=100):
    raw = []
    with open(txt_path) as f:
        for line in f:
            val = line.strip()
            if val:
                raw.append(float(val))
            if len(raw) >= max_iter:
                break
    
    arr = np.array(raw, dtype=float)
    if len(arr) < max_iter:
        last_val = arr[-1] if len(arr) > 0 else 0.0
        arr = np.pad(arr, (0, max_iter - len(arr)), mode='constant', constant_values=last_val)
        
    return arr

acc_base = extract_acc_from_log(BASE_LOG_FILE, 100)
acc_no_imp  = extract_acc_from_log(NO_IMP_FILE, 100)

def moving_average(arr, window=15):
    """Centered moving average with edge-padding."""
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]

smooth_base = moving_average(acc_base, window=1)
smooth_no_imp  = moving_average(acc_no_imp, window=1)

x = np.arange(1, 101)

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

fig, ax = plt.subplots(figsize=(8, 4.5))

# Use distinct but professional colors
C_BASE = "#D62728"   # Red (Baseline)
C_NOIMP  = "#2CA02C" # Green

ax.plot(x, smooth_base,    color=C_BASE,   linewidth=2.2, label="Full Functionality")
ax.plot(x, smooth_no_imp,  color=C_NOIMP,  linewidth=2.0, linestyle="--", label="No Feedback")

ax.set_xlabel("Iteration")
ax.set_ylabel("Accuracy")
ax.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
ax.grid(True, linestyle="--", alpha=0.4)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))

# Dynamic y limits
all_smooth = np.concatenate([smooth_base, smooth_no_imp])
ymin = max(0.0, np.min(all_smooth) - 0.05)
ymax = min(1.0, np.max(all_smooth) + 0.05)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
fig.savefig(OUT_PNG)
fig.savefig(OUT_PDF)
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
