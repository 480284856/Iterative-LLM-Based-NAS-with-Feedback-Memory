#!/usr/bin/env python3
"""Generate a side-by-side accuracy curve figure for the paper."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

ACC_QWEN25 = REPO_ROOT / "output" / "output_qwen2p5-7b-instruct-3-fixed-seed" / "acc.txt"
ACC_QWEN3 = REPO_ROOT / "output" / "output_qwen3-4b-instruct-2507-2" / "acc.txt"
ACC_DEEPSEEK = REPO_ROOT / "output" / "deepseek_coder_6.7b_instruct" / "acc.txt"
ACC_QWEN25_CODER = REPO_ROOT / "output" / "output_qwen2p5_coder" / "acc.txt"
OUT_FILE = SCRIPT_DIR / "accuracy_curve.png"

WINDOW = 5


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    pad = w // 2
    x_pad = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.convolve(x_pad, np.ones(w) / w, mode="valid")


def plot_panel(ax, acc, title, ylim):
    n = len(acc)
    iterations = np.arange(1, n + 1, dtype=float)
    acc_smooth = moving_average(acc, WINDOW)

    ax.plot(iterations, acc, alpha=0.35, color="steelblue", linewidth=0.8, label="Raw")
    ax.plot(iterations, acc_smooth, color="coral", linewidth=2, label=f"Smoothed (w={WINDOW})")

    ax.set_xlabel("Successful evaluation index", fontsize=10)
    ax.set_ylabel("One-epoch accuracy", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, n)
    ax.set_ylim(ylim)


def main():
    acc_25 = np.loadtxt(ACC_QWEN25, dtype=np.float64)
    acc_3 = np.loadtxt(ACC_QWEN3, dtype=np.float64)
    acc_ds = np.loadtxt(ACC_DEEPSEEK, dtype=np.float64)
    acc_qc = np.loadtxt(ACC_QWEN25_CODER, dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ((ax1, ax2), (ax3, ax4)) = axes

    plot_panel(ax1, acc_25, "Qwen2.5-7B-Instruct (77 evals/100 iterations)", ylim=(0.30, 0.65))
    plot_panel(ax2, acc_3, "Qwen3-4B-Instruct (41 evals/100 iterations)", ylim=(0.05, 0.45))
    plot_panel(ax3, acc_ds, "DeepSeek-Coder-6.7B-Instruct (39 evals/100 iterations)", ylim=(0.30, 0.70))
    plot_panel(ax4, acc_qc, "Qwen2.5-Coder-7B-Instruct (56 evals/100 iterations)", ylim=(0.20, 0.70))

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_FILE}")


if __name__ == "__main__":
    main()
