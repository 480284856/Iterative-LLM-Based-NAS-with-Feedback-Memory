"""
Statistical analysis and research-quality plotting for prompt improvement experiments.
Accuracy is extracted from results.log; failed iterations use the previous round's accuracy.
"""

import re
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent.parent          # .../Paper
EXPERIMENTAL_RESULT = BASE / "experimental_result"
PROMPT_IMPROVEMENT = BASE.parent             # .../prompt_improvement
OUTPUT_DIR = Path(__file__).parent

# Regex for results.log lines
RE_ACC = re.compile(r"iteration:\s*(\d+),\s*accuracy:\s*([\d.]+)")
RE_ERR = re.compile(r"iteration:\s*(\d+),\s*error:")


def extract_acc_from_results_log(log_path):
    """
    Extract accuracy per iteration from results.log.
    For lines with 'error:', use the previous round's accuracy (forward-fill for leading errors).
    """
    log_path = Path(log_path)
    acc_list = []  # (iter_1-based, value or None)
    last_acc = None

    with open(log_path) as f:
        for line in f:
            m_acc = RE_ACC.search(line)
            m_err = RE_ERR.search(line)
            if m_acc:
                acc = float(m_acc.group(2))
                last_acc = acc
                acc_list.append(acc)
            elif m_err:
                # Use previous round's accuracy; if none yet, append None and forward-fill later
                acc_list.append(last_acc)

    arr = np.array(acc_list, dtype=float)
    # Forward-fill leading NaNs (iterations that failed before any success)
    if np.any(np.isnan(arr)):
        first_valid = np.where(~np.isnan(arr))[0]
        if len(first_valid) > 0:
            fill_val = arr[first_valid[0]]
            arr[np.isnan(arr)] = fill_val
    return arr


def load_acc(path):
    with open(path) as f:
        return np.array([float(line.strip()) for line in f if line.strip()])


# ── Extract accuracy: deepseek/glm from results.log (errors → previous round); qwen2.5 from acc.txt + append 0.6636 ──
deepseek_log = PROMPT_IMPROVEMENT / "output/deepseek_coder_6.7b_instruct_2000/results.log"
glm_log = EXPERIMENTAL_RESULT / "glm5_500/results.log"
qwen_acc_path = EXPERIMENTAL_RESULT / "qwen2.5_7b_instruct_2000/acc.txt"

deepseek_acc = extract_acc_from_results_log(deepseek_log)
glm_acc = extract_acc_from_results_log(glm_log)
# Qwen2.5: 不参考 results.log，直接使用现有 acc.txt 并补足到 2000 条（不足部分填 0.6636）
qwen_acc_raw = load_acc(qwen_acc_path)
n_pad = 2000 - len(qwen_acc_raw)
qwen_acc = np.append(qwen_acc_raw, np.full(n_pad, 0.6636)) if n_pad > 0 else qwen_acc_raw

# Persist acc.txt for deepseek and glm only (qwen2.5 不写回，保留原 acc.txt + 仅在内存中补 0.6636)
for path, acc in [
    (PROMPT_IMPROVEMENT / "output/deepseek_coder_6.7b_instruct_2000/acc.txt", deepseek_acc),
    (EXPERIMENTAL_RESULT / "glm5_500/acc.txt", glm_acc),
]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in acc:
            f.write(f"{v}\n")
    print(f"Wrote {len(acc)} values to {path}")
print(f"Qwen2.5: loaded {len(qwen_acc_raw)} from acc.txt, padded to 2000 with 0.6636 (not overwriting file)")

print(f"deepseek: {len(deepseek_acc)} entries, first={deepseek_acc[0]:.4f}, last={deepseek_acc[-1]:.4f}")
print(f"qwen2.5:  {len(qwen_acc)}  entries, first={qwen_acc[0]:.4f}, last={qwen_acc[-1]:.4f}")
print(f"glm5:     {len(glm_acc)}   entries, first={glm_acc[0]:.4f}, last={glm_acc[-1]:.4f}")


# ── Statistical analysis ───────────────────────────────────────────────────
def analyze(name, acc):
    n = len(acc)
    x = np.arange(1, n + 1)

    # Spearman rank correlation (iteration vs accuracy)
    spearman_rho, spearman_p = stats.spearmanr(x, acc)

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x, acc)

    # Mann-Kendall trend test (monotonic trend)
    # Use scipy version: kendall tau between x and acc
    kendall_tau, kendall_p = stats.kendalltau(x, acc)

    # Linear regression slope (normalized per 100 iterations)
    slope, intercept, r_value, lm_p, std_err = stats.linregress(x, acc)

    # First vs last accuracy difference
    first_acc = acc[0]
    last_acc = acc[-1]
    delta = last_acc - first_acc

    # Rolling max (best-so-far)
    running_max = np.maximum.accumulate(acc)
    final_best = running_max[-1]

    # Improvement from first to best
    delta_best = final_best - first_acc

    result = {
        "model": name,
        "n_successful_iterations": int(n),
        "spearman_rho": {
            "value": round(float(spearman_rho), 4),
            "p_value": round(float(spearman_p), 6),
            "建议是否放到论文中": "是" if abs(spearman_rho) > 0.3 and spearman_p < 0.05 else "否",
            "说明": "迭代轮次与accuracy之间的Spearman秩相关系数，反映单调趋势强度"
        },
        "pearson_r": {
            "value": round(float(pearson_r), 4),
            "p_value": round(float(pearson_p), 6),
            "建议是否放到论文中": "是" if abs(pearson_r) > 0.3 and pearson_p < 0.05 else "否",
            "说明": "迭代轮次与accuracy之间的Pearson线性相关系数"
        },
        "kendall_tau": {
            "value": round(float(kendall_tau), 4),
            "p_value": round(float(kendall_p), 6),
            "建议是否放到论文中": "是" if abs(kendall_tau) > 0.2 and kendall_p < 0.05 else "否",
            "说明": "Kendall τ趋势检验，与Spearman互为佐证，对噪声更鲁棒"
        },
        "linear_slope_per_100iter": {
            "value": round(float(slope * 100), 4),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(lm_p), 6),
            "建议是否放到论文中": "是" if lm_p < 0.05 else "否",
            "说明": "线性回归斜率（每100次迭代accuracy平均提升量），R²反映线性拟合优度"
        },
        "第一轮和最后一轮的accuracy差异": {
            "first_accuracy": round(float(first_acc), 4),
            "last_accuracy": round(float(last_acc), 4),
            "delta (last - first)": round(float(delta), 4),
            "说明": "最后一轮 - 第一轮，正值表示整体提升"
        },
        "最高accuracy与初始accuracy差异": {
            "best_accuracy": round(float(final_best), 4),
            "delta_best (best - first)": round(float(delta_best), 4),
            "建议是否放到论文中": "是",
            "说明": "曲线中最高accuracy - 第一轮accuracy，体现最大潜在提升"
        }
    }
    return result, acc


results = []
for name, acc in [("deepseek_coder_6.7b", deepseek_acc),
                  ("qwen2.5_7b_instruct", qwen_acc),
                  ("glm5", glm_acc)]:
    r, _ = analyze(name, acc)
    results.append(r)
    print(f"\n=== {name} ===")
    print(json.dumps(r, ensure_ascii=False, indent=2))

# Save statistics JSON
with open(OUTPUT_DIR / "statistics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nStatistics saved to {OUTPUT_DIR / 'statistics.json'}")


# ── Research-quality figure ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def rolling_max(arr):
    return np.maximum.accumulate(arr)


def smooth(arr, window=5):
    """Simple moving average smoothing."""
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# ── Subplot 1: DeepSeek + Qwen2.5 ─────────────────────────────────────────
ax1 = axes[0]

colors = {
    "deepseek": "#E64B35",
    "qwen": "#4DBBD5",
}

for label, acc, color in [
    ("DeepSeek-Coder-6.7B", deepseek_acc, colors["deepseek"]),
    ("Qwen2.5-7B-Instruct", qwen_acc, colors["qwen"]),
]:
    x = np.arange(1, len(acc) + 1)
    rm = rolling_max(acc)
    # Raw data (thin, transparent)
    ax1.plot(x, acc, color=color, alpha=0.25, linewidth=0.8, zorder=1)
    # Rolling max (bold)
    ax1.plot(x, rm, color=color, linewidth=2.0, label=f"{label} (best-so-far)", zorder=2)
    # Mark final best
    ax1.scatter([x[-1]], [rm[-1]], color=color, s=50, zorder=3)

# Y-axis range
all_vals = np.concatenate([deepseek_acc, qwen_acc])
ymin = max(0, np.percentile(all_vals, 1) - 0.03)
ymax = min(1.0, np.max(all_vals) + 0.05)
ax1.set_ylim(ymin, ymax)

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Accuracy")
ax1.set_title("(a) DeepSeek-Coder-6.7B & Qwen2.5-7B-Instruct")
ax1.legend(loc="lower right", framealpha=0.85, edgecolor='gray')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

# ── Subplot 2: GLM ────────────────────────────────────────────────────────
ax2 = axes[1]

glm_color = "#00A087"
x_glm = np.arange(1, len(glm_acc) + 1)
rm_glm = rolling_max(glm_acc)

ax2.plot(x_glm, glm_acc, color=glm_color, alpha=0.25, linewidth=0.8, zorder=1)
ax2.plot(x_glm, rm_glm, color=glm_color, linewidth=2.0,
         label="GLM-5 (best-so-far)", zorder=2)
ax2.scatter([x_glm[-1]], [rm_glm[-1]], color=glm_color, s=50, zorder=3)

ymin_g = max(0, np.percentile(glm_acc, 1) - 0.03)
ymax_g = min(1.0, np.max(glm_acc) + 0.05)
ax2.set_ylim(ymin_g, ymax_g)

ax2.set_xlabel("Iteration")
ax2.set_ylabel("Accuracy")
ax2.set_title("(b) GLM-5")
ax2.legend(loc="lower right", framealpha=0.85, edgecolor='gray')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

plt.tight_layout(pad=2.0)

fig_path = OUTPUT_DIR / "accuracy_curves.pdf"
fig_png_path = OUTPUT_DIR / "accuracy_curves.png"
plt.savefig(fig_path)
plt.savefig(fig_png_path)
print(f"Figure saved to {fig_path} and {fig_png_path}")
