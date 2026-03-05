"""
Statistical analysis and research-quality plotting for prompt improvement experiments.
Accuracy is extracted from results.log; failed iterations use the previous round's accuracy.
"""

import re
import numpy as np
import json
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent.parent.parent   # .../Paper
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


# ── Load accuracy arrays ──────────────────────────────────────────────────
# DeepSeek: extracted from results.log (2000 iterations, errors filled with prev acc)
deepseek_log = EXPERIMENTAL_RESULT / "cifar10/deepseek_coder_6.7b_instruct_2000/results.log"
deepseek_acc = extract_acc_from_results_log(deepseek_log)

# Qwen2.5: extracted from full 56k-line log (2000 iterations)
qwen_acc_path = EXPERIMENTAL_RESULT / "cifar10/qwen2.5_7b_instruct_2000/qwen2.5_7b_instruct_2000_acc.txt"
qwen_acc = load_acc(qwen_acc_path)

# GLM5: all-successful results.log (100 iterations, no errors)
glm_log = EXPERIMENTAL_RESULT / "cifar10/glm5_cifar10/results.log"
glm_acc = extract_acc_from_results_log(glm_log)

# Count truly-successful (non-filled) iterations per model
def count_successful_from_log(log_path):
    """Count lines with 'accuracy:' in results.log (true successes, not error-filled)."""
    count = 0
    with open(log_path) as f:
        for line in f:
            if RE_ACC.search(line):
                count += 1
    return count

deepseek_n_success = count_successful_from_log(deepseek_log)
qwen_n_success = 376   # counted from full log in extract_qwen_acc.py
glm_n_success = count_successful_from_log(glm_log)

print(f"deepseek: {len(deepseek_acc)} total iters, {deepseek_n_success} successful, "
      f"first={deepseek_acc[0]:.4f}, last={deepseek_acc[-1]:.4f}")
print(f"qwen2.5:  {len(qwen_acc)} total iters, {qwen_n_success} successful, "
      f"first={qwen_acc[0]:.4f}, last={qwen_acc[-1]:.4f}")
print(f"glm5:     {len(glm_acc)} total iters, {glm_n_success} successful, "
      f"first={glm_acc[0]:.4f}, last={glm_acc[-1]:.4f}")

print(f"deepseek: {len(deepseek_acc)} entries, first={deepseek_acc[0]:.4f}, last={deepseek_acc[-1]:.4f}")
print(f"qwen2.5:  {len(qwen_acc)}  entries, first={qwen_acc[0]:.4f}, last={qwen_acc[-1]:.4f}")
print(f"glm5:     {len(glm_acc)}   entries, first={glm_acc[0]:.4f}, last={glm_acc[-1]:.4f}")


# ── Statistical analysis ───────────────────────────────────────────────────
def analyze(name, acc, n_success=None):
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
        "n_total_iterations": int(n),
        "n_successful_iterations": int(n_success) if n_success is not None else int(n),
        "success_rate": round(float(n_success / n) if n_success is not None else 1.0, 4),
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
for name, acc, n_succ in [
    ("deepseek_coder_6.7b", deepseek_acc, deepseek_n_success),
    ("qwen2.5_7b_instruct", qwen_acc, qwen_n_success),
    ("glm5", glm_acc, glm_n_success),
]:
    r, _ = analyze(name, acc, n_success=n_succ)
    results.append(r)
    print(f"\n=== {name} ===")
    print(json.dumps(r, ensure_ascii=False, indent=2))

# Save statistics JSON
with open(OUTPUT_DIR / "statistics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nStatistics saved to {OUTPUT_DIR / 'statistics.json'}")
