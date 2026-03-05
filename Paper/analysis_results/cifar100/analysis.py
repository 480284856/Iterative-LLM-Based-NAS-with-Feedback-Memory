"""
Statistical analysis for prompt improvement experiments (CIFAR-100 dataset).
Accuracy is extracted from results.log; failed iterations use the previous round's accuracy.
"""

import re
import numpy as np
import json
from scipy import stats
from pathlib import Path

BASE = Path(__file__).parent.parent.parent   # .../Paper
EXPERIMENTAL_RESULT = BASE / "experimental_result"
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
    acc_list = []
    last_acc = None

    if not log_path.exists():
        return np.array([], dtype=float)

    with open(log_path) as f:
        for line in f:
            m_acc = RE_ACC.search(line)
            m_err = RE_ERR.search(line)
            if m_acc:
                acc = float(m_acc.group(2))
                last_acc = acc
                acc_list.append(acc)
            elif m_err:
                acc_list.append(last_acc)

    arr = np.array(acc_list, dtype=float)
    if len(arr) > 0 and np.any(np.isnan(arr)):
        first_valid = np.where(~np.isnan(arr))[0]
        if len(first_valid) > 0:
            fill_val = arr[first_valid[0]]
            arr[np.isnan(arr)] = fill_val
    return arr


def count_successful_from_log(log_path):
    """Count lines with 'accuracy:' in results.log (true successes, not error-filled)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return 0
    count = 0
    with open(log_path) as f:
        for line in f:
            if RE_ACC.search(line):
                count += 1
    return count


# ── Load accuracy arrays ──────────────────────────────────────────────────
# DeepSeek: extracted from results.log
deepseek_log = EXPERIMENTAL_RESULT / "cifar100/deepseek_coder_6.7b_instruct_2000/results.log"
deepseek_acc = extract_acc_from_results_log(deepseek_log)
deepseek_n_success = count_successful_from_log(deepseek_log)

# Qwen2.5: extracted from results.log
qwen_log = EXPERIMENTAL_RESULT / "cifar100/qwen2.5_7b_instruct_2000/results.log"
qwen_acc = extract_acc_from_results_log(qwen_log)
qwen_n_success = count_successful_from_log(qwen_log)

# GLM5: extracted from results.log
glm_log = EXPERIMENTAL_RESULT / "cifar100/glm5_100/results.log"
glm_acc = extract_acc_from_results_log(glm_log)
glm_n_success = count_successful_from_log(glm_log)


print(f"deepseek: {len(deepseek_acc)} total iters, {deepseek_n_success} successful")
if len(deepseek_acc) > 0:
    print(f"first={deepseek_acc[0]:.4f}, last={deepseek_acc[-1]:.4f}")

print(f"qwen2.5:  {len(qwen_acc)} total iters, {qwen_n_success} successful")
if len(qwen_acc) > 0:
    print(f"first={qwen_acc[0]:.4f}, last={qwen_acc[-1]:.4f}")

print(f"glm5:     {len(glm_acc)} total iters, {glm_n_success} successful")
if len(glm_acc) > 0:
    print(f"first={glm_acc[0]:.4f}, last={glm_acc[-1]:.4f}")


# ── Statistical analysis ───────────────────────────────────────────────────
def analyze(name, acc, n_success=None):
    n = len(acc)
    if n == 0:
        return None, acc

    x = np.arange(1, n + 1)

    spearman_rho, spearman_p = stats.spearmanr(x, acc)
    pearson_r, pearson_p = stats.pearsonr(x, acc)
    kendall_tau, kendall_p = stats.kendalltau(x, acc)
    slope, intercept, r_value, lm_p, std_err = stats.linregress(x, acc)

    first_acc = acc[0]
    last_acc = acc[-1]
    delta = last_acc - first_acc

    running_max = np.maximum.accumulate(acc)
    final_best = running_max[-1]
    delta_best = final_best - first_acc

    result = {
        "model": name,
        "n_total_iterations": int(n),
        "n_successful_iterations": int(n_success) if n_success is not None else int(n),
        "success_rate": round(float(n_success / n) if n_success is not None else 1.0, 4),
        "spearman_rho": {
            "value": round(float(spearman_rho), 4),
            "p_value": round(float(spearman_p), 6),
        },
        "pearson_r": {
            "value": round(float(pearson_r), 4),
            "p_value": round(float(pearson_p), 6),
        },
        "kendall_tau": {
            "value": round(float(kendall_tau), 4),
            "p_value": round(float(kendall_p), 6),
        },
        "linear_slope_per_100iter": {
            "value": round(float(slope * 100), 4),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(lm_p), 6),
        },
        "first_accuracy": round(float(first_acc), 4),
        "last_accuracy": round(float(last_acc), 4),
        "best_accuracy": round(float(final_best), 4),
        "delta_last_first": round(float(delta), 4),
        "delta_best_first": round(float(delta_best), 4),
    }
    return result, acc


results = []
for name, acc, n_succ in [
    ("deepseek_coder_6.7b", deepseek_acc, deepseek_n_success),
    ("qwen2.5_7b_instruct", qwen_acc, qwen_n_success),
    ("glm5", glm_acc, glm_n_success),
]:
    if len(acc) > 0:
        r, _ = analyze(name, acc, n_success=n_succ)
        if r is not None:
            results.append(r)
            print(f"\n=== {name} ===")
            print(json.dumps(r, ensure_ascii=False, indent=2))

# Save statistics JSON
output_json = OUTPUT_DIR / "statistics.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nStatistics saved to {output_json}")
