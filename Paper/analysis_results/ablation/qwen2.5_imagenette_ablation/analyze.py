import re
import numpy as np

RE_ACC = re.compile(r"iteration:\s*(\d+),\s*accuracy:\s*([\d.]+)")
RE_ERR = re.compile(r"iteration:\s*(\d+),\s*error:")

def analyze_log(log_path, max_iter=100):
    raw_accs = []
    success_count = 0
    error_count = 0
    with open(log_path) as f:
        for line in f:
            if m := RE_ACC.search(line):
                raw_accs.append(float(m.group(2)))
                success_count += 1
            elif RE_ERR.search(line):
                error_count += 1
            if success_count + error_count >= max_iter:
                break
                
    if not raw_accs:
        return 0, 0, 0, 0
    return max(raw_accs), np.mean(raw_accs[-5:]), success_count, error_count

qw_base = analyze_log("base_results.log")
qw_no_imp = analyze_log("no_improver.log")

print("Qwen Base Iter 1-100    : Peak={:.3f}, last 5 mean={:.3f}, successes={}, errors={}".format(*qw_base))
print("Qwen w/o Improver       : Peak={:.3f}, last 5 mean={:.3f}, successes={}, errors={}".format(*qw_no_imp))
