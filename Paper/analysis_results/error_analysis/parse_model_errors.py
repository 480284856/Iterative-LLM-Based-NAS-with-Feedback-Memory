import os
from collections import defaultdict

def categorize_error(line):
    if "Code extraction failed" in line: return "Extraction Error"
    if "OutOfMemoryError" in line or "CUDA out of memory" in line: return "CUDA OOM"
    if "Code validation failed" in line:
        if any(x in line.lower() for x in ["mat1 and mat2", "shape", "output size is too small", "input dimension", "size mismatch", "expected input", "given groups="]):
            return "Shape Mismatch"
        if "NameError:" in line: return "NameError"
        if "AttributeError:" in line: return "AttributeError"
        if "TypeError:" in line: return "TypeError"
        if "RuntimeError:" in line:
            if "Expected all tensors to be on the same device" in line: return "Device Mismatch"
            return "Other Val Error"
        return "Other Val Error"
    return "Other Error"

def analyze_dataset_models(dataset_path):
    for model_dir in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_dir)
        if not os.path.isdir(model_path): continue
        log_file = os.path.join(model_path, "results.log")
        if not os.path.exists(log_file):
            log_file = os.path.join(model_path, f"{model_dir}.log")
            if not os.path.exists(log_file): continue
        
        errors = defaultdict(int)
        total = 0
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("iteration:") and ", error:" in line:
                    error_msg = line.split(", error:", 1)[1].strip()
                    if ", timestamp:" in error_msg: error_msg = error_msg.rsplit(", timestamp:", 1)[0].strip()
                    errors[categorize_error(error_msg)] += 1
                    total += 1
                elif line.startswith("[History] Updated entry for iteration ") and " with result: error: " in line:
                    error_msg = line.split(" with result: error: ", 1)[1].strip()
                    errors[categorize_error(error_msg)] += 1
                    total += 1
        
        print(f"Model: {model_dir} | Total Errors: {total}")
        for k, v in sorted(errors.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k}: {v} ({v/total*100:.1f}%)" if total else f"  {k}: 0")
    print("")

base_dir = "/home/qu/Desktop/nngpt/prompt_improvement/Paper/experimental_result"
print("=== CIFAR-10 ===")
analyze_dataset_models(os.path.join(base_dir, "cifar10"))
print("=== ImageNette ===")
analyze_dataset_models(os.path.join(base_dir, "imagenette"))
