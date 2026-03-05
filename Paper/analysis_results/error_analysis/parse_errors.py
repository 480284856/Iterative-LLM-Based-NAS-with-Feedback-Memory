import os
from collections import defaultdict

def categorize_error(line):
    if "Code extraction failed" in line:
        return "Extraction Error (Syntax Error)"
    
    if "OutOfMemoryError" in line or "CUDA out of memory" in line:
        return "CUDA OutOfMemory"
    
    if "Code validation failed" in line:
        if "mat1 and mat2 shapes cannot be multiplied" in line or \
           "shape" in line and "is invalid for input" in line or \
           "Calculated output size" in line and "Output size is too small" in line or \
           "Input dimension should be at least" in line or \
           "size mismatch" in line.lower() or \
           "expected input" in line and "to have" in line and "channels" in line or \
           "tensor size" in line.lower() or \
           "given groups=" in line.lower():
            return "Tensor Shape Mismatch"
            
        if "NameError:" in line:
            return "NameError (Undefined Variables)"
            
        if "AttributeError:" in line:
            return "AttributeError (Undefined Modules)"
            
        if "TypeError:" in line:
            return "TypeError"
            
        if "RuntimeError:" in line:
            if "Expected all tensors to be on the same device" in line:
                return "Device Mismatch"
            return "Other Validation RuntimeError"
            
        if "SyntaxError:" in line:
            return "SyntaxError"
            
        return "Other Validation Error"
        
    if "RuntimeError" in line:
        return "Execution RuntimeError"
        
    return "Other Error"

def analyze_dataset(name, dataset_path):
    print(f"--- {name} ---")
    dataset_errors = defaultdict(int)
    total_errors = 0
    
    for model_dir in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        log_file = os.path.join(model_path, "results.log")
        if not os.path.exists(log_file):
            continue
            
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("iteration:") and ", error:" in line:
                    error_msg = line.split(", error:", 1)[1].strip()
                    if ", timestamp:" in error_msg:
                        error_msg = error_msg.rsplit(", timestamp:", 1)[0].strip()
                        
                    category = categorize_error(error_msg)
                    dataset_errors[category] += 1
                    total_errors += 1
                    
    print(f"Total Errors: {total_errors}")
    sorted_errors = sorted(dataset_errors.items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_errors:
        print(f"{cat}: {count} ({count/total_errors*100:.1f}%)")
    print("\n")

def main():
    base_dir = "/home/qu/Desktop/nngpt/prompt_improvement/Paper/experimental_result"
    cifar_dir = os.path.join(base_dir, "cifar10")
    if os.path.exists(cifar_dir): analyze_dataset("CIFAR-10", cifar_dir)
    imagenet_dir = os.path.join(base_dir, "imagenette")
    if os.path.exists(imagenet_dir): analyze_dataset("ImageNette", imagenet_dir)

if __name__ == "__main__":
    main()
