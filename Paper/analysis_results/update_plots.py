import os
import re

scripts = [
    "cifar10/deepseek/plot_deepseek.py",
    "cifar10/qwen2p5/plot_qwen.py",
    "cifar10/glm5/plot_glm5.py",
    "cifar100/deepseek/plot_deepseek.py",
    "cifar100/qwen2p5/plot_qwen.py",
    "cifar100/glm5/plot_glm5.py",
    "imagenette/deepseek/plot_deepseek.py",
    "imagenette/qwen2p5/plot_qwen2p5.py",
    "imagenette/glm5/plot_glm5.py",
    "ablation/deepseek_coder_6.7b_cifar10_ablation/plot_ablation.py",
    "ablation/deepseek_coder_6.7b_cifar100_ablation/plot_ablation.py",
    "ablation/deepseek_coder_6.7b_imagenette_ablation/plot_ablation.py"
]

for script_path in scripts:
    if not os.path.exists(script_path):
        print(f"File not found: {script_path}")
        continue
    
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. Update legends
    content = re.sub(r'label="Accuracy \(error.*?prev\)"', 'label="Accuracy"', content)
    content = re.sub(r'label="Smoothed \(w=\d+\)"', 'label="Smoothed"', content)
    
    # 2. Update titles (Remove " – Accuracy over XXX Iterations" or " - CIFAR... Accuracy over...")
    content = re.sub(r'ax\.set_title\(".*?DeepSeek-Coder-6\.7B-Instruct.*?"\)', 'ax.set_title("DeepSeek-Coder-6.7B-Instruct")', content)
    content = re.sub(r'ax\.set_title\(".*?Qwen2\.5-7B-Instruct.*?"\)', 'ax.set_title("Qwen2.5-7B-Instruct")', content)
    content = re.sub(r'ax\.set_title\(".*?GLM-5.*?"\)', 'ax.set_title("GLM-5")', content)
    
    # 3. Update fonts robustly in plt.rcParams.update
    content = re.sub(r'"font\.size":\s*\d+,', '"font.size": 22,', content)
    content = re.sub(r'"axes\.titlesize":\s*\d+,', '"axes.titlesize": 24,', content)
    content = re.sub(r'"axes\.labelsize":\s*\d+,', '"axes.labelsize": 22,', content)
    content = re.sub(r'"legend\.fontsize":\s*\d+,', '"legend.fontsize": 18,', content)
    content = re.sub(r'"xtick\.labelsize":\s*\d+,', '"xtick.labelsize": 18,', content)
    content = re.sub(r'"ytick\.labelsize":\s*\d+,', '"ytick.labelsize": 18,', content)
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Updated {script_path}")

