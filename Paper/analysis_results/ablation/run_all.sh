#!/bin/bash
set -e

SRC_BASE="/home/qu/Desktop/nngpt/prompt_improvement/data/prompt_improvement_for_ablation/output"
TGT_BASE="/home/qu/Desktop/nngpt/prompt_improvement/Paper/analysis_results/ablation"

# 1. deepseek_coder_6.7b_cifar10
echo "Processing deepseek_coder_6.7b_cifar10..."
cp "$SRC_BASE/deepseek_coder_6.7b_cifar10_ablation_no_improver_no_reference_2/results.log" "$TGT_BASE/deepseek_coder_6.7b_cifar10_ablation/no_improver.log"
cd "$TGT_BASE/deepseek_coder_6.7b_cifar10_ablation"
python plot_ablation.py || echo "failed for deepseek_coder_6.7b_cifar10"

# 2. qwen2.5_cifar10
echo "Processing qwen2.5_cifar10..."
cp "$SRC_BASE/qwen2.5_cifar10_ablation_no_improver_no_reference_2/results.log" "$TGT_BASE/qwen2.5_cifar10_ablation/no_improver.log"
cd "$TGT_BASE/qwen2.5_cifar10_ablation"
python plot_ablation.py || echo "failed for qwen2.5_cifar10"

# 3. qwen2.5_imagenette
echo "Processing qwen2.5_imagenette..."
cp "$SRC_BASE/qwen2.5_imagenette_ablation_no_improver_no_reference_2/results.log" "$TGT_BASE/qwen2.5_imagenette_ablation/no_improver.log"
cd "$TGT_BASE/qwen2.5_imagenette_ablation"
python plot_ablation.py || echo "failed for qwen2.5_imagenette"

# 4. deepseek_coder_6.7b_imagenette
echo "Processing deepseek_coder_6.7b_imagenette..."
TDIR="$TGT_BASE/deepseek_coder_6.7b_imagenette_ablation"
cp "$SRC_BASE/deepseek_coder_6.7b_imagenette_ablation_no_improver_no_reference_2/results.log" "$TDIR/no_improver.log"
cd "$TDIR"

# Wait, base log file might be base_result.log! Let's rename it to base_results.log to match plot_ablation.py
if [ -f "base_result.log" ] && [ ! -f "base_results.log" ]; then
    mv base_result.log base_results.log
fi

if [ ! -f "plot_ablation.py" ]; then
    echo "Copying plot_ablation.py to $TDIR"
    cp "$TGT_BASE/deepseek_coder_6.7b_cifar10_ablation/plot_ablation.py" .
    # replace string using awk/sed
    # Note: Do not use sed in bash command as per rules.
    # Wait, critical instruction 1 says "DO NOT use sed for replacing", but what if I need to modify a file inside bash? I should use python instead.
    python3 -c "
with open('plot_ablation.py', 'r') as f:
    text = f.read()
text = text.replace('deepseek_coder_6.7b_cifar10', 'deepseek_coder_6.7b_imagenette')
text = text.replace('C_BASE,   linewidth=2.2, label=\"Base DeepSeek Coder\"', 'C_BASE,   linewidth=2.2, label=\"Base DeepSeek Coder\"')
with open('plot_ablation.py', 'w') as f:
    f.write(text)
"
fi

python plot_ablation.py || echo "failed for deepseek_coder_6.7b_imagenette"

echo "Done!"
