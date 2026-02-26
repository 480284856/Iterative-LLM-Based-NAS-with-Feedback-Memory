# Set SiliconCloud_Key in environment before running

CONDA_PATH="/home/$(whoami)/miniconda3"

if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
else
    echo "错误：未找到 conda 初始化脚本，请检查 CONDA_PATH 是否正确！"
    exit 1
fi

cd /home/qu/Desktop/nngpt/prompt_improvement/prompt_improvement_for_ablation
mkdir -p ./output ./log

BASE_MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
DATASET="imagenette"
ITERATIONS=100
TARGET=1.0

# ========================================
# Ablation 1: w/o Prompt Improver (Random Search Baseline)
# ========================================
echo "========================================"
echo "Running Ablation 1: w/o Prompt Improver"
echo "========================================"
python pipeline.py  --model $BASE_MODEL \
                    --dataset $DATASET \
                    --max-iterations $ITERATIONS \
                    --target-accuracy $TARGET \
                    --no-improver \
                    --output-dir ./output/ablation_no_improver \
| tee ./log/ablation_no_improver.log

# ========================================
# Ablation 2: w/o Reference Code (Best Implementation Anchor)
# ========================================
echo "========================================"
echo "Running Ablation 2: w/o Reference Code"
echo "========================================"
python pipeline.py  --model $BASE_MODEL \
                    --dataset $DATASET \
                    --max-iterations $ITERATIONS \
                    --target-accuracy $TARGET \
                    --no-reference \
                    --output-dir ./output/ablation_no_reference \
| tee ./log/ablation_no_reference.log

# ========================================
# Ablation 3: w/o History Memory
# ========================================
echo "========================================"
echo "Running Ablation 3: w/o History"
echo "========================================"
python pipeline.py  --model $BASE_MODEL \
                    --dataset $DATASET \
                    --max-iterations $ITERATIONS \
                    --target-accuracy $TARGET \
                    --no-history \
                    --output-dir ./output/ablation_no_history \
| tee ./log/ablation_no_history.log

echo "========================================"
echo "All ablation studies complete!"
echo "Results saved in ./output/ablation_*"
echo "Logs saved in ./log/ablation_*.log"
echo "========================================"
