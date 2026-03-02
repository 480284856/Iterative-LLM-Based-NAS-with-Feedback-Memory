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

MODELS=("deepseek-ai/deepseek-coder-6.7b-instruct" "Qwen/Qwen2.5-7B-Instruct")
MODEL_NAMES=("deepseek_coder_6.7b" "qwen2.5")
DATASETS=("cifar10" "imagenette")   # 新增 imagenette 数据集
ITERATIONS=100
TARGET=1.0

for DATASET in "${DATASETS[@]}"; do
    for i in "${!MODELS[@]}"; do
        BASE_MODEL="${MODELS[$i]}"
        MODEL_NAME="${MODEL_NAMES[$i]}"

        # ========================================
        # Ablation 1: w/o Prompt Improver (Random Search Baseline)
        # ========================================
        echo "========================================"
        echo "Running Ablation 1: w/o Prompt Improver for $MODEL_NAME on $DATASET"
        echo "========================================"
        python pipeline.py  --model "$BASE_MODEL" \
                            --dataset $DATASET \
                            --max-iterations $ITERATIONS \
                            --target-accuracy $TARGET \
                            --no-improver \
                            --no-reference \
                            --output-dir ./output/${MODEL_NAME}_${DATASET}_ablation_no_improver_no_reference_2 \
        | tee ./log/${MODEL_NAME}_${DATASET}_ablation_no_improver_no_reference_2.log
    done
done

echo "========================================"
echo "All ablation studies complete!"
echo "Results saved in ./output/*_ablation_*"
echo "Logs saved in ./log/*_ablation_*.log"
echo "========================================"
