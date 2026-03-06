# Set your API key in the environment before running, e.g.:
#   export SiliconCloud_Key="your-key"
CONDA_PATH="/home/$(whoami)/miniconda3"

if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
else
    echo "错误：未找到 conda 初始化脚本，请检查 CONDA_PATH 是否正确！"
    exit 1
fi

SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
cd "$SCRIPT_DIR"
mkdir -p ./output ./log

MODELS=("Qwen/Qwen2.5-7B-Instruct" "deepseek-ai/deepseek-coder-6.7b-instruct")
MODEL_NAMES=("qwen2.5_7b_instruct" "deepseek_coder_6.7b")
DATASETS=("cifar10" "cifar100" "imagenette")
ITERATIONS=2000
TARGET=1.0

for DATASET in "${DATASETS[@]}"; do
    for i in "${!MODELS[@]}"; do
        MODEL="${MODELS[$i]}"
        MODEL_NAME="${MODEL_NAMES[$i]}"

        echo "========================================"
        echo "Running $MODEL_NAME on $DATASET"
        echo "========================================"
        python pipeline.py  --model "$MODEL" \
                            --dataset "$DATASET" \
                            --max-iterations $ITERATIONS \
                            --target-accuracy $TARGET \
                            --output-dir ./output/${MODEL_NAME}_${DATASET} \
        | tee ./log/${MODEL_NAME}_${DATASET}.log
    done
done

# python pipeline.py  --model Pro/zai-org/GLM-5 \
#                     --remote \
#                     --dataset cifar100 \
#                     --max-iterations 100 \
#                     --target-accuracy 1 \
#                     --output-dir ./output/glm5_cifar100 \
# | tee ./log/glm5_cifar100.log

# python pipeline.py  --model Pro/zai-org/GLM-5 \
#                     --remote \
#                     --dataset cifar10 \
#                     --max-iterations 100 \
#                     --target-accuracy 1 \
#                     --output-dir ./output/glm5_cifar10 \
# | tee ./log/glm5_cifar10.log

echo "========================================"
echo "All experiments complete!"
echo "Results saved in ./output/"
echo "Logs saved in ./log/"
echo "========================================"