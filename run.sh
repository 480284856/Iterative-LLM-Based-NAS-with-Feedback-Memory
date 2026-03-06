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

python pipeline.py  --model Qwen/Qwen2.5-7B-Instruct \
                    --dataset cifar100 \
                    --max-iterations 2000 \
                    --target-accuracy 1.0 \
                    --output-dir ./output/qwen2.5_7b_instruct_cifar100_2000_latest \
| tee ./log/qwen2.5_7b_instruct_cifar100_2000_latest.log

python pipeline.py  --model deepseek-ai/deepseek-coder-6.7b-instruct \
                    --dataset cifar100 \
                    --max-iterations 2000 \
                    --target-accuracy 1.0 \
                    --output-dir ./output/deepseek_coder_6.7b_instruct_cifar100_2000 \
| tee ./log/deepseek_coder_6.7b_instruct_cifar100_2000.log

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

# copy the output to the data directory
# chmod +w ../data
# cp -rf ./output/glm5_cifar100 ../data/glm5_cifar100
# cp -rf ./output/glm5_cifar10 ../data/glm5_cifar10
# chmod -w ../data