# Set SiliconCloud_Key in environment before running

CONDA_PATH="/home/$(whoami)/miniconda3"

if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
else
    echo "错误：未找到 conda 初始化脚本，请检查 CONDA_PATH 是否正确！"
    exit 1
fi

cd /home/gu/prompt_improvement
mkdir -p ./output ./log

python pipeline.py  --model Qwen/Qwen2.5-7B-Instruct \
                    --dataset imagenette \
                    --max-iterations 2000 \
                    --target-accuracy 1.0 \
                    --output-dir ./output/qwen2.5_7b_instruct_2000 \
| tee ./log/qwen2.5_7b_instruct_2000.log

python pipeline.py  --model deepseek-ai/deepseek-coder-6.7b-instruct \
                    --dataset imagenette \
                    --max-iterations 2000 \
                    --target-accuracy 1.0 \
                    --output-dir ./output/deepseek_coder_6.7b_instruct_2000 \
| tee ./log/deepseek_coder_6.7b_instruct_2000.log

python pipeline.py  --model Pro/zai-org/GLM-5 \
                    --remote \
                    --dataset imagenette \
                    --max-iterations 100 \
                    --target-accuracy 1 \
                    --output-dir ./output/glm5_500 \
| tee ./log/glm5_500.log