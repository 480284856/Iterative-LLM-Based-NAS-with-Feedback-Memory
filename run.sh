# Set HF_TOKEN in environment before running

CONDA_PATH="/home/$(whoami)/miniconda3"

if [ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
else
    echo "错误：未找到 conda 初始化脚本，请检查 CONDA_PATH 是否正确！"
    exit 1
fi


conda activate promptimprove

cd /home/gu/Desktop/prompt_improvement

python pipeline.py  --model zai-org/glm-4-9b \
                    --max-iterations 100 \
                    --output-dir /home/gu/Desktop/prompt_improvement/output/glm4_9b

python pipeline.py  --model deepseek-ai/deepseek-coder-6.7b-instruct \
                    --max-iterations 100 \
                    --output-dir /home/gu/Desktop/prompt_improvement/output/deepseek_coder_6.7b_instruct

python pipeline.py  --model codellama/CodeLlama-7b-Python-hf \
                    --max-iterations 100 \
                    --output-dir /home/gu/Desktop/prompt_improvement/output/codellama_7b_Python_hf
