#!/bin/bash

# =================================================================
# 脚本功能：同步 prompt_improvement_for_ablation 目录到远程主机
# 使用方式：bash sync_to_remotes.sh
# =================================================================

# 定义本地源目录（绝对路径）
# 注意：末尾不加斜杠会连同目录名一起同步到远程的 ~ 下
SOURCE_DIR="/home/qu/Desktop/nngpt/prompt_improvement/prompt_improvement_for_ablation"

# 定义目标主机名（对应 ~/.ssh/config 中的 Host）
HOSTS=("gu@10.85.13.59")

# 同步中需要排除的文件/文件夹
EXCLUDES=(
    "--exclude=.git"
    "--exclude=__pycache__"
    "--exclude=log"
    "--exclude=output"
    "--exclude=output_old"
    "--exclude=data"
)

# 开始循环同步
for HOST in "${HOSTS[@]}"; do
    echo "----------------------------------------------------"
    echo "正在同步到主机: $HOST ..."
    
    # 执行 rsync
    # -a: 归档模式 (保留权限、链接等)
    # -v: 显示详细过程
    # -z: 传输时压缩
    rsync -avz "${EXCLUDES[@]}" "$SOURCE_DIR" "$HOST":~/
    
    if [ $? -eq 0 ]; then
        echo "✅ 主机 $HOST 同步完成。"
    else
        echo "❌ 主位 $HOST 同步失败，请检查 SSH 连接。"
    fi
done

echo "----------------------------------------------------"
echo "所有任务执行完毕。"
