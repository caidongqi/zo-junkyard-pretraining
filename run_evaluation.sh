#!/bin/bash
# 快速运行checkpoint evaluation的脚本

# ============================================================
# Hugging Face 连接配置（解决SSL错误）
# ============================================================
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/.cache/huggingface
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
export DATASETS_CACHE=~/.cache/huggingface/datasets

# 如果需要使用代理，取消下面的注释并设置
# export HTTP_PROXY=http://your-proxy:port
# export HTTPS_PROXY=http://your-proxy:port

# ============================================================

# 默认checkpoint路径（你可以修改这个路径）
CHECKPOINT_PATH=""

# 如果命令行提供了checkpoint路径，使用命令行参数
if [ $# -ge 1 ]; then
    CHECKPOINT_PATH="$1"
else
    echo "错误: 请提供checkpoint路径"
    echo ""
    echo "使用方法:"
    echo "  ./run_evaluation.sh <checkpoint_path>"
    echo ""
    echo "示例:"
    echo "  ./run_evaluation.sh logs/path/to/checkpoint"
    exit 1
fi

echo "HF连接配置: $HF_ENDPOINT"
echo "=========================================="
echo "运行下游任务评估"
echo "=========================================="
echo "Checkpoint路径: $CHECKPOINT_PATH"
echo ""

# 检查checkpoint是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint目录不存在: $CHECKPOINT_PATH"
    exit 1
fi

# 运行下游任务评估 (SST-2, SQuAD, LAMBADA)
CUDA_VISIBLE_DEVICES=4 python evaluate_downstream_tasks.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --datasets sst2 squad lambada \
    --max_samples 128 \
    --block_size 256 \
    --device cuda

echo ""
echo "=========================================="
echo "评估完成!"
echo "=========================================="

