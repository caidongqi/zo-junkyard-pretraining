#!/bin/bash

# FineWeb 数据集使用示例
# 本脚本展示如何使用 FineWeb 系列数据集进行训练

echo "=========================================="
echo "FineWeb 数据集使用示例"
echo "=========================================="
echo ""

# 示例 1: 使用 FineWeb-Edu 10BT 进行快速测试
echo "示例 1: 快速测试 (FineWeb-Edu 10BT + 20M 模型)"
echo "预计时间: 15-30 分钟"
echo "命令:"
echo "./parallel_sweep.sh --model-size 20M --dataset fineweb-edu-10bt --max-samples 10000 --epochs 5 --modes ZO --query-budgets 1,2,4 --parallel 2"
echo ""
echo "是否运行此示例? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    ./parallel_sweep.sh --model-size 20M --dataset fineweb-edu-10bt --max-samples 10000 --epochs 5 --modes ZO --query-budgets 1,2,4 --parallel 2
fi
echo ""

# 示例 2: 使用 FineWeb-Edu 进行标准实验
echo "示例 2: 标准实验 (FineWeb-Edu + 200M 模型)"
echo "预计时间: 2-4 小时"
echo "命令:"
echo "./parallel_sweep.sh --model-size 200M --dataset fineweb-edu --max-samples 30000 --epochs 10 --modes ZO,FO --parallel 4"
echo ""
echo "是否运行此示例? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    ./parallel_sweep.sh --model-size 200M --dataset fineweb-edu --max-samples 30000 --epochs 10 --modes ZO,FO --parallel 4
fi
echo ""

# 示例 3: 对比不同模型在 FineWeb 上的表现
echo "示例 3: 模型规模对比 (FineWeb-Edu 10BT)"
echo "预计时间: 1-2 小时"
echo "命令:"
echo "./parallel_sweep.sh --model-size 20M --dataset fineweb-edu-10bt --max-samples 20000 --epochs 10"
echo "./parallel_sweep.sh --model-size 200M --dataset fineweb-edu-10bt --max-samples 20000 --epochs 10"
echo ""
echo "是否运行此示例? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    echo "运行 20M 模型..."
    ./parallel_sweep.sh --model-size 20M --dataset fineweb-edu-10bt --max-samples 20000 --epochs 10
    echo "运行 200M 模型..."
    ./parallel_sweep.sh --model-size 200M --dataset fineweb-edu-10bt --max-samples 20000 --epochs 10
fi
echo ""

# 示例 4: 直接使用 Python 脚本
echo "示例 4: 直接使用 Python 脚本"
echo "命令:"
echo "python reproduce_zo_paper.py \\"
echo "    --mode ZO \\"
echo "    --scope full \\"
echo "    --query_budget_q 8 \\"
echo "    --learning_rate 1e-3 \\"
echo "    --epochs 10 \\"
echo "    --batch_size 4 \\"
echo "    --optimizer mudamw \\"
echo "    --model_size 200M \\"
echo "    --dataset fineweb-edu-10bt \\"
echo "    --max_samples 20000"
echo ""
echo "是否运行此示例? (y/n)"
read -r response
if [[ "$response" == "y" ]]; then
    python reproduce_zo_paper.py \
        --mode ZO \
        --scope full \
        --query_budget_q 8 \
        --learning_rate 1e-3 \
        --epochs 10 \
        --batch_size 4 \
        --optimizer mudamw \
        --model_size 200M \
        --dataset fineweb-edu-10bt \
        --max_samples 20000
fi
echo ""

echo "=========================================="
echo "更多信息请查看: README_model_dataset.md"
echo "=========================================="

