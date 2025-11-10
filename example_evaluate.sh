#!/bin/bash
# 评估Checkpoint的示例脚本
# Example script for evaluating checkpoints

# 设置checkpoint路径（可以根据实际情况修改）
CHECKPOINT_PATH="/data/cdq/current_project/zo-test-cdq/logs/parallel_sweep_20251109_221054/experiments/Instruct_20M_full_bs32_q8_bp1_optmudamw_lr1e-3_ct0.01_ns10.0/logs/parallel_sweep_20251109_221054/experiments/Instruct_20M_full_bs32_q8_bp1_optmudamw_lr1e-3_ct0.01_ns10.0/checkpoint"

# 示例1: 评估在cosmopedia数据集上的loss
echo "========================================="
echo "示例1: 评估在cosmopedia数据集上"
echo "========================================="
python evaluate_checkpoint.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset cosmopedia \
    --batch_size 8 \
    --block_size 128 \
    --max_samples 10000 \
    --output_file results/eval_cosmopedia.json

# 示例2: 评估在fineweb-edu-10bt数据集上的loss
echo ""
echo "========================================="
echo "示例2: 评估在fineweb-edu-10bt数据集上"
echo "========================================="
python evaluate_checkpoint.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset fineweb-edu-10bt \
    --batch_size 4 \
    --block_size 128 \
    --max_samples 5000 \
    --output_file results/eval_fineweb_edu.json

# 示例3: 评估在wikitext-103数据集上的loss
echo ""
echo "========================================="
echo "示例3: 评估在wikitext-103数据集上"
echo "========================================="
python evaluate_checkpoint.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset wikitext-103 \
    --batch_size 8 \
    --block_size 128 \
    --output_file results/eval_wikitext.json

# 示例4: 评估在tinystories数据集上的loss（快速测试）
echo ""
echo "========================================="
echo "示例4: 评估在tinystories数据集上（快速测试）"
echo "========================================="
python evaluate_checkpoint.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset tinystories \
    --batch_size 16 \
    --block_size 128 \
    --max_samples 5000 \
    --max_batches 100 \
    --output_file results/eval_tinystories.json

echo ""
echo "========================================="
echo "评估完成！结果保存在 results/ 目录"
echo "========================================="

