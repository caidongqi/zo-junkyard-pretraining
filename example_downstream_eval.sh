#!/bin/bash
# 下游任务评估示例脚本
# Example script for downstream task evaluation

echo "=========================================="
echo "下游任务评估示例 (Downstream Task Evaluation Examples)"
echo "=========================================="

# 设置变量（请根据实际情况修改）
# Set variables (modify according to your setup)
CHECKPOINT_PATH="/data/cdq/current_project/zo-test-cdq/logs/parallel_sweep_20251109_221054/experiments/Instruct_20M_full_bs32_q8_bp1_optmudamw_lr1e-3_ct0.01_ns10.0/logs/parallel_sweep_20251109_221054/experiments/Instruct_20M_full_bs32_q8_bp1_optmudamw_lr1e-3_ct0.01_ns10.0/checkpoint"
OUTPUT_DIR="results/downstream_eval"

mkdir -p $OUTPUT_DIR

echo ""
echo "示例 1: 文本生成评估"
echo "Example 1: Text Generation Evaluation"
echo "=========================================="

python downstream_evaluation.py \
    --task generation \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --prompts "Hello, how are you?" "What is AI?" "Once upon a time" \
    --max_length 80 \
    --temperature 0.8 \
    --num_generations 3 \
    --output_file "$OUTPUT_DIR/generation_eval.json"

echo ""
echo "示例 2: SST-2分类任务评估（微调）"
echo "Example 2: SST-2 Classification with Finetuning"
echo "=========================================="

python downstream_evaluation.py \
    --task sst2 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --finetune \
    --finetune_epochs 3 \
    --finetune_lr 1e-4 \
    --batch_size 32 \
    --output_file "$OUTPUT_DIR/sst2_finetuned.json"

echo ""
echo "示例 3: SST-2零样本评估（不微调）"
echo "Example 3: SST-2 Zero-shot Evaluation (No Finetuning)"
echo "=========================================="

python downstream_evaluation.py \
    --task sst2 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --no_finetune \
    --batch_size 32 \
    --output_file "$OUTPUT_DIR/sst2_zeroshot.json"

echo ""
echo "示例 4: 对比预训练模型 vs 从头训练模型"
echo "Example 4: Compare Pretrained vs From-Scratch Model"
echo "=========================================="

python downstream_evaluation.py \
    --task sst2 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --compare_from_scratch \
    --model_size 200M \
    --finetune \
    --finetune_epochs 3 \
    --batch_size 32 \
    --output_file "$OUTPUT_DIR/comparison_pretrained_vs_scratch.json"

echo ""
echo "示例 5: 对比两个不同checkpoint"
echo "Example 5: Compare Two Different Checkpoints"
echo "=========================================="

# 假设有两个checkpoint（请根据实际情况修改）
CHECKPOINT_1="$CHECKPOINT_PATH"
CHECKPOINT_2="logs/parallel_sweep_20251106_123930/experiments/ZO_full_bs2_q2_bpN_A_optmudamw_lr1e-3/checkpoint"

python downstream_evaluation.py \
    --task sst2 \
    --checkpoint_path "$CHECKPOINT_1" \
    --baseline_checkpoint "$CHECKPOINT_2" \
    --finetune \
    --finetune_epochs 3 \
    --batch_size 32 \
    --output_file "$OUTPUT_DIR/comparison_two_checkpoints.json"

echo ""
echo "示例 6: 完整评估（生成 + 分类）"
echo "Example 6: Full Evaluation (Generation + Classification)"
echo "=========================================="

python downstream_evaluation.py \
    --task all \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --compare_from_scratch \
    --prompts "Hello!" "What is machine learning?" "The future of AI" \
    --max_length 100 \
    --num_generations 2 \
    --finetune \
    --finetune_epochs 2 \
    --batch_size 32 \
    --output_file "$OUTPUT_DIR/full_evaluation.json"

echo ""
echo "=========================================="
echo "所有示例运行完成！"
echo "All examples completed!"
echo "结果保存在: $OUTPUT_DIR"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="

