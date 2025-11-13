#!/bin/bash
# 基准测试评估示例脚本
# Example script for common benchmark evaluation

# 设置checkpoint路径（请根据实际情况修改）
CHECKPOINT_PATH="/data/cdq/current_project/zo-test-cdq/logs/parallel_sweep_20251112_003916/experiments/Instruct_20M_full_bs4_blk512_q8_bp1_optmudamw_lr1e-3_blend0.8_ct0.01_ns10.0/logs/parallel_sweep_20251112_003916/experiments/Instruct_20M_full_bs4_blk512_q8_bp1_optmudamw_lr1e-3_blend0.8_ct0.01_ns10.0/checkpoint"

# 设置GPU
export CUDA_VISIBLE_DEVICES=4


# ============================================================================
# 示例 1: 评估 ARC-Easy (零样本)
# Example 1: Evaluate ARC-Easy (zero-shot)
# ============================================================================
echo "Example 1: ARC-Easy Zero-shot Evaluation"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark arc_easy \
    --n_shot 0 \
    --output_file results/arc_easy_zeroshot.json


# ============================================================================
# 示例 2: 评估 ARC-Challenge (零样本)
# Example 2: Evaluate ARC-Challenge (zero-shot)
# ============================================================================
echo ""
echo "Example 2: ARC-Challenge Zero-shot Evaluation"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark arc_challenge \
    --n_shot 0 \
    --output_file results/arc_challenge_zeroshot.json


# ============================================================================
# 示例 3: 评估 HellaSwag (零样本)
# Example 3: Evaluate HellaSwag (zero-shot)
# ============================================================================
echo ""
echo "Example 3: HellaSwag Zero-shot Evaluation"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark hellaswag \
    --n_shot 0 \
    --output_file results/hellaswag_zeroshot.json


# ============================================================================
# 示例 4: 评估 WinoGrande (零样本)
# Example 4: Evaluate WinoGrande (zero-shot)
# ============================================================================
echo ""
echo "Example 4: WinoGrande Zero-shot Evaluation"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark winogrande \
    --n_shot 0 \
    --output_file results/winogrande_zeroshot.json


# ============================================================================
# 示例 5: 评估 PIQA (零样本)
# Example 5: Evaluate PIQA (zero-shot)
# ============================================================================
echo ""
echo "Example 5: PIQA Zero-shot Evaluation"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark piqa \
    --n_shot 0 \
    --output_file results/piqa_zeroshot.json


# ============================================================================
# 示例 6: 评估所有基准测试 (零样本)
# Example 6: Evaluate All Benchmarks (zero-shot)
# ============================================================================
echo ""
echo "Example 6: Evaluate All Benchmarks (zero-shot)"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark all \
    --n_shot 0 \
    --output_file results/all_benchmarks_zeroshot.json \
    --verbose


# ============================================================================
# 示例 7: Few-shot 评估 (5-shot)
# Example 7: Few-shot Evaluation (5-shot)
# ============================================================================
echo ""
echo "Example 7: HellaSwag 5-shot Evaluation"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark hellaswag \
    --n_shot 5 \
    --output_file results/hellaswag_5shot.json


# ============================================================================
# 示例 8: 快速测试（限制样本数量）
# Example 8: Quick Test (limited samples)
# ============================================================================
echo ""
echo "Example 8: Quick Test on ARC-Easy (100 samples)"
python evaluate_benchmarks.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --benchmark arc_easy \
    --n_shot 0 \
    --max_samples 100 \
    --output_file results/arc_easy_quick_test.json \
    --verbose


# ============================================================================
# 示例 9: 列出所有支持的基准测试
# Example 9: List all supported benchmarks
# ============================================================================
echo ""
echo "Example 9: List all supported benchmarks"
python evaluate_benchmarks.py --list_benchmarks


# ============================================================================
# 使用说明
# Usage Notes
# ============================================================================
# 
# 1. 修改 CHECKPOINT_PATH 为你的模型路径
#    Update CHECKPOINT_PATH to your model path
#
# 2. 修改 CUDA_VISIBLE_DEVICES 设置使用的GPU
#    Update CUDA_VISIBLE_DEVICES to set which GPU to use
#
# 3. 支持的基准测试:
#    Supported benchmarks:
#    - arc_easy: AI2 Reasoning Challenge (Easy)
#    - arc_challenge: AI2 Reasoning Challenge (Challenge)
#    - hellaswag: Commonsense Reasoning
#    - winogrande: Pronoun Disambiguation
#    - piqa: Physical Interaction QA
#    - boolq: Boolean Questions
#    - openbookqa: Open Book QA
#    - all: 评估所有基准测试
#
# 4. n_shot 参数:
#    - 0: zero-shot (推荐，最常用)
#    - 5: 5-shot (few-shot learning)
#    - 可以设置为其他值
#
# 5. 评估时间估计:
#    Estimated evaluation time:
#    - ARC-Easy: ~5-10分钟
#    - HellaSwag: ~20-30分钟 (样本较多)
#    - WinoGrande: ~10-15分钟
#    - PIQA: ~5-10分钟
#    - 所有基准测试: ~1-2小时
#
# 6. 使用 --max_samples 可以快速测试
#    Use --max_samples for quick testing
#

