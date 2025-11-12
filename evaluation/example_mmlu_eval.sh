#!/bin/bash
# MMLU 评估示例脚本
# Example script for MMLU evaluation

# ============================================================================
# 示例 1: Zero-shot 评估（所有学科）
# Example 1: Zero-shot evaluation (all subjects)
# ============================================================================
echo "Example 1: Zero-shot evaluation on all MMLU subjects"
CUDA_VISIBLE_DEVICES=4 python evaluate_mmlu.py \
    --output_file results/mmlu_zeroshot_all.json \
    --n_shot 0 \
    --checkpoint_path /data/cdq/current_project/zo-test-cdq/logs/parallel_sweep_20251112_000045/experiments/FO_500M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251112_000045/experiments/FO_500M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/checkpoint


# ============================================================================
# 示例 2: 5-shot 评估（标准MMLU设置）
# Example 2: 5-shot evaluation (standard MMLU setting)
# ============================================================================
echo ""
echo "Example 2: 5-shot evaluation on all MMLU subjects"
python evaluate_mmlu.py \
    --checkpoint_path logs/parallel_sweep_20251112_000045/experiments/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251112_000045/experiments/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/checkpoint \
    --n_shot 5 \
    --output_file results/mmlu_5shot_all.json


# ============================================================================
# 示例 3: 只评估特定学科（STEM领域）
# Example 3: Evaluate specific subjects (STEM)
# ============================================================================
echo ""
echo "Example 3: Evaluate STEM subjects only"
python evaluate_mmlu.py \
    --checkpoint_path logs/parallel_sweep_20251112_000045/experiments/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251112_000045/experiments/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/checkpoint \
    --subjects abstract_algebra astronomy college_mathematics college_physics \
    --n_shot 5 \
    --output_file results/mmlu_5shot_stem.json


# ============================================================================
# 示例 4: 评估计算机科学相关学科
# Example 4: Evaluate computer science subjects
# ============================================================================
echo ""
echo "Example 4: Evaluate computer science subjects"
python evaluate_mmlu.py \
    --checkpoint_path logs/parallel_sweep_20251112_000045/experiments/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251112_000045/experiments/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/checkpoint \
    --subjects college_computer_science high_school_computer_science computer_security machine_learning \
    --n_shot 5 \
    --output_file results/mmlu_5shot_cs.json \
    --verbose


# ============================================================================
# 示例 5: 列出所有可用的学科
# Example 5: List all available subjects
# ============================================================================
echo ""
echo "Example 5: List all available MMLU subjects"
python evaluate_mmlu.py --list_subjects


# ============================================================================
# 使用说明
# Usage Notes
# ============================================================================
# 
# 1. 替换 checkpoint_path 为你自己的模型路径
#    Replace checkpoint_path with your own model path
#
# 2. n_shot 参数：
#    - 0: zero-shot（无示例）
#    - 5: 5-shot（MMLU标准设置，推荐）
#    - 可以设置为其他值，但5是最常用的
#
# 3. subjects 参数：
#    - 不指定则评估所有57个学科
#    - 指定特定学科可以加快评估速度
#    - 使用 --list_subjects 查看所有可用学科
#
# 4. 输出文件会保存为JSON格式，包含：
#    - 总体准确率
#    - 各类别（STEM、人文、社会科学等）准确率
#    - 每个学科的详细结果
#

