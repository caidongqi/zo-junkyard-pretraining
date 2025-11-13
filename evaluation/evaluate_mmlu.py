"""
MMLU 评估脚本 (MMLU Evaluation Script)

Massive Multitask Language Understanding (MMLU) 基准测试
包含57个学科的多项选择题，涵盖STEM、人文、社会科学等领域

功能 (Features):
1. 支持zero-shot和few-shot评估
2. 评估所有57个MMLU学科
3. 计算每个学科和总体的准确率
4. 支持多种模型checkpoint格式
5. 详细的结果报告和分析

用法 (Usage):
    # Zero-shot评估
    python evaluate_mmlu.py \
        --checkpoint_path path/to/checkpoint \
        --n_shot 0 \
        --output_file results/mmlu_zeroshot.json
    
    # 5-shot评估（标准MMLU设置）
    python evaluate_mmlu.py \
        --checkpoint_path path/to/checkpoint \
        --n_shot 5 \
        --output_file results/mmlu_5shot.json
    
    # 只评估特定学科
    python evaluate_mmlu.py \
        --checkpoint_path path/to/checkpoint \
        --subjects abstract_algebra astronomy \
        --n_shot 5
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Config,
)


# ==================== MMLU 学科分类 ====================

MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
    'clinical_knowledge', 'college_biology', 'college_chemistry',
    'college_computer_science', 'college_mathematics', 'college_medicine',
    'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics',
    'formal_logic', 'global_facts', 'high_school_biology',
    'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography',
    'high_school_government_and_politics', 'high_school_macroeconomics',
    'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging',
    'human_sexuality', 'international_law', 'jurisprudence',
    'logical_fallacies', 'machine_learning', 'management', 'marketing',
    'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
    'professional_law', 'professional_medicine', 'professional_psychology',
    'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
    'virology', 'world_religions'
]

# 学科分类
SUBJECT_CATEGORIES = {
    'STEM': [
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_physics',
        'computer_security', 'conceptual_physics', 'electrical_engineering',
        'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_mathematics',
        'high_school_physics', 'high_school_statistics', 'machine_learning'
    ],
    'Humanities': [
        'formal_logic', 'high_school_european_history', 'high_school_us_history',
        'high_school_world_history', 'international_law', 'jurisprudence',
        'logical_fallacies', 'moral_disputes', 'moral_scenarios', 'philosophy',
        'prehistory', 'professional_law', 'world_religions'
    ],
    'Social Sciences': [
        'business_ethics', 'clinical_knowledge', 'college_medicine', 'econometrics',
        'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_microeconomics',
        'high_school_psychology', 'human_aging', 'human_sexuality', 'management',
        'marketing', 'medical_genetics', 'miscellaneous', 'nutrition',
        'professional_accounting', 'professional_medicine', 'professional_psychology',
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'virology'
    ],
    'Other': [
        'anatomy', 'global_facts'
    ]
}


# ==================== 模型加载 ====================

def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """
    从checkpoint加载GPT-2模型
    
    Args:
        checkpoint_path: checkpoint目录路径
        device: 设备
    
    Returns:
        model: 加载的GPT-2模型
        tokenizer: 对应的tokenizer
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 加载tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if tokenizer_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print("Warning: Tokenizer not found in checkpoint, using GPT2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.2f}M parameters")
    
    return model, tokenizer


# ==================== MMLU 数据处理 ====================

def format_subject(subject: str) -> str:
    """格式化学科名称（将下划线替换为空格，首字母大写）"""
    return " ".join(word.capitalize() for word in subject.split("_"))


def format_example(question: str, choices: List[str], include_answer: bool = False, answer: Optional[str] = None) -> str:
    """
    格式化一个MMLU问题为文本
    
    Args:
        question: 问题文本
        choices: 选项列表 [A, B, C, D]
        include_answer: 是否包含答案（用于few-shot示例）
        answer: 答案（0, 1, 2, 3对应A, B, C, D）
    
    Returns:
        格式化的问题文本
    """
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    
    if include_answer and answer is not None:
        prompt += f"Answer: {chr(65 + int(answer))}\n"
    else:
        prompt += "Answer:"
    
    return prompt


def get_few_shot_examples(dataset, n_shot: int = 5) -> List[Dict]:
    """
    从dev集中获取few-shot示例
    
    Args:
        dataset: MMLU数据集
        n_shot: few-shot示例数量
    
    Returns:
        few-shot示例列表
    """
    if n_shot == 0:
        return []
    
    # 从dev集中获取前n_shot个示例
    examples = []
    for i in range(min(n_shot, len(dataset['dev']))):
        example = dataset['dev'][i]
        examples.append({
            'question': example['question'],
            'choices': example['choices'],
            'answer': example['answer']
        })
    
    return examples


def build_prompt(question: str, choices: List[str], few_shot_examples: List[Dict], subject: str = "") -> str:
    """
    构建完整的评估prompt
    
    Args:
        question: 当前问题
        choices: 当前选项
        few_shot_examples: few-shot示例
        subject: 学科名称（可选）
    
    Returns:
        完整的prompt
    """
    prompt = ""
    
    # 添加学科信息（可选）
    if subject:
        prompt += f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    
    # 添加few-shot示例
    for example in few_shot_examples:
        prompt += format_example(
            example['question'],
            example['choices'],
            include_answer=True,
            answer=example['answer']
        )
        prompt += "\n"
    
    # 添加当前问题
    prompt += format_example(question, choices, include_answer=False)
    
    return prompt


# ==================== 模型评估 ====================

def get_answer_logits(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    获取模型对于A/B/C/D四个选项的logits
    
    Args:
        model: GPT-2模型
        tokenizer: tokenizer
        prompt: 输入prompt
        device: 设备
    
    Returns:
        四个选项的logits (shape: [4])
    """
    model.eval()
    
    # 编码prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # 获取最后一个token的logits（这是模型对下一个token的预测）
    last_token_logits = logits[0, -1, :]  # [vocab_size]
    
    # 获取A/B/C/D对应的token ID
    answer_tokens = {
        'A': tokenizer.encode('A', add_special_tokens=False)[0],
        'B': tokenizer.encode('B', add_special_tokens=False)[0],
        'C': tokenizer.encode('C', add_special_tokens=False)[0],
        'D': tokenizer.encode('D', add_special_tokens=False)[0],
    }
    
    # 提取A/B/C/D的logits
    answer_logits = torch.tensor([
        last_token_logits[answer_tokens['A']].item(),
        last_token_logits[answer_tokens['B']].item(),
        last_token_logits[answer_tokens['C']].item(),
        last_token_logits[answer_tokens['D']].item(),
    ])
    
    return answer_logits


def evaluate_subject(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    subject: str,
    n_shot: int = 5,
    device: str = 'cuda',
    verbose: bool = False
) -> Dict:
    """
    评估模型在单个学科上的表现
    
    Args:
        model: GPT-2模型
        tokenizer: tokenizer
        subject: 学科名称
        n_shot: few-shot示例数量
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        包含准确率等指标的字典
    """
    # 加载数据集
    try:
        dataset = load_dataset("cais/mmlu", subject)
    except Exception as e:
        print(f"Error loading subject {subject}: {e}")
        return None
    
    # 获取few-shot示例
    few_shot_examples = get_few_shot_examples(dataset, n_shot=n_shot)
    
    # 评估测试集
    test_data = dataset['test']
    correct = 0
    total = len(test_data)
    
    predictions = []
    
    if verbose:
        iterator = tqdm(test_data, desc=f"Evaluating {subject}")
    else:
        iterator = test_data
    
    for example in iterator:
        question = example['question']
        choices = example['choices']
        answer = example['answer']
        
        # 构建prompt
        prompt = build_prompt(question, choices, few_shot_examples, subject)
        
        # 获取模型预测
        answer_logits = get_answer_logits(model, tokenizer, prompt, device)
        predicted_answer = answer_logits.argmax().item()
        
        # 检查是否正确
        is_correct = (predicted_answer == int(answer))
        if is_correct:
            correct += 1
        
        predictions.append({
            'question': question,
            'choices': choices,
            'true_answer': int(answer),
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'logits': answer_logits.tolist()
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    results = {
        'subject': subject,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': predictions[:10] if not verbose else predictions  # 只保存前10个预测（节省空间）
    }
    
    return results


def evaluate_all_subjects(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    subjects: Optional[List[str]] = None,
    n_shot: int = 5,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    评估模型在所有（或指定）学科上的表现
    
    Args:
        model: GPT-2模型
        tokenizer: tokenizer
        subjects: 要评估的学科列表（None表示评估所有学科）
        n_shot: few-shot示例数量
        device: 设备
        verbose: 是否打印详细信息
    
    Returns:
        包含所有学科结果的字典
    """
    if subjects is None:
        subjects = MMLU_SUBJECTS
    
    print(f"\n{'='*80}")
    print(f"Evaluating MMLU ({len(subjects)} subjects, {n_shot}-shot)")
    print(f"{'='*80}\n")
    
    all_results = {}
    category_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for subject in subjects:
        print(f"\n[{subjects.index(subject)+1}/{len(subjects)}] Evaluating {format_subject(subject)}...")
        
        result = evaluate_subject(
            model=model,
            tokenizer=tokenizer,
            subject=subject,
            n_shot=n_shot,
            device=device,
            verbose=False
        )
        
        if result:
            all_results[subject] = result
            accuracy = result['accuracy']
            correct = result['correct']
            total = result['total']
            
            print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
            
            # 更新分类统计
            for category, category_subjects in SUBJECT_CATEGORIES.items():
                if subject in category_subjects:
                    category_results[category]['correct'] += correct
                    category_results[category]['total'] += total
                    break
    
    # 计算总体和分类准确率
    total_correct = sum(r['correct'] for r in all_results.values())
    total_questions = sum(r['total'] for r in all_results.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
    
    # 计算各类别准确率
    category_accuracies = {}
    for category, stats in category_results.items():
        if stats['total'] > 0:
            category_accuracies[category] = {
                'accuracy': stats['correct'] / stats['total'],
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    # 打印总结
    print(f"\n{'='*80}")
    print("MMLU Evaluation Summary")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_questions})")
    print(f"\nCategory Accuracies:")
    for category, stats in sorted(category_accuracies.items()):
        print(f"  {category:20s}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    print(f"{'='*80}\n")
    
    return {
        'overall': {
            'accuracy': overall_accuracy,
            'correct': total_correct,
            'total': total_questions
        },
        'categories': category_accuracies,
        'subjects': all_results,
        'config': {
            'n_shot': n_shot,
            'num_subjects': len(subjects)
        }
    }


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="MMLU评估脚本 - 评估模型在MMLU基准测试上的表现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必需参数
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='模型checkpoint目录路径'
    )
    
    # 可选参数
    parser.add_argument(
        '--subjects',
        nargs='+',
        default=None,
        help='要评估的学科列表（默认评估所有学科）'
    )
    
    parser.add_argument(
        '--n_shot',
        type=int,
        default=5,
        help='Few-shot示例数量（0表示zero-shot，默认: 5）'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备（默认: cuda）'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='结果输出文件路径（JSON格式）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='打印详细的评估信息'
    )
    
    parser.add_argument(
        '--list_subjects',
        action='store_true',
        help='列出所有可用的MMLU学科'
    )
    
    args = parser.parse_args()
    
    # 列出学科
    if args.list_subjects:
        print("Available MMLU Subjects:")
        print("=" * 80)
        for category, subjects in SUBJECT_CATEGORIES.items():
            print(f"\n{category}:")
            for subject in subjects:
                print(f"  - {subject}")
        print("\n" + "=" * 80)
        print(f"Total: {len(MMLU_SUBJECTS)} subjects")
        return
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 验证学科
    if args.subjects:
        invalid_subjects = [s for s in args.subjects if s not in MMLU_SUBJECTS]
        if invalid_subjects:
            print(f"Error: Invalid subjects: {invalid_subjects}")
            print(f"Use --list_subjects to see all available subjects")
            return
    
    print("=" * 80)
    print("MMLU Evaluation Configuration")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"N-shot: {args.n_shot}")
    print(f"Subjects: {len(args.subjects) if args.subjects else len(MMLU_SUBJECTS)}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # 加载模型
    print("\nLoading model...")
    model, tokenizer = load_checkpoint(args.checkpoint_path, device=args.device)
    
    # 评估MMLU
    results = evaluate_all_subjects(
        model=model,
        tokenizer=tokenizer,
        subjects=args.subjects,
        n_shot=args.n_shot,
        device=args.device,
        verbose=args.verbose
    )
    
    # 添加元数据
    results['metadata'] = {
        'checkpoint_path': str(args.checkpoint_path),
        'n_shot': args.n_shot,
        'timestamp': datetime.now().isoformat(),
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    # 保存结果
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()

