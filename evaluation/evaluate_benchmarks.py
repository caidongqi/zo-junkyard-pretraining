"""
通用基准测试评估脚本 (Common Benchmarks Evaluation Script)

支持的基准测试:
- ARC-Easy & ARC-Challenge: AI2 Reasoning Challenge (科学问答)
- HellaSwag: 常识推理和句子补全
- WinoGrande: 代词消歧
- PIQA: Physical Interaction QA (物理常识问答)
- BoolQ: Yes/No问答
- OpenBookQA: 开放书籍问答

功能特性:
1. 支持zero-shot和few-shot评估
2. 自动计算准确率
3. 详细的错误分析
4. 保存评估结果为JSON

用法 (Usage):
    # 评估ARC-Easy
    python evaluate_benchmarks.py \
        --checkpoint_path path/to/checkpoint \
        --benchmark arc_easy \
        --n_shot 0
    
    # 评估HellaSwag
    python evaluate_benchmarks.py \
        --checkpoint_path path/to/checkpoint \
        --benchmark hellaswag \
        --n_shot 5
    
    # 评估所有基准测试
    python evaluate_benchmarks.py \
        --checkpoint_path path/to/checkpoint \
        --benchmark all \
        --n_shot 0
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
)


# ==================== 支持的基准测试 ====================

SUPPORTED_BENCHMARKS = {
    'arc_easy': {
        'name': 'ARC-Easy',
        'description': 'AI2 Reasoning Challenge (Easy)',
        'dataset': 'allenai/ai2_arc',
        'config': 'ARC-Easy',
        'num_choices': 4,
    },
    'arc_challenge': {
        'name': 'ARC-Challenge',
        'description': 'AI2 Reasoning Challenge (Challenge)',
        'dataset': 'allenai/ai2_arc',
        'config': 'ARC-Challenge',
        'num_choices': 4,
    },
    'hellaswag': {
        'name': 'HellaSwag',
        'description': 'Commonsense Reasoning and Sentence Completion',
        'dataset': 'Rowan/hellaswag',
        'config': None,
        'num_choices': 4,
    },
    'winogrande': {
        'name': 'WinoGrande',
        'description': 'Pronoun Disambiguation',
        'dataset': 'winogrande',
        'config': 'winogrande_xl',
        'num_choices': 2,
    },
    'piqa': {
        'name': 'PIQA',
        'description': 'Physical Interaction QA',
        'dataset': 'baber/piqa',
        'config': None,
        'num_choices': 2,
    },
    'boolq': {
        'name': 'BoolQ',
        'description': 'Boolean Questions',
        'dataset': 'boolq',
        'config': None,
        'num_choices': 2,
    },
    'openbookqa': {
        'name': 'OpenBookQA',
        'description': 'Open Book Question Answering',
        'dataset': 'openbookqa',
        'config': 'main',
        'num_choices': 4,
    },
}


# ==================== 模型加载 ====================

def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """加载模型checkpoint"""
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


# ==================== 数据格式化函数 ====================

def format_arc_example(example: Dict, include_answer: bool = False) -> str:
    """格式化ARC问题"""
    question = example['question']
    choices = example['choices']
    
    prompt = f"Question: {question}\n"
    
    # 处理choices（可能是字典格式）
    if isinstance(choices, dict):
        labels = choices['label']
        texts = choices['text']
        for label, text in zip(labels, texts):
            prompt += f"{label}. {text}\n"
    else:
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
    
    if include_answer:
        answer = example['answerKey']
        prompt += f"Answer: {answer}\n"
    else:
        prompt += "Answer:"
    
    return prompt


def format_hellaswag_example(example: Dict, include_answer: bool = False) -> str:
    """格式化HellaSwag问题"""
    ctx = example['ctx']
    endings = example['endings']
    
    prompt = f"Context: {ctx}\n\n"
    prompt += "Which ending makes the most sense?\n"
    
    for i, ending in enumerate(endings):
        prompt += f"{chr(65 + i)}. {ending}\n"
    
    if include_answer:
        answer = int(example['label'])
        prompt += f"Answer: {chr(65 + answer)}\n"
    else:
        prompt += "Answer:"
    
    return prompt


def format_winogrande_example(example: Dict, include_answer: bool = False) -> str:
    """格式化WinoGrande问题"""
    sentence = example['sentence']
    option1 = example['option1']
    option2 = example['option2']
    
    prompt = f"Sentence: {sentence}\n"
    prompt += f"A. {option1}\n"
    prompt += f"B. {option2}\n"
    
    if include_answer:
        answer = int(example['answer']) - 1  # WinoGrande uses 1/2, convert to 0/1
        prompt += f"Answer: {chr(65 + answer)}\n"
    else:
        prompt += "Answer:"
    
    return prompt


def format_piqa_example(example: Dict, include_answer: bool = False) -> str:
    """格式化PIQA问题"""
    goal = example['goal']
    sol1 = example['sol1']
    sol2 = example['sol2']
    
    prompt = f"Goal: {goal}\n\n"
    prompt += "Which solution is correct?\n"
    prompt += f"A. {sol1}\n"
    prompt += f"B. {sol2}\n"
    
    if include_answer:
        answer = int(example['label'])
        prompt += f"Answer: {chr(65 + answer)}\n"
    else:
        prompt += "Answer:"
    
    return prompt


def format_boolq_example(example: Dict, include_answer: bool = False) -> str:
    """格式化BoolQ问题"""
    passage = example['passage']
    question = example['question']
    
    prompt = f"Passage: {passage}\n\n"
    prompt += f"Question: {question}\n"
    prompt += "A. Yes\n"
    prompt += "B. No\n"
    
    if include_answer:
        answer = 0 if example['answer'] else 1  # True -> A, False -> B
        prompt += f"Answer: {chr(65 + answer)}\n"
    else:
        prompt += "Answer:"
    
    return prompt


def format_openbookqa_example(example: Dict, include_answer: bool = False) -> str:
    """格式化OpenBookQA问题"""
    question_stem = example['question_stem']
    choices = example['choices']
    
    prompt = f"Question: {question_stem}\n"
    
    if isinstance(choices, dict):
        labels = choices['label']
        texts = choices['text']
        for label, text in zip(labels, texts):
            prompt += f"{label}. {text}\n"
    
    if include_answer:
        answer = example['answerKey']
        prompt += f"Answer: {answer}\n"
    else:
        prompt += "Answer:"
    
    return prompt


# 格式化函数映射
FORMAT_FUNCTIONS = {
    'arc_easy': format_arc_example,
    'arc_challenge': format_arc_example,
    'hellaswag': format_hellaswag_example,
    'winogrande': format_winogrande_example,
    'piqa': format_piqa_example,
    'boolq': format_boolq_example,
    'openbookqa': format_openbookqa_example,
}


# ==================== Few-shot示例构建 ====================

def build_prompt_with_examples(
    example: Dict,
    few_shot_examples: List[Dict],
    benchmark: str,
    instruction: str = ""
) -> str:
    """构建包含few-shot示例的完整prompt"""
    prompt = ""
    
    # 添加任务指令
    if instruction:
        prompt += instruction + "\n\n"
    
    # 添加few-shot示例
    format_fn = FORMAT_FUNCTIONS[benchmark]
    for few_shot_ex in few_shot_examples:
        prompt += format_fn(few_shot_ex, include_answer=True)
        prompt += "\n"
    
    # 添加当前问题
    prompt += format_fn(example, include_answer=False)
    
    return prompt


# ==================== 模型评估 ====================

def get_answer_choice_logits(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_choices: int,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    获取模型对于选项的logits
    
    Returns:
        选项的logits (shape: [num_choices])
    """
    model.eval()
    
    # 编码prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 获取模型输出
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # 最后一个token的logits
    
    # 获取选项对应的token
    choice_tokens = [tokenizer.encode(chr(65 + i), add_special_tokens=False)[0] 
                     for i in range(num_choices)]
    
    # 提取选项的logits
    choice_logits = torch.tensor([logits[token].item() for token in choice_tokens])
    
    return choice_logits


def get_correct_answer_index(example: Dict, benchmark: str) -> int:
    """获取正确答案的索引"""
    if benchmark in ['arc_easy', 'arc_challenge', 'openbookqa']:
        answer = example['answerKey']
        # 可能是A/B/C/D或1/2/3/4
        if answer.isdigit():
            return int(answer) - 1
        else:
            return ord(answer.upper()) - ord('A')
    elif benchmark == 'hellaswag':
        return int(example['label'])
    elif benchmark == 'winogrande':
        return int(example['answer']) - 1  # 1/2 -> 0/1
    elif benchmark in ['piqa', 'boolq']:
        if 'label' in example:
            return int(example['label'])
        else:
            return 0 if example['answer'] else 1
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def evaluate_benchmark(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    benchmark: str,
    n_shot: int = 0,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    评估模型在单个基准测试上的表现
    
    Args:
        model: GPT-2模型
        tokenizer: tokenizer
        benchmark: 基准测试名称
        n_shot: few-shot示例数量
        device: 设备
        max_samples: 最大评估样本数
        verbose: 是否显示详细信息
    
    Returns:
        评估结果字典
    """
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    
    bench_info = SUPPORTED_BENCHMARKS[benchmark]
    print(f"\n{'='*80}")
    print(f"Evaluating {bench_info['name']}")
    print(f"Description: {bench_info['description']}")
    print(f"N-shot: {n_shot}")
    print(f"{'='*80}\n")
    
    # 加载数据集
    try:
        if bench_info['config']:
            dataset = load_dataset(bench_info['dataset'], bench_info['config'])
        else:
            # 对于某些数据集，尝试使用trust_remote_code或不同的加载方式
            if benchmark == 'piqa':
                # PIQA数据集在新版本中可能需要特殊处理
                try:
                    # 首先尝试使用allenai/piqa
                    dataset = load_dataset('allenai/piqa', trust_remote_code=True)
                except Exception as e1:
                    print(f"First attempt failed: {e1}")
                    try:
                        # 尝试使用piqa（不带组织名）
                        dataset = load_dataset('piqa', trust_remote_code=True)
                    except Exception as e2:
                        print(f"Second attempt failed: {e2}")
                        # 最后尝试直接从URL加载JSONL
                        try:
                            dataset = load_dataset('json', data_files={
                                'train': 'https://storage.googleapis.com/ai2-mosaic/public/piqa/train/train.jsonl',
                                'validation': 'https://storage.googleapis.com/ai2-mosaic/public/piqa/valid/valid.jsonl'
                            })
                        except Exception as e3:
                            print(f"All PIQA loading methods failed: {e3}")
                            raise e3
            else:
                try:
                    dataset = load_dataset(bench_info['dataset'], trust_remote_code=True)
                except:
                    # 如果trust_remote_code失败，尝试不使用
                    dataset = load_dataset(bench_info['dataset'], trust_remote_code=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Trying alternative loading method...")
        # 尝试不使用trust_remote_code
        try:
            if bench_info['config']:
                dataset = load_dataset(bench_info['dataset'], bench_info['config'], trust_remote_code=False)
            else:
                dataset = load_dataset(bench_info['dataset'], trust_remote_code=False)
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            return None
    
    # 确定使用哪个split
    if 'test' in dataset:
        test_split = 'test'
    elif 'validation' in dataset:
        test_split = 'validation'
    else:
        test_split = list(dataset.keys())[0]
    
    print(f"Using split: {test_split}")
    test_data = dataset[test_split]
    
    # 获取few-shot示例
    few_shot_examples = []
    if n_shot > 0:
        # 从训练集或验证集获取示例
        if 'train' in dataset:
            few_shot_split = 'train'
        elif 'validation' in dataset and test_split != 'validation':
            few_shot_split = 'validation'
        else:
            few_shot_split = None
        
        if few_shot_split:
            for i in range(min(n_shot, len(dataset[few_shot_split]))):
                few_shot_examples.append(dataset[few_shot_split][i])
    
    # 限制样本数量
    if max_samples:
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    
    # 评估
    correct = 0
    total = len(test_data)
    predictions = []
    
    # 任务指令
    instruction = f"The following are questions about {bench_info['description'].lower()}."
    
    iterator = tqdm(test_data, desc=f"Evaluating {bench_info['name']}") if verbose else test_data
    
    for example in iterator:
        # 构建prompt
        prompt = build_prompt_with_examples(
            example=example,
            few_shot_examples=few_shot_examples,
            benchmark=benchmark,
            instruction="" if n_shot == 0 else instruction
        )
        
        # 获取模型预测
        choice_logits = get_answer_choice_logits(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_choices=bench_info['num_choices'],
            device=device
        )
        
        predicted_idx = choice_logits.argmax().item()
        correct_idx = get_correct_answer_index(example, benchmark)
        
        is_correct = (predicted_idx == correct_idx)
        if is_correct:
            correct += 1
        
        predictions.append({
            'predicted': predicted_idx,
            'correct': correct_idx,
            'is_correct': is_correct,
            'logits': choice_logits.tolist()
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"\n{bench_info['name']} Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return {
        'benchmark': benchmark,
        'name': bench_info['name'],
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'n_shot': n_shot,
        'predictions': predictions[:100] if not verbose else predictions  # 保存前100个
    }


def evaluate_all_benchmarks(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    benchmarks: List[str],
    n_shot: int = 0,
    device: str = 'cuda',
    max_samples: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """评估模型在多个基准测试上的表现"""
    results = {
        'overall': {},
        'benchmarks': {},
        'config': {
            'n_shot': n_shot,
            'max_samples': max_samples,
        }
    }
    
    all_correct = 0
    all_total = 0
    
    for benchmark in benchmarks:
        result = evaluate_benchmark(
            model=model,
            tokenizer=tokenizer,
            benchmark=benchmark,
            n_shot=n_shot,
            device=device,
            max_samples=max_samples,
            verbose=verbose
        )
        
        if result:
            results['benchmarks'][benchmark] = result
            all_correct += result['correct']
            all_total += result['total']
    
    # 计算总体准确率
    overall_accuracy = all_correct / all_total if all_total > 0 else 0.0
    results['overall'] = {
        'accuracy': overall_accuracy,
        'correct': all_correct,
        'total': all_total
    }
    
    # 打印总结
    print(f"\n{'='*80}")
    print("Overall Summary")
    print(f"{'='*80}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({all_correct}/{all_total})")
    print("\nPer-Benchmark Results:")
    for benchmark, result in results['benchmarks'].items():
        print(f"  {result['name']:20s}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")
    print(f"{'='*80}\n")
    
    return results


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(
        description="通用基准测试评估脚本",
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
    
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='基准测试名称 (或 "all" 评估所有)'
    )
    
    # 可选参数
    parser.add_argument(
        '--n_shot',
        type=int,
        default=0,
        help='Few-shot示例数量（默认: 0）'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大评估样本数（默认: 全部）'
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
        help='显示详细评估信息'
    )
    
    parser.add_argument(
        '--list_benchmarks',
        action='store_true',
        help='列出所有支持的基准测试'
    )
    
    args = parser.parse_args()
    
    # 列出基准测试
    if args.list_benchmarks:
        print("Supported Benchmarks:")
        print("=" * 80)
        for key, info in SUPPORTED_BENCHMARKS.items():
            print(f"\n{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Dataset: {info['dataset']}")
            print(f"  Number of choices: {info['num_choices']}")
        print("\n" + "=" * 80)
        return
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 确定要评估的基准测试
    if args.benchmark == 'all':
        benchmarks = list(SUPPORTED_BENCHMARKS.keys())
    else:
        if args.benchmark not in SUPPORTED_BENCHMARKS:
            print(f"Error: Unknown benchmark '{args.benchmark}'")
            print(f"Use --list_benchmarks to see all available benchmarks")
            return
        benchmarks = [args.benchmark]
    
    print("=" * 80)
    print("Benchmark Evaluation Configuration")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"N-shot: {args.n_shot}")
    print(f"Max samples: {args.max_samples or 'All'}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # 加载模型
    print("\nLoading model...")
    model, tokenizer = load_checkpoint(args.checkpoint_path, device=args.device)
    
    # 评估基准测试
    if len(benchmarks) == 1:
        results = evaluate_benchmark(
            model=model,
            tokenizer=tokenizer,
            benchmark=benchmarks[0],
            n_shot=args.n_shot,
            device=args.device,
            max_samples=args.max_samples,
            verbose=args.verbose
        )
        results = {'benchmarks': {benchmarks[0]: results}}
    else:
        results = evaluate_all_benchmarks(
            model=model,
            tokenizer=tokenizer,
            benchmarks=benchmarks,
            n_shot=args.n_shot,
            device=args.device,
            max_samples=args.max_samples,
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

