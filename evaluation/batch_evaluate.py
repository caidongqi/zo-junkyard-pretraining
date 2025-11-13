"""
批量评估 Checkpoints
Batch evaluate multiple checkpoints on multiple datasets

用法 (Usage):
    # 单个checkpoint，多个数据集
    python batch_evaluate.py \
        --checkpoint_path path/to/checkpoint \
        --datasets cosmopedia fineweb-edu-10bt wikitext-103 \
        --output_dir results/batch_eval
    
    # 多个checkpoints（自动查找实验目录）
    python batch_evaluate.py \
        --experiments_dir logs/parallel_sweep_20251104_152749/experiments \
        --datasets cosmopedia tinystories \
        --output_dir results/multi_checkpoint_eval
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import torch
import pandas as pd
from tqdm import tqdm

from evaluate_checkpoint import load_checkpoint, evaluate_loss
from data import get_dataloader, DATASET_CONFIGS


def find_checkpoints(experiments_dir: Path) -> List[Path]:
    """
    在实验目录中查找所有 checkpoint
    
    Args:
        experiments_dir: 实验目录路径
    
    Returns:
        checkpoint 路径列表
    """
    checkpoints = []
    
    if not experiments_dir.exists():
        print(f"Warning: Experiments directory does not exist: {experiments_dir}")
        return checkpoints
    
    # 递归查找所有包含 config.json 的 checkpoint 目录
    for path in experiments_dir.rglob("checkpoint"):
        config_file = path / "config.json"
        if config_file.exists():
            checkpoints.append(path)
    
    return checkpoints


def evaluate_checkpoint_on_datasets(
    checkpoint_path: Path,
    datasets: List[str],
    batch_size: int = 4,
    block_size: int = 128,
    max_samples: int = None,
    max_batches: int = None,
    device: str = 'cuda',
    cache_dir: str = 'cache',
    force_reload: bool = False,
) -> List[Dict[str, Any]]:
    """
    在多个数据集上评估单个 checkpoint
    
    Args:
        checkpoint_path: checkpoint 路径
        datasets: 数据集名称列表
        其他参数同 evaluate_checkpoint.py
    
    Returns:
        评估结果列表
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    # 加载 checkpoint（只加载一次）
    try:
        model, tokenizer, config = load_checkpoint(checkpoint_path, device=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return results
    
    # 在每个数据集上评估
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"Evaluating on dataset: {dataset_name}")
        print(f"{'='*80}")
        
        try:
            # 加载数据集
            dataloader = get_dataloader(
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                batch_size=batch_size,
                block_size=block_size,
                max_samples=max_samples,
                cache_dir=cache_dir,
                force_reload=force_reload
            )
            
            # 评估
            avg_loss, perplexity, total_tokens = evaluate_loss(
                model=model,
                dataloader=dataloader,
                device=device,
                max_batches=max_batches
            )
            
            # 保存结果
            result = {
                'checkpoint_path': str(checkpoint_path),
                'checkpoint_name': checkpoint_path.parent.parent.name if checkpoint_path.parent.parent.name != 'experiments' else checkpoint_path.parent.name,
                'dataset': dataset_name,
                'avg_loss': avg_loss,
                'perplexity': perplexity,
                'total_tokens': total_tokens,
                'batch_size': batch_size,
                'block_size': block_size,
                'max_samples': max_samples,
                'model_config': config,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            print(f"\nResults for {dataset_name}:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Perplexity: {perplexity:.4f}")
            print(f"  Total Tokens: {total_tokens:,}")
            
        except Exception as e:
            print(f"Error evaluating on {dataset_name}: {e}")
            continue
    
    return results


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """
    保存评估结果到文件
    
    Args:
        results: 评估结果列表
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整的 JSON 结果
    json_file = output_dir / "full_results.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {json_file}")
    
    # 创建摘要 CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'checkpoint': result['checkpoint_name'],
            'dataset': result['dataset'],
            'loss': result['avg_loss'],
            'perplexity': result['perplexity'],
            'tokens': result['total_tokens'],
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        csv_file = output_dir / "summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"Summary saved to: {csv_file}")
        
        # 打印摘要表格
        print("\n" + "="*80)
        print("Summary Table")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="批量评估 checkpoints 在多个数据集上的 loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Checkpoint 参数（两种模式：单个 checkpoint 或实验目录）
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        '--checkpoint_path',
        type=str,
        help='单个 checkpoint 目录路径'
    )
    checkpoint_group.add_argument(
        '--experiments_dir',
        type=str,
        help='实验目录路径（自动查找所有 checkpoint）'
    )
    
    # 数据集参数
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        required=True,
        help='要评估的数据集名称列表（空格分隔）'
    )
    
    # 评估参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小 (默认: 4)'
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=128,
        help='文本块大小/序列长度 (默认: 128)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大样本数 (默认: None)'
    )
    
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='最大评估 batch 数 (默认: None)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备 (默认: cuda)'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache',
        help='数据集缓存目录 (默认: cache)'
    )
    
    parser.add_argument(
        '--force_reload',
        action='store_true',
        help='强制重新加载数据集'
    )
    
    # 输出参数
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='结果输出目录'
    )
    
    args = parser.parse_args()
    
    # 检查 CUDA 可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # 验证数据集名称
    invalid_datasets = [d for d in args.datasets if d not in DATASET_CONFIGS]
    if invalid_datasets:
        print(f"Error: Invalid datasets: {invalid_datasets}")
        print(f"Available datasets: {list(DATASET_CONFIGS.keys())}")
        return
    
    print("=" * 80)
    print("Batch Checkpoint Evaluation")
    print("=" * 80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Block size: {args.block_size}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)
    
    # 获取 checkpoint 列表
    if args.checkpoint_path:
        checkpoints = [Path(args.checkpoint_path)]
    else:
        experiments_dir = Path(args.experiments_dir)
        checkpoints = find_checkpoints(experiments_dir)
        print(f"\nFound {len(checkpoints)} checkpoints in {experiments_dir}")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
    
    if not checkpoints:
        print("Error: No checkpoints found!")
        return
    
    # 批量评估
    all_results = []
    for checkpoint_path in checkpoints:
        results = evaluate_checkpoint_on_datasets(
            checkpoint_path=checkpoint_path,
            datasets=args.datasets,
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_samples=args.max_samples,
            max_batches=args.max_batches,
            device=args.device,
            cache_dir=args.cache_dir,
            force_reload=args.force_reload
        )
        all_results.extend(results)
    
    # 保存结果
    if all_results:
        output_dir = Path(args.output_dir)
        save_results(all_results, output_dir)
        print(f"\n✓ Batch evaluation completed! Results saved to {output_dir}")
    else:
        print("\n✗ No results generated!")


if __name__ == "__main__":
    main()
