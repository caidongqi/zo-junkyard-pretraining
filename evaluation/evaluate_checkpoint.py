"""
评估Checkpoint在不同数据集上的Loss
Evaluate checkpoint loss on various datasets

用法 (Usage):
    python evaluate_checkpoint.py --checkpoint_path PATH --dataset cosmopedia --batch_size 4
    
示例 (Examples):
    # 评估FO训练的checkpoint在cosmopedia数据集上的loss
    python evaluate_checkpoint.py \
        --checkpoint_path logs/parallel_sweep_20251104_152749/experiments/FO_full_bs2_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251104_152749/experiments/FO_full_bs2_qN_A_bpN_A_optmudamw_lr1e-3/checkpoint \
        --dataset cosmopedia \
        --batch_size 8 \
        --block_size 128 \
        --max_samples 10000
    
    # 评估在fineweb-edu-10bt数据集上
    python evaluate_checkpoint.py \
        --checkpoint_path path/to/checkpoint \
        --dataset fineweb-edu-10bt \
        --batch_size 4
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm

# 导入自定义模块
from data import get_dataloader, list_available_datasets, DATASET_CONFIGS


def load_checkpoint(checkpoint_path, device='cuda'):
    """
    从指定路径加载checkpoint
    
    Args:
        checkpoint_path: checkpoint目录路径
        device: 设备 ('cuda' 或 'cpu')
    
    Returns:
        model: 加载的模型
        tokenizer: 对应的tokenizer
        config: 模型配置
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # 加载tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if tokenizer_path.exists():
        print(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print("Warning: Tokenizer not found in checkpoint, using GPT2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型配置
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        print(f"Loading model config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Model config: vocab_size={config['vocab_size']}, "
              f"n_positions={config['n_positions']}, "
              f"n_embd={config['n_embd']}, "
              f"n_layer={config['n_layer']}, "
              f"n_head={config['n_head']}")
    else:
        raise ValueError(f"Config file not found: {config_path}")
    
    # 加载模型
    print(f"Loading model weights...")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    
    return model, tokenizer, config


def evaluate_loss(model, dataloader, device='cuda', max_batches=None):
    """
    在数据集上评估模型的loss
    
    Args:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 设备
        max_batches: 最大评估batch数，None表示评估全部
    
    Returns:
        avg_loss: 平均loss
        perplexity: 困惑度
        num_tokens: 评估的token总数
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", total=max_batches if max_batches else len(dataloader))
        for batch_idx, batch in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
            
            # 将数据移到设备上
            if isinstance(batch, torch.Tensor):
                inputs = batch.to(device)
                labels = batch.to(device)
            else:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device) if 'labels' in batch else inputs
            
            # 前向传播
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            
            # 累计loss
            batch_size = inputs.size(0)
            seq_length = inputs.size(1)
            num_tokens_in_batch = batch_size * seq_length
            
            total_loss += loss.item() * num_tokens_in_batch
            total_tokens += num_tokens_in_batch
            num_batches += 1
            
            # 更新进度条
            current_avg_loss = total_loss / total_tokens
            current_perplexity = torch.exp(torch.tensor(current_avg_loss)).item()
            pbar.set_postfix({
                'loss': f'{current_avg_loss:.4f}',
                'ppl': f'{current_perplexity:.2f}',
                'tokens': f'{total_tokens:,}'
            })
    
    # 计算平均loss和困惑度
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity, total_tokens


def main():
    parser = argparse.ArgumentParser(
        description="评估checkpoint在不同数据集上的loss",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 必需参数
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        required=True,
        help='Checkpoint目录路径'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help='要评估的数据集名称'
    )
    
    # 可选参数
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
        help='最大样本数，None表示使用推荐值 (默认: None)'
    )
    
    parser.add_argument(
        '--max_batches',
        type=int,
        default=None,
        help='最大评估batch数，None表示评估全部 (默认: None)'
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
        help='强制重新加载数据集（忽略缓存）'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='结果输出文件路径（JSON格式），默认打印到控制台'
    )
    
    parser.add_argument(
        '--list_datasets',
        action='store_true',
        help='列出所有可用的数据集'
    )
    
    args = parser.parse_args()
    
    # 列出数据集
    if args.list_datasets:
        list_available_datasets()
        return
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("=" * 80)
    print("Checkpoint Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Block size: {args.block_size}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max batches: {args.max_batches}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # 加载checkpoint
    print("\n[1/3] Loading checkpoint...")
    model, tokenizer, config = load_checkpoint(args.checkpoint_path, device=args.device)
    
    # 加载数据集
    print("\n[2/3] Loading dataset...")
    dataloader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        block_size=args.block_size,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        force_reload=args.force_reload
    )
    
    # 评估loss
    print("\n[3/3] Evaluating loss...")
    avg_loss, perplexity, total_tokens = evaluate_loss(
        model=model,
        dataloader=dataloader,
        device=args.device,
        max_batches=args.max_batches
    )
    
    # 打印结果
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Total Tokens: {total_tokens:,}")
    print("=" * 80)
    
    # 保存结果
    results = {
        'checkpoint_path': str(args.checkpoint_path),
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'block_size': args.block_size,
        'max_samples': args.max_samples,
        'max_batches': args.max_batches,
        'device': args.device,
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens,
        'model_config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()


