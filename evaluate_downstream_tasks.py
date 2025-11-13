#!/usr/bin/env python3
"""
下游任务评估脚本
评估checkpoint在SST-2, SQuAD和LAMBADA上的性能

使用方法:
    python evaluate_downstream_tasks.py --checkpoint_path <path_to_checkpoint>
    
示例:
    python evaluate_downstream_tasks.py \
        --checkpoint_path logs/parallel_sweep_20251112_000045_baselines/experiments/FO_200M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251112_000045/experiments/FO_200M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/checkpoint
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Iterable

import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm

# ============================================================================
# Hugging Face 连接配置（解决SSL错误）
# ============================================================================
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

# 配置requests重试
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    retry_strategy = Retry(total=10, backoff_factor=2, 
                          status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
except:
    pass


def load_checkpoint(checkpoint_path, device='cuda'):
    """从指定路径加载checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint路径不存在: {checkpoint_path}")
    
    print(f"\n{'='*80}")
    print(f"加载Checkpoint")
    print(f"{'='*80}")
    print(f"路径: {checkpoint_path}")
    
    # 加载tokenizer
    tokenizer_path = checkpoint_path / "tokenizer"
    if tokenizer_path.exists():
        print(f"✓ 加载tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        print("⚠ Tokenizer未找到，使用默认GPT2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型配置
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ 模型配置: {config['n_embd']}d, {config['n_layer']}层, {config['n_head']}头")
    else:
        raise ValueError(f"配置文件未找到: {config_path}")
    
    # 加载模型
    print(f"✓ 加载模型权重...")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型加载成功! 参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"{'='*80}\n")
    
    return model, tokenizer, config


def prepare_texts(dataset_id: str, max_samples: int = 128, max_retries: int = 5):
    """准备数据集文本（带重试机制，优先使用本地缓存）"""
    from datasets import load_dataset
    
    texts = []
    
    # 带重试的数据集加载（优先使用缓存）
    for attempt in range(1, max_retries + 1):
        try:
            # 优先使用本地缓存，如果不存在才下载
            download_mode = "reuse_cache_if_exists"
            
            if dataset_id == "sst2":
                dataset = load_dataset("glue", "sst2", split="validation", download_mode=download_mode)
                for example in dataset.select(range(min(max_samples, len(dataset)))):
                    texts.append(example["sentence"])
                    
            elif dataset_id == "squad":
                dataset = load_dataset("squad", split="validation", download_mode=download_mode)
                count = 0
                for example in dataset:
                    answers = example.get("answers", {}).get("text", [])
                    answer_text = answers[0] if answers else ""
                    text = " ".join(
                        filter(
                            None,
                            [example.get("context"), example.get("question"), answer_text],
                        )
                    )
                    if text:
                        texts.append(text)
                        count += 1
                        if count >= max_samples:
                            break
                            
            elif dataset_id == "lambada":
                dataset = load_dataset("lambada", split="test", download_mode=download_mode)
                for example in dataset.select(range(min(max_samples, len(dataset)))):
                    texts.append(example["text"])
            else:
                raise ValueError(f"Unknown dataset id: {dataset_id}")
            
            break  # 成功加载，跳出重试循环
            
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"⚠ 加载数据集失败 (尝试 {attempt}/{max_retries}): {e}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"✗ 数据集加载失败，已重试 {max_retries} 次")
                raise
    
    return texts


def compute_perplexity(model, tokenizer, texts: list, device: str, block_size: int = 256):
    """计算perplexity"""
    model_was_training = model.training
    model.eval()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    
    print(f"  计算perplexity (共{len(texts)}个样本)...")
    
    for text in tqdm(texts, desc="  处理中", leave=False):
        if not text:
            continue
        
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=block_size,
        )
        input_ids = encoded["input_ids"].to(device)
        labels = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        
        token_count = shift_labels.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count
    
    if model_was_training:
        model.train()
    
    if total_tokens == 0:
        return float("inf")
    
    return math.exp(total_loss / total_tokens)


def compute_accuracy_lambada(model, tokenizer, texts: list, device: str, block_size: int = 256):
    """计算LAMBADA准确率"""
    model_was_training = model.training
    model.eval()
    
    correct_predictions = 0
    total_samples = 0
    
    print(f"  计算LAMBADA准确率 (共{len(texts)}个样本)...")
    
    for text in tqdm(texts, desc="  处理中", leave=False):
        text = text.strip()
        if not text or ' ' not in text:
            continue
        
        # 分割上下文和最后一个词
        try:
            context, target_word = text.rsplit(' ', 1)
        except ValueError:
            continue
        
        total_samples += 1
        
        inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=block_size - 1).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # 获取对下一个词的预测 logits
            last_token_logits = outputs.logits[0, -1, :]
        
        # 找到概率最高的 token
        predicted_token_id = torch.argmax(last_token_logits).item()
        
        # 正确解码：将预测的token添加到上下文中，然后解码整个序列
        context_token_ids = inputs['input_ids'][0].tolist()
        full_predicted_ids = context_token_ids + [predicted_token_id]
        
        # 解码完整序列
        full_predicted_text = tokenizer.decode(full_predicted_ids, skip_special_tokens=True)
        
        # 提取预测的最后一个词
        predicted_last_word = full_predicted_text[len(tokenizer.decode(context_token_ids, skip_special_tokens=True)):].strip()
        
        # 比较预测词和目标词（不区分大小写）
        if predicted_last_word.lower() == target_word.lower():
            correct_predictions += 1
    
    if model_was_training:
        model.train()
    
    if total_samples == 0:
        return 0.0
    
    return correct_predictions / total_samples


def evaluate_dataset(model, tokenizer, dataset_name: str, device: str, 
                     max_samples: int = 128, block_size: int = 256):
    """评估单个数据集"""
    print(f"\n{'='*80}")
    print(f"评估数据集: {dataset_name.upper()}")
    print(f"{'='*80}")
    
    # 准备数据
    print(f"  加载数据集...")
    texts = prepare_texts(dataset_name, max_samples=max_samples)
    print(f"  ✓ 加载了 {len(texts)} 个样本")
    
    results = {}
    
    # 计算perplexity
    try:
        ppl = compute_perplexity(model, tokenizer, texts, device, block_size)
        results['perplexity'] = ppl
        print(f"  ✓ Perplexity: {ppl:.4f}")
    except Exception as e:
        results['perplexity_error'] = str(e)
        print(f"  ✗ Perplexity计算失败: {e}")
    
    # 对LAMBADA计算准确率
    if dataset_name == "lambada":
        try:
            acc = compute_accuracy_lambada(model, tokenizer, texts, device, block_size)
            results['accuracy'] = acc
            print(f"  ✓ Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        except Exception as e:
            results['accuracy_error'] = str(e)
            print(f"  ✗ Accuracy计算失败: {e}")
    
    print(f"{'='*80}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="评估checkpoint在下游任务上的性能",
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
    
    # 可选参数
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['sst2', 'squad', 'lambada'],
        choices=['sst2', 'squad', 'lambada'],
        help='要评估的数据集 (默认: sst2 squad lambada)'
    )
    
    parser.add_argument(
        '--max_samples',
        type=int,
        default=128,
        help='每个数据集的最大样本数 (默认: 128)'
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=256,
        help='序列长度 (默认: 256)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备 (默认: cuda)'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='结果输出文件。默认保存到checkpoint目录下的downstream_evaluation_results.json'
    )
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 打印配置
    print("\n" + "=" * 80)
    print("下游任务评估配置")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"数据集: {', '.join(args.datasets)}")
    print(f"Max samples: {args.max_samples}")
    print(f"Block size: {args.block_size}")
    print(f"Device: {args.device}")
    print(f"HF镜像: {os.environ.get('HF_ENDPOINT', '默认')}")
    print("=" * 80)
    
    # 加载checkpoint
    model, tokenizer, config = load_checkpoint(args.checkpoint_path, device=args.device)
    
    # 评估每个数据集
    all_results = {}
    for dataset_name in args.datasets:
        try:
            results = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                device=args.device,
                max_samples=args.max_samples,
                block_size=args.block_size
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"\n✗ 评估 {dataset_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
            continue
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    for dataset_name, results in all_results.items():
        print(f"\n{dataset_name.upper()}:")
        if 'perplexity' in results:
            print(f"  Perplexity: {results['perplexity']:.4f}")
        if 'accuracy' in results:
            print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        if 'error' in results:
            print(f"  Error: {results['error']}")
    print("=" * 80)
    
    # 保存结果
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path(args.checkpoint_path).parent / "downstream_evaluation_results.json"
    
    output_data = {
        'checkpoint_path': str(args.checkpoint_path),
        'timestamp': datetime.now().isoformat(),
        'config': {
            'max_samples': args.max_samples,
            'block_size': args.block_size,
            'device': args.device,
            'datasets': args.datasets
        },
        'model_config': config,
        'results': all_results
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

