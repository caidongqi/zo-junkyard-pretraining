"""
下游任务评估脚本 (Downstream Task Evaluation)

功能 (Features):
1. 文本生成对话任务 (Text Generation / Dialogue)
2. 分类任务 (Classification Tasks)
   - SST-2 情感分类
   - 其他GLUE任务
3. 对比预训练模型 vs 从头训练模型 (Pretrained vs From-Scratch)

用法 (Usage):
    # 评估文本生成能力
    python downstream_evaluation.py \
        --task generation \
        --checkpoint_path path/to/checkpoint \
        --prompts "Hello, how are you?" "What is AI?"
    
    # 评估SST-2分类任务（使用checkpoint）
    python downstream_evaluation.py \
        --task sst2 \
        --checkpoint_path path/to/checkpoint \
        --compare_baseline \
        --output_file results/sst2_eval.json
    
    # 对比预训练模型和从头训练模型在SST-2上的表现
    python downstream_evaluation.py \
        --task sst2 \
        --checkpoint_path path/to/finetuned/checkpoint \
        --baseline_checkpoint path/to/scratch/checkpoint \
        --output_file results/comparison.json
"""

import argparse
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Config,
)


# ==================== 模型加载 (Model Loading) ====================

def load_gpt2_checkpoint(checkpoint_path: str, device: str = 'cuda') -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """
    加载GPT-2风格的checkpoint（从reproduce_zo_paper.py训练的模型）
    
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
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.2f}M parameters")
    
    return model, tokenizer


def load_from_scratch_gpt2(model_size: str = '200M', device: str = 'cuda') -> Tuple[GPT2LMHeadModel, AutoTokenizer]:
    """
    创建一个从头初始化的GPT-2模型（未训练）
    
    Args:
        model_size: 模型大小
        device: 设备
    
    Returns:
        model: 从头初始化的GPT-2模型
        tokenizer: tokenizer
    """
    from model import create_model
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = create_model(model_size=model_size, vocab_size=len(tokenizer))
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"From-scratch model created: {total_params / 1e6:.2f}M parameters")
    
    return model, tokenizer


# ==================== 文本生成任务 (Text Generation) ====================

def generate_text(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    device: str = 'cuda'
) -> List[str]:
    """
    使用模型生成文本
    
    Args:
        model: GPT-2模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_length: 最大生成长度
        temperature: 温度参数
        top_k: top-k采样
        top_p: nucleus采样
        num_return_sequences: 生成序列数量
        device: 设备
    
    Returns:
        生成的文本列表
    """
    model.eval()
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 生成
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # 解码输出
    generated_texts = []
    for sequence in output_sequences:
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts


def evaluate_generation(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: str = 'cuda',
    **generation_kwargs
) -> Dict:
    """
    评估模型的文本生成能力
    
    Args:
        model: GPT-2模型
        tokenizer: tokenizer
        prompts: 提示列表
        device: 设备
        **generation_kwargs: 生成参数
    
    Returns:
        包含生成结果的字典
    """
    results = {
        'prompts': [],
        'generations': [],
        'model_info': {
            'num_parameters': sum(p.numel() for p in model.parameters()),
        }
    }
    
    print("\n" + "=" * 80)
    print("Text Generation Evaluation")
    print("=" * 80)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 80)
        
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            **generation_kwargs
        )
        
        for i, text in enumerate(generated_texts, 1):
            print(f"Generation {i}: {text}")
        
        results['prompts'].append(prompt)
        results['generations'].append(generated_texts)
    
    print("=" * 80)
    
    return results


# ==================== 分类任务 (Classification Tasks) ====================

def prepare_sst2_dataset(tokenizer: AutoTokenizer, batch_size: int = 32, max_length: int = 128):
    """
    准备SST-2数据集
    
    Args:
        tokenizer: tokenizer
        batch_size: 批次大小
        max_length: 最大序列长度
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    dataset = load_dataset("glue", "sst2")
    
    def preprocess(example):
        return tokenizer(
            example["sentence"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    
    tokenized = dataset.map(preprocess, batched=False)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in ["input_ids", "attention_mask", "label"]]
    )
    
    def collate_fn(features):
        batch = {}
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["input_ids"], dtype=torch.long) for f in features],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f.get("attention_mask", [1]*len(f["input_ids"])), dtype=torch.long) for f in features],
            batch_first=True,
            padding_value=0
        )
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch
    
    train_loader = DataLoader(tokenized["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader


def adapt_gpt2_for_classification(
    gpt2_model: GPT2LMHeadModel,
    num_labels: int = 2,
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    将GPT-2模型适配为分类模型
    
    Args:
        gpt2_model: GPT-2语言模型
        num_labels: 分类标签数量
        device: 设备
    
    Returns:
        分类模型
    """
    class GPT2ForClassification(torch.nn.Module):
        def __init__(self, gpt2_model, num_labels):
            super().__init__()
            self.transformer = gpt2_model.transformer
            self.config = gpt2_model.config
            self.num_labels = num_labels
            
            # 添加分类头
            self.classifier = torch.nn.Linear(self.config.n_embd, num_labels)
            
        def forward(self, input_ids, attention_mask=None, labels=None):
            # 获取transformer输出
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)
            
            # 使用最后一个token的hidden state（类似于GPT-2的做法）
            # 或者使用平均pooling
            if attention_mask is not None:
                # 使用最后一个非padding token
                sequence_lengths = attention_mask.sum(dim=1) - 1
                pooled_output = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]
            else:
                pooled_output = hidden_states[:, -1, :]
            
            # 分类
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            
            return {'loss': loss, 'logits': logits}
    
    model = GPT2ForClassification(gpt2_model, num_labels)
    model.to(device)
    return model


def evaluate_classification(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    task_name: str = 'Classification'
) -> Dict:
    """
    评估分类任务性能（零样本，不进行微调）
    
    Args:
        model: 分类模型
        dataloader: 数据加载器
        device: 设备
        task_name: 任务名称
    
    Returns:
        包含评估指标的字典
    """
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    
    print(f"\nEvaluating {task_name}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{task_name} Evaluation"):
            for k in batch:
                batch[k] = batch[k].to(device)
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            logits = outputs['logits']
            preds = logits.argmax(dim=-1)
            
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].numel()
            
            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item()
                num_batches += 1
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    
    accuracy = correct / max(1, total)
    avg_loss = total_loss / max(1, num_batches)
    
    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total,
        'predictions': all_preds[:100],  # 只保存前100个预测
        'labels': all_labels[:100],
    }
    
    print(f"\n{task_name} Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"  Loss: {avg_loss:.4f}")
    
    return results


def finetune_and_evaluate_classification(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = 'cuda',
    lr: float = 1e-4,
    epochs: int = 3,
    task_name: str = 'Classification'
) -> Dict:
    """
    微调并评估分类任务性能
    
    Args:
        model: 分类模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        lr: 学习率
        epochs: 训练轮数
        task_name: 任务名称
    
    Returns:
        包含训练和评估结果的字典
    """
    from torch.optim import AdamW
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    training_history = {
        'train_losses': [],
        'val_accuracies': [],
        'val_losses': [],
    }
    
    print(f"\nFinetuning on {task_name}...")
    print(f"Learning rate: {lr}, Epochs: {epochs}")
    
    for epoch in range(epochs):
        # 训练
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            for k in batch:
                batch[k] = batch[k].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = epoch_loss / num_batches
        training_history['train_losses'].append(avg_train_loss)
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                logits = outputs['logits']
                preds = logits.argmax(dim=-1)
                
                val_correct += (preds == batch["labels"]).sum().item()
                val_total += batch["labels"].numel()
                val_loss += outputs['loss'].item()
                val_batches += 1
        
        val_accuracy = val_correct / max(1, val_total)
        avg_val_loss = val_loss / max(1, val_batches)
        
        training_history['val_accuracies'].append(val_accuracy)
        training_history['val_losses'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f} ({val_correct}/{val_total})")
    
    # 最终评估
    final_results = evaluate_classification(model, val_loader, device, task_name)
    final_results['training_history'] = training_history
    
    return final_results


# ==================== SST-2 分类任务评估 ====================

def evaluate_sst2_task(
    checkpoint_path: Optional[str] = None,
    baseline_checkpoint: Optional[str] = None,
    compare_from_scratch: bool = False,
    model_size: str = '200M',
    device: str = 'cuda',
    batch_size: int = 32,
    finetune: bool = True,
    finetune_lr: float = 1e-4,
    finetune_epochs: int = 3,
) -> Dict:
    """
    评估SST-2情感分类任务
    
    Args:
        checkpoint_path: 主模型checkpoint路径
        baseline_checkpoint: 基线模型checkpoint路径（可选）
        compare_from_scratch: 是否对比从头训练的模型
        model_size: 模型大小（用于from scratch）
        device: 设备
        batch_size: 批次大小
        finetune: 是否进行微调
        finetune_lr: 微调学习率
        finetune_epochs: 微调轮数
    
    Returns:
        评估结果字典
    """
    results = {
        'task': 'sst2',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': batch_size,
            'finetune': finetune,
            'finetune_lr': finetune_lr,
            'finetune_epochs': finetune_epochs,
        },
        'models': {}
    }
    
    # 准备数据集
    print("\n" + "=" * 80)
    print("SST-2 Sentiment Classification Task")
    print("=" * 80)
    
    # 使用GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    print("\nLoading SST-2 dataset...")
    train_loader, val_loader = prepare_sst2_dataset(tokenizer, batch_size=batch_size)
    
    # 评估主模型
    if checkpoint_path:
        print(f"\n[Model 1] Evaluating checkpoint: {checkpoint_path}")
        gpt2_model, _ = load_gpt2_checkpoint(checkpoint_path, device)
        cls_model = adapt_gpt2_for_classification(gpt2_model, num_labels=2, device=device)
        
        if finetune:
            model_results = finetune_and_evaluate_classification(
                cls_model, train_loader, val_loader, device, 
                lr=finetune_lr, epochs=finetune_epochs, task_name='SST-2 (Finetuned Checkpoint)'
            )
        else:
            model_results = evaluate_classification(
                cls_model, val_loader, device, task_name='SST-2 (Zero-shot Checkpoint)'
            )
        
        results['models']['checkpoint'] = {
            'path': str(checkpoint_path),
            'results': model_results
        }
    
    # 评估基线模型
    if baseline_checkpoint:
        print(f"\n[Model 2] Evaluating baseline checkpoint: {baseline_checkpoint}")
        gpt2_model, _ = load_gpt2_checkpoint(baseline_checkpoint, device)
        cls_model = adapt_gpt2_for_classification(gpt2_model, num_labels=2, device=device)
        
        if finetune:
            model_results = finetune_and_evaluate_classification(
                cls_model, train_loader, val_loader, device,
                lr=finetune_lr, epochs=finetune_epochs, task_name='SST-2 (Baseline)'
            )
        else:
            model_results = evaluate_classification(
                cls_model, val_loader, device, task_name='SST-2 (Baseline Zero-shot)'
            )
        
        results['models']['baseline'] = {
            'path': str(baseline_checkpoint),
            'results': model_results
        }
    
    # 评估从头训练的模型
    if compare_from_scratch:
        print(f"\n[Model 3] Evaluating from-scratch model (size: {model_size})")
        gpt2_model, _ = load_from_scratch_gpt2(model_size=model_size, device=device)
        cls_model = adapt_gpt2_for_classification(gpt2_model, num_labels=2, device=device)
        
        if finetune:
            model_results = finetune_and_evaluate_classification(
                cls_model, train_loader, val_loader, device,
                lr=finetune_lr, epochs=finetune_epochs, task_name='SST-2 (From Scratch)'
            )
        else:
            model_results = evaluate_classification(
                cls_model, val_loader, device, task_name='SST-2 (From Scratch Zero-shot)'
            )
        
        results['models']['from_scratch'] = {
            'model_size': model_size,
            'results': model_results
        }
    
    # 打印对比结果
    print("\n" + "=" * 80)
    print("SST-2 Task Comparison Summary")
    print("=" * 80)
    for model_name, model_data in results['models'].items():
        model_results = model_data['results']
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {model_results['accuracy']:.4f}")
        print(f"  Loss: {model_results['loss']:.4f}")
        if 'training_history' in model_results:
            print(f"  Final Val Accuracy: {model_results['training_history']['val_accuracies'][-1]:.4f}")
    print("=" * 80)
    
    return results


# ==================== 主函数 (Main Function) ====================

def main():
    parser = argparse.ArgumentParser(
        description="下游任务评估（文本生成 + 分类任务）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 任务选择
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['generation', 'sst2', 'all'],
        help='评估任务类型'
    )
    
    # 模型路径
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='主模型checkpoint路径'
    )
    
    parser.add_argument(
        '--baseline_checkpoint',
        type=str,
        default=None,
        help='基线模型checkpoint路径（用于对比）'
    )
    
    parser.add_argument(
        '--compare_from_scratch',
        action='store_true',
        help='对比从头训练的模型'
    )
    
    parser.add_argument(
        '--model_size',
        type=str,
        default='200M',
        choices=['20M', '200M', '500M', '1B'],
        help='从头训练模型的大小'
    )
    
    # 生成任务参数
    parser.add_argument(
        '--prompts',
        nargs='+',
        default=[
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Tell me a story about",
        ],
        help='文本生成提示列表'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='生成文本的最大长度'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='生成温度'
    )
    
    parser.add_argument(
        '--num_generations',
        type=int,
        default=3,
        help='每个提示生成的文本数量'
    )
    
    # 分类任务参数
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批次大小'
    )
    
    parser.add_argument(
        '--finetune',
        action='store_true',
        default=True,
        help='是否微调（默认True）'
    )
    
    parser.add_argument(
        '--no_finetune',
        action='store_true',
        help='不进行微调，只做零样本评估'
    )
    
    parser.add_argument(
        '--finetune_lr',
        type=float,
        default=1e-4,
        help='微调学习率'
    )
    
    parser.add_argument(
        '--finetune_epochs',
        type=int,
        default=3,
        help='微调轮数'
    )
    
    # 通用参数
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备'
    )
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='结果输出文件路径（JSON格式）'
    )
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 处理finetune参数
    if args.no_finetune:
        args.finetune = False
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'tasks': {}
    }
    
    # 执行任务
    if args.task in ['generation', 'all']:
        if not args.checkpoint_path:
            print("Error: --checkpoint_path is required for generation task")
            return
        
        print("\n" + "=" * 80)
        print("TASK 1: TEXT GENERATION")
        print("=" * 80)
        
        # 加载模型
        model, tokenizer = load_gpt2_checkpoint(args.checkpoint_path, args.device)
        
        # 评估生成
        gen_results = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            prompts=args.prompts,
            device=args.device,
            max_length=args.max_length,
            temperature=args.temperature,
            num_return_sequences=args.num_generations,
        )
        
        all_results['tasks']['generation'] = gen_results
        
        # 如果需要对比
        if args.baseline_checkpoint or args.compare_from_scratch:
            print("\n" + "=" * 80)
            print("GENERATION COMPARISON")
            print("=" * 80)
            
            if args.baseline_checkpoint:
                print("\n[Baseline Model]")
                baseline_model, baseline_tokenizer = load_gpt2_checkpoint(
                    args.baseline_checkpoint, args.device
                )
                baseline_gen_results = evaluate_generation(
                    model=baseline_model,
                    tokenizer=baseline_tokenizer,
                    prompts=args.prompts,
                    device=args.device,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    num_return_sequences=args.num_generations,
                )
                all_results['tasks']['generation_baseline'] = baseline_gen_results
            
            if args.compare_from_scratch:
                print("\n[From-Scratch Model]")
                scratch_model, scratch_tokenizer = load_from_scratch_gpt2(
                    model_size=args.model_size, device=args.device
                )
                scratch_gen_results = evaluate_generation(
                    model=scratch_model,
                    tokenizer=scratch_tokenizer,
                    prompts=args.prompts,
                    device=args.device,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    num_return_sequences=args.num_generations,
                )
                all_results['tasks']['generation_from_scratch'] = scratch_gen_results
    
    if args.task in ['sst2', 'all']:
        print("\n" + "=" * 80)
        print("TASK 2: SST-2 SENTIMENT CLASSIFICATION")
        print("=" * 80)
        
        sst2_results = evaluate_sst2_task(
            checkpoint_path=args.checkpoint_path,
            baseline_checkpoint=args.baseline_checkpoint,
            compare_from_scratch=args.compare_from_scratch,
            model_size=args.model_size,
            device=args.device,
            batch_size=args.batch_size,
            finetune=args.finetune,
            finetune_lr=args.finetune_lr,
            finetune_epochs=args.finetune_epochs,
        )
        
        all_results['tasks']['sst2'] = sst2_results
    
    # 保存结果
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    main()







