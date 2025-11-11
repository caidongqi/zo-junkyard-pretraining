import argparse
import csv
import json
import logging
import math
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import transformers
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入自定义模块
from model import create_model
from data import get_dataloader

from optimizer import MuDaMWOptimizer, CustomAdamOptimizer

from generate_instruction import generate_instruct_directions_hybrid

# --- 常量配置 (Constants) ---

DEFAULT_INSTRUCT_COSINE_TARGET = 0.9
DEFAULT_INSTRUCT_NOISE_SCALE = 0.5

# --- 辅助函数：学习率调度器 (Helper Function: Learning Rate Scheduler) ---
def get_cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float
):
    """
    根据当前步数计算学习率，包含线性预热和余弦退火。
    """
    # 1. 线性预热阶段
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    # 2. 超过总步数，返回最小学习率
    if step > total_steps:
        return min_lr
    # 3. 余弦退火阶段
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# --- 1. 配置与模型定义 (Configuration and Model Definition) ---
# 注意: create_model 函数现在从 model.py 导入

def get_trainable_parameters(model, scope='full'):
    """
    根据指定的范围，返回可训练的参数列表。
    - 'full': 返回所有参数。
    - 'reduced': 只返回最后一层 (MLP + LayerNorm) 的参数。
    """
    if scope == 'full':
        print("Training scope: full model.")
        return list(model.parameters())
    
    elif scope == 'reduced':
        print("Training scope: reduced (only the last transformer block's MLP and LayerNorm).")
        trainable_params = []
        # 选择最后一层进行训练，以复现论文的核心发现
        last_block = model.transformer.h[-1]
        for name, param in last_block.named_parameters():
            if 'mlp' in name or 'ln_2' in name:
                trainable_params.append(param)
    
        # 同时，输出层也需要训练
        for param in model.lm_head.parameters():
            trainable_params.append(param)
            
        return trainable_params
    else:
        raise ValueError(f"Unknown training scope: {scope}")

# --- 2. 数据加载与预处理 (Data Loading and Preprocessing) ---
# 注意: get_dataloader 函数现在从 data.py 导入


# --- 3. 核心算法：ZO 梯度估计 (Core Algorithm: ZO Gradient Estimator) ---

@torch.no_grad()
def zo_gradient_estimator(
    model,
    trainable_params,
    loss_fn,
    inputs,
    labels,
    q,
    epsilon,
    device,
    manual_directions=None,
    data_provider=None,
):
    """ZO梯度估计器，支持可迭代的手动方向序列，可选地为每个查询提供独立数据。"""
    # 关闭dropout，加速且去噪
    was_training = model.training
    model.eval()

    # 记录可训练参数的原值
    original = [p.data.clone() for p in trainable_params]

    def compute_loss(batch_inputs, batch_labels):
        logits = model(batch_inputs).logits
        # Shift logits and labels for next-token prediction
        # logits[..., :-1, :] predicts labels[..., 1:]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_labels[..., 1:].contiguous()
        return loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    def get_batch():
        if data_provider is None:
            return inputs, labels
        batch_inputs, batch_labels = data_provider()
        if batch_inputs.device != device:
            batch_inputs = batch_inputs.to(device)
        if batch_labels.device != device:
            batch_labels = batch_labels.to(device)
        return batch_inputs, batch_labels

    # 计算原始参数位置的loss（用于记录）
    batch_inputs, batch_labels = get_batch()
    base_loss = compute_loss(batch_inputs, batch_labels)

    grads = [torch.zeros_like(p.data) for p in trainable_params]
    used_directions = 0
    manual_used = 0

    manual_iter = None
    if manual_directions is not None:
        manual_iter = iter(manual_directions)

    if manual_iter is not None:
        while True:
            if q is not None and manual_used >= q:
                break
            try:
                raw_direction = next(manual_iter)
            except StopIteration:
                break
            if raw_direction is None:
                continue

            direction = []
            for p, d in zip(trainable_params, raw_direction):
                dt = d.detach()
                if dt.device != device or dt.dtype != p.data.dtype:
                    dt = dt.to(device=device, dtype=p.data.dtype)
                direction.append(dt)

            batch_inputs, batch_labels = get_batch()

            for p, p0, d in zip(trainable_params, original, direction):
                p.data = p0 + epsilon * d
            loss_pos = compute_loss(batch_inputs, batch_labels)

            for p, p0, d in zip(trainable_params, original, direction):
                p.data = p0 - epsilon * d
            loss_neg = compute_loss(batch_inputs, batch_labels)

            for p, p0 in zip(trainable_params, original):
                p.data = p0.clone()

            proj = (loss_pos - loss_neg) / (2 * epsilon)
            for gi, d in enumerate(direction):
                grads[gi].add_(proj * d)

            used_directions += 1
            manual_used += 1

    remaining_q = 0
    if q is not None:
        remaining_q = max(q - manual_used, 0)

    # 随机方向部分，继续使用种子以节省显存
    seeds = []
    proj_grads = []
    for _ in range(remaining_q):
        seed = torch.randint(0, 2**31 - 1, ()).item()
        seeds.append(seed)

        batch_inputs, batch_labels = get_batch()

        torch.manual_seed(seed)
        for p in trainable_params:
            z = torch.randn_like(p.data)
            p.data = p.data + epsilon * z
        loss_pos = compute_loss(batch_inputs, batch_labels)

        torch.manual_seed(seed)
        for p, p0 in zip(trainable_params, original):
            z = torch.randn_like(p.data)
            p.data = p0 - epsilon * z
        loss_neg = compute_loss(batch_inputs, batch_labels)

        for p, p0 in zip(trainable_params, original):
            p.data = p0.clone()

        proj_grads.append(((loss_pos - loss_neg) / (2 * epsilon)).item())
        
        # 清理显存（每个query后）
        if (isinstance(device, str) and device == 'cuda') or (hasattr(device, 'type') and device.type == 'cuda'):
            torch.cuda.empty_cache()

    # 重建随机方向贡献
    for seed, proj in zip(seeds, proj_grads):
        torch.manual_seed(seed)
        for gi, p in enumerate(trainable_params):
            z = torch.randn_like(p.data)
            grads[gi].add_(proj * z)
        used_directions += 1

    if used_directions > 0:
        for gi in range(len(grads)):
            grads[gi].div_(float(used_directions))

    if was_training:
        model.train()

    return grads, base_loss


def save_latest_checkpoint(
    model,
    tokenizer,
    checkpoint_dir,
    optimizer_state=None,
    metadata=None,
    logger=None,
):
    if not checkpoint_dir:
        return

    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_path)
    if tokenizer is not None:
        tokenizer.save_pretrained(ckpt_path / "tokenizer")

    if optimizer_state is not None:
        torch.save(optimizer_state, ckpt_path / "optimizer.pt")

    if metadata is not None:
        with open(ckpt_path / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    if logger is not None:
        logger.info("Checkpoint saved to %s", ckpt_path)


def compute_backprop_gradients(model, trainable_params, loss_fn, inputs, labels):
    """执行一次标准BP，返回loss和每个参数的梯度副本。"""
    # 清理之前的梯度，使用set_to_none=True以释放显存
    model.zero_grad(set_to_none=True)
    
    with torch.enable_grad():
        logits = model(inputs).logits
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss.backward()

    grads = []
    for p in trainable_params:
        if p.grad is None:
            grads.append(torch.zeros_like(p.data))
        else:
            grads.append(p.grad.detach().clone())

    # 清理梯度，使用set_to_none=True以释放显存
    model.zero_grad(set_to_none=True)
    
    # 清理显存缓存
    if inputs.device.type == 'cuda':
        torch.cuda.empty_cache()

    return loss.detach(), grads

# --- 4. 训练循环 (Training Loops) ---

def train(
    mode,
    scope,
    q,
    lr,
    epochs,
    batch_size,
    device,
    plot_file,
    csv_file=None,
    log_interval=10,
    optimizer_type='sgd',
    bp_interval=None,
    queries_use_different_data=False,
    model_size='200M',
    dataset_name='cosmopedia-100k',
    max_samples=None,
    block_size=128,
    checkpoint_dir=None,
    logger=None,
    run_name=None,
    bp_dataset_name=None,
    bp_max_samples=None,
    blend_bp_gradient=False,
    instruct_cosine_target=DEFAULT_INSTRUCT_COSINE_TARGET,
    instruct_noise_scale=DEFAULT_INSTRUCT_NOISE_SCALE,
    # 新增LR调度器参数
    use_lr_scheduler=False,
    warmup_steps=300,
    min_lr=1e-6,
    # 新增梯度累积参数
    gradient_accumulation_steps=1,
):
    """主训练函数"""
    
    # 设置
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = create_model(model_size=model_size, vocab_size=len(tokenizer)).to(device)
    
    # 检查block_size是否超过模型的最大位置编码
    max_positions = model.config.n_positions
    if block_size > max_positions:
        print(f"⚠️  Warning: block_size ({block_size}) exceeds model's max positions ({max_positions})")
        print(f"   Automatically adjusting block_size to {max_positions}")
        block_size = max_positions
    
    dataloader = get_dataloader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        batch_size=batch_size,
        block_size=block_size,
        max_samples=max_samples,
    )
    
    # 为BP创建单独的dataloader（如果指定了不同的数据集）
    bp_dataloader = None
    if bp_dataset_name is not None and bp_dataset_name != dataset_name:
        print(f"Creating separate BP dataloader with dataset: {bp_dataset_name}")
        # BP dataloader也使用调整后的block_size
        bp_dataloader = get_dataloader(
            tokenizer=tokenizer,
            dataset_name=bp_dataset_name,
            batch_size=batch_size,
            block_size=block_size,
            max_samples=bp_max_samples,
        )
        if logger:
            logger.info("Separate BP dataloader created with dataset: %s", bp_dataset_name)

    csv_path = Path(csv_file) if csv_file else None
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(
            "Starting training run '%s' with configuration: mode=%s scope=%s q=%s lr=%s epochs=%s batch_size=%s gradient_accumulation_steps=%s effective_batch_size=%s block_size=%s optimizer=%s bp_interval=%s blend_bp_gradient=%s instruct_cosine_target=%s instruct_noise_scale=%s device=%s dataset=%s model_size=%s max_samples=%s bp_dataset=%s bp_max_samples=%s use_lr_scheduler=%s warmup_steps=%s min_lr=%s",
            run_name or "unnamed",
            mode,
            scope,
            q,
            lr,
            epochs,
            batch_size,
            gradient_accumulation_steps,
            batch_size * gradient_accumulation_steps,
            block_size,
            optimizer_type,
            bp_interval,
            blend_bp_gradient,
            instruct_cosine_target,
            instruct_noise_scale,
            device,
            dataset_name,
            model_size,
            max_samples,
            bp_dataset_name or "same_as_main",
            bp_max_samples or "default",
            use_lr_scheduler,
            warmup_steps if use_lr_scheduler else 'N/A',
            min_lr if use_lr_scheduler else 'N/A',
        )
    
    # 确定可训练参数
    trainable_params = get_trainable_parameters(model, scope)
    params_trainable = sum(p.numel() for p in trainable_params)
    params_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {params_trainable / 1e6:.2f}M ({params_trainable*100/params_total:.2f}% of total)")
    if logger:
        logger.info(
            "Trainable parameters: %.2fM (%.2f%% of total %.2fM)",
            params_trainable / 1e6,
            params_trainable * 100 / params_total,
            params_total / 1e6,
        )

    # 冻结非训练参数
    for p in model.parameters():
        p.requires_grad = False
    for p in trainable_params:
        p.requires_grad = True

    zo_like_modes = {'ZO', 'Calibrate', 'Instruct'}

    query_batch_provider = None
    if queries_use_different_data and mode in zo_like_modes:
        query_dataloader = DataLoader(dataloader.dataset, batch_size=batch_size, shuffle=True)
        query_iter = iter(query_dataloader)

        def _next_query_batch():
            nonlocal query_iter
            try:
                batch = next(query_iter)
            except StopIteration:
                query_iter = iter(query_dataloader)
                batch = next(query_iter)
            batch = batch.to(device)
            labels = batch.clone()
            return batch, labels

        query_batch_provider = _next_query_batch
        print("ZO queries will use fresh data batches per direction.")
        if logger:
            logger.info("ZO queries will use fresh data batches per direction.")

    # 为BP创建单独的batch provider（如果使用单独的数据集）
    bp_batch_provider = None
    if bp_dataloader is not None:
        bp_iter = iter(bp_dataloader)

        def _next_bp_batch():
            nonlocal bp_iter
            try:
                batch = next(bp_iter)
            except StopIteration:
                bp_iter = iter(bp_dataloader)
                batch = next(bp_iter)
            batch = batch.to(device)
            labels = batch.clone()
            return batch, labels

        bp_batch_provider = _next_bp_batch
        print(f"BP will use separate dataset: {bp_dataset_name}")
        if logger:
            logger.info("BP will use separate dataset: %s", bp_dataset_name)

    if mode in {'Calibrate', 'Instruct'}:
        if bp_interval is None or bp_interval <= 0:
            raise ValueError(f"Mode '{mode}' requires bp_interval > 0.")

    # 初始化优化器和损失函数
    optimizer = None
    if mode == 'FO':
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(trainable_params, lr=lr)
        elif optimizer_type == 'adam':
            optimizer = CustomAdamOptimizer(trainable_params, lr=lr)
        elif optimizer_type == 'mudamw':
            optimizer = MuDaMWOptimizer(trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        print(f"Using optimizer: {optimizer_type.upper()}")
        if logger:
            logger.info("Using optimizer: %s", optimizer_type.upper())
    elif mode in zo_like_modes:
        if optimizer_type == 'sgd':
            optimizer = None  # 使用 vanilla SGD（手动更新）
            print(f"Using optimizer: Vanilla SGD (manual update)")
            if logger:
                logger.info("Using optimizer: Vanilla SGD (manual update)")
        elif optimizer_type == 'adam':
            optimizer = CustomAdamOptimizer(trainable_params, lr=lr)
            print(f"Using optimizer: Custom Adam")
            if logger:
                logger.info("Using optimizer: Custom Adam")
        elif optimizer_type == 'mudamw':
            optimizer = MuDaMWOptimizer(trainable_params, lr=lr)
            print(f"Using optimizer: MuDaMW")
            if logger:
                logger.info("Using optimizer: MuDaMW")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    loss_fn = CrossEntropyLoss()
    
    losses = []
    
    # 初始化CSV日志
    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'epoch',
                'step',
                'mode',
                'scope',
                'q',
                'initial_lr',
                'current_lr',
                'batch_size',
                'gradient_accumulation_steps',
                'effective_batch_size',
                'optimizer',
                'bp_interval',
                'loss',
                'grad_norm'
            ])
    
    # 开始训练
    model.train()
    step = 0
    # 计算总优化步数（考虑梯度累积）
    batches_per_epoch = len(dataloader)
    total_optimization_steps = (batches_per_epoch * epochs) // gradient_accumulation_steps
    total_steps = total_optimization_steps  # 用于LR调度器
    
    last_metrics = {
        'loss': None,
        'grad_norm': None,
        'epoch': None,
        'step': None,
    }
    
    # 梯度累积计数器
    accumulation_counter = 0
    accumulated_loss = 0.0
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        if logger:
            logger.info("Epoch %s/%s started", epoch + 1, epochs)
        for batch_idx, batch in enumerate(pbar):
            # --- 学习率计算 ---
            if use_lr_scheduler:
                current_lr = get_cosine_schedule_with_warmup(
                    step=step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    max_lr=lr,  # args.learning_rate 作为最大学习率
                    min_lr=min_lr,
                )
                # 动态更新优化器内的学习率
                if optimizer is not None:
                    if hasattr(optimizer, 'param_groups'):
                        # PyTorch 标准优化器
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                    elif hasattr(optimizer, 'lr'):
                        # 自定义优化器（CustomAdamOptimizer, MuDaMWOptimizer）
                        optimizer.lr = current_lr
            else:
                current_lr = lr # 如果不使用调度器，则保持固定学习率

            inputs = batch.to(device)
            labels = inputs.clone()

            grad_norm = 0.0  # 默认值
            
            if mode == 'FO':
                # FO模式：支持梯度累积
                logits = model(inputs).logits
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # 梯度累积：损失除以累积步数以保持梯度尺度
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                
                # 累积损失（用于记录）
                accumulated_loss += loss.item()
                accumulation_counter += 1
                
                # 只在累积到指定步数时才更新参数
                should_update = (accumulation_counter >= gradient_accumulation_steps)
                
                if should_update:
                    # 计算梯度范数（用于记录）
                    grad_norm_sq = 0.0
                    for p in trainable_params:
                        if p.grad is not None:
                            grad_norm_sq += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
                    grad_norm = math.sqrt(grad_norm_sq)
                    
                    # 更新参数
                    optimizer.step()
                    
                    # 清零梯度
                    if hasattr(optimizer, 'zero_grad'):
                        optimizer.zero_grad()
                    else:
                        for p in trainable_params:
                            if p.grad is not None:
                                p.grad.zero_()
                    
                    # 计算平均损失
                    loss = torch.tensor(accumulated_loss / accumulation_counter)
                    
                    # 重置累积计数器
                    accumulation_counter = 0
                    accumulated_loss = 0.0
                    
                    # 增加优化步数
                    step += 1
                else:
                    # 如果还在累积阶段，跳过日志记录和checkpoint
                    continue
            
            elif mode in zo_like_modes:
                # ZO模式：每个batch都是一次优化步骤
                should_use_bp = (
                    mode in {'Calibrate', 'Instruct'}
                    and bp_interval is not None
                    and bp_interval > 0
                    and ((step + 1) % bp_interval == 0)
                )

                bp_grads = None
                if should_use_bp:
                    # 在计算BP梯度前清理显存
                    if (isinstance(device, str) and device == 'cuda') or (hasattr(device, 'type') and device.type == 'cuda'):
                        torch.cuda.empty_cache()
                    
                    # 如果有单独的BP数据集，则使用它；否则使用当前训练batch
                    if bp_batch_provider is not None:
                        bp_inputs, bp_labels = bp_batch_provider()
                        _, bp_grads = compute_backprop_gradients(model, trainable_params, loss_fn, bp_inputs, bp_labels)
                    else:
                        _, bp_grads = compute_backprop_gradients(model, trainable_params, loss_fn, inputs, labels)
                    
                    # 计算BP梯度后清理显存
                    if (isinstance(device, str) and device == 'cuda') or (hasattr(device, 'type') and device.type == 'cuda'):
                        torch.cuda.empty_cache()

                epsilon = 1e-4  # 增大扰动大小以提高数值稳定性

                manual_dirs = None
                if mode == 'Instruct' and should_use_bp and bp_grads is not None:
                    manual_dirs = generate_instruct_directions_hybrid(
                        bp_grads=bp_grads,
                        q=q,
                        cosine_target=instruct_cosine_target,
                        noise_scale=instruct_noise_scale,
                        device=device,
                    )
                    if manual_dirs is None:
                        total_norm_sq = 0.0
                        for g in bp_grads:
                            total_norm_sq += float(torch.sum(g * g).item())
                        total_norm = math.sqrt(total_norm_sq)
                        if total_norm > 0.0:
                            manual_dirs = ([g / total_norm for g in bp_grads],)

                grad_paramwise, loss = zo_gradient_estimator(
                    model,
                    trainable_params,
                    loss_fn,
                    inputs,
                    labels,
                    q,
                    epsilon,
                    device,
                    manual_directions=manual_dirs,
                    data_provider=query_batch_provider,
                )

                # 在Calibrate模式下使用BP梯度
                if mode == 'Calibrate' and should_use_bp and bp_grads is not None:
                    grad_paramwise = bp_grads
                
                # 在Instruct模式下，可选择混合BP和ZO梯度
                if mode == 'Instruct' and blend_bp_gradient and should_use_bp and bp_grads is not None:
                    grad_paramwise = [
                        0.5 * (gz + gb)
                        for gz, gb in zip(grad_paramwise, bp_grads)
                    ]

                grad_norm_sq = 0.0
                for g in grad_paramwise:
                    if g is not None:
                        grad_norm_sq += float(torch.sum(g.detach() * g.detach()).item())
                grad_norm = math.sqrt(grad_norm_sq)

                if optimizer is None: # 手动 SGD 更新
                    for p, g in zip(trainable_params, grad_paramwise):
                        if g is None:
                            continue
                        p.data -= current_lr * g # 使用动态学习率
                else: # 使用 Adam 或 MuDaMW
                    # 优化器的学习率已在循环开始时更新
                    optimizer.step(grads=grad_paramwise)
                
                # ZO模式：每个batch后增加step
                step += 1

            losses.append(loss.item())
            current_step = step
            last_metrics.update({
                'loss': float(loss.item()),
                'grad_norm': float(grad_norm),
                'epoch': epoch + 1,
                'step': current_step,
            })

            # 记录到CSV / 日志 / checkpoint（每 log_interval 步）
            should_log_step = (log_interval > 0) and (current_step % log_interval == 0)
            if should_log_step:
                timestamp_dt = datetime.now()
                timestamp = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
                row_q = q if mode in zo_like_modes else 'N/A'
                row_bp_interval = bp_interval if mode in {'Calibrate', 'Instruct'} else 'N/A'

                if csv_path:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            timestamp,
                            epoch + 1,
                            current_step,
                            mode,
                            scope,
                            row_q,
                            lr, # 初始最大学习率
                            current_lr, # 当前学习率
                            batch_size,
                            gradient_accumulation_steps if mode == 'FO' else 1,
                            batch_size * (gradient_accumulation_steps if mode == 'FO' else 1),
                            optimizer_type,
                            row_bp_interval,
                            loss.item(),
                            grad_norm
                        ])

                if logger:
                    logger.info(
                        "step=%s epoch=%s loss=%.6f grad_norm=%.6f lr=%.6e mode=%s scope=%s q=%s bp_interval=%s",
                        current_step,
                        epoch + 1,
                        loss.item(),
                        grad_norm,
                        current_lr,
                        mode,
                        scope,
                        row_q,
                        row_bp_interval,
                    )

                if checkpoint_path:
                    optimizer_state = None
                    if optimizer is not None and hasattr(optimizer, 'state_dict'):
                        optimizer_state = optimizer.state_dict()

                    metadata = {
                        'timestamp': timestamp_dt.isoformat(),
                        'run_name': run_name,
                        'mode': mode,
                        'scope': scope,
                        'epoch': epoch + 1,
                        'step': current_step,
                        'q': q if mode in zo_like_modes else None,
                        'learning_rate': lr, # 初始最大学习率
                        'current_learning_rate': current_lr, # 当前学习率
                        'batch_size': batch_size,
                        'gradient_accumulation_steps': gradient_accumulation_steps if mode == 'FO' else 1,
                        'effective_batch_size': batch_size * (gradient_accumulation_steps if mode == 'FO' else 1),
                        'optimizer': optimizer_type,
                        'bp_interval': bp_interval if mode in {'Calibrate', 'Instruct'} else None,
                        'loss': float(loss.item()),
                        'grad_norm': float(grad_norm),
                        'device': device,
                        'model_size': model_size,
                        'dataset': dataset_name,
                        'checkpoint_type': 'periodic',
                        'instruct_cosine_target': instruct_cosine_target if mode == 'Instruct' else None,
                        'instruct_noise_scale': instruct_noise_scale if mode == 'Instruct' else None,
                        'use_lr_scheduler': use_lr_scheduler,
                        'warmup_steps': warmup_steps if use_lr_scheduler else None,
                        'min_lr': min_lr if use_lr_scheduler else None,
                    }
                    save_latest_checkpoint(
                        model,
                        tokenizer,
                        checkpoint_path,
                        optimizer_state=optimizer_state,
                        metadata=metadata,
                        logger=logger,
                    )
            
            postfix = {
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}", # 在进度条中显示当前学习率
                "grad_norm": f"{grad_norm:.4f}",
                "opt": optimizer_type
            }
            if mode in zo_like_modes:
                postfix["queries"] = f"{q}"
                if mode in {'Calibrate', 'Instruct'} and bp_interval is not None and bp_interval > 0:
                    postfix["bp_int"] = bp_interval

            pbar.set_postfix(postfix)
        if logger:
            logger.info("Epoch %s/%s completed", epoch + 1, epochs)

    if checkpoint_path and last_metrics['loss'] is not None:
        optimizer_state = None
        if optimizer is not None and hasattr(optimizer, 'state_dict'):
            optimizer_state = optimizer.state_dict()
        final_metadata = {
            'timestamp': datetime.now().isoformat(),
            'run_name': run_name,
            'mode': mode,
            'scope': scope,
            'epoch': last_metrics['epoch'],
            'step': last_metrics['step'],
            'q': q if mode in zo_like_modes else None,
            'learning_rate': lr,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps if mode == 'FO' else 1,
            'effective_batch_size': batch_size * (gradient_accumulation_steps if mode == 'FO' else 1),
            'optimizer': optimizer_type,
            'bp_interval': bp_interval if mode in {'Calibrate', 'Instruct'} else None,
            'loss': last_metrics['loss'],
            'grad_norm': last_metrics['grad_norm'],
            'device': device,
            'model_size': model_size,
            'dataset': dataset_name,
            'checkpoint_type': 'final',
            'instruct_cosine_target': instruct_cosine_target if mode == 'Instruct' else None,
            'instruct_noise_scale': instruct_noise_scale if mode == 'Instruct' else None,
            'use_lr_scheduler': use_lr_scheduler,
            'warmup_steps': warmup_steps if use_lr_scheduler else None,
            'min_lr': min_lr if use_lr_scheduler else None,
        }
        save_latest_checkpoint(
            model,
            tokenizer,
            checkpoint_path,
            optimizer_state=optimizer_state,
            metadata=final_metadata,
            logger=logger,
        )

    if logger:
        logger.info("Training complete. Total steps: %s", step)

    # --- 5. 结果可视化 (Result Visualization) ---
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    q_text = q if mode in zo_like_modes else 'N/A'
    bp_text = bp_interval if mode in {'Calibrate', 'Instruct'} else 'N/A'
    scheduler_text = f"Scheduler(warmup={warmup_steps})" if use_lr_scheduler else "Fixed LR"
    plt.title(
        f'Training Loss Curve\nMode={mode}, Scope={scope}, q={q_text}, BP-Interval={bp_text}, '
        f'LR={lr}, Optimizer={optimizer_type.upper()}, {scheduler_text}'
    )
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True)
    plt.savefig(plot_file)
    print(f"\nTraining finished. Loss curve saved to '{plot_file}'")


# --- 6. 主程序入口 (Main Entry Point) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce 'Zeroth Order Optimization for Pretraining Language Models' paper's vanilla solver experiment.")
    parser.add_argument("--mode", type=str, required=True, choices=['FO', 'ZO', 'Calibrate', 'Instruct'], help="Optimization mode.")
    parser.add_argument("--scope", type=str, default='reduced', choices=['full', 'reduced'], help="Training scope: 'full' model or 'reduced' (last layer only).")
    parser.add_argument("--query_budget_q", type=int, default=1, help="Query budget (q) for ZO. Number of random directions.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate (acts as max_lr if scheduler is used).")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (only for FO mode). Effective batch size = batch_size * gradient_accumulation_steps.")
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'mudamw'], 
                        help="Optimizer type: SGD (vanilla), Adam, or MuDaMW.")
    parser.add_argument("--csv_file", type=str, default=None, help="CSV file to save training logs.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval for CSV (every N steps).")
    parser.add_argument("--bp_interval", type=int, default=0, help="Backprop interval for hybrid modes (Calibrate/Instruct). Set > 0 to enable.")
    parser.add_argument(
        "--queries_use_different_data",
        action="store_true", default=True,
        help="Use a fresh data batch for each ZO query instead of reusing the training batch.",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Directory to store run logs. Defaults to logs/<run_name>_<timestamp>.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to store the latest checkpoint. Defaults to <log_dir>/checkpoint.")
    parser.add_argument("--disable_checkpoint", action="store_true", help="Disable checkpoint saving.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name to organize logs and checkpoints.")
    
    # 模型和数据集参数
    parser.add_argument("--model_size", type=str, default='20M', 
                        choices=['20M', '200M', '500M', '1B'],
                        help="Model size: 20M (fast), 200M (GPT-2 Small), 500M (medium), 1B (large).")
    parser.add_argument("--dataset", type=str, default='cosmopedia-100k',
                        choices=['cosmopedia-100k', 'cosmopedia', 'wikitext-103', 'openwebtext', 
                                'c4', 'tinystories', 'pile-subset', 'fineweb', 'fineweb-edu', 
                                'fineweb-edu-10bt'],
                        help="Dataset name for training.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use from dataset. None=use recommended value.")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Sequence length (block size) for tokenization (default: 128).")
    
    # BP数据集参数（用于Calibrate/Instruct模式）
    parser.add_argument("--bp_dataset", type=str, default=None,
                        choices=['cosmopedia-100k', 'cosmopedia', 'wikitext-103', 'openwebtext', 
                                'c4', 'tinystories', 'pile-subset', 'fineweb', 'fineweb-edu', 
                                'fineweb-edu-10bt'],
                        help="Separate dataset for BP gradient computation (Calibrate/Instruct modes). If not specified, uses same as --dataset.")
    parser.add_argument("--bp_max_samples", type=int, default=None,
                        help="Maximum number of samples to use from BP dataset. None=use recommended value.")
    
    # 梯度混合参数（用于Instruct模式）
    parser.add_argument("--blend_bp_gradient", action="store_true", default=True,
                        help="In Instruct mode, blend BP gradient with ZO gradient (average of both). Only effective when bp_interval > 0.")
    parser.add_argument("--instruct_cosine_target", type=float, default=DEFAULT_INSTRUCT_COSINE_TARGET,
                        help="Target cosine similarity for hybrid instruct direction generation.")
    parser.add_argument("--instruct_noise_scale", type=float, default=DEFAULT_INSTRUCT_NOISE_SCALE,
                        help="Noise scale for hybrid instruct direction generation.")

    # 新增：学习率调度器参数
    parser.add_argument("--use_lr_scheduler", action="store_true",
                        help="Enable cosine learning rate scheduler with warmup.")
    parser.add_argument("--warmup_steps", type=int, default=300,
                        help="Number of warmup steps for the LR scheduler.")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine annealing.")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    run_start = datetime.now()
    timestamp_str = run_start.strftime("%Y%m%d_%H%M%S")

    q_str = args.query_budget_q if args.mode in {'ZO', 'Calibrate', 'Instruct'} else 'na'
    bp_str = args.bp_interval if args.mode in {'Calibrate', 'Instruct'} else 'na'
    default_run_name = (
        f"{args.mode}_{args.scope}_q{q_str}_bp{bp_str}_opt{args.optimizer}_lr{args.learning_rate}"
        f"_bs{args.batch_size}_ct{args.instruct_cosine_target}_ns{args.instruct_noise_scale}"
    )
    run_name = args.run_name or default_run_name

    if args.log_dir:
        run_log_dir = Path(args.log_dir)
        if not run_log_dir.is_absolute():
            run_log_dir = Path.cwd() / run_log_dir
    else:
        base_log_dir = Path.cwd() / "logs"
        run_log_dir = base_log_dir / f"{run_name}_{timestamp_str}"

    run_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs will be stored in {run_log_dir}")

    log_file = run_log_dir / f"training_{timestamp_str}.log"
    logger = logging.getLogger(f"reproduce_zo_paper.{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info("Run directory initialized at %s", run_log_dir)

    if args.csv_file is None:
        csv_file_path = run_log_dir / f"{run_name}.csv"
    else:
        csv_file_path = Path(args.csv_file)
        if not csv_file_path.is_absolute():
            csv_file_path = run_log_dir / csv_file_path

    if args.disable_checkpoint:
        checkpoint_path = None
    else:
        if args.checkpoint_dir:
            checkpoint_path = Path(args.checkpoint_dir)
            if not checkpoint_path.is_absolute():
                checkpoint_path = run_log_dir / args.checkpoint_dir
        else:
            checkpoint_path = run_log_dir / "checkpoint"

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plot_filename = (
        f"{args.mode}_{args.scope}_q{q_str}_bp{bp_str}_opt{args.optimizer}_lr{args.learning_rate}"
        f"_bs{args.batch_size}_ct{args.instruct_cosine_target}_ns{args.instruct_noise_scale}.png"
    )

    bp_interval_arg = args.bp_interval if args.bp_interval > 0 else None

    train(
        mode=args.mode,
        scope=args.scope,
        q=args.query_budget_q,
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        plot_file=results_dir / plot_filename,
        csv_file=str(csv_file_path),
        log_interval=args.log_interval,
        optimizer_type=args.optimizer,
        bp_interval=bp_interval_arg,
        queries_use_different_data=args.queries_use_different_data,
        model_size=args.model_size,
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        block_size=args.block_size,
        checkpoint_dir=str(checkpoint_path) if checkpoint_path else None,
        logger=logger,
        run_name=run_name,
        bp_dataset_name=args.bp_dataset,
        bp_max_samples=args.bp_max_samples,
        blend_bp_gradient=args.blend_bp_gradient,
        instruct_cosine_target=args.instruct_cosine_target,
        instruct_noise_scale=args.instruct_noise_scale,
        # 传递调度器参数
        use_lr_scheduler=args.use_lr_scheduler,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        # 传递梯度累积参数
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print(f"CSV metrics saved to {csv_file_path}")
    if checkpoint_path:
        print(f"Latest checkpoint stored at {checkpoint_path}")
        logger.info("Latest checkpoint stored at %s", checkpoint_path)
    else:
        print("Checkpoint saving disabled.")
        logger.info("Checkpoint saving disabled.")
    print(f"Training logs saved to {log_file}")
    logger.info("Training logs saved to %s", log_file)

    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()