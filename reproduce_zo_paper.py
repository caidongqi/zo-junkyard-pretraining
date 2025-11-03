import argparse
import time
import math
import os
import pickle
import csv
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import transformers
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 常量配置 (Constants) ---

INSTRUCT_COSINE_TARGET = 0.9

# --- 1. 配置与模型定义 (Configuration and Model Definition) ---

def create_model(vocab_size):
    """
    创建一个参数量级在 20M 左右的轻量级 GPT-2 模型。
    这与论文中使用的 Llama2-20M 类似，用于复现实验。
    """
    print("Initializing a small GPT-2 model (~20M parameters)...")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=768,
        n_layer=24,
        n_head=32,
        bos_token_id=vocab_size - 1,
        eos_token_id=vocab_size - 1,
    )
    model = GPT2LMHeadModel(config)
    
    # 计算并打印模型参数量
    params_total = sum(p.numel() for p in model.parameters())
    print(f"Model created. Total parameters: {params_total / 1e6:.2f}M")
    
    return model

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

def get_dataloader(tokenizer, batch_size=4, block_size=128, cache_dir="cache"):
    """
    加载 cosmopedia-100k 数据集并进行预处理。
    支持缓存以避免重复加载。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # 创建缓存文件名
    cache_file = cache_dir / f"dataset_bs{block_size}_samples20000.pkl"
    
    # 检查缓存是否存在
    if cache_file.exists():
        print(f"Loading dataset from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            examples = pickle.load(f)
        print(f"Dataset loaded from cache. Total samples: {len(examples)}")
        return DataLoader(examples, batch_size=batch_size, shuffle=True)
    
    print("Loading and preprocessing cosmopedia-100k dataset...")
    # 为了快速演示，我们只使用一小部分数据
    dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train", streaming=True)
    dataset = dataset.take(20000) # 使用20k个样本进行训练

    def tokenize_function(examples):
        # 使用长文本合并和分块的方式来创建样本
        text = "".join(examples["text"])
        return tokenizer(text, truncation=False)

    tokenized_texts = []
    for example in tqdm(dataset, desc="Reading dataset"):
        tokenized_texts.extend(tokenize_function(example)["input_ids"])
    
    # 将所有文本分块
    examples = []
    for i in range(0, len(tokenized_texts) - block_size + 1, block_size):
        examples.append(torch.tensor(tokenized_texts[i:i + block_size], dtype=torch.long))

    print(f"Dataset prepared. Total samples: {len(examples)}")
    
    # 保存到缓存
    print(f"Saving dataset to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(examples, f)
    
    return DataLoader(examples, batch_size=batch_size, shuffle=True)


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
        return loss_fn(logits.view(-1, logits.size(-1)), batch_labels.view(-1))

    def get_batch():
        if data_provider is None:
            return inputs, labels
        batch_inputs, batch_labels = data_provider()
        if batch_inputs.device != device:
            batch_inputs = batch_inputs.to(device)
        if batch_labels.device != device:
            batch_labels = batch_labels.to(device)
        return batch_inputs, batch_labels

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

    return grads


def compute_backprop_gradients(model, trainable_params, loss_fn, inputs, labels):
    """执行一次标准BP，返回loss和每个参数的梯度副本。"""
    for p in trainable_params:
        if p.grad is not None:
            p.grad.zero_()

    with torch.enable_grad():
        logits = model(inputs).logits
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss.backward()

    grads = []
    for p in trainable_params:
        if p.grad is None:
            grads.append(torch.zeros_like(p.data))
        else:
            grads.append(p.grad.detach().clone())

    model.zero_grad(set_to_none=False)

    return loss.detach(), grads


def generate_instruct_directions(bp_grads, q, cosine_target, total_norm, total_norm_sq):
    """生成与BP梯度方向保持给定余弦相似度的方向迭代器。"""
    if not bp_grads or q is None or q <= 0:
        return None

    if total_norm <= 1e-12:
        return None

    cosine_target = float(cosine_target)
    cosine_target = max(min(cosine_target, 0.9999), -0.9999)
    orth_scale = math.sqrt(max(0.0, 1.0 - cosine_target ** 2))
    eps = 1e-12

    def generator():
        for _ in range(q):
            noises = None
            for _attempt in range(6):
                noises = [torch.randn_like(g) for g in bp_grads]
                dot = 0.0
                for noise, grad in zip(noises, bp_grads):
                    dot += float(torch.sum(noise * grad).item())
                proj_coeff = dot / (total_norm_sq + eps)
                for i in range(len(noises)):
                    noises[i] = noises[i] - proj_coeff * bp_grads[i]

                noise_norm_sq = 0.0
                for noise in noises:
                    noise_norm_sq += float(torch.sum(noise * noise).item())

                if noise_norm_sq > eps:
                    noise_norm = math.sqrt(noise_norm_sq)
                    for i in range(len(noises)):
                        noises[i] = noises[i] / (noise_norm + eps)
                    break
            else:
                noises = [torch.randn_like(g) for g in bp_grads]
                noise_norm_sq = sum(float(torch.sum(n * n).item()) for n in noises) + eps
                noise_norm = math.sqrt(noise_norm_sq)
                for i in range(len(noises)):
                    noises[i] = noises[i] / (noise_norm + eps)

            direction = []
            for grad, noise in zip(bp_grads, noises):
                dir_tensor = cosine_target * grad + orth_scale * total_norm * noise
                direction.append(dir_tensor)
            yield direction

    return generator()


# --- 3.5. 优化器实现 (Optimizer Implementations) ---

def _zeropower_via_newtonschulz5_torch(G, steps=5):
    """Newton-Schulz 迭代计算零次幂投影（用于 MuDaMW）"""
    assert G.dim() == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T
    X = X / (torch.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X

def _adjust_lr_for_muon_torch(lr, shape):
    """调整学习率（用于 MuDaMW）"""
    A, B = shape[:2]
    return lr * (0.2 * math.sqrt(max(A, B)))

class CustomAdamOptimizer:
    """自定义 Adam 优化器（可用于 FO 和 ZO）"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, grads=None):
        """
        执行一步优化更新
        grads: 可选，外部提供的梯度列表（用于 ZO）。如果为 None，则使用参数的 .grad
        """
        self.t += 1
        
        for i, p in enumerate(self.params):
            if grads is not None:
                g = grads[i]
            else:
                if p.grad is None:
                    continue
                g = p.grad.data
            
            if g is None:
                continue
            
            # Adam 更新
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

class MuDaMWOptimizer:
    """MuDaMW 优化器（基于 flwr_server.py 的实现）"""
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6, 
                 weight_decay=0.0, correct_bias=True, cautious=False, hidden_size=768):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.correct_bias = correct_bias
        self.cautious = cautious
        self.hidden_size = hidden_size
        self.t = 0
        self.exp_avg = [torch.zeros_like(p.data) for p in self.params]
        self.exp_avg_sq = [torch.zeros_like(p.data) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, grads=None):
        """
        执行一步优化更新
        grads: 可选，外部提供的梯度列表（用于 ZO）。如果为 None，则使用参数的 .grad
        """
        self.t += 1
        
        for i, p in enumerate(self.params):
            if grads is not None:
                g = grads[i]
            else:
                if p.grad is None:
                    continue
                g = p.grad.data
            
            if g is None:
                continue
            
            # Decoupled weight decay
            if self.weight_decay > 0.0:
                p.data.add_(p.data, alpha=-self.lr * self.weight_decay)
            
            # 一二阶动量
            self.exp_avg[i] = self.beta1 * self.exp_avg[i] + (1.0 - self.beta1) * g
            self.exp_avg_sq[i] = self.beta2 * self.exp_avg_sq[i] + (1.0 - self.beta2) * (g ** 2)
            
            denom = torch.sqrt(self.exp_avg_sq[i]) + self.epsilon
            
            # 偏置校正
            step_size = self.lr
            if self.correct_bias:
                bias_correction1 = 1.0 - (self.beta1 ** self.t)
                bias_correction2 = 1.0 - (self.beta2 ** self.t)
                step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
            
            # 归一化梯度（可选 cautious）
            if self.cautious:
                mask = (self.exp_avg[i] * g > 0).float()
                scale = mask.numel() / (mask.sum() + 1.0)
                mask = mask * scale
                norm_grad = (self.exp_avg[i] * mask) / denom
            else:
                norm_grad = self.exp_avg[i] / denom
            
            # 一维向量 → 2D 正交化
            if norm_grad.dim() == 1 and norm_grad.numel() % self.hidden_size == 0 and norm_grad.numel() > self.hidden_size:
                G = norm_grad.reshape(self.hidden_size, -1)
                adj_lr = _adjust_lr_for_muon_torch(step_size, G.shape)
                G = _zeropower_via_newtonschulz5_torch(G, steps=5)
                norm_grad = G.reshape(-1)
                step = adj_lr
            elif norm_grad.dim() == 2:
                # 对于 2D 参数（如权重矩阵），直接应用正交化
                adj_lr = _adjust_lr_for_muon_torch(step_size, norm_grad.shape)
                norm_grad = _zeropower_via_newtonschulz5_torch(norm_grad, steps=5)
                step = adj_lr
            else:
                step = step_size
            
            p.data.add_(norm_grad, alpha=-step)

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
):
    """主训练函数"""
    
    # 设置
    transformers.logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = create_model(len(tokenizer)).to(device)
    dataloader = get_dataloader(tokenizer, batch_size)
    
    # 确定可训练参数
    trainable_params = get_trainable_parameters(model, scope)
    params_trainable = sum(p.numel() for p in trainable_params)
    params_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {params_trainable / 1e6:.2f}M ({params_trainable*100/params_total:.2f}% of total)")

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
    elif mode in zo_like_modes:
        if optimizer_type == 'sgd':
            optimizer = None  # 使用 vanilla SGD（手动更新）
            print(f"Using optimizer: Vanilla SGD (manual update)")
        elif optimizer_type == 'adam':
            optimizer = CustomAdamOptimizer(trainable_params, lr=lr)
            print(f"Using optimizer: Custom Adam")
        elif optimizer_type == 'mudamw':
            optimizer = MuDaMWOptimizer(trainable_params, lr=lr)
            print(f"Using optimizer: MuDaMW")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    loss_fn = CrossEntropyLoss()
    
    losses = []
    
    # 初始化CSV日志
    if csv_file:
        csv_path = Path(csv_file)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'epoch',
                'step',
                'mode',
                'scope',
                'q',
                'lr',
                'batch_size',
                'optimizer',
                'bp_interval',
                'loss',
                'grad_norm'
            ])
    
    # 开始训练
    model.train()
    step = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            inputs = batch.to(device)
            labels = inputs.clone()

            grad_norm = 0.0  # 默认值
            
            if mode == 'FO':
                if hasattr(optimizer, 'zero_grad'):
                    optimizer.zero_grad()
                logits = model(inputs).logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                
                # 计算梯度范数（用于记录）
                grad_norm_sq = 0.0
                for p in trainable_params:
                    if p.grad is not None:
                        grad_norm_sq += float(torch.sum(p.grad.detach() * p.grad.detach()).item())
                grad_norm = math.sqrt(grad_norm_sq)
                
                optimizer.step()
            
            elif mode in zo_like_modes:
                should_use_bp = (
                    mode in {'Calibrate', 'Instruct'}
                    and bp_interval is not None
                    and bp_interval > 0
                    and ((step + 1) % bp_interval == 0)
                )

                bp_grads = None
                if should_use_bp:
                    loss, bp_grads = compute_backprop_gradients(model, trainable_params, loss_fn, inputs, labels)
                else:
                    with torch.no_grad():
                        current_logits = model(inputs).logits
                        loss = loss_fn(current_logits.view(-1, current_logits.size(-1)), labels.view(-1))

                epsilon = 1e-4  # 增大扰动大小以提高数值稳定性

                manual_dirs = None
                total_norm_sq = None
                total_norm = None
                if mode == 'Instruct' and should_use_bp and bp_grads is not None:
                    total_norm_sq = 0.0
                    for g in bp_grads:
                        total_norm_sq += float(torch.sum(g * g).item())
                    total_norm = math.sqrt(total_norm_sq)
                    if total_norm > 0.0:
                        manual_dirs = generate_instruct_directions(
                            bp_grads,
                            q,
                            INSTRUCT_COSINE_TARGET,
                            total_norm,
                            total_norm_sq,
                        )
                    if manual_dirs is None and total_norm is not None and total_norm > 0.0:
                        manual_dirs = ([g / total_norm for g in bp_grads],)

                grad_paramwise = zo_gradient_estimator(
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

                # TODO: 把这里二者的关系变成一个超参数。
                if mode == 'Calibrate' and should_use_bp and bp_grads is not None:
                    # grad_paramwise = [
                        # 0.5 * (gz + gb)
                        # for gz, gb in zip(grad_paramwise, bp_grads)
                    # ]
                    grad_paramwise = bp_grads

                grad_norm_sq = 0.0
                for g in grad_paramwise:
                    if g is not None:
                        grad_norm_sq += float(torch.sum(g.detach() * g.detach()).item())
                grad_norm = math.sqrt(grad_norm_sq)

                if optimizer is None:
                    for p, g in zip(trainable_params, grad_paramwise):
                        if g is None:
                            continue
                        p.data -= lr * g
                else:
                    optimizer.step(grads=grad_paramwise)

            losses.append(loss.item())
            
            # 记录到CSV（每log_interval步记录一次）
            if csv_file and step % log_interval == 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row_q = q if mode in zo_like_modes else 'N/A'
                    row_bp_interval = bp_interval if mode in {'Calibrate', 'Instruct'} else 'N/A'
                    writer.writerow([
                        timestamp,
                        epoch + 1,
                        step,
                        mode,
                        scope,
                        row_q,
                        lr,
                        batch_size,
                        optimizer_type,
                        row_bp_interval,
                        loss.item(),
                        grad_norm
                    ])
            
            postfix = {
                "loss": f"{loss.item():.4f}",
                "grad_norm": f"{grad_norm:.4f}",
                "opt": optimizer_type
            }
            if mode in zo_like_modes:
                postfix["queries"] = f"{q}"
                if mode in {'Calibrate', 'Instruct'} and bp_interval is not None and bp_interval > 0:
                    postfix["bp_int"] = bp_interval

            pbar.set_postfix(postfix)
            
            step += 1

    # --- 5. 结果可视化 (Result Visualization) ---
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    q_text = q if mode in zo_like_modes else 'N/A'
    bp_text = bp_interval if mode in {'Calibrate', 'Instruct'} else 'N/A'
    plt.title(
        f'Training Loss Curve\nMode={mode}, Scope={scope}, q={q_text}, BP-Interval={bp_text}, '
        f'LR={lr}, Optimizer={optimizer_type.upper()}'
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
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
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
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建文件名
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    q_str = args.query_budget_q if args.mode in {'ZO', 'Calibrate', 'Instruct'} else 'na'
    bp_str = args.bp_interval if args.mode in {'Calibrate', 'Instruct'} else 'na'
    plot_filename = f"{args.mode}_{args.scope}_q{q_str}_bp{bp_str}_opt{args.optimizer}_lr{args.learning_rate}_bs{args.batch_size}.png"
    
    # 生成CSV文件名（如果未指定）
    if args.csv_file is None:
        csv_filename = f"{args.mode}_{args.scope}_q{q_str}_bp{bp_str}_opt{args.optimizer}_lr{args.learning_rate}_bs{args.batch_size}.csv"
        csv_file = results_dir / csv_filename
    else:
        csv_file = args.csv_file
    
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
        csv_file=csv_file,
        log_interval=args.log_interval,
        optimizer_type=args.optimizer,
        bp_interval=bp_interval_arg,
        queries_use_different_data=args.queries_use_different_data,
    )