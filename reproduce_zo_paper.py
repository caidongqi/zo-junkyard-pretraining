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
def zo_gradient_estimator(model, trainable_params, loss_fn, inputs, labels, q, epsilon, device):
    # 关闭dropout，加速且去噪
    was_training = model.training
    model.eval()

    # 记录可训练参数的原值
    original = [p.data.clone() for p in trainable_params]

    def f_loss():
        logits = model(inputs).logits
        return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    # 累加“投影标量 g_i”与其对应的方向种子，节省显存
    seeds = []
    proj_grads = []

    for _ in range(q):
        seed = torch.randint(0, 2**31-1, ()).item()
        seeds.append(seed)

        # +εz
        torch.manual_seed(seed)
        for p in trainable_params:
            z = torch.randn_like(p.data)
            p.data = p.data + epsilon * z
        loss_pos = f_loss()

        # -εz（从 +εz 直接到 -εz = 原值 - εz，因此需要再减 2εz）
        torch.manual_seed(seed)
        for p, p0 in zip(trainable_params, original):
            z = torch.randn_like(p.data)
            p.data = p.data - 2 * epsilon * z
        loss_neg = f_loss()

        # 恢复到原值
        for p, p0 in zip(trainable_params, original):
            p.data = p0.clone()

        proj_grads.append(((loss_pos - loss_neg) / (2 * epsilon)).item())

    # 用投影标量与同一随机种子重建 z 并得到“参数形状的梯度”
    grads = [torch.zeros_like(p.data) for p in trainable_params]
    denom = float(len(proj_grads))

    for seed, g in zip(seeds, proj_grads):
        torch.manual_seed(seed)
        for gi, p in enumerate(trainable_params):
            z = torch.randn_like(p.data)
            grads[gi].add_(g * z)

    # 平均
    for gi in range(len(grads)):
        grads[gi].div_(denom)

    # 恢复训练状态
    if was_training:
        model.train()

    # 返回“按参数形状”的梯度列表，训练循环里逐参数应用更新（可加权重衰减、跳过bias/LN）
    return grads


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

def train(mode, scope, q, lr, epochs, batch_size, device, plot_file, csv_file=None, log_interval=10, optimizer_type='sgd'):
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
    elif mode == 'ZO':
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
            writer.writerow(['timestamp', 'epoch', 'step', 'mode', 'scope', 'q', 'lr', 'batch_size', 'optimizer', 'loss', 'grad_norm'])
    
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
            
            elif mode == 'ZO':
                # 在更新前计算当前步的损失用于记录
                with torch.no_grad():
                    current_logits = model(inputs).logits
                    loss = loss_fn(current_logits.view(-1, current_logits.size(-1)), labels.view(-1))

                # 使用 ZO 估计梯度
                epsilon = 1e-4 # 增大扰动大小以提高数值稳定性
                grad_paramwise = zo_gradient_estimator(model, trainable_params, loss_fn, inputs, labels, q, epsilon, device)
                
                # 添加梯度范数监控（组合范数）
                grad_norm_sq = 0.0
                for g in grad_paramwise:
                    if g is not None:
                        grad_norm_sq += float(torch.sum(g.detach() * g.detach()).item())
                grad_norm = math.sqrt(grad_norm_sq)
                
                # 应用梯度更新
                if optimizer is None:
                    # 手动应用梯度更新 (Vanilla SGD step)
                    for p, g in zip(trainable_params, grad_paramwise):
                        if g is None:
                            continue
                        p.data -= lr * g
                else:
                    # 使用自定义优化器（Adam 或 MuDaMW）
                    optimizer.step(grads=grad_paramwise)

            losses.append(loss.item())
            
            # 记录到CSV（每log_interval步记录一次）
            if csv_file and step % log_interval == 0:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, epoch+1, step, mode, scope, q if mode=='ZO' else 'N/A', lr, batch_size, optimizer_type, loss.item(), grad_norm])
            
            if mode == 'ZO':
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "grad_norm": f"{grad_norm:.4f}",
                    "queries": f"{q}",
                    "opt": optimizer_type
                })
            else:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "grad_norm": f"{grad_norm:.4f}",
                    "opt": optimizer_type
                })
            
            step += 1

    # --- 5. 结果可视化 (Result Visualization) ---
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.title(f'Training Loss Curve\nMode={mode}, Scope={scope}, q={q if mode=="ZO" else "N/A"}, LR={lr}, Optimizer={optimizer_type.upper()}')
    plt.xlabel('Training Steps')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True)
    plt.savefig(plot_file)
    print(f"\nTraining finished. Loss curve saved to '{plot_file}'")


# --- 6. 主程序入口 (Main Entry Point) ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce 'Zeroth Order Optimization for Pretraining Language Models' paper's vanilla solver experiment.")
    parser.add_argument("--mode", type=str, required=True, choices=['FO', 'ZO'], help="Optimization mode: First-Order (FO) or Zeroth-Order (ZO).")
    parser.add_argument("--scope", type=str, default='reduced', choices=['full', 'reduced'], help="Training scope: 'full' model or 'reduced' (last layer only).")
    parser.add_argument("--query_budget_q", type=int, default=1, help="Query budget (q) for ZO. Number of random directions.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'mudamw'], 
                        help="Optimizer type: SGD (vanilla), Adam, or MuDaMW.")
    parser.add_argument("--csv_file", type=str, default=None, help="CSV file to save training logs.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval for CSV (every N steps).")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 创建文件名
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plot_filename = f"{args.mode}_{args.scope}_q{args.query_budget_q if args.mode == 'ZO' else 'na'}_opt{args.optimizer}_lr{args.learning_rate}_bs{args.batch_size}.png"
    
    # 生成CSV文件名（如果未指定）
    if args.csv_file is None:
        csv_filename = f"{args.mode}_{args.scope}_q{args.query_budget_q if args.mode == 'ZO' else 'na'}_opt{args.optimizer}_lr{args.learning_rate}_bs{args.batch_size}.csv"
        csv_file = results_dir / csv_filename
    else:
        csv_file = args.csv_file
    
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
        optimizer_type=args.optimizer
    )