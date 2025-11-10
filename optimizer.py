# --- 3.5. 优化器实现 (Optimizer Implementations) ---
import math
import torch

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

    def state_dict(self):
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            't': self.t,
            'm': [m.clone() for m in self.m],
            'v': [v.clone() for v in self.v],
        }

    def load_state_dict(self, state):
        self.lr = state.get('lr', self.lr)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.epsilon = state.get('epsilon', self.epsilon)
        self.t = state.get('t', 0)

        m_state = state.get('m', [])
        v_state = state.get('v', [])
        if len(m_state) != len(self.m) or len(v_state) != len(self.v):
            raise ValueError("State dict does not match parameter groups for CustomAdamOptimizer")

        for i in range(len(self.m)):
            self.m[i] = m_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)
            self.v[i] = v_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)

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

    def state_dict(self):
        return {
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'correct_bias': self.correct_bias,
            'cautious': self.cautious,
            'hidden_size': self.hidden_size,
            't': self.t,
            'exp_avg': [buf.clone() for buf in self.exp_avg],
            'exp_avg_sq': [buf.clone() for buf in self.exp_avg_sq],
        }

    def load_state_dict(self, state):
        self.lr = state.get('lr', self.lr)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.epsilon = state.get('epsilon', self.epsilon)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        self.correct_bias = state.get('correct_bias', self.correct_bias)
        self.cautious = state.get('cautious', self.cautious)
        self.hidden_size = state.get('hidden_size', self.hidden_size)
        self.t = state.get('t', 0)

        exp_avg_state = state.get('exp_avg', [])
        exp_avg_sq_state = state.get('exp_avg_sq', [])
        if len(exp_avg_state) != len(self.exp_avg) or len(exp_avg_sq_state) != len(self.exp_avg_sq):
            raise ValueError("State dict does not match parameter groups for MuDaMWOptimizer")

        for i in range(len(self.exp_avg)):
            self.exp_avg[i] = exp_avg_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)
            self.exp_avg_sq[i] = exp_avg_sq_state[i].clone().to(device=self.params[i].device, dtype=self.params[i].dtype)