# server.py (最终修正版)
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
import argparse
import numpy as np
from model import get_model, get_model_parameters
import csv
import os
import subprocess
import time
import json

# --- 默认配置（可被 CLI 覆盖，运行时不修改全局，仅作默认值） ---
DEFAULT_MUON_LR = 0.01
DEFAULT_MUON_BETA1 = 0.9
DEFAULT_MUON_BETA2 = 0.999
DEFAULT_MUON_EPSILON = 1e-8
DEFAULT_PERTURBATION_H = 0.01

# 单独的 Adam 超参（用于 BP_ADAM 与 ZO_ADAM）
DEFAULT_ADAM_LR = 0.001
DEFAULT_ADAM_BETA1 = 0.9
DEFAULT_ADAM_BETA2 = 0.999
DEFAULT_ADAM_EPSILON = 1e-8

def _zeropower_via_newtonschulz5_np(G, steps=5):
    # G: np.ndarray, 2D
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G
    if G.shape[0] > G.shape[1]:
        X = X.T
    X = X / (np.linalg.norm(X) + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X

def _adjust_lr_for_muon_np(lr, shape):
    # shape: (A, B)
    A, B = shape[:2]
    return lr * (0.2 * np.sqrt(max(A, B)))

class ZOCloudMuonStrategy(FedAvg):
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        min_fit_clients: int,
        min_available_clients: int,
        train_mode: str = "ZO_MUON",  # 可选: BP_ADAM / ZO_ADAM / ZO_MUON
        block_until_clients: int = 0,
        muon_lr: float = DEFAULT_MUON_LR,
        muon_beta1: float = DEFAULT_MUON_BETA1,
        muon_beta2: float = DEFAULT_MUON_BETA2,
        muon_epsilon: float = DEFAULT_MUON_EPSILON,
        perturbation_h: float = DEFAULT_PERTURBATION_H,
        # Adam 超参
        adam_lr: float = DEFAULT_ADAM_LR,
        adam_beta1: float = DEFAULT_ADAM_BETA1,
        adam_beta2: float = DEFAULT_ADAM_BETA2,
        adam_epsilon: float = DEFAULT_ADAM_EPSILON,
    ):
        super().__init__(
            initial_parameters=initial_parameters,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            # 告诉 FedAvg 我们不需要它的评估逻辑
            min_evaluate_clients=0,
            evaluate_fn=None,
        )
        # 修改点 1: 初始化一个变量来暂存每轮的模型参数
        self.current_parameters: Optional[Parameters] = initial_parameters
        
        # 初始化 Muon/Adam 状态变量
        model_dim = len(parameters_to_ndarrays(initial_parameters)[0])
        self.m = np.zeros(model_dim, dtype=np.float32)
        self.v = np.zeros(model_dim, dtype=np.float32)
        self.t = 0
        # Adam 独立状态（用于 BP_ADAM / ZO_ADAM）
        self.adam_m = np.zeros(model_dim, dtype=np.float32)
        self.adam_v = np.zeros(model_dim, dtype=np.float32)
        self.adam_t = 0
        self.train_mode = train_mode.upper()
        self.block_until_clients = int(block_until_clients)
        # 保存优化器与ZO扰动超参
        self.muon_lr = float(muon_lr)
        self.muon_beta1 = float(muon_beta1)
        self.muon_beta2 = float(muon_beta2)
        self.muon_epsilon = float(muon_epsilon)
        self.perturbation_h = float(perturbation_h)
        # 保存 Adam 超参（与 Muon 区分开）
        self.adam_lr = float(adam_lr)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_epsilon = float(adam_epsilon)

    # 修改点 2: 覆盖 configure_fit 来 "捕获" 当前轮次的模型参数
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """在每轮开始时，保存当前模型参数。"""
        self.current_parameters = parameters
        # 可选：阻塞等待直到有足够客户端连接
        target = self.block_until_clients
        if target and target > 0:
            while True:
                available = 0
                try:
                    available = client_manager.num_available()  # 优先使用 Flower API（若存在）
                except Exception:
                    try:
                        # 退化方案：all() 可能返回 {client_id: ClientProxy}
                        available = len(getattr(client_manager, "all")())
                    except Exception:
                        available = 0
                if available >= target:
                    break
                time.sleep(1)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """根据训练模式执行聚合与更新。"""
        if not results:
            return None, {}
        
        # 模式 1: BP + Adam -> 客户端仅上报梯度；服务器端聚合梯度并执行 Adam 更新
        if self.train_mode == "BP_ADAM":
            if self.current_parameters is None:
                return None, {}
            current_params_flat = parameters_to_ndarrays(self.current_parameters)[0]
            model_dim = len(current_params_flat)

            # 记录客户端指标（如有）
            client_losses = []
            # 按样本数加权聚合梯度
            grad_sum = np.zeros(model_dim, dtype=np.float32)
            total_examples = 0
            for _, fit_res in results:
                metrics = fit_res.metrics or {}
                if "avg_loss" in metrics:
                    client_losses.append(float(metrics["avg_loss"]))
                arrs = parameters_to_ndarrays(fit_res.parameters)
                if not arrs:
                    continue
                grad_vec = arrs[0].astype(np.float32, copy=False)
                n = int(getattr(fit_res, "num_examples", 1) or 1)
                if grad_vec.shape[0] != model_dim:
                    # 跳过形状不匹配的梯度
                    continue
                grad_sum += grad_vec * float(n)
                total_examples += n

            if total_examples == 0:
                return None, {}
            estimated_grad = grad_sum / float(total_examples)

            # 写入轮次平均 loss（可选）
            if client_losses:
                log_dir = getattr(self, "log_dir", os.path.join("logs", "default"))
                os.makedirs(log_dir, exist_ok=True)
                metrics_path = os.path.join(log_dir, "metrics.csv")
                write_header = not os.path.exists(metrics_path)
                with open(metrics_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["round", "avg_loss"]) 
                    writer.writerow([server_round, float(np.mean(client_losses))])
                print(f"服务器轮次 {server_round}: 平均loss={float(np.mean(client_losses)):.6f} 已写入 {metrics_path}")

            # 使用独立 Adam 状态进行服务器端更新（不使用 Muon 状态/超参）
            self.adam_t += 1
            self.adam_m = self.adam_beta1 * self.adam_m + (1 - self.adam_beta1) * estimated_grad
            self.adam_v = self.adam_beta2 * self.adam_v + (1 - self.adam_beta2) * (estimated_grad ** 2)
            m_hat = self.adam_m / (1 - self.adam_beta1 ** self.adam_t)
            v_hat = self.adam_v / (1 - self.adam_beta2 ** self.adam_t)

            lr = self.adam_lr
            updated_params_flat = current_params_flat - lr * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)
            print(f"服务器轮次 {server_round}: BP 模式下已在服务器端通过 Adam 更新模型（独立状态）。")

            return ndarrays_to_parameters([updated_params_flat]), {}
            
        # 新协议：客户端仅上传随机种子与标量损失，服务器端用同一 seed 重建方向
        if self.current_parameters is None:
            return None, {}
        current_params_flat = parameters_to_ndarrays(self.current_parameters)[0]
        model_dim = len(current_params_flat)

        grad_sum = np.zeros(model_dim, dtype=np.float32)
        sample_count = 0
        for _, fit_res in results:
            data = parameters_to_ndarrays(fit_res.parameters)
            seeds_arr, scalar_pairs = data[0], data[1]
            seeds_arr = seeds_arr.astype(np.int64, copy=False)
            for seed, (loss_pos, loss_neg) in zip(seeds_arr.tolist(), scalar_pairs.tolist()):
                rng = np.random.default_rng(int(seed))
                u_i = rng.standard_normal(model_dim, dtype=np.float32)
                norm = np.linalg.norm(u_i)
                if norm == 0:
                    u_i = rng.standard_normal(model_dim, dtype=np.float32)
                    norm = np.linalg.norm(u_i) + 1e-12
                u_i /= norm
                diff = float(loss_pos) - float(loss_neg)
                grad_sum += (diff / (2 * self.perturbation_h)) * u_i
                sample_count += 1

        if sample_count == 0:
            return None, {}
        estimated_grad = grad_sum / float(sample_count)
        
        print(f"服务器轮次 {server_round}: 近似梯度已计算，范数: {np.linalg.norm(estimated_grad)}")

        # --- 聚合并记录客户端上报的 avg_pair_loss ---
        client_losses = []
        log_dir = getattr(self, "log_dir", os.path.join("logs", "default"))
        os.makedirs(log_dir, exist_ok=True)
        metrics_path = os.path.join(log_dir, "metrics.csv")
        client_metrics_path = os.path.join(log_dir, "client_metrics.csv")
        write_header = not os.path.exists(metrics_path)
        write_client_header = not os.path.exists(client_metrics_path)
        with open(metrics_path, mode="a", newline="") as f, open(client_metrics_path, mode="a", newline="") as fc:
            writer = csv.writer(f)
            writer_c = csv.writer(fc)
            if write_header:
                writer.writerow(["round", "avg_pair_loss"])
            if write_client_header:
                writer_c.writerow(["round", "client_id", "avg_pair_loss", "train_mode", "local_steps", "client_lr", "seq_length", "batch_size", "use_fineweb", "num_directions", "perturbation_h"])
            for _, fit_res in results:
                metrics = fit_res.metrics or {}
                if "avg_pair_loss" in metrics:
                    client_losses.append(float(metrics["avg_pair_loss"]))
                    writer_c.writerow([
                        server_round,
                        metrics.get("client_id", ""),
                        metrics.get("avg_pair_loss", ""),
                        metrics.get("train_mode", ""),
                        metrics.get("local_steps", ""),
                        metrics.get("client_lr", ""),
                        metrics.get("seq_length", ""),
                        metrics.get("batch_size", ""),
                        metrics.get("use_fineweb", ""),
                        metrics.get("num_directions", ""),
                        metrics.get("perturbation_h", ""),
                    ])
            if client_losses:
                round_loss = float(np.mean(client_losses))
                writer.writerow([server_round, round_loss])
                print(f"服务器轮次 {server_round}: 平均loss={round_loss:.6f} 已写入 {metrics_path}")

        # 修改点 3: 使用 self.current_parameters 作为更新前的模型参数
        
        # ZO 分支：根据 train_mode 选择 Adam 或 Muon，使用对应的独立状态
        if self.train_mode == "ZO_ADAM":
            # Adam 状态
            self.adam_t += 1
            self.adam_m = self.adam_beta1 * self.adam_m + (1 - self.adam_beta1) * estimated_grad
            self.adam_v = self.adam_beta2 * self.adam_v + (1 - self.adam_beta2) * (estimated_grad ** 2)
            m_hat = self.adam_m / (1 - self.adam_beta1 ** self.adam_t)
            v_hat = self.adam_v / (1 - self.adam_beta2 ** self.adam_t)
            lr = self.adam_lr
            updated_params_flat = current_params_flat - lr * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)
            print(f"服务器轮次 {server_round}: 模型已通过 Adam 更新（ZO_ADAM）。")
        else:
            # MuDaMW（AdamW with optional Muon-style ortho on 2D reshape）
            self.t += 1

            # 超参（若未设置则给默认）
            mu_beta1 = getattr(self, "muon_beta1", 0.9)
            mu_beta2 = getattr(self, "muon_beta2", 0.999)
            mu_eps   = getattr(self, "muon_epsilon", 1e-6)
            lr       = getattr(self, "muon_lr", 1e-3)
            wd       = getattr(self, "muon_weight_decay", 0.0)
            correct_bias = getattr(self, "muon_correct_bias", True)
            cautious     = getattr(self, "muon_cautious", False)
            hidden_size  = getattr(self, "muon_hidden_size", 768)

            # 状态初始化（exp_avg / exp_avg_sq）
            if not hasattr(self, "mu_exp_avg"):
                self.mu_exp_avg = np.zeros_like(estimated_grad, dtype=estimated_grad.dtype)
            if not hasattr(self, "mu_exp_avg_sq"):
                self.mu_exp_avg_sq = np.zeros_like(estimated_grad, dtype=estimated_grad.dtype)

            # decoupled weight decay（先衰减，再加更新）
            if wd > 0.0:
                current_params_flat = current_params_flat + (-lr * wd) * current_params_flat

            # 一二阶动量
            self.mu_exp_avg = mu_beta1 * self.mu_exp_avg + (1.0 - mu_beta1) * estimated_grad
            self.mu_exp_avg_sq = mu_beta2 * self.mu_exp_avg_sq + (1.0 - mu_beta2) * (estimated_grad ** 2)

            denom = np.sqrt(self.mu_exp_avg_sq) + mu_eps

            # 偏置校正
            step_size = lr
            if correct_bias:
                bias_correction1 = 1.0 - (mu_beta1 ** self.t)
                bias_correction2 = 1.0 - (mu_beta2 ** self.t)
                step_size = step_size * np.sqrt(bias_correction2) / bias_correction1

            # 归一化梯度（可选 cautious）
            if cautious:
                mask = (self.mu_exp_avg * estimated_grad > 0).astype(estimated_grad.dtype)
                # 与 mudamw 中同款缩放（防止全 0）
                scale = mask.size / (mask.sum() + 1.0)
                mask = mask * scale
                norm_grad = (self.mu_exp_avg * mask) / denom
            else:
                norm_grad = self.mu_exp_avg / denom

            # 一维向量 → 2D 正交化（与 mudamw 逻辑一致）
            if norm_grad.ndim == 1 and norm_grad.size % hidden_size == 0 and norm_grad.size > hidden_size:
                G = norm_grad.reshape(hidden_size, -1)
                adj_lr = _adjust_lr_for_muon_np(step_size, G.shape)
                G = _zeropower_via_newtonschulz5_np(G, steps=5).astype(norm_grad.dtype)
                norm_grad = G.reshape(-1)
                step = adj_lr
            else:
                step = step_size

            updated_params_flat = current_params_flat - step * norm_grad
            print(f"服务器轮次 {server_round}: 模型已通过 MuDaMW 更新（ZO_MUDAMW）。")

        # 将更新后的参数返回给 Flower 服务器，它将在下一轮使用
        return ndarrays_to_parameters([updated_params_flat]), {}

# --- 启动服务器 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZO-FL Server Launcher")
    parser.add_argument("--train-mode", type=str, default=os.getenv("TRAIN_MODE", "ZO_MUON"),
                        choices=["BP_ADAM", "ZO_ADAM", "ZO_MUON"], help="训练模式")
    parser.add_argument("--num-rounds", type=int, default=int(os.getenv("NUM_ROUNDS", "10")), help="训练轮次")
    parser.add_argument("--server-address", type=str, default=os.getenv("SERVER_ADDRESS", "0.0.0.0:8080"), help="服务器地址")
    parser.add_argument("--min-fit-clients", type=int, default=int(os.getenv("MIN_FIT_CLIENTS", "2")), help="每轮最少参与训练的客户端数")
    parser.add_argument("--min-available-clients", type=int, default=int(os.getenv("MIN_AVAILABLE_CLIENTS", "2")), help="最少可用客户端数")
    parser.add_argument("--exp-name", type=str, default=os.getenv("EXP_NAME", ""), help="实验名称（留空则按参数自动生成）")
    # 优化器/ZO相关超参
    parser.add_argument("--muon-lr", type=float, default=float(os.getenv("MUON_LR", DEFAULT_MUON_LR)), help="Muon 学习率")
    parser.add_argument("--muon-beta1", type=float, default=float(os.getenv("MUON_BETA1", DEFAULT_MUON_BETA1)), help="beta1")
    parser.add_argument("--muon-beta2", type=float, default=float(os.getenv("MUON_BETA2", DEFAULT_MUON_BETA2)), help="beta2")
    parser.add_argument("--muon-eps", type=float, default=float(os.getenv("MUON_EPSILON", DEFAULT_MUON_EPSILON)), help="Muon epsilon")
    # Adam 超参（用于 BP_ADAM / ZO_ADAM）
    parser.add_argument("--adam-lr", type=float, default=float(os.getenv("ADAM_LR", DEFAULT_ADAM_LR)), help="Adam 学习率")
    parser.add_argument("--adam-beta1", type=float, default=float(os.getenv("ADAM_BETA1", DEFAULT_ADAM_BETA1)), help="Adam beta1")
    parser.add_argument("--adam-beta2", type=float, default=float(os.getenv("ADAM_BETA2", DEFAULT_ADAM_BETA2)), help="Adam beta2")
    parser.add_argument("--adam-eps", type=float, default=float(os.getenv("ADAM_EPS", DEFAULT_ADAM_EPSILON)), help="Adam epsilon")
    parser.add_argument("--perturbation-h", type=float, default=float(os.getenv("PERTURBATION_H", DEFAULT_PERTURBATION_H)), help="ZO 扰动常数 h")
    parser.add_argument("--block-until-clients", type=int, default=int(os.getenv("BLOCK_UNTIL_CLIENTS", "0")), help="在第一轮前阻塞等待到达的最小客户端数（0 为不阻塞）")
    args = parser.parse_args()

    model = get_model()
    initial_parameters = ndarrays_to_parameters([get_model_parameters(model).numpy()])

    strategy = ZOCloudMuonStrategy(
        initial_parameters=initial_parameters,
        min_available_clients=args.min_available_clients,
        min_fit_clients=args.min_fit_clients,
        train_mode=args.train_mode,
        block_until_clients=args.block_until_clients,
        muon_lr=args.muon_lr,
        muon_beta1=args.muon_beta1,
        muon_beta2=args.muon_beta2,
        muon_epsilon=args.muon_eps,
        perturbation_h=args.perturbation_h,
        adam_lr=args.adam_lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_eps,
    )

    # 构造实验名
    exp_name = args.exp_name.strip()
    if not exp_name:
        if args.train_mode == "ZO_MUON":
            exp_name = (
                f"{args.train_mode}_R{args.num_rounds}_F{args.min_fit_clients}_A{args.min_available_clients}"
                f"_LR{args.muon_lr}_B1{args.muon_beta1}_B2{args.muon_beta2}_H{args.perturbation_h}"
            )
        else:
            exp_name = (
                f"{args.train_mode}_R{args.num_rounds}_F{args.min_fit_clients}_A{args.min_available_clients}"
                f"_LR{args.adam_lr}_B1{args.adam_beta1}_B2{args.adam_beta2}"
            )
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # 将 log_dir 挂到 strategy 以便写入
    setattr(strategy, "log_dir", log_dir)

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    # 训练结束后自动绘图
    metrics_csv = os.path.join(log_dir, "metrics.csv")
    if os.path.exists(metrics_csv):
        try:
            subprocess.run([
                "python",
                os.path.join(os.path.dirname(__file__), "plot_metrics.py"),
                "--metrics", metrics_csv,
                "--output", os.path.join(log_dir, "loss_curve.png"),
            ], check=True)
        except Exception:
            pass

    # 保存一次服务端参数及运行配置，便于复现实验
    server_params = {
        "exp_name": exp_name,
        "train_mode": args.train_mode,
        "num_rounds": args.num_rounds,
        "server_address": args.server_address,
        "min_fit_clients": args.min_fit_clients,
        "min_available_clients": args.min_available_clients,
        "muon_lr": args.muon_lr,
        "muon_beta1": args.muon_beta1,
        "muon_beta2": args.muon_beta2,
        "muon_epsilon": args.muon_eps,
        "perturbation_h": args.perturbation_h,
        "block_until_clients": args.block_until_clients,
        "log_dir": log_dir,
    }
    try:
        with open(os.path.join(log_dir, "server_params.json"), "w", encoding="utf-8") as f:
            json.dump(server_params, f, ensure_ascii=False, indent=2)
    except Exception:
        pass