import torch
import time

def generate_hybrid_directions(bp_grads, q, cosine_target, noise_scale, device=None):
    """
    生成一个混合方向：保留高能量的梯度信号，并在其余维度上添加随机噪声。

    Args:
        bp_grads (list of torch.Tensor): 原始的BP梯度列表。
        q (int): 要生成的方向数量。
        cosine_target (float): 目标余弦相似度，用于确定高能部分的能量阈值。
        noise_scale (float): 噪声强度。值越大，噪声越强，最终相似度可能越低。
        device (torch.device, optional): 计算设备。

    Yields:
        list of torch.Tensor: 生成的混合方向。
    """
    # --- 1. 初始化和预处理 (与原函数类似) ---
    if not bp_grads or q <= 0:
        return

    if device is None:
        device = bp_grads[0].device if hasattr(bp_grads[0], 'device') else torch.device('cpu')

    # 将所有梯度扁平化为一个向量
    grad_flat = torch.cat([g.flatten().to(device) for g in bp_grads])
    d = grad_flat.numel()
    if d == 0:
        return

    total_norm = torch.norm(grad_flat)
    total_norm_sq = total_norm**2
    
    if total_norm <= 1e-12:
        return

    # --- 2. 确定高能量维度的阈值和索引 (与原函数类似) ---
    min_similarity = max(min(abs(cosine_target), 0.9999), 0.9)
    energy_threshold = min_similarity**2 * total_norm_sq

    abs_sq = grad_flat.abs().pow(2)
    
    # 动态确定需要的top-k数量以满足能量阈值
    k = min(64, d)
    while True:
        values, indices = torch.topk(abs_sq, k, largest=True)
        captured_energy = values.sum().item()
        if captured_energy >= energy_threshold or k >= d:
            break
        k = min(k * 2, d)

    cumsum_values = torch.cumsum(values, dim=0)
    effective_rank_idx = torch.searchsorted(cumsum_values, torch.tensor(energy_threshold, device=device))
    effective_rank = int(effective_rank_idx.item()) + 1
    
    # 最终确定的高能量维度索引
    high_energy_indices = indices[:effective_rank]

    # --- 3. 生成器循环，每次生成一个混合方向 ---
    for i in range(q):
        # --- 3a. 构建混合方向：保留高能部分 + 添加噪声 ---
        hybrid_flat = torch.zeros_like(grad_flat)
        
        # 步骤A: 保留高能量方向
        hybrid_flat[high_energy_indices] = grad_flat[high_energy_indices]

        # 步骤B: 在剩下的方向上加随机噪声
        # 创建一个mask来标识低能量维度
        low_energy_mask = torch.ones_like(grad_flat, dtype=torch.bool)
        low_energy_mask[high_energy_indices] = False
        
        num_low_energy_dims = low_energy_mask.sum().item()

        if num_low_energy_dims > 0 and noise_scale > 0:
            # 生成与低能量维度数量相同的标准正态分布噪声
            # 将噪声的强度缩放到与梯度范数相关的水平，使其效果更可控
            noise_magnitude = noise_scale * (total_norm / (d**0.5))
            noise = torch.randn(num_low_energy_dims, device=device) * noise_magnitude
            
            # 将噪声应用到低能量维度上
            hybrid_flat[low_energy_mask] = noise

        # --- 3b. 计算并报告统计数据 ---
        hybrid_norm = torch.norm(hybrid_flat)
        if hybrid_norm <= 1e-12:
            final_similarity = 0.0
        else:
            # 计算最终的余弦相似度
            dot_product = torch.dot(hybrid_flat, grad_flat)
            final_similarity = dot_product / (hybrid_norm * total_norm)

        print("-" * 60)
        print(f"生成方向 #{i+1}:")
        print(f"  - 梯度总维度 (d): {d}")
        print(f"  - 高能量维度数量 (effective_rank): {effective_rank} ({effective_rank / d:.2%} of total)")
        print(f"  - 噪声强度 (noise_scale): {noise_scale}")
        print(f"  - 最终与原梯度的余弦相似度: {final_similarity.item():.6f}")
        print("-" * 60)

        # --- 3c. 重塑形状并产出结果 ---
        direction = []
        start = 0
        for g in bp_grads:
            sz = g.numel()
            chunk = hybrid_flat[start:start+sz].view_as(g)
            if chunk.device != g.device:
                chunk = chunk.to(g.device)
            direction.append(chunk)
            start += sz
        
        yield direction

# ==============================================================================
# 主测试脚本
# ==============================================================================
if __name__ == "__main__":
    # --- 可调参数 ---
    # 1. BP梯度的总维度
    DIMENSIONALITY = 20000000
    
    # 2. 目标余弦相似度 (用于决定保留多少高能维度)
    #    值越高，保留的维度越多 (effective_rank 会变大)
    COSINE_TARGET = 0.9
    
    # 3. 噪声强度 (noise_scale)
    #    0.0  : 低能维度为0，等价于稀疏投影
    #    0.5  : 较弱的噪声
    #    1.0  : 标准强度的噪声
    #    5.0  : 很强的噪声，会导致相似度显著下降
    NOISE_SCALE = 0.5
    
    # 4. 要生成的方向数量
    Q = 1

    # --- 模拟环境设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试将在设备: {device} 上运行\n")

    # 创建一个模拟的BP梯度
    # 使用 randn().pow(3) 来创建一个能量分布更不均匀的梯度，使其更接近真实场景
    print("正在创建模拟BP梯度...")
    mock_gradient = torch.randn(DIMENSIONALITY, device=device).pow(3)
    bp_grads = [mock_gradient] # 包装在列表中以匹配函数签名

    # --- 运行生成器并获取结果 ---
    print("开始生成混合方向...")
    hybrid_direction_generator = generate_hybrid_directions(
        bp_grads=bp_grads,
        q=Q,
        cosine_target=COSINE_TARGET,
        noise_scale=NOISE_SCALE,
        device=device
    )

    # 迭代生成器以触发计算和打印
    for generated_direction in hybrid_direction_generator:
        # 在这里可以对生成的方向做进一步处理
        # 例如： my_optimizer.step(generated_direction)
        pass

    print("\n测试完成。")
    print("你可以尝试调整脚本顶部的 DIMENSIONALITY, COSINE_TARGET, 和 NOISE_SCALE 来观察结果变化。")