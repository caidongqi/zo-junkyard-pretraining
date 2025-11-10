import torch
import torch.nn.functional as F

def compare_grad_cos_similarity(*grads_list):
    """
    比较多个梯度组之间的平均余弦相似度。
    每个梯度组是一个list，对应每个参数的梯度张量。

    参数:
        *grads_list: 任意数量的梯度组, 每个都是 List[Tensor]
                     例如: grads_bp, grads_zo, grads_noisy_bp, ...
    返回:
        sim_matrix: Tensor [n, n]，每个元素是两组梯度的平均余弦相似度
    """
    n = len(grads_list)
    if n < 2:
        raise ValueError("需要至少两个梯度组进行比较")

    # 将每组梯度拼接成一个长向量
    flat_grads = []
    for grads in grads_list:
        flat = torch.cat([g.view(-1) for g in grads]).float()
        flat_grads.append(flat / (flat.norm() + 1e-12))  # 归一化

    # 计算两两余弦相似度
    sim_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = torch.dot(flat_grads[i], flat_grads[j]).item()

    return sim_matrix

# def main():
#     # 假设有三组梯度
#     loss_bp, grads_bp = compute_backprop_gradients(model, params, loss_fn, x, y)
#     loss_zo, grads_zo = compute_backprop_gradients(model, params, loss_fn, x, y)  # 举例
#     loss_rand, grads_rand = compute_backprop_gradients(model, params, loss_fn, x, y)

#     sim = compare_grad_cos_similarity(grads_bp, grads_zo, grads_rand)
#     print(sim)

# if __name__ == "__main__":
#     main()