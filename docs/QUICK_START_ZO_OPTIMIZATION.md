# ZO优化功能快速开始指南

## 🚀 快速开始

### 1. 基本使用（内存优化自动启用）

```bash
# 标准ZO训练（内存优化已自动启用）
python reproduce_zo_paper.py \
    --mode ZO \
    --query_budget_q 4 \
    --model_size 20M \
    --batch_size 4 \
    --learning_rate 1e-3
```

### 2. 启用并行Q值计算

```bash
# 适合q值较大的场景
python reproduce_zo_paper.py \
    --mode ZO \
    --query_budget_q 16 \
    --parallel_q_computation \
    --parallel_batch_size 4 \
    --model_size 200M \
    --batch_size 4 \
    --learning_rate 1e-3
```

### 3. 在Instruct模式中使用

```bash
# Instruct模式也支持这些优化
python reproduce_zo_paper.py \
    --mode Instruct \
    --query_budget_q 10 \
    --bp_interval 5 \
    --parallel_q_computation \
    --parallel_batch_size 4 \
    --model_size 200M
```

## 🧪 运行测试

验证优化功能是否正常工作：

```bash
python test_zo_optimization.py
```

这将运行以下测试：
1. 内存优化验证
2. 并行计算性能对比
3. 向后兼容性测试

## 📊 参数选择指南

### parallel_batch_size 选择

| q值范围 | 推荐batch_size | 说明 |
|--------|---------------|------|
| 1-4    | 不启用并行     | 顺序计算已足够高效 |
| 5-8    | 2-4           | 小批次并行 |
| 9-16   | 4-6           | 中等批次 |
| 17+    | 6-8           | 较大批次 |

### 何时使用并行计算

**推荐使用场景**：
- ✅ q值 >= 8
- ✅ GPU内存充足
- ✅ 需要快速迭代

**不推荐场景**：
- ❌ q值 <= 4（收益不明显）
- ❌ GPU内存紧张
- ❌ 调试阶段（顺序计算更容易追踪）

## 💾 内存节省估算

| 模型大小 | 原始额外内存 | 优化后额外内存 | 节省 |
|---------|------------|--------------|------|
| 20M     | ~80MB      | ~10MB        | ~70MB |
| 200M    | ~800MB     | ~100MB       | ~700MB |
| 500M    | ~2GB       | ~250MB       | ~1.75GB |
| 1B      | ~4GB       | ~500MB       | ~3.5GB |

## ⚡ 性能提升

并行计算模式的典型性能提升：

| q值 | batch_size | 性能提升 |
|-----|-----------|---------|
| 8   | 4         | 3-8%    |
| 16  | 4         | 5-12%   |
| 32  | 8         | 8-15%   |

*注：实际提升取决于硬件和模型大小*

## 🔍 实际示例

### 示例1: 小模型快速实验

```bash
python reproduce_zo_paper.py \
    --mode ZO \
    --scope full \
    --query_budget_q 4 \
    --learning_rate 1e-3 \
    --epochs 1 \
    --batch_size 8 \
    --model_size 20M \
    --dataset cosmopedia-100k \
    --max_samples 1000
```

### 示例2: 大模型训练（启用所有优化）

```bash
python reproduce_zo_paper.py \
    --mode Instruct \
    --scope full \
    --query_budget_q 16 \
    --bp_interval 10 \
    --learning_rate 1e-4 \
    --epochs 1 \
    --batch_size 4 \
    --model_size 200M \
    --dataset fineweb-edu \
    --max_samples 10000 \
    --parallel_q_computation \
    --parallel_batch_size 4 \
    --optimizer mudamw \
    --use_lr_scheduler \
    --warmup_steps 300
```

### 示例3: 调试和验证

```bash
# 使用小数据量验证
python reproduce_zo_paper.py \
    --mode ZO \
    --query_budget_q 2 \
    --model_size 20M \
    --max_samples 100 \
    --log_interval 1
```

## 📝 Shell脚本示例

创建一个批量测试脚本 `test_zo_optimization_batch.sh`:

```bash
#!/bin/bash

# 测试不同q值和并行设置的组合

echo "测试ZO优化功能..."

# 测试1: 基线（不启用并行）
python reproduce_zo_paper.py \
    --mode ZO \
    --query_budget_q 8 \
    --model_size 20M \
    --max_samples 500 \
    --run_name "baseline_q8"

# 测试2: 启用并行 batch_size=4
python reproduce_zo_paper.py \
    --mode ZO \
    --query_budget_q 8 \
    --parallel_q_computation \
    --parallel_batch_size 4 \
    --model_size 20M \
    --max_samples 500 \
    --run_name "parallel_q8_bs4"

# 测试3: 大q值 + 并行
python reproduce_zo_paper.py \
    --mode ZO \
    --query_budget_q 32 \
    --parallel_q_computation \
    --parallel_batch_size 8 \
    --model_size 20M \
    --max_samples 500 \
    --run_name "parallel_q32_bs8"

echo "测试完成！查看logs/目录获取结果"
```

运行：
```bash
chmod +x test_zo_optimization_batch.sh
./test_zo_optimization_batch.sh
```

## ❓ 常见问题

### Q1: 启用并行计算后结果是否会改变？

**A:** 不会。并行计算只是改变了计算的组织方式，数学上完全等价。可能会有极小的数值误差（1e-6量级），这是浮点运算的正常现象。

### Q2: 内存优化会影响梯度质量吗？

**A:** 不会。内存优化只是改变了参数的存储方式，不改变梯度计算的逻辑。

### Q3: 我的旧训练脚本还能用吗？

**A:** 完全可以！所有优化都是向后兼容的。不添加新参数时，行为与原来完全一致。

### Q4: parallel_batch_size应该设置多大？

**A:** 推荐值是4-8。过大的值收益递减，还可能导致内存问题。

### Q5: 什么时候能看到明显的性能提升？

**A:** 
- 内存优化：立即生效，所有ZO方法都受益
- 并行计算：q >= 8时能看到明显效果

## 🐛 问题排查

### 内存不足错误

```bash
# 减小batch_size或parallel_batch_size
python reproduce_zo_paper.py \
    --mode ZO \
    --batch_size 2 \  # 减小
    --parallel_batch_size 2 \  # 减小
    ...
```

### 速度没有提升

1. 检查q值是否足够大（建议 >= 8）
2. 尝试调整parallel_batch_size
3. 确保GPU不是满载（使用`nvidia-smi`查看）

## 📚 更多信息

- 详细文档：`ZO_OPTIMIZATION_README.md`
- 测试代码：`test_zo_optimization.py`
- 主程序：`reproduce_zo_paper.py`

## 🎯 最佳实践总结

1. **始终受益于内存优化**：无需任何配置
2. **大q值使用并行**：q >= 8时启用`--parallel_q_computation`
3. **合理设置批次**：`--parallel_batch_size 4-8`
4. **监控资源**：使用`nvidia-smi`监控GPU使用
5. **逐步调优**：从小参数开始，逐步增加

---

**享受更高效的ZO训练！** 🚀

