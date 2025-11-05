# BP数据集分离功能使用说明

## 功能简介

现在支持为 `Calibrate` 和 `Instruct` 模式中的BP梯度计算使用单独的数据集，这样可以让：
- **ZO查询**使用一个数据集（例如：大规模、低质量数据）
- **BP梯度**使用另一个数据集（例如：小规模、高质量数据）

## 使用方法

### 1. 命令行直接使用

在运行 `reproduce_zo_paper.py` 时，可以使用以下参数：

```bash
python reproduce_zo_paper.py \
    --mode Instruct \
    --scope full \
    --query_budget_q 64 \
    --bp_interval 1 \
    --dataset fineweb-edu-10bt \
    --max_samples 10000 \
    --bp_dataset cosmopedia-100k \
    --bp_max_samples 5000 \
    --learning_rate 1e-3 \
    --optimizer mudamw \
    --epochs 5 \
    --batch_size 2
```

**参数说明：**
- `--dataset`: 主训练数据集（ZO查询使用）
- `--max_samples`: 主数据集的最大样本数
- `--bp_dataset`: BP梯度计算使用的数据集（可选）
- `--bp_max_samples`: BP数据集的最大样本数（可选）

### 2. 在 parallel_sweep.sh 中使用

编辑 `parallel_sweep.sh` 文件，设置以下变量：

```bash
# 主数据集配置
DATASET="fineweb-edu-10bt"  # ZO查询使用的数据集
MAX_SAMPLES="10000"          # 主数据集样本数

# BP数据集配置
BP_DATASET="cosmopedia-100k"  # BP梯度计算使用的数据集
BP_MAX_SAMPLES="5000"         # BP数据集样本数
```

或通过命令行参数：

```bash
bash parallel_sweep.sh \
    --modes Instruct \
    --dataset fineweb-edu-10bt \
    --max-samples 10000 \
    --bp-dataset cosmopedia-100k \
    --bp-max-samples 5000 \
    --query-budgets 64 \
    --bp-intervals 1 \
    --parallel 4 \
    --gpus "0,1,2,3"
```

## 应用场景示例

### 场景1：使用大规模低质量数据进行ZO，高质量数据指导BP

```bash
# ZO使用大规模FineWeb数据（噪声较多但量大）
# BP使用精选的Cosmopedia数据（质量高但量小）
python reproduce_zo_paper.py \
    --mode Instruct \
    --dataset fineweb \
    --max_samples 50000 \
    --bp_dataset cosmopedia \
    --bp_max_samples 10000 \
    --query_budget_q 128 \
    --bp_interval 1
```

### 场景2：使用通用数据进行ZO，领域特定数据指导BP

```bash
# ZO使用通用的OpenWebText
# BP使用特定领域的数据集
python reproduce_zo_paper.py \
    --mode Instruct \
    --dataset openwebtext \
    --bp_dataset pile-subset \
    --query_budget_q 64 \
    --bp_interval 2
```

### 场景3：只使用单一数据集（默认行为）

```bash
# 不指定bp_dataset，则BP和ZO使用相同数据集
python reproduce_zo_paper.py \
    --mode Instruct \
    --dataset cosmopedia-100k \
    --query_budget_q 64 \
    --bp_interval 1
```

## 注意事项

1. **兼容性**：此功能仅在 `Calibrate` 和 `Instruct` 模式下生效，`FO` 和 `ZO` 模式不受影响

2. **默认行为**：如果不指定 `--bp_dataset`，系统会自动使用主数据集进行BP计算（保持向后兼容）

3. **数据集选择建议**：
   - BP数据集通常应该是**高质量**的，因为它直接影响梯度方向
   - ZO数据集可以是**大规模**的，用于充分探索参数空间

4. **性能考虑**：
   - 使用单独的BP数据集会额外加载一个dataloader
   - 确保有足够的内存来加载两个数据集

## 可用数据集列表

- `cosmopedia-100k`: 高质量合成教育数据（推荐用于BP）
- `cosmopedia`: Cosmopedia完整版
- `wikitext-103`: 维基百科文本
- `openwebtext`: 开源WebText
- `c4`: 大规模清洗网页数据
- `tinystories`: 简单故事数据集
- `pile-subset`: The Pile子集
- `fineweb`: FineWeb完整版（推荐用于ZO）
- `fineweb-edu`: FineWeb教育子集
- `fineweb-edu-10bt`: FineWeb-Edu 10BT采样

## 日志输出示例

当使用单独的BP数据集时，会看到如下日志：

```
Creating separate BP dataloader with dataset: cosmopedia-100k
BP will use separate dataset: cosmopedia-100k
[Instruct] Perturbation 1/64: cosine_similarity=0.900123, target=0.900000
...
```

这表明系统正在使用两个独立的数据集。

