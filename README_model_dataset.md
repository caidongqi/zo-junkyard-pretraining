# 模型和数据集配置说明

本文档说明如何使用新的模型和数据集配置系统。

## 快速开始

### 1. 查看可用的模型和数据集

```bash
# 查看所有可用的模型配置
python model.py --list

# 查看所有可用的数据集
python data.py --list

# 查看特定数据集的详细信息
python data.py --info cosmopedia-100k
```

### 2. 使用不同的模型和数据集运行实验

```bash
# 使用默认配置 (200M模型 + cosmopedia-100k数据集)
./parallel_sweep.sh

# 使用小型模型 (20M) 进行快速测试
./parallel_sweep.sh --model-size 20M --max-samples 5000

# 使用大型模型 (1B) 和完整 Cosmopedia 数据集
./parallel_sweep.sh --model-size 1B --dataset cosmopedia --max-samples 100000

# 使用中型模型 (500M) 和 WikiText-103 数据集
./parallel_sweep.sh --model-size 500M --dataset wikitext-103
```

## 可用的模型配置

| 模型大小 | 参数量 | 嵌入维度 | 层数 | 注意力头 | 适用场景 |
|---------|--------|---------|------|---------|---------|
| `20M`   | ~20M   | 256     | 6    | 4       | 快速原型验证，资源受限环境 |
| `200M`  | ~200M  | 768     | 12   | 12      | 标准实验，类似 GPT-2 Small |
| `500M`  | ~500M  | 1024    | 24   | 16      | 中型规模，平衡性能和成本 |
| `1B`    | ~1B    | 1536    | 24   | 24      | 大规模实验，需要较大显存 |

### 模型选择建议

- **快速调试和原型**: 使用 `20M`
- **标准实验**: 使用 `200M` (默认)
- **性能评估**: 使用 `500M` 或 `1B`
- **论文复现**: 根据原论文使用的模型规模选择

## 可用的数据集

| 数据集 | 大小 | 样本数 | 推荐用途 | 推荐样本数 |
|--------|------|--------|---------|-----------|
| `cosmopedia-100k` | ~100K docs | 100K | 快速实验 (默认) | 20,000 |
| `cosmopedia` | ~30M docs | 30M+ | 大规模高质量预训练 | 100,000 |
| `wikitext-103` | ~100M tokens | - | 经典语言建模基准 | 全部 |
| `openwebtext` | ~40GB | 8M+ | 真实网络文本分布 | 50,000 |
| `c4` | ~750GB | 365M | 超大规模预训练 | 100,000 |
| `tinystories` | ~2M stories | 2M | 小模型调试 | 50,000 |
| `pile-subset` | ~200GB | - | 多领域多样化数据 | 100,000 |
| `fineweb` | ~15T tokens | - | **主流高质量预训练 (强烈推荐)** | 100,000 |
| `fineweb-edu` | ~1.3T tokens | - | **教育质量内容 (强烈推荐)** | 50,000 |
| `fineweb-edu-10bt` | ~10B tokens | - | 快速实验的高质量数据 | 30,000 |

### 数据集选择建议

- **快速测试**: `cosmopedia-100k` (默认) 或 `fineweb-edu-10bt` 或 `tinystories`
- **标准预训练**: `fineweb-edu` (强烈推荐) 或 `cosmopedia` 或 `openwebtext`
- **主流大规模预训练**: `fineweb` 或 `fineweb-edu` (业界主流选择)
- **经典基准**: `wikitext-103`
- **超大规模预训练**: `fineweb` 或 `c4` 或 `pile-subset`
- **多样性**: `openwebtext` 或 `pile-subset`

**注意**: FineWeb 系列数据集是目前业界主流的预训练数据集，质量经过精心优化，强烈推荐用于正式训练。

## 使用示例

### 示例 1: 快速原型验证

使用小模型和少量数据快速测试 ZO 优化器:

```bash
./parallel_sweep.sh \
    --model-size 20M \
    --dataset cosmopedia-100k \
    --max-samples 5000 \
    --epochs 5 \
    --modes ZO \
    --query-budgets 1,2,4,8 \
    --parallel 4
```

### 示例 2: 标准实验 (默认配置)

使用 200M 模型和标准数据集进行完整实验:

```bash
./parallel_sweep.sh \
    --model-size 200M \
    --dataset cosmopedia-100k \
    --epochs 10 \
    --modes ZO,FO \
    --parallel 8
```

### 示例 3: 大规模实验

使用 1B 模型和大规模数据集:

```bash
./parallel_sweep.sh \
    --model-size 1B \
    --dataset cosmopedia \
    --max-samples 100000 \
    --epochs 20 \
    --batch-sizes 4 \
    --parallel 16 \
    --gpus "0,1,2,3,4,5,6,7"
```

### 示例 4: 对比不同模型规模

在不同模型规模上对比 ZO 和 FO:

```bash
# 20M 模型
./parallel_sweep.sh --model-size 20M --dataset cosmopedia-100k --max-samples 10000

# 200M 模型
./parallel_sweep.sh --model-size 200M --dataset cosmopedia-100k --max-samples 20000

# 500M 模型
./parallel_sweep.sh --model-size 500M --dataset cosmopedia-100k --max-samples 20000

# 1B 模型
./parallel_sweep.sh --model-size 1B --dataset cosmopedia-100k --max-samples 20000
```

### 示例 5: 对比不同数据集

在不同数据集上评估模型性能:

```bash
# FineWeb-Edu 10BT (主流推荐，快速实验)
./parallel_sweep.sh --dataset fineweb-edu-10bt --max-samples 30000

# FineWeb-Edu (主流推荐，高质量)
./parallel_sweep.sh --dataset fineweb-edu --max-samples 50000

# FineWeb (主流推荐，超大规模)
./parallel_sweep.sh --dataset fineweb --max-samples 100000

# Cosmopedia (高质量合成数据)
./parallel_sweep.sh --dataset cosmopedia-100k --max-samples 20000

# WikiText (维基百科)
./parallel_sweep.sh --dataset wikitext-103

# OpenWebText (真实网络文本)
./parallel_sweep.sh --dataset openwebtext --max-samples 30000

# TinyStories (简单故事)
./parallel_sweep.sh --dataset tinystories --max-samples 20000
```

## 直接使用 Python 脚本

你也可以直接调用 Python 脚本而不通过 bash 脚本:

```bash
python reproduce_zo_paper.py \
    --mode ZO \
    --scope full \
    --query_budget_q 8 \
    --learning_rate 1e-3 \
    --epochs 10 \
    --batch_size 4 \
    --optimizer mudamw \
    --model_size 200M \
    --dataset cosmopedia-100k \
    --max_samples 20000
```

## 缓存机制

数据集会自动缓存到 `cache/` 目录，缓存文件名格式为:
```
dataset_{dataset_name}_bs{block_size}_samples{max_samples}.pkl
```

如果需要强制重新加载数据集，可以删除相应的缓存文件。

## 注意事项

### 显存需求

不同模型规模的显存需求估计:

- `20M`: ~1-2 GB
- `200M`: ~4-8 GB
- `500M`: ~10-16 GB
- `1B`: ~20-32 GB

实际显存使用还取决于:
- 批次大小 (`--batch-sizes`)
- 序列长度
- 是否使用梯度累积
- ZO 查询预算 (`--query-budgets`)

### 训练时间

不同配置的训练时间差异很大:

- `20M + 5K samples`: ~10-30 分钟
- `200M + 20K samples`: ~1-3 小时
- `500M + 50K samples`: ~4-8 小时
- `1B + 100K samples`: ~12-24 小时

使用并行训练可以显著减少总时间。

### 磁盘空间

数据集缓存会占用磁盘空间:

- `cosmopedia-100k` (20K samples): ~100-500 MB
- `cosmopedia` (100K samples): ~500MB-2GB
- `wikitext-103`: ~500MB
- `openwebtext` (50K samples): ~1-3 GB
- `c4` (100K samples): ~2-5 GB
- `fineweb-edu-10bt` (30K samples): ~500MB-1GB
- `fineweb-edu` (50K samples): ~1-3 GB
- `fineweb` (100K samples): ~2-5 GB

## 故障排除

### 问题 1: CUDA OOM (显存不足)

**解决方案**:
- 减小批次大小: `--batch-sizes 1` 或 `--batch-sizes 2`
- 使用更小的模型: `--model-size 20M`
- 减小查询预算: `--query-budgets 1,2,4`
- 减小并行任务数: `--parallel 2`

### 问题 2: 数据集加载失败

**解决方案**:
- 检查网络连接 (需要访问 HuggingFace Hub)
- 删除缓存文件重新下载: `rm cache/dataset_*.pkl`
- 尝试使用流式加载的数据集

### 问题 3: 训练速度慢

**解决方案**:
- 使用更小的模型进行测试
- 减少样本数: `--max-samples 5000`
- 增加批次大小 (如果显存允许)
- 使用多GPU并行: `--parallel 8 --gpus "0,1,2,3,4,5,6,7"`

## 扩展和自定义

### 添加新的模型配置

编辑 `model.py`，在 `MODEL_CONFIGS` 字典中添加新配置:

```python
'2B': {
    'n_embd': 2048,
    'n_layer': 32,
    'n_head': 32,
    'n_positions': 2048,
    'description': '超大型模型，约2B参数',
    'estimated_params': '2B',
}
```

### 添加新的数据集

编辑 `data.py`，在 `DATASET_CONFIGS` 字典中添加新配置:

```python
'my-dataset': {
    'hf_path': 'username/dataset-name',
    'split': 'train',
    'text_field': 'text',
    'streaming': True,
    'description': '我的自定义数据集',
    'recommended_samples': 50000,
    'language': 'en',
    'size': '~1M documents',
}
```

## 参考文档

- [parallel_sweep.sh 使用说明](README_parallel.md)
- [绘图功能说明](README_plotting.md)
- [参数扫描说明](README_sweep.md)

