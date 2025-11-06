# Checkpoint 评估工具

这个工具用于评估训练好的 checkpoint 在不同数据集上的 loss 和困惑度（perplexity）。

## 功能特点

- ✅ 加载任意 GPT-2 架构的 checkpoint
- ✅ 支持多种预训练数据集（cosmopedia, fineweb, wikitext等）
- ✅ 自动计算平均 loss 和困惑度
- ✅ 支持结果保存为 JSON 格式
- ✅ 支持批量评估多个数据集

## 安装依赖

```bash
pip install torch transformers datasets tqdm
```

## 基本用法

### 1. 单数据集评估

```bash
python evaluate_checkpoint.py \
    --checkpoint_path path/to/checkpoint \
    --dataset cosmopedia \
    --batch_size 8 \
    --block_size 128 \
    --max_samples 10000
```

### 2. 列出可用数据集

```bash
python evaluate_checkpoint.py --list_datasets
```

### 3. 保存结果到文件

```bash
python evaluate_checkpoint.py \
    --checkpoint_path path/to/checkpoint \
    --dataset fineweb-edu-10bt \
    --batch_size 4 \
    --output_file results/eval_results.json
```

## 参数说明

### 必需参数

- `--checkpoint_path`: Checkpoint 目录路径
- `--dataset`: 数据集名称（使用 `--list_datasets` 查看所有可用选项）

### 可选参数

- `--batch_size`: 批次大小（默认: 4）
- `--block_size`: 序列长度（默认: 128）
- `--max_samples`: 最大样本数，None 表示使用推荐值（默认: None）
- `--max_batches`: 最大评估 batch 数，用于快速测试（默认: None，评估全部）
- `--device`: 计算设备，'cuda' 或 'cpu'（默认: cuda）
- `--cache_dir`: 数据集缓存目录（默认: cache）
- `--force_reload`: 强制重新加载数据集，忽略缓存
- `--output_file`: 结果输出文件路径（JSON格式）

## 可用数据集

### Cosmopedia 系列
- `cosmopedia`: 完整版（web_samples_v2，30M+ 文档）
- `cosmopedia-100k`: 100K 样本快速版本
- `cosmopedia-stories`: 故事子集
- `cosmopedia-khanacademy`: Khan Academy 教育内容
- `cosmopedia-openstax`: OpenStax 教科书内容

### FineWeb 系列
- `fineweb`: 完整版（15T tokens）
- `fineweb-edu`: 教育子集（1.3T tokens）
- `fineweb-edu-10bt`: 10BT 采样版本

### 其他数据集
- `wikitext-103`: 维基百科文本
- `openwebtext`: 开源 WebText 复现
- `c4`: Colossal Clean Crawled Corpus
- `tinystories`: 简单故事数据集
- `pile-subset`: The Pile 无版权子集

## 使用示例

### 示例 1: 评估 FO 训练的模型在 cosmopedia 上的表现

```bash
python evaluate_checkpoint.py \
    --checkpoint_path logs/parallel_sweep_20251104_152749/experiments/FO_full_bs2_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251104_152749/experiments/FO_full_bs2_qN_A_bpN_A_optmudamw_lr1e-3/checkpoint \
    --dataset cosmopedia \
    --batch_size 8 \
    --block_size 128 \
    --max_samples 10000 \
    --output_file results/eval_fo_cosmopedia.json
```

### 示例 2: 快速测试（只评估 100 个 batch）

```bash
python evaluate_checkpoint.py \
    --checkpoint_path path/to/checkpoint \
    --dataset tinystories \
    --batch_size 16 \
    --max_batches 100
```

### 示例 3: 使用批量评估脚本

创建了一个批量评估脚本 `batch_evaluate.py`，可以一次性在多个数据集上评估：

```bash
python batch_evaluate.py \
    --checkpoint_path path/to/checkpoint \
    --datasets cosmopedia fineweb-edu-10bt wikitext-103 \
    --batch_size 8 \
    --output_dir results/batch_eval
```

### 示例 4: 使用 Shell 脚本批量评估

```bash
# 修改 example_evaluate.sh 中的 checkpoint 路径
nano example_evaluate.sh

# 运行批量评估
bash example_evaluate.sh
```

## 输出格式

### 控制台输出

```
================================================================================
Evaluation Results
================================================================================
Average Loss: 2.345678
Perplexity: 10.4321
Total Tokens: 1,280,000
================================================================================
```

### JSON 输出格式

```json
{
  "checkpoint_path": "path/to/checkpoint",
  "dataset": "cosmopedia",
  "batch_size": 8,
  "block_size": 128,
  "max_samples": 10000,
  "max_batches": null,
  "device": "cuda",
  "avg_loss": 2.345678,
  "perplexity": 10.4321,
  "total_tokens": 1280000,
  "model_config": {
    "vocab_size": 50257,
    "n_positions": 1024,
    "n_embd": 768,
    "n_layer": 12,
    "n_head": 12
  },
  "timestamp": "2025-11-05T12:34:56.789"
}
```

## 批量评估工具

`batch_evaluate.py` 提供了批量评估功能：

```bash
# 在多个数据集上评估单个 checkpoint
python batch_evaluate.py \
    --checkpoint_path path/to/checkpoint \
    --datasets cosmopedia fineweb-edu-10bt wikitext-103 tinystories \
    --batch_size 8 \
    --output_dir results/batch_eval

# 评估多个 checkpoint（自动查找实验目录）
python batch_evaluate.py \
    --experiments_dir logs/parallel_sweep_20251104_152749/experiments \
    --datasets cosmopedia fineweb-edu-10bt \
    --batch_size 4 \
    --output_dir results/multi_checkpoint_eval
```

## 性能优化建议

1. **使用缓存**: 首次运行会缓存数据集，后续运行会更快
2. **调整 batch_size**: 根据 GPU 显存调整，更大的 batch_size 通常更快
3. **限制样本数**: 使用 `--max_samples` 进行快速评估
4. **限制 batch 数**: 使用 `--max_batches` 进行极快测试

## 注意事项

1. **显存要求**: 确保有足够的 GPU 显存。如果显存不足，可以：
   - 减小 `--batch_size`
   - 减小 `--block_size`
   - 使用 `--device cpu`（但会很慢）

2. **数据集大小**: 某些数据集（如 c4, fineweb）非常大，建议：
   - 设置合理的 `--max_samples`
   - 首次下载需要时间，请耐心等待

3. **Checkpoint 格式**: 工具期望 checkpoint 目录包含：
   - `config.json`: 模型配置
   - `model.safetensors` 或 `pytorch_model.bin`: 模型权重
   - `tokenizer/`: tokenizer 文件（可选，会使用 GPT-2 tokenizer）

## 故障排除

### 问题 1: CUDA out of memory

```bash
# 解决方案: 减小 batch_size
python evaluate_checkpoint.py ... --batch_size 2
```

### 问题 2: 数据集加载失败

```bash
# 解决方案: 强制重新加载
python evaluate_checkpoint.py ... --force_reload
```

### 问题 3: Config name is missing

这个问题已经修复。如果仍然遇到，请确保 `data.py` 中的数据集配置包含 `dataset_name` 字段。

## 扩展功能

### 添加自定义数据集

在 `data.py` 的 `DATASET_CONFIGS` 字典中添加新配置：

```python
'my-dataset': {
    'hf_path': 'path/to/dataset',
    'split': 'train',
    'text_field': 'text',
    'streaming': True,
    'description': '我的自定义数据集',
    'recommended_samples': 10000,
    'language': 'en',
    'size': '~1M documents',
    'dataset_name': 'subset_name',  # 如果需要
},
```

## 相关文件

- `evaluate_checkpoint.py`: 单数据集评估脚本
- `batch_evaluate.py`: 批量评估脚本
- `example_evaluate.sh`: Shell 脚本示例
- `data.py`: 数据集配置
- `model.py`: 模型配置

## License

与项目主 LICENSE 相同





