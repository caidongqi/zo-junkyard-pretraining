# ZO Optimization 实验代码说明

## 代码结构

```
zo-test-cdq/
├── cores/                    # 核心功能模块
│   ├── data.py              # 数据处理
│   ├── model.py             # 模型定义
│   ├── optimizer.py         # 优化器实现
│   ├── training_management.py  # 训练管理和评估
│   └── instruct_params_manager.py  # Instruct模式参数管理
├── evaluation/              # 评估脚本
│   ├── plot_csv_loss.py     # CSV损失曲线绘制
│   ├── evaluate_checkpoint.py
│   ├── downstream_evaluation.py
│   └── ...
├── parallel_sweep_*.sh      # 并行实验脚本
│   ├── parallel_sweep_0.5B_ours.sh      # 0.5B模型 - 我们的方法
│   ├── parallel_sweep_0.5B_baselines.sh # 0.5B模型 - 基线方法
│   ├── parallel_sweep_1B_ours.sh         # 1B模型 - 我们的方法
│   └── parallel_sweep_1B_baselines.sh    # 1B模型 - 基线方法
├── reproduce_zo_paper.py    # 主训练脚本
├── cache/                   # 数据集缓存目录（自动生成）
└── logs/                    # 实验日志和结果目录
```

## 任务说明

需要运行以下4个脚本进行实验：

1. `parallel_sweep_0.5B_ours.sh` - 0.5B模型，我们的方法（Instruct模式）
2. `parallel_sweep_0.5B_baselines.sh` - 0.5B模型，基线方法（FO/ZO模式）
3. `parallel_sweep_1B_ours.sh` - 1B模型，我们的方法（Instruct模式）
4. `parallel_sweep_1B_baselines.sh` - 1B模型，基线方法（FO/ZO模式）

## 重要注意事项

### Batch Size 配置

- **Baseline脚本** (`*_baselines.sh`): 
  - Batch size 可以根据GPU显存情况**随意调大**
  - 在脚本中修改 `BATCH_SIZES` 参数，例如：`BATCH_SIZES=(4 8 16)` 或 `BATCH_SIZES=(32)`

- **Ours脚本** (`*_ours.sh`):
  - Batch size **不建议调太大**
  - 因为端侧资源受限，可能无法运行过大的batch size
  - 默认配置为 `BATCH_SIZES=(4)`，建议保持或小幅调整

### 数据集缓存

在开始大规模实验前，建议先运行一个脚本将数据集保存到缓存中，避免重复加载：

```bash
# 先运行一个简单的实验，让数据集pkl保存到cache目录
# 这个过程大约需要1小时
bash parallel_sweep_0.5B_ours.sh --gpus "0" --parallel 1
```

数据集缓存文件会保存在 `cache/` 目录下，格式为 `dataset_<dataset_name>_blk<block_size>_samples<max_samples>.pkl`

### GPU 分配

通过脚本中的 `GPU_IDS` 参数来分配GPU：

```bash
# 方式1: 在脚本中直接修改（推荐）
# 编辑脚本，找到 GPU_IDS="4" 这一行，修改为：
GPU_IDS="0,1,2,3"  # 使用GPU 0,1,2,3

# 方式2: 通过命令行参数传递
bash parallel_sweep_0.5B_ours.sh --gpus "0,1,2,3"
```

GPU分配示例：
- 单GPU: `GPU_IDS="0"`
- 多GPU: `GPU_IDS="0,1,2,3"`
- 自动检测: `GPU_IDS=""` (留空会自动检测所有可用GPU)

## 运行实验

### 基本运行

```bash
# 运行0.5B模型实验
bash parallel_sweep_0.5B_ours.sh
bash parallel_sweep_0.5B_baselines.sh

# 运行1B模型实验
bash parallel_sweep_1B_ours.sh
bash parallel_sweep_1B_baselines.sh
```

### 自定义GPU和并行数

```bash
# 指定GPU和最大并行任务数
bash parallel_sweep_0.5B_ours.sh --gpus "0,1,2,3" --parallel 8
```

## 输出结果

### 输出目录结构

实验完成后，结果保存在 `logs/parallel_sweep_<timestamp>/` 目录下：

```
logs/parallel_sweep_20251112_194831/
├── experiments/
│   └── <experiment_name>/
│       ├── logs/
│       │   └── csv_logs_<...>/
│       │       └── <experiment_name>.csv          # 训练损失CSV文件
│       └── evaluation_results.jsonl               # 评估结果JSONL文件
└── job_logs/                                      # 任务日志
```

### 关键输出文件

1. **CSV文件路径示例**:
   ```
   /data/cdq/current_project/zo-test-cdq/logs/parallel_sweep_20251112_194831/experiments/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3/logs/parallel_sweep_20251112_194831/csv_logs_FO_full_4_8_1_0.1_1e-3_mudamw_1_10_0.01_10.0/FO_20M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3.csv
   ```

2. **评估结果文件**:
   ```
   logs/parallel_sweep_<timestamp>/experiments/<experiment_name>/evaluation_results.jsonl
   ```

## 结果可视化

### 使用 plot_csv_loss.py 绘制对比曲线

1. **收集CSV文件**: 将所有需要对比的CSV文件放到一个文件夹中

```bash
# 创建对比文件夹
mkdir -p plots/csv_data_0.5B

# 复制CSV文件到该文件夹（从不同实验目录中）
cp logs/parallel_sweep_*/experiments/*/logs/*/csv_logs*/*.csv plots/csv_data_0.5B/
```

2. **运行绘图脚本**:

```bash
# 基本用法
python evaluation/plot_csv_loss.py --folder plots/csv_data_0.5B

# 带平滑和保存选项
python evaluation/plot_csv_loss.py \
    --folder plots/csv_data_0.5B \
    --smooth-window 20 \
    --max-points 500 \
    --output plots/loss_comparison_0.5B.png \
    --title "0.5B Model Loss Comparison"
```

### 绘图参数说明

- `--folder`: CSV文件所在文件夹（会递归搜索）
- `--smooth-window`: 移动平均窗口大小（默认10）
- `--max-points`: 每个系列的最大点数（用于降采样）
- `--output`: 输出图片路径
- `--title`: 图表标题

## 实验流程总结

1. **准备阶段**（可选但推荐）:
   ```bash
   # 先运行一个实验缓存数据集（约1小时）
   bash parallel_sweep_0.5B_ours.sh --gpus "0" --parallel 1
   ```

2. **运行实验**:
   ```bash
   # 根据GPU资源调整GPU_IDS和并行数
   bash parallel_sweep_0.5B_ours.sh --gpus "0,1,2,3" --parallel 8
   bash parallel_sweep_0.5B_baselines.sh --gpus "0,1,2,3" --parallel 8
   bash parallel_sweep_1B_ours.sh --gpus "0,1,2,3" --parallel 8
   bash parallel_sweep_1B_baselines.sh --gpus "0,1,2,3" --parallel 8
   ```

3. **收集结果**:
   - 从 `logs/parallel_sweep_<timestamp>/experiments/` 目录收集CSV文件
   - 收集 `evaluation_results.jsonl` 文件

4. **可视化对比**:
   ```bash
   # 将CSV文件整理到文件夹后
   python evaluation/plot_csv_loss.py --folder <csv_folder> --output <output_png>
   ```

## 常见问题

### Q: 如何查看实验进度？
A: 查看 `logs/parallel_sweep_<timestamp>/job_logs/` 目录下的日志文件

### Q: 实验失败怎么办？
A: 脚本会继续运行其他实验，失败信息会记录在日志中。可以单独重新运行失败的实验。

### Q: 如何修改实验参数？
A: 直接编辑对应的 `parallel_sweep_*.sh` 脚本，修改相应的配置参数（如学习率、batch size等）

### Q: 如何只运行部分实验？
A: 可以在脚本中注释掉不需要的参数组合，或者使用脚本的命令行参数来限制实验范围

