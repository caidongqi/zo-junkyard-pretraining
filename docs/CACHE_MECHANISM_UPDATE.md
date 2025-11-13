# 数据缓存机制优化说明

## 🎯 更新内容

修改了数据集缓存文件的命名规则，使其更加清晰易懂。

## 📝 修改详情

### 旧命名规则
```python
cache_file = f"dataset_{dataset_name}_bs{block_size}_samples{max_samples}.pkl"
```

**问题**：`bs` 容易与 batch_size 混淆，但实际上指的是 block_size（序列长度）

**示例**：
```
dataset_cosmopedia-100k_bs128_samples20000.pkl  # bs=block_size=128
dataset_cosmopedia-100k_bs512_samples20000.pkl  # bs=block_size=512
```

### 新命名规则
```python
cache_file = f"dataset_{dataset_name}_blk{block_size}_samples{max_samples}.pkl"
```

**改进**：使用 `blk` 明确表示 block_size，避免混淆

**示例**：
```
dataset_cosmopedia-100k_blk128_samples20000.pkl  # blk=block_size=128
dataset_cosmopedia-100k_blk512_samples20000.pkl  # blk=block_size=512
```

## 🔑 关键特性

### ✅ 不同 batch_size 共用同一个 pkl

这是**核心特性**！缓存文件名中**不包含** batch_size，因此：

```bash
# 这两个命令会使用同一个 pkl 文件
python reproduce_zo_paper.py --batch_size 4 --block_size 128  # 使用: blk128
python reproduce_zo_paper.py --batch_size 8 --block_size 128  # 使用: blk128（同一个文件）
python reproduce_zo_paper.py --batch_size 16 --block_size 128 # 使用: blk128（同一个文件）
```

### ❌ 不同 block_size 需要不同的 pkl

因为数据已经被切分成固定长度的块：

```bash
# 这两个命令会使用不同的 pkl 文件
python reproduce_zo_paper.py --block_size 128  # 使用: blk128（每个样本128 tokens）
python reproduce_zo_paper.py --block_size 512  # 使用: blk512（每个样本512 tokens）
```

## 📊 缓存文件命名规则

### 命名格式
```
dataset_{数据集名称}_blk{块大小}_samples{样本数}.pkl
```

### 各部分说明

| 部分 | 说明 | 示例 |
|------|------|------|
| `dataset_` | 固定前缀 | `dataset_` |
| `{数据集名称}` | 数据集名称 | `cosmopedia-100k` |
| `_blk{块大小}` | 序列长度（block_size） | `_blk512` |
| `_samples{样本数}` | 最大样本数 | `_samples20000` |
| `.pkl` | 文件扩展名 | `.pkl` |

### 完整示例

```
dataset_cosmopedia-100k_blk128_samples20000.pkl
dataset_cosmopedia-100k_blk512_samples20000.pkl
dataset_fineweb-edu_blk2048_samples10000.pkl
dataset_wikitext-103_blk256_samples5000.pkl
```

## 🔄 向后兼容性

### 旧缓存文件的处理

如果你的 `cache/` 目录中有旧命名格式的文件：

```bash
# 旧文件
dataset_cosmopedia-100k_bs128_samples20000.pkl
dataset_cosmopedia-100k_bs512_samples20000.pkl
```

**选项1：重命名（推荐）**
```bash
cd cache/
mv dataset_cosmopedia-100k_bs128_samples20000.pkl dataset_cosmopedia-100k_blk128_samples20000.pkl
mv dataset_cosmopedia-100k_bs512_samples20000.pkl dataset_cosmopedia-100k_blk512_samples20000.pkl
```

**选项2：删除并重新生成**
```bash
rm cache/dataset_*_bs*.pkl  # 删除旧文件
# 重新运行训练，会自动生成新格式的缓存
```

**选项3：不处理**
- 旧文件不会自动使用，会重新生成新格式的缓存
- 旧文件可以手动删除以释放空间

### 批量重命名脚本

创建 `rename_cache.sh`:

```bash
#!/bin/bash
# 批量重命名旧格式的缓存文件

cd cache/

for file in dataset_*_bs*.pkl; do
    if [ -f "$file" ]; then
        new_file=$(echo "$file" | sed 's/_bs/_blk/')
        echo "Renaming: $file -> $new_file"
        mv "$file" "$new_file"
    fi
done

echo "Done!"
```

运行：
```bash
chmod +x rename_cache.sh
./rename_cache.sh
```

## 💡 使用场景示例

### 场景1：相同数据不同 batch_size 训练

```bash
# 实验1: batch_size=4
python reproduce_zo_paper.py \
    --mode ZO \
    --batch_size 4 \
    --block_size 512 \
    --dataset cosmopedia-100k

# 实验2: batch_size=8（复用同一个pkl）
python reproduce_zo_paper.py \
    --mode ZO \
    --batch_size 8 \
    --block_size 512 \
    --dataset cosmopedia-100k

# 两次训练都使用：dataset_cosmopedia-100k_blk512_samples20000.pkl
```

**结果**：
- ✅ 第一次训练会创建 pkl 缓存
- ✅ 第二次训练直接加载缓存（节省时间）
- ✅ 磁盘空间只占用一份

### 场景2：不同 block_size 的对比实验

```bash
# 实验1: block_size=128
python reproduce_zo_paper.py \
    --mode ZO \
    --block_size 128 \
    --dataset cosmopedia-100k

# 实验2: block_size=512（需要新的pkl）
python reproduce_zo_paper.py \
    --mode ZO \
    --block_size 512 \
    --dataset cosmopedia-100k

# 生成两个不同的缓存文件：
# - dataset_cosmopedia-100k_blk128_samples20000.pkl
# - dataset_cosmopedia-100k_blk512_samples20000.pkl
```

**结果**：
- ✅ 每个 block_size 都有独立的缓存
- ✅ 后续使用相同 block_size 时可以复用

### 场景3：并行实验（parallel_sweep.sh）

```bash
# parallel_sweep.sh 中的配置
BATCH_SIZES=(4 8 16)
BLOCK_SIZES=(512 1024 2048)

# 对于相同的 block_size，不同的 batch_size 会共用缓存
# 例如：
# - bs=4, blk=512  -> 使用 blk512 缓存
# - bs=8, blk=512  -> 使用 blk512 缓存（复用）
# - bs=16, blk=512 -> 使用 blk512 缓存（复用）
```

**优势**：
- ⚡ 大幅减少数据加载时间
- 💾 节省磁盘空间
- 🔄 提高实验效率

## 📈 性能影响

### 缓存命中情况对比

#### 修改前（误解场景）
如果缓存名包含 batch_size：
```
dataset_cosmopedia-100k_bs4_blk512_samples20000.pkl
dataset_cosmopedia-100k_bs8_blk512_samples20000.pkl
dataset_cosmopedia-100k_bs16_blk512_samples20000.pkl
```
- ❌ 3个文件
- ❌ 重复数据
- ❌ 浪费磁盘空间

#### 修改后（实际情况）
```
dataset_cosmopedia-100k_blk512_samples20000.pkl
```
- ✅ 1个文件
- ✅ 所有batch_size共用
- ✅ 节省空间和时间

### 时间节省估算

对于一个典型的数据集加载（20K samples, block_size=512）：

| 操作 | 时间 |
|------|------|
| 首次加载（生成缓存） | ~2-5分钟 |
| 从缓存加载 | ~5-10秒 |
| **节省时间** | **~2-4分钟/次** |

如果运行10个不同 batch_size 的实验：
- 无缓存复用：~20-50分钟
- 有缓存复用：~5-10秒 + 首次5分钟 = ~5-6分钟
- **总节省**：~15-44分钟

## 🔍 验证方法

### 检查缓存是否被复用

```bash
# 运行第一次实验
python reproduce_zo_paper.py --batch_size 4 --block_size 512

# 查看输出，应该显示：
# "Saving dataset to cache: cache/dataset_..._blk512_samples20000.pkl"

# 运行第二次实验（不同batch_size）
python reproduce_zo_paper.py --batch_size 8 --block_size 512

# 查看输出，应该显示：
# "Loading dataset from cache: cache/dataset_..._blk512_samples20000.pkl"
# "Creating DataLoader with batch_size=8"
```

### 检查缓存文件

```bash
# 查看 cache 目录
ls -lh cache/

# 应该看到新格式的文件名
# dataset_cosmopedia-100k_blk512_samples20000.pkl
```

### 确认不同 batch_size 使用同一文件

```bash
# 启用调试信息
python reproduce_zo_paper.py --batch_size 4 --block_size 512 2>&1 | grep "cache"
python reproduce_zo_paper.py --batch_size 8 --block_size 512 2>&1 | grep "cache"

# 两次都应该显示相同的缓存文件路径
```

## 📋 总结

### 主要改进
1. ✅ 将 `bs` 改为 `blk`，命名更清晰
2. ✅ 明确说明不同 batch_size 共用缓存
3. ✅ 添加 DataLoader 创建时的日志输出
4. ✅ 更新文档说明

### 用户收益
- 🚀 **加快实验速度**：不同 batch_size 的实验可以复用缓存
- 💾 **节省磁盘空间**：避免重复存储相同数据
- 📝 **更清晰的命名**：`blk` 明确表示 block_size
- 🔍 **更好的可追溯性**：日志清楚显示使用的 batch_size

### 技术细节
- pkl 文件存储的是 `List[torch.Tensor]`，每个 tensor 形状为 `[block_size]`
- DataLoader 在运行时动态组织 batch，batch_size 不影响存储格式
- 缓存键值：`(dataset_name, block_size, max_samples)`

---

**更新日期**：2025-11-12  
**版本**：1.1  
**状态**：已完成 ✅

