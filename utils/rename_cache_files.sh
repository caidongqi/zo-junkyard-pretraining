#!/bin/bash

# 批量重命名旧格式的数据集缓存文件
# 旧格式: dataset_*_bs{block_size}_samples*.pkl
# 新格式: dataset_*_blk{block_size}_samples*.pkl

echo "=========================================="
echo "数据集缓存文件重命名工具"
echo "=========================================="
echo ""

CACHE_DIR="cache"

# 检查缓存目录是否存在
if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ 缓存目录 '$CACHE_DIR' 不存在"
    echo "   请确保在项目根目录运行此脚本"
    exit 1
fi

cd "$CACHE_DIR" || exit 1

# 查找所有旧格式的文件
OLD_FILES=$(ls dataset_*_bs*.pkl 2>/dev/null)

if [ -z "$OLD_FILES" ]; then
    echo "✅ 没有找到需要重命名的旧格式文件"
    echo "   所有缓存文件已经是新格式 (blk)"
    exit 0
fi

echo "找到以下旧格式的缓存文件："
echo ""
echo "$OLD_FILES" | nl -w2 -s'. '
echo ""

# 统计数量
FILE_COUNT=$(echo "$OLD_FILES" | wc -l)
echo "共 $FILE_COUNT 个文件需要重命名"
echo ""

# 询问用户确认
read -p "是否继续重命名这些文件? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

echo ""
echo "开始重命名..."
echo ""

# 重命名文件
RENAMED=0
FAILED=0

for file in $OLD_FILES; do
    if [ -f "$file" ]; then
        # 使用 sed 将 _bs 替换为 _blk
        new_file=$(echo "$file" | sed 's/_bs/_blk/')
        
        if [ "$file" != "$new_file" ]; then
            echo "重命名: $file"
            echo "    -> $new_file"
            
            # 检查新文件名是否已存在
            if [ -f "$new_file" ]; then
                echo "    ⚠️  警告: 目标文件已存在，跳过"
                ((FAILED++))
            else
                mv "$file" "$new_file"
                if [ $? -eq 0 ]; then
                    echo "    ✅ 成功"
                    ((RENAMED++))
                else
                    echo "    ❌ 失败"
                    ((FAILED++))
                fi
            fi
            echo ""
        fi
    fi
done

cd - > /dev/null || exit 1

echo "=========================================="
echo "重命名完成！"
echo "=========================================="
echo ""
echo "统计："
echo "  成功: $RENAMED 个文件"
echo "  失败: $FAILED 个文件"
echo ""

if [ $RENAMED -gt 0 ]; then
    echo "✅ 缓存文件已更新为新格式"
    echo ""
    echo "新格式说明："
    echo "  - 使用 'blk' 表示 block_size（序列长度）"
    echo "  - 不同的 batch_size 会共用同一个 pkl 文件"
    echo "  - 格式: dataset_{name}_blk{block_size}_samples{count}.pkl"
fi

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  部分文件重命名失败，请检查错误信息"
fi

