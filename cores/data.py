"""
数据集配置文件 (Dataset Configuration)
定义不同的预训练数据集及其加载方式
"""

from pathlib import Path
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset as HFDataset


# ============================================================================
# 数据集配置字典
# ============================================================================

DATASET_CONFIGS = {
    # Cosmopedia-100k: 高质量合成教育数据
    # 优点：数据质量高，适合快速实验
    # 缺点：数据量较小
    'cosmopedia-100k': {
        'hf_path': 'HuggingFaceTB/cosmopedia-100k',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': '高质量合成教育数据集，100k样本，适合快速实验',
        'recommended_samples': 20000,  # 推荐使用的样本数
        'language': 'en',
        'size': '~100K documents',
    },
    
    # Cosmopedia (完整版): 更大规模的高质量合成数据
    # 优点：数据量大，质量高
    # 缺点：下载和处理较慢
    'cosmopedia': {
        'hf_path': 'HuggingFaceTB/cosmopedia',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'Cosmopedia完整版，30M+样本，高质量合成教育数据（web_samples_v2）',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~30M documents',
        'dataset_name': 'web_samples_v2',  # 默认使用web_samples_v2子集
    },
    
    # Cosmopedia - Stories: 故事子集
    'cosmopedia-stories': {
        'hf_path': 'HuggingFaceTB/cosmopedia',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'Cosmopedia stories子集，故事数据',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~3M documents',
        'dataset_name': 'stories',
    },
    
    # Cosmopedia - KhanAcademy: Khan Academy教育内容
    'cosmopedia-khanacademy': {
        'hf_path': 'HuggingFaceTB/cosmopedia',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'Cosmopedia KhanAcademy子集，高质量教育内容',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~2M documents',
        'dataset_name': 'khanacademy',
    },
    
    # Cosmopedia - OpenStax: OpenStax教科书内容
    'cosmopedia-openstax': {
        'hf_path': 'HuggingFaceTB/cosmopedia',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'Cosmopedia OpenStax子集，教科书内容',
        'recommended_samples': 30000,
        'language': 'en',
        'size': '~1M documents',
        'dataset_name': 'openstax',
    },
    
    # WikiText-103: 维基百科文本
    # 优点：经典预训练数据集，广泛使用
    # 缺点：数据量相对较小
    'wikitext-103': {
        'hf_path': 'wikitext',
        'split': 'train',
        'text_field': 'text',
        'streaming': False,
        'description': '维基百科文本，经典语言建模数据集',
        'recommended_samples': None,  # 使用全部数据
        'language': 'en',
        'size': '~100M tokens',
        'dataset_name': 'wikitext-103-v1',
    },
    
    # OpenWebText: 开源的 WebText 复现版本
    # 优点：数据多样性好，接近真实网络文本分布
    # 缺点：数据集较大，需要较长处理时间
    'openwebtext': {
        'hf_path': 'openwebtext',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'OpenWebText，开源WebText复现，8M+网页文档',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~8M documents, ~40GB',
    },
    
    # C4 (Colossal Clean Crawled Corpus): 超大规模清洗网页数据
    # 优点：数据量巨大，质量经过清洗
    # 缺点：非常大，需要大量存储和处理时间
    'c4': {
        'hf_path': 'c4',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'C4数据集，超大规模清洗网页语料，适合大规模预训练',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~365M documents, ~750GB',
        'dataset_name': 'en',
    },
    
    # TinyStories: 简单故事数据集
    # 优点：数据简单，模型容易学习，适合调试
    # 缺点：任务相对简单，不适合评估复杂能力
    'tinystories': {
        'hf_path': 'roneneldan/TinyStories',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'TinyStories，简单故事数据集，适合小模型和快速实验',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~2M stories',
    },
    
    # The Pile (子集): 多样化大规模预训练语料
    # 优点：数据多样性极佳，包含多个领域
    # 缺点：完整版非常大，建议使用子集
    'pile-subset': {
        'hf_path': 'monology/pile-uncopyrighted',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'The Pile无版权子集，多样化高质量预训练数据',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~200GB',
    },
    
    # FineWeb: 高质量预训练数据集（完整版）
    # 优点：主流预训练数据，质量高，15T tokens
    # 缺点：数据量巨大
    'fineweb': {
        'hf_path': 'HuggingFaceFW/fineweb',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'FineWeb完整版，15T tokens，主流高质量预训练数据',
        'recommended_samples': 100000,
        'language': 'en',
        'size': '~15T tokens',
        'dataset_name': 'default',
    },
    
    # FineWeb-Edu: FineWeb教育子集
    # 优点：教育内容质量高，1.3T tokens
    # 缺点：仍然很大
    'fineweb-edu': {
        'hf_path': 'HuggingFaceFW/fineweb-edu',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'FineWeb教育子集，1.3T tokens，高质量教育内容',
        'recommended_samples': 50000,
        'language': 'en',
        'size': '~1.3T tokens',
        'dataset_name': 'default',
    },
    
    # FineWeb-Edu-10BT: FineWeb-Edu 10BT采样版本
    # 优点：适合快速实验，质量高
    # 缺点：相对较小
    'fineweb-edu-10bt': {
        'hf_path': 'HuggingFaceFW/fineweb-edu',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'FineWeb-Edu 10BT采样，适合快速实验的高质量教育数据',
        'recommended_samples': 30000, #maximum 1.53B
        'language': 'en',
        'size': '~10B tokens',
        'dataset_name': 'sample-10BT',
    },
}


# ============================================================================
# 数据加载函数 (核心修改部分)
# ============================================================================

def get_dataloader(
    tokenizer,
    dataset_name='cosmopedia-100k',
    batch_size=4,
    block_size=128,
    max_samples=None,
    cache_dir="cache",
    force_reload=False,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    num_proc=None,
):
    """
    加载指定数据集并创建 DataLoader（使用 datasets.save_to_disk 高效缓存）
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available options: {available}"
        )
    
    config = DATASET_CONFIGS[dataset_name]
    
    if max_samples is None:
        max_samples = config.get('recommended_samples', 20000)

    cache_base_dir = Path(cache_dir)
    cache_path = cache_base_dir / f"dataset_{dataset_name}_blk{block_size}_samples{max_samples}"

    if not force_reload and cache_path.exists():
        print(f"Loading processed dataset from disk cache: {cache_path}")
        try:
            processed_dataset = HFDataset.load_from_disk(str(cache_path))
            print(f"Dataset loaded from cache. Total blocks: {len(processed_dataset)}")
        except Exception as e:
            print(f"Failed to load from cache: {e}. Re-processing dataset.")
            # 清理损坏的缓存并重新处理
            import shutil
            shutil.rmtree(cache_path, ignore_errors=True)
            return get_dataloader(
                tokenizer, dataset_name, batch_size, block_size, max_samples,
                cache_dir, True, num_workers, pin_memory, prefetch_factor,
                persistent_workers, num_proc
            )
    else:
        print(f"Cache not found or `force_reload` is True. Processing dataset from scratch.")
        cache_base_dir.mkdir(exist_ok=True)
        
        # --- 1. 加载原始数据集 ---
        print(f"Loading raw dataset: {dataset_name}")
        load_kwargs = {
            'path': config['hf_path'],
            'split': config['split'],
            'name': config.get('dataset_name'),
            'download_mode': 'reuse_cache_if_exists',
        }
        # 无论如何，先以流式加载来节省初始内存，特别是对于大文件
        dataset = load_dataset(**load_kwargs, streaming=True)

        if max_samples:
            dataset = dataset.take(max_samples)
            print(f"Streaming: taking the first {max_samples} samples.")
        
        # --- 2. 【关键修改】将流式数据集实体化 ---
        # 这是创建缓存所必需的步骤。它会迭代数据流并将所有样本加载到内存中。
        print("Materializing streaming dataset into a standard dataset...")
        dataset_list = list(tqdm(dataset, desc="Loading samples into memory"))
        dataset = HFDataset.from_list(dataset_list)
        print(f"Materialized dataset with {len(dataset)} samples.")

        # --- 3. Tokenization ---
        if num_proc is None:
            num_proc = os.cpu_count() or 1
        
        def tokenize_function(examples):
            # 移除值为None的文本，避免tokenizer报错
            texts = [text for text in examples[config['text_field']] if text is not None]
            return tokenizer(texts, add_special_tokens=False)

        print(f"Tokenizing dataset with {num_proc} processes...")
        # 现在 'dataset' 是一个标准Dataset，可以安全地使用多进程map
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Running tokenizer on dataset",
        )

        # --- 4. 组合和分块 ---
        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        print(f"Grouping texts into chunks of {block_size}...")
        processed_dataset = tokenized_dataset.map(
            group_texts,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            desc=f"Grouping texts into chunks of {block_size}",
        )
        
        # --- 5. 保存到磁盘缓存 ---
        # 现在 processed_dataset 是一个标准的Dataset，可以安全地保存
        print(f"Saving processed dataset to disk cache: {cache_path}")
        processed_dataset.save_to_disk(str(cache_path))
        print(f"Dataset prepared and cached. Total blocks: {len(processed_dataset)}")

    # --- 6. 创建 DataLoader ---
    processed_dataset.set_format(type='torch', columns=['input_ids', 'labels'])

    print(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}")
    return DataLoader(
        processed_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )


def get_dataset_info(dataset_name='cosmopedia-100k'):
    """
    获取数据集配置信息（不加载数据）
    
    Args:
        dataset_name: 数据集名称
    
    Returns:
        dict: 数据集配置字典
    """
    if dataset_name not in DATASET_CONFIGS:
        available = ', '.join(DATASET_CONFIGS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available options: {available}"
        )
    
    return DATASET_CONFIGS[dataset_name].copy()


def list_available_datasets():
    """
    列出所有可用的数据集配置
    """
    print("Available dataset configurations:")
    print("=" * 100)
    for name, config in DATASET_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  HuggingFace path: {config['hf_path']}")
        print(f"  Size: {config['size']}")
        print(f"  Language: {config['language']}")
        print(f"  Recommended samples: {config.get('recommended_samples', 'All')}")
        print(f"  Streaming: {config['streaming']}")
    print("=" * 100)


def print_performance_tips():
    """
    打印DataLoader性能优化建议
    """
    import os
    cpu_count = os.cpu_count()
    
    print("\n" + "=" * 80)
    print("DataLoader 性能优化指南")
    print("=" * 80)
    
    print("\n1. CPU相关参数:")
    print(f"   - 当前系统CPU核心数: {cpu_count}")
    print(f"   - 推荐 num_proc (tokenization进程数): {cpu_count} (使用所有核心)")
    print(f"   - 推荐 num_workers (DataLoader进程数): 4-8")
    print("   - 注意: num_workers过大会增加内存占用和进程通信开销")
    
    print("\n2. GPU相关参数:")
    print("   - pin_memory=True: 使用固定内存，加速CPU->GPU传输")
    print("   - 仅在GPU训练时启用，CPU训练时设为False")
    
    print("\n3. 预取参数:")
    print("   - prefetch_factor=2-4: 每个worker预取的batch数")
    print("   - 增大可减少等待时间，但会增加内存占用")
    
    print("\n4. Worker持久化:")
    print("   - persistent_workers=True: 保持worker进程存活")
    print("   - 避免每个epoch重新创建进程，但会占用更多内存")
    
    print("\n5. Tokenization优化:")
    print("   - 使用batched=True进行批量tokenization")
    print("   - batch_size=1000: 每批处理的样本数")
    print("   - 多进程并行加速（num_proc参数）")
    
    print("\n6. 缓存策略:")
    print("   - 首次加载会创建.pkl缓存文件")
    print("   - 后续加载直接读取缓存，极大加速")
    print("   - 使用force_reload=True强制重新处理")
    
    print("\n7. 关于GPU加速Tokenization:")
    print("   - ❌ Tokenization不适合GPU加速")
    print("   - ✓ GPU擅长: 大规模并行浮点运算（训练/推理）")
    print("   - ✓ CPU擅长: 字符串处理、查找、分割（tokenization）")
    print("   - ✓ 正确做法: 使用多进程CPU并行tokenization")
    
    print("\n8. 性能测试建议:")
    print("   - 先用小数据集测试最佳参数组合")
    print("   - 监控: CPU使用率、内存占用、GPU利用率")
    print("   - 调整num_workers和num_proc找到平衡点")
    
    print("\n9. 推荐配置示例:")
    print("   # 快速实验（小数据集）")
    print("   dataloader = get_dataloader(tokenizer, 'cosmopedia-100k',")
    print("                               batch_size=8, num_workers=4, num_proc=8)")
    print()
    print("   # 大规模训练（大数据集）")
    print(f"   dataloader = get_dataloader(tokenizer, 'fineweb-edu',")
    print(f"                               batch_size=32, num_workers=8, num_proc={cpu_count})")
    print()
    print("   # CPU训练")
    print("   dataloader = get_dataloader(tokenizer, 'cosmopedia-100k',")
    print("                               batch_size=4, num_workers=2, pin_memory=False)")
    
    print("\n" + "=" * 80)
    print("提示: 使用 python cores/data.py --perf 查看此指南")
    print("=" * 80 + "\n")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset configuration viewer and performance guide")
    parser.add_argument(
        '--list', 
        action='store_true', 
        help='List all available dataset configurations'
    )
    parser.add_argument(
        '--info', 
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help='Show detailed information about a specific dataset'
    )
    parser.add_argument(
        '--perf',
        action='store_true',
        help='Show performance optimization tips for DataLoader'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_datasets()
    elif args.info:
        info = get_dataset_info(args.info)
        print(f"\nDataset: {args.info}")
        print("=" * 80)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("=" * 80)
    elif args.perf:
        print_performance_tips()
    else:
        parser.print_help()

