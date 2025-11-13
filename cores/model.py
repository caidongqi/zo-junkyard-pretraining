"""
模型配置文件 (Model Configuration)
定义不同规模的 GPT-2 模型配置，用于预训练实验
"""

from transformers import GPT2Config, GPT2LMHeadModel


# ============================================================================
# 模型配置字典
# ============================================================================

MODEL_CONFIGS = {
    # 小型模型 (~20M 参数)
    # 适合快速实验和资源受限环境
    '20M': {
        'n_embd': 256,      # 嵌入维度
        'n_layer': 6,       # Transformer 层数
        'n_head': 4,        # 注意力头数
        'n_positions': 1024, # 最大序列长度
        'description': '超小型模型，约20M参数，适合快速原型验证',
        'estimated_params': '20M',
    },
    
    # 中小型模型 (~200M 参数)
    # 类似 GPT-2 Small 的规模
    '200M': {
        'n_embd': 768,      # 嵌入维度
        'n_layer': 12,      # Transformer 层数
        'n_head': 12,       # 注意力头数
        'n_positions': 1024, # 最大序列长度
        'description': '中小型模型，约200M参数，类似GPT-2 Small',
        'estimated_params': '200M',
    },
    
    # 中型模型 (~500M 参数)
    # 介于 GPT-2 Medium 和 Large 之间
    '500M': {
        'n_embd': 1024,     # 嵌入维度
        'n_layer': 24,      # Transformer 层数
        'n_head': 16,       # 注意力头数
        'n_positions': 1024, # 最大序列长度
        'description': '中型模型，约500M参数，平衡性能和计算成本',
        'estimated_params': '500M',
    },
    
    # 大型模型 (~1B 参数)
    # 类似 GPT-2 Large/XL 的规模
    '1B': {
        'n_embd': 1536,     # 嵌入维度
        'n_layer': 24,      # Transformer 层数
        'n_head': 24,       # 注意力头数
        'n_positions': 1024, # 最大序列长度
        'description': '大型模型，约1B参数，需要较大显存',
        'estimated_params': '1B',
    },
}


# ============================================================================
# 模型创建函数
# ============================================================================

def create_model(model_size='200M', vocab_size=50257):
    """
    根据指定的模型规模创建 GPT-2 模型
    
    Args:
        model_size: 模型大小标识，可选 '20M', '200M', '500M', '1B'
        vocab_size: 词汇表大小，默认为 GPT-2 的标准词汇表大小
    
    Returns:
        GPT2LMHeadModel: 初始化的模型
    
    Examples:
        >>> model = create_model('200M', vocab_size=50257)
        >>> model = create_model('1B', vocab_size=50304)  # 扩展词汇表后
    """
    if model_size not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available options: {available}"
        )
    
    config_dict = MODEL_CONFIGS[model_size]
    print(f"Creating model: {model_size}")
    print(f"Description: {config_dict['description']}")
    print(f"Configuration:")
    print(f"  - Embedding dim: {config_dict['n_embd']}")
    print(f"  - Layers: {config_dict['n_layer']}")
    print(f"  - Attention heads: {config_dict['n_head']}")
    print(f"  - Max position: {config_dict['n_positions']}")
    print(f"  - Vocab size: {vocab_size}")
    
    # 创建 GPT-2 配置
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=config_dict['n_positions'],
        n_embd=config_dict['n_embd'],
        n_layer=config_dict['n_layer'],
        n_head=config_dict['n_head'],
        bos_token_id=vocab_size - 1,  # 使用词汇表最后一个 token 作为 BOS
        eos_token_id=vocab_size - 1,  # 使用词汇表最后一个 token 作为 EOS
    )
    
    # 创建模型
    model = GPT2LMHeadModel(config)
    
    # 计算实际参数量
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created successfully!")
    print(f"Total parameters: {params_total:,} ({params_total / 1e6:.2f}M)")
    print(f"Trainable parameters: {params_trainable:,} ({params_trainable / 1e6:.2f}M)")
    
    return model


def get_model_info(model_size='200M'):
    """
    获取指定模型规模的配置信息（不创建模型）
    
    Args:
        model_size: 模型大小标识
    
    Returns:
        dict: 模型配置字典
    """
    if model_size not in MODEL_CONFIGS:
        available = ', '.join(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available options: {available}"
        )
    
    return MODEL_CONFIGS[model_size].copy()


def list_available_models():
    """
    列出所有可用的模型配置
    """
    print("Available model configurations:")
    print("=" * 80)
    for size, config in MODEL_CONFIGS.items():
        print(f"\n{size}:")
        print(f"  Description: {config['description']}")
        print(f"  Estimated params: {config['estimated_params']}")
        print(f"  n_embd={config['n_embd']}, n_layer={config['n_layer']}, "
              f"n_head={config['n_head']}, n_positions={config['n_positions']}")
    print("=" * 80)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model configuration viewer")
    parser.add_argument(
        '--list', 
        action='store_true', 
        help='List all available model configurations'
    )
    parser.add_argument(
        '--create', 
        type=str, 
        choices=['20M', '200M', '500M', '1B'],
        help='Create a model of specified size'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=50257,
        help='Vocabulary size (default: 50257)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    elif args.create:
        model = create_model(args.create, vocab_size=args.vocab_size)
        print("\nModel created successfully!")
    else:
        parser.print_help()

