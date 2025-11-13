#!/usr/bin/env python3
"""
论文绘图脚本：绘制 Loss vs CO2 Emission 图

读取CSV文件，横坐标为loss，纵坐标为CO2 emission。
CO2计算规则：
- FO和ZO开头的文件：CO2 = steps * 100
- Instruct开头的文件：CO2 = steps（正常值）
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# ==================== 配置区域：在这里修改CSV文件路径 ====================
CSV_FILES = [
    'plots/csv_data_200M/FO_200M_full_bs4_blk512_qN_A_bpN_A_optmudamw_lr1e-3.csv',
    'plots/csv_data_200M/ZO_200M_full_bs4_blk512_q8_bpN_A_optmudamw_lr1e-3.csv',
    'plots/csv_data_200M/Instruct_200M_full_bs4_blk512_q8_bp1_optmudamw_lr1e-3_blend0.2_ct0.01_ns10.0.csv',
]

LABEL_CURATED = ["FO", "ZO", "Ours"]  # 自定义标签列表（如果为空则自动生成）
OUTPUT_FILE = 'plots/loss_vs_co2_200M.png'  # 输出文件路径
TITLE = 'Loss vs CO₂ Emission'  # 图表标题（None表示使用默认标题）

# 图表配置
FIGSIZE = (4, 2)  # 图表大小（宽, 高）
DPI = 300  # 分辨率
LINEWIDTH = 2.0  # 线宽
# ========================================================================


def calculate_co2_emission(steps: List[int], filename: str) -> List[int]:
    """根据文件名计算CO2排放量"""
    # FO和ZO开头：steps * 100
    if filename.startswith('FO'):
        return [step * 100 * 10 for step in steps]
    # Instruct开头：正常steps
    elif filename.startswith('Instruct'):
        return steps
    else:
        # 默认情况：尝试从文件名判断
        filename_lower = filename.lower()
        if 'fo' in filename_lower:
            return [step * 100 * 10 for step in steps]
        else:
            return steps


def simplify_label(filename: str) -> str:
    """简化标签名称，提取关键信息"""
    # 提取模型大小和方法
    if 'FO' in filename:
        return 'FO'
    elif 'ZO' in filename:
        return 'ZO'
    elif 'Instruct' in filename:
        return 'Instruct'
    else:
        # 如果无法识别，返回文件名（去掉路径）
        return Path(filename).stem


def load_csv_data(csv_path: Path) -> Tuple[List[float], List[int], str]:
    """从CSV文件加载数据"""
    df = pd.read_csv(csv_path)
    
    # 检查必需的列
    if 'step' not in df.columns or 'loss' not in df.columns:
        raise ValueError(f"CSV文件缺少必需的列: {csv_path}")
    
    if df.empty:
        raise ValueError(f"CSV文件为空: {csv_path}")
    
    steps = df['step'].values.tolist()
    losses = df['loss'].values.tolist()
    
    # 计算CO2排放量
    filename = csv_path.stem
    co2_emissions = calculate_co2_emission(steps, filename)
    
    # 简化标签
    label = simplify_label(filename)
    
    return losses, co2_emissions, label


def plot_loss_vs_co2(
    data_list: List[Tuple[List[float], List[int], str]],
    output: str,
    title: str = None,
) -> None:
    """绘制 Loss vs CO2 Emission 图"""
    if not data_list:
        print("错误: 没有数据可绘制")
        return
    
    # 创建图表
    plt.figure(figsize=FIGSIZE, dpi=DPI)
    
    # 定义颜色和线型（论文风格）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':']
    
    # 绘制每条曲线
    for idx, (losses, co2_emissions, label) in enumerate(data_list):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        
        plt.plot(
            losses,
            co2_emissions,
            label=label,
            linewidth=LINEWIDTH,
            alpha=0.8,
            color=color,
            linestyle=linestyle,
        )
    
    # 设置标签和标题
    plt.xlabel('Loss', fontsize=12, fontweight='bold')
    plt.ylabel('CO₂ Emission', fontsize=12, fontweight='bold')
    plt.yscale('log')
    plt.gca().invert_xaxis()  # 反转x轴，使Loss从大到小显示
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    
    # 优化图例：放在右上角，更简洁
    plt.legend(
        loc='lower right',
        fontsize=10,
        framealpha=0.3,
        fancybox=True,
        shadow=False,
        edgecolor='gray',
        frameon=True,
        # bbox_to_anchor=(0.5, 1),
    )
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 美化图表
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("读取CSV文件...")
    print("=" * 60)
    
    # 检查自定义标签数量是否匹配
    use_custom_labels = len(LABEL_CURATED) == len(CSV_FILES) and len(LABEL_CURATED) > 0
    if not use_custom_labels and len(LABEL_CURATED) > 0:
        print(f"警告: 自定义标签数量({len(LABEL_CURATED)})与CSV文件数量({len(CSV_FILES)})不匹配，将使用自动生成的标签")
        use_custom_labels = False
    
    # 加载数据
    data_list = []
    for idx, csv_file_str in enumerate(CSV_FILES):
        csv_file = Path(csv_file_str)
        
        if not csv_file.exists():
            print(f"错误: 文件不存在: {csv_file}")
            return
        
        print(f"读取: {csv_file.name}")
        try:
            losses, co2_emissions, auto_label = load_csv_data(csv_file)
            
            # 使用自定义标签或自动生成的标签
            if use_custom_labels:
                label = LABEL_CURATED[idx]
            else:
                label = auto_label
            
            data_list.append((losses, co2_emissions, label))
            print(f"  - 标签: {label}")
            print(f"  - 数据点数: {len(losses)}")
            print(f"  - Loss范围: [{min(losses):.4f}, {max(losses):.4f}]")
            print(f"  - CO₂范围: [{min(co2_emissions)}, {max(co2_emissions)}]")
        except Exception as e:
            print(f"  - 错误: {e}")
            return
    
    print("=" * 60)
    print("绘制图表...")
    print("=" * 60)
    
    # 绘制图表
    plot_loss_vs_co2(
        data_list=data_list,
        output=OUTPUT_FILE,
        title=TITLE,
    )
    
    print("=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
