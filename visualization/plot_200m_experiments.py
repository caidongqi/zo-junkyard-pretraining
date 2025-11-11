#!/usr/bin/env python3
"""
绘制所有200M实验的loss曲线对比图
包括FO和Instruct模式的所有实验
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def find_csv_files():
    """查找所有200M实验的CSV文件"""
    csv_files = {}
    
    # 查找所有200M的CSV文件
    pattern = "logs/parallel_sweep_*/experiments/*200M*/logs/*/csv_logs_*/*200M*.csv"
    all_files = glob.glob(pattern)
    
    if not all_files:
        print(f"未找到200M实验的CSV文件 (pattern: {pattern})")
        return csv_files
    
    # 按实验名称分组，选择最新的文件
    experiments = {}
    for csv_file in all_files:
        # 从路径中提取实验名称
        parts = csv_file.split('/')
        for part in parts:
            if '200M' in part and ('FO' in part or 'Instruct' in part):
                exp_name = part
                break
        
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(csv_file)
    
    # 为每个实验选择最新的文件
    for exp_name, files in experiments.items():
        latest_file = max(files, key=os.path.getmtime)
        csv_files[exp_name] = latest_file
        print(f"找到实验 {exp_name}: {latest_file}")
    
    return csv_files

def load_csv_data(csv_file):
    """加载CSV数据"""
    try:
        df = pd.read_csv(csv_file)
        if 'step' in df.columns and 'loss' in df.columns:
            return df
        else:
            print(f"警告: {csv_file} 缺少必要的列")
            print(f"  现有列: {df.columns.tolist()}")
            return None
    except Exception as e:
        print(f"读取 {csv_file} 失败: {e}")
        return None

def plot_loss_curves(csv_files, output_file):
    """绘制loss曲线对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 定义颜色和样式
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    color_idx = 0
    
    for exp_name, csv_file in sorted(csv_files.items()):
        df = load_csv_data(csv_file)
        
        if df is None:
            continue
        
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        # 从实验名称提取信息
        if 'FO' in exp_name:
            label_prefix = 'FO (BP only)'
        elif 'Instruct' in exp_name:
            # 提取q值
            if '_q8_' in exp_name:
                q_val = 'q=8'
            elif '_q64_' in exp_name:
                q_val = 'q=64'
            else:
                q_val = 'q=?'
            label_prefix = f'Instruct ({q_val})'
        else:
            label_prefix = exp_name
        
        # 绘制训练loss
        if 'step' in df.columns and 'loss' in df.columns:
            steps = df['step'].values
            losses = df['loss'].values
            
            # 过滤掉无效值
            valid_mask = pd.notna(steps) & pd.notna(losses)
            steps = steps[valid_mask]
            losses = losses[valid_mask]
            
            if len(steps) > 0:
                # 每隔10步取一次值（因为log_interval=10）
                step_mask = (steps % 10 == 0) | (steps == steps[-1])
                sampled_steps = steps[step_mask]
                sampled_losses = losses[step_mask]
                
                if len(sampled_steps) > 0:
                    label = f"{label_prefix} (train)"
                    ax.plot(sampled_steps, sampled_losses, label=label, color=color, 
                           linestyle='-', linewidth=2, alpha=0.7)
        
        # 绘制评估loss（如果有）
        if 'eval_loss' in df.columns and 'step' in df.columns:
            eval_mask = df['eval_loss'].notna() & (df['eval_loss'] != '')
            if eval_mask.any():
                eval_steps = df.loc[eval_mask, 'step'].values
                eval_losses = df.loc[eval_mask, 'eval_loss'].values
                
                try:
                    eval_losses = pd.to_numeric(eval_losses, errors='coerce')
                    valid_eval_mask = pd.notna(eval_losses)
                    eval_steps = eval_steps[valid_eval_mask]
                    eval_losses = eval_losses[valid_eval_mask]
                    
                    if len(eval_steps) > 0:
                        label = f"{label_prefix} (eval)"
                        ax.plot(eval_steps, eval_losses, label=label, 
                               color=color, linestyle='--', 
                               linewidth=2, alpha=0.8, marker='o', markersize=4)
                except Exception as e:
                    print(f"处理 {exp_name} 的评估loss时出错: {e}")
    
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Loss Curves: All 200M Model Experiments', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 2000)  # 限制横坐标范围为0-2000步
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存图表到: {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("开始绘制所有200M实验的loss曲线")
    print("=" * 60)
    
    # 查找CSV文件
    csv_files = find_csv_files()
    
    if not csv_files:
        print("错误: 未找到任何200M实验的CSV文件")
        return
    
    # 创建输出目录
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    
    # 绘制对比图
    print("\n绘制所有200M实验的对比图...")
    plot_loss_curves(
        csv_files,
        output_dir / 'loss_curves_all_200m.png'
    )
    
    print("\n" + "=" * 60)
    print("图表绘制完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()


