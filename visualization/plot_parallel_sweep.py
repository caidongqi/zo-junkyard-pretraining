#!/usr/bin/env python3
"""
绘制parallel_sweep.sh实验结果的loss曲线对比图
- FO模式（纯BP）
- Instruct模式（BP辅助ZO）
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def find_latest_sweep_dir():
    """查找最新的parallel_sweep目录"""
    pattern = "logs/parallel_sweep_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
    
    # 按修改时间排序，返回最新的
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

def find_csv_files(sweep_dir):
    """查找FO和Instruct实验的CSV文件"""
    csv_files = {}
    
    # 查找FO实验的CSV文件
    fo_pattern = f"{sweep_dir}/experiments/FO_*/logs/*/csv_logs_*/FO_*.csv"
    fo_files = glob.glob(fo_pattern)
    if fo_files:
        latest_fo = max(fo_files, key=os.path.getmtime)
        csv_files['FO'] = latest_fo
        print(f"找到FO实验: {latest_fo}")
    else:
        print(f"未找到FO实验的CSV文件 (pattern: {fo_pattern})")
    
    # 查找Instruct实验的CSV文件
    instruct_pattern = f"{sweep_dir}/experiments/Instruct_*/logs/*/csv_logs_*/Instruct_*.csv"
    instruct_files = glob.glob(instruct_pattern)
    if instruct_files:
        latest_instruct = max(instruct_files, key=os.path.getmtime)
        csv_files['Instruct'] = latest_instruct
        print(f"找到Instruct实验: {latest_instruct}")
    else:
        print(f"未找到Instruct实验的CSV文件 (pattern: {instruct_pattern})")
    
    return csv_files

def load_csv_data(csv_file):
    """加载CSV数据"""
    try:
        df = pd.read_csv(csv_file)
        # CSV列名: timestamp, epoch, step, mode, scope, q, lr, batch_size, optimizer, bp_interval, loss, grad_norm
        if 'step' in df.columns and 'loss' in df.columns:
            return df
        else:
            print(f"警告: {csv_file} 缺少必要的列")
            print(f"  现有列: {df.columns.tolist()}")
            return None
    except Exception as e:
        print(f"读取 {csv_file} 失败: {e}")
        return None

def plot_loss_curves(csv_files, output_file, sweep_dir):
    """绘制loss曲线对比图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 实验标签和颜色
    exp_configs = {
        'FO': {
            'label': 'FO (BP only)',
            'color': '#1f77b4',  # 蓝色
        },
        'Instruct': {
            'label': 'Instruct (BP-assisted ZO)',
            'color': '#ff7f0e',  # 橙色
        },
    }
    
    for exp_id in ['FO', 'Instruct']:
        if exp_id not in csv_files:
            continue
        
        csv_file = csv_files[exp_id]
        df = load_csv_data(csv_file)
        
        if df is None:
            continue
        
        config = exp_configs[exp_id]
        
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
                step_mask = (steps % 10 == 0) | (steps == steps[-1])  # 包含最后一个点
                sampled_steps = steps[step_mask]
                sampled_losses = losses[step_mask]
                
                if len(sampled_steps) > 0:
                    label = f"{config['label']} (train)"
                    ax.plot(sampled_steps, sampled_losses, label=label, color=config['color'], 
                           linestyle='-', linewidth=2, alpha=0.7)
        
        # 绘制评估loss（如果有）
        if 'eval_loss' in df.columns and 'step' in df.columns:
            # 只取有评估loss的步骤
            eval_mask = df['eval_loss'].notna() & (df['eval_loss'] != '')
            if eval_mask.any():
                eval_steps = df.loc[eval_mask, 'step'].values
                eval_losses = df.loc[eval_mask, 'eval_loss'].values
                
                # 转换为数值类型
                try:
                    eval_losses = pd.to_numeric(eval_losses, errors='coerce')
                    valid_eval_mask = pd.notna(eval_losses)
                    eval_steps = eval_steps[valid_eval_mask]
                    eval_losses = eval_losses[valid_eval_mask]
                    
                    if len(eval_steps) > 0:
                        label = f"{config['label']} (eval)"
                        ax.plot(eval_steps, eval_losses, label=label, 
                               color=config['color'], linestyle='--', 
                               linewidth=2, alpha=0.8, marker='o', markersize=4)
                except Exception as e:
                    print(f"处理 {exp_id} 的评估loss时出错: {e}")
    
    # 从sweep_dir提取配置信息
    sweep_name = os.path.basename(sweep_dir)
    
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Loss Curves: FO vs Instruct (500M Model, bs=32, q=8)', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"已保存图表到: {output_file}")
    plt.close()

def main():
    print("=" * 60)
    print("开始绘制parallel_sweep实验结果")
    print("=" * 60)
    
    # 查找最新的sweep目录
    sweep_dir = find_latest_sweep_dir()
    if not sweep_dir:
        print("错误: 未找到parallel_sweep目录")
        return
    
    print(f"使用sweep目录: {sweep_dir}")
    
    # 查找CSV文件
    csv_files = find_csv_files(sweep_dir)
    
    if not csv_files:
        print("错误: 未找到任何CSV文件")
        return
    
    if len(csv_files) < 2:
        print(f"警告: 只找到 {len(csv_files)} 个实验的CSV文件，将只绘制找到的实验")
    
    # 创建输出目录
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    
    # 绘制对比图
    print("\n绘制FO vs Instruct对比图...")
    plot_loss_curves(
        csv_files,
        output_dir / 'loss_curves_parallel_sweep.png',
        sweep_dir
    )
    
    print("\n" + "=" * 60)
    print("图表绘制完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()


