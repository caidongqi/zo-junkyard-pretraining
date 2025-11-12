#!/usr/bin/env python3
"""
Script to plot all CSV files from a parallel_sweep_xx folder as curves.
Usage: python plot_parallel_sweep.py <folder_path>
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import numpy as np

def find_all_csv_files(folder_path):
    """
    Find all CSV files in the given folder recursively.
    """
    csv_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)

def extract_experiment_name(csv_path):
    """
    Extract a meaningful experiment name from the CSV file path.
    """
    # Get the CSV filename without extension
    filename = Path(csv_path).stem
    return filename

def smooth_curve(data, window_size=10):
    """
    Apply moving average smoothing to data.
    
    Args:
        data: numpy array or pandas Series
        window_size: size of the smoothing window
    
    Returns:
        smoothed numpy array
    """
    if len(data) < window_size:
        return data
    
    # Convert to numpy array if needed
    if hasattr(data, 'values'):
        data = data.values
    
    # Apply moving average
    cumsum = np.cumsum(np.insert(data, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    
    # Pad the beginning to maintain the same length
    padding = data[:window_size-1]
    smoothed = np.concatenate([padding, smoothed])
    
    return smoothed

def plot_all_csvs(folder_path, output_dir=None, smooth_window=10):
    """
    Plot all CSV files from the folder.
    Creates separate plots for loss and grad_norm.
    
    Args:
        folder_path: path to the parallel_sweep folder
        output_dir: directory to save plots (default: folder_path/plots)
        smooth_window: window size for moving average smoothing (default: 10)
    """
    # Find all CSV files
    csv_files = find_all_csv_files(folder_path)
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(folder_path, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Training Curves - {Path(folder_path).name} (Smoothed, window={smooth_window})', 
                 fontsize=16, fontweight='bold')
    
    # Color map for different experiments
    colors = plt.cm.tab20(np.linspace(0, 1, len(csv_files)))
    
    # Track valid files for legend
    valid_experiments = []
    
    # Plot each CSV file
    for idx, csv_file in enumerate(csv_files):
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            # Skip empty files
            if df.empty:
                print(f"Skipping empty file: {csv_file}")
                continue
            
            # Get experiment name
            exp_name = extract_experiment_name(csv_file)
            
            # Check required columns
            if 'step' not in df.columns or 'loss' not in df.columns:
                print(f"Skipping {csv_file}: missing required columns")
                continue
            
            # Apply smoothing
            steps = df['step'].values
            loss_smoothed = smooth_curve(df['loss'].values, smooth_window)
            
            # Plot loss (convert to numpy to avoid indexing issues)
            axes[0].plot(steps, loss_smoothed, 
                        label=exp_name, 
                        color=colors[idx],
                        linewidth=1.5,
                        alpha=0.8)
            
            # Plot grad_norm if available
            if 'grad_norm' in df.columns:
                grad_norm_smoothed = smooth_curve(df['grad_norm'].values, smooth_window)
                axes[1].plot(steps, grad_norm_smoothed, 
                            label=exp_name,
                            color=colors[idx],
                            linewidth=1.5,
                            alpha=0.8)
            
            valid_experiments.append(exp_name)
            print(f"Plotted: {exp_name}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    if not valid_experiments:
        print("No valid data to plot")
        return
    
    # Configure loss plot
    axes[0].set_xlabel('Training Step', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Steps', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                   fontsize=8, framealpha=0.9)
    
    # Configure grad_norm plot
    axes[1].set_xlabel('Training Step', fontsize=12)
    axes[1].set_ylabel('Gradient Norm', fontsize=12)
    axes[1].set_title('Gradient Norm over Steps', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                   fontsize=8, framealpha=0.9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'all_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also create individual plots for each metric
    create_individual_plots(folder_path, csv_files, output_dir, smooth_window)
    
    plt.close()

def create_individual_plots(folder_path, csv_files, output_dir, smooth_window=10):
    """
    Create separate plots for loss and grad_norm.
    
    Args:
        folder_path: path to the parallel_sweep folder
        csv_files: list of CSV file paths
        output_dir: directory to save plots
        smooth_window: window size for moving average smoothing
    """
    # Plot only loss
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(csv_files)))
    
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if df.empty or 'step' not in df.columns or 'loss' not in df.columns:
                continue
            
            exp_name = extract_experiment_name(csv_file)
            steps = df['step'].values
            loss_smoothed = smooth_curve(df['loss'].values, smooth_window)
            
            plt.plot(steps, loss_smoothed, 
                    label=exp_name,
                    color=colors[idx],
                    linewidth=2,
                    alpha=0.8)
        except Exception as e:
            continue
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss - {Path(folder_path).name} (Smoothed, window={smooth_window})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    
    loss_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved to: {loss_path}")
    plt.close()
    
    # Plot only grad_norm
    plt.figure(figsize=(12, 6))
    
    has_grad_norm = False
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if df.empty or 'step' not in df.columns or 'grad_norm' not in df.columns:
                continue
            
            exp_name = extract_experiment_name(csv_file)
            steps = df['step'].values
            grad_norm_smoothed = smooth_curve(df['grad_norm'].values, smooth_window)
            
            plt.plot(steps, grad_norm_smoothed, 
                    label=exp_name,
                    color=colors[idx],
                    linewidth=2,
                    alpha=0.8)
            has_grad_norm = True
        except Exception as e:
            continue
    
    if has_grad_norm:
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Gradient Norm', fontsize=12)
        plt.title(f'Gradient Norm - {Path(folder_path).name} (Smoothed, window={smooth_window})', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
        plt.tight_layout()
        
        grad_path = os.path.join(output_dir, 'gradient_norm.png')
        plt.savefig(grad_path, dpi=300, bbox_inches='tight')
        print(f"Gradient norm plot saved to: {grad_path}")
    
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_parallel_sweep.py <folder_path> [output_dir] [smooth_window]")
        print("\nExample:")
        print("  python plot_parallel_sweep.py logs/parallel_sweep_20251112_003830")
        print("  python plot_parallel_sweep.py logs/parallel_sweep_20251112_003830 plots/my_plots")
        print("  python plot_parallel_sweep.py logs/parallel_sweep_20251112_003830 plots/my_plots 20")
        print("\nArguments:")
        print("  folder_path    - Path to parallel_sweep folder")
        print("  output_dir     - (Optional) Output directory for plots")
        print("  smooth_window  - (Optional) Smoothing window size (default: 10)")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    smooth_window = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)
    
    print(f"Processing folder: {folder_path}")
    print(f"Smooth window size: {smooth_window}")
    print("=" * 60)
    
    plot_all_csvs(folder_path, output_dir, smooth_window)
    
    print("=" * 60)
    print("Done!")

if __name__ == "__main__":
    main()

