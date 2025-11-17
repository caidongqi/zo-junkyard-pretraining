#!/usr/bin/env python3
"""Plot training losses from CSV files in a parallel_sweep folder.

This script recursively scans a directory for CSV files containing training logs,
extracts the loss values, optionally applies smoothing and downsampling, and
renders a combined matplotlib plot.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def extract_losses_from_csv(path: Path) -> Tuple[List[int], List[float]]:
    """Extract step and loss columns from a CSV file.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Tuple of (steps, losses) lists
    """
    try:
        df = pd.read_csv(path)
        
        # Check for required columns
        if 'step' not in df.columns or 'loss' not in df.columns:
            return [], []
        
        # Skip empty dataframes
        if df.empty:
            return [], []
        
        steps = df['step'].values.tolist()
        losses = df['loss'].values.tolist()
        
        return steps, losses
        
    except Exception as e:
        print(f"Warning: Failed to read {path}: {e}")
        return [], []


def smooth_series(values: List[float], window: int) -> List[float]:
    """Apply trailing moving average smoothing.
    
    Args:
        values: Input data series
        window: Size of the smoothing window
        
    Returns:
        Smoothed series of the same length
    """
    if window is None or window <= 1 or not values:
        return values[:]

    window = min(window, len(values))
    acc = 0.0
    rolling: deque[float] = deque()
    smoothed: List[float] = []

    for value in values:
        rolling.append(value)
        acc += value
        if len(rolling) > window:
            acc -= rolling.popleft()
        smoothed.append(acc / len(rolling))

    return smoothed


def downsample_series(
    steps: List[int],
    values: List[float],
    max_points: Optional[int]
) -> Tuple[List[int], List[float]]:
    """Downsample a series to a maximum number of points.
    
    Args:
        steps: Step numbers
        values: Loss values
        max_points: Maximum number of points to keep
        
    Returns:
        Tuple of (downsampled_steps, downsampled_values)
    """
    if not values:
        return [], []

    if max_points is None or max_points <= 0 or len(values) <= max_points:
        return steps[:], values[:]

    total = len(values)
    selected_indices: List[int] = []
    for i in range(max_points):
        idx = int(round(i * (total - 1) / (max_points - 1))) if max_points > 1 else 0
        selected_indices.append(idx)

    downsampled_steps = [steps[idx] for idx in selected_indices]
    downsampled_values = [values[idx] for idx in selected_indices]
    return downsampled_steps, downsampled_values


def iter_csv_files(root: Path) -> Iterable[Path]:
    """Recursively find all CSV files under the root directory.
    
    Args:
        root: Root directory to search
        
    Yields:
        Path objects for each CSV file found
    """
    for csv_file in sorted(root.rglob("*.csv")):
        if csv_file.is_file():
            yield csv_file


def load_all_losses(
    root: Path,
    max_points: Optional[int],
    smooth_window: Optional[int],
) -> Dict[str, Tuple[List[int], List[float]]]:
    """Load and process loss data from all CSV files.
    
    Args:
        root: Root directory containing CSV files
        max_points: Maximum points per series after downsampling
        smooth_window: Window size for moving average smoothing
        
    Returns:
        Dictionary mapping experiment names to (steps, losses) tuples
    """
    series: Dict[str, Tuple[List[int], List[float]]] = {}
    
    for csv_file in iter_csv_files(root):
        steps, losses = extract_losses_from_csv(csv_file)
        
        if not losses:
            continue
        
        # Apply smoothing
        smoothed = smooth_series(losses, smooth_window)
        
        # Downsample
        ds_steps, ds_losses = downsample_series(steps, smoothed, max_points)
        
        # Use filename without extension as label
        label = csv_file.stem
        series[label] = (ds_steps, ds_losses)
        
    return series


def plot_series(
    series: Dict[str, Tuple[List[int], List[float]]],
    output: Optional[Path],
    title: Optional[str],
    smooth_window: int,
) -> None:
    """Plot all loss series on a single figure.
    
    Args:
        series: Dictionary of experiment names to (steps, losses) tuples
        output: Optional output path to save the figure
        title: Optional custom title for the plot
        smooth_window: Smoothing window size (for display in title)
    """
    if not series:
        print("No loss data found in CSV files.")
        return

    plt.figure(figsize=(4, 2))
    
    for label, (steps, losses) in sorted(series.items()):
        # if label.startswith('FO') or label.startswith('ZO'):
            # steps = 4 * np.array(steps)
        plt.plot(steps, losses, label=label, linewidth=1.5, alpha=0.8)

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    
    # Build title
    if title:
        plot_title = title
    else:
        plot_title = "Training Loss from CSV Files"
    
    if smooth_window and smooth_window > 1:
        plot_title += f" (smoothed, window={smooth_window})"
    
    plt.title(plot_title)
    plt.xlim(0,2000)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot training losses from CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all CSVs in a folder with default settings
  python plot_csv_loss.py logs/parallel_sweep_20251112_003830
  
  # Apply smoothing and save to file
  python plot_csv_loss.py logs/parallel_sweep_20251112_003830 --smooth-window 20 --output plots/loss.png
  
  # Downsample to 500 points and use custom title
  python plot_csv_loss.py logs/parallel_sweep_20251112_003830 --max-points 500 --title "My Experiment"
        """
    )
    parser.add_argument(
        "--folder",
        type=Path,
        help="Root directory to search for CSV files (searches recursively).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Maximum points per series after downsampling (default: no downsampling).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Apply a trailing moving-average with this window size (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot image instead of displaying it.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title (default: generic).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if not args.folder.exists():
        print(f"Error: Directory '{args.folder}' does not exist.")
        return
    
    if not args.folder.is_dir():
        print(f"Error: '{args.folder}' is not a directory.")
        return
    
    print(f"Searching for CSV files in: {args.folder}")
    print(f"Smooth window: {args.smooth_window}")
    if args.max_points:
        print(f"Max points per series: {args.max_points}")
    print("=" * 60)
    
    series = load_all_losses(args.folder, args.max_points, args.smooth_window)
    
    if series:
        print(f"Found {len(series)} CSV file(s) with loss data")
    
    plot_series(series, args.output, args.title, args.smooth_window)
    
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()

