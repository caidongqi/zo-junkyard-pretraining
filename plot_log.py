#!/usr/bin/env python3
"""Plot training losses extracted from job log files.

This script scans directories matching ``job_logs_*`` under the given root,
parses ``*.log`` files to extract ``loss=`` entries, optionally downsamples the
series for readability, and renders a combined matplotlib plot. Use
``--output`` to save the figure or rely on the interactive ``plt.show()``.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
LOSS_PATTERN = re.compile(r"loss\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def extract_losses_from_log(path: Path) -> List[float]:
    losses: List[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = strip_ansi(raw_line)
            match = LOSS_PATTERN.search(line)
            if match:
                try:
                    losses.append(float(match.group(1)))
                except ValueError:
                    continue
    return losses


def downsample_series(values: List[float], max_points: Optional[int]) -> Tuple[List[int], List[float]]:
    if not values:
        return [], []

    steps = list(range(1, len(values) + 1))
    if max_points is None or max_points <= 0 or len(values) <= max_points:
        return steps, values[:]

    total = len(values)
    selected_indices: List[int] = []
    for i in range(max_points):
        idx = int(round(i * (total - 1) / (max_points - 1))) if max_points > 1 else 0
        selected_indices.append(idx)

    downsampled_steps = [steps[idx] for idx in selected_indices]
    downsampled_values = [values[idx] for idx in selected_indices]
    return downsampled_steps, downsampled_values


def iter_log_files(root: Path) -> Iterable[Path]:
    for job_dir in sorted(root.glob("job_logs_*")):
        if not job_dir.is_dir():
            continue
        for log_file in sorted(job_dir.glob("*.log")):
            if log_file.is_file():
                yield log_file


def load_all_losses(root: Path, max_points: Optional[int]) -> Dict[str, Tuple[List[int], List[float]]]:
    series: Dict[str, Tuple[List[int], List[float]]] = {}
    for log_file in iter_log_files(root):
        losses = extract_losses_from_log(log_file)
        if not losses:
            continue
        steps, downsampled = downsample_series(losses, max_points)
        series[log_file.stem] = (steps, downsampled)
    return series


def plot_series(
    series: Dict[str, Tuple[List[int], List[float]]],
    output: Optional[Path],
    title: Optional[str],
) -> None:
    if not series:
        print("No loss data found in job logs.")
        return

    plt.figure(figsize=(14, 7))
    for label, (steps, losses) in series.items():
        plt.plot(steps, losses, label=label)

    plt.xlabel(
        "Logged Steps (downsampled)"
        if any(len(steps) > 1 for steps, _ in series.values())
        else "Logged Step"
    )
    plt.ylabel("Loss")
    plt.title(title or "Training Loss from job_logs_*")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot losses from job log files.")
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to search for job_logs_* folders (default: current directory).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=500,
        help="Maximum points per series after downsampling (default: 500).",
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


# python plot_log.py --log-root /data/cdq/current_project/zo-test-cdq/log_data/ --output plots/log_plot.png
def main() -> None:
    args = parse_args()
    series = load_all_losses(args.log_root, args.max_points)
    plot_series(series, args.output, args.title)


if __name__ == "__main__":
    main()

