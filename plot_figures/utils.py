#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of plotting utility functions
Common functionality shared across multiple plotting files

Author: Auto-generated
Date: 2025-12-25
"""

import re

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import rcParams

LIGHT_COLORS = [
    "#ea5762",
    "#f89226",
    "#4aaca0",
    "#46626d",
]

DARK_COLORS = {
    "adamw": "#c41e3a",
    "muon sphere": "#1ec4a8",
    "spectral sphere": "#2e9d18",
    "muon": "#1e3ac4",
}


def setup_plt_style():
    """Set plotting style"""
    # Use serif font (DejaVu Serif as primary)
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["DejaVu Serif"]
    rcParams["font.size"] = 14
    rcParams["axes.labelsize"] = 14
    rcParams["axes.titlesize"] = 16
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12
    rcParams["legend.fontsize"] = 16
    rcParams["figure.titlesize"] = 24

    # Set lines and markers
    rcParams["lines.linewidth"] = 2.5
    rcParams["lines.markersize"] = 4

    # Set axes
    rcParams["axes.linewidth"] = 1.0
    rcParams["xtick.major.width"] = 1.0
    rcParams["ytick.major.width"] = 1.0
    rcParams["xtick.minor.width"] = 0.8
    rcParams["ytick.minor.width"] = 0.8

    # Use high-quality output
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = 0.05

    # Use PDF Type 42 font
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


def lighten_color(color: str, amount: float = 0.35) -> tuple:
    """
    Mix color with white to make it "slightly lighter".
    amount ∈ [0, 1], larger values make the color brighter.

    Args:
        color: Color string (e.g., "#006400")
        amount: Mixing ratio

    Returns:
        tuple: RGB color tuple (r, g, b)
    """
    r, g, b = mcolors.to_rgb(color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)


def parse_training_log_file(log_file, min_step=2000, max_step=6500):
    """
    Parse training log file to extract iteration steps and training loss values

    Args:
        log_file: Path to log file
        min_step: Minimum step number
        max_step: Maximum step number

    Returns:
        iterations: List of iteration steps
        losses: List of loss values
    """
    iterations = []
    losses = []

    # Regular expression to match training log lines
    pattern = r"iteration\s+(\d+)/.*lm loss:\s+([\d.E+-]+)"

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                # Only keep data within specified range
                if min_step <= iteration <= max_step:
                    loss = float(match.group(2))
                    iterations.append(iteration)
                    losses.append(loss)

    return np.array(iterations), np.array(losses)


def save_figure(fig, output_file, formats=["pdf", "png", "eps"]):
    """
    Save figure in multiple formats

    Args:
        fig: Matplotlib figure object
        output_file: Output file path (without extension)
        formats: List of formats to save
    """
    base_name = output_file.replace(".pdf", "")

    for fmt in formats:
        file_path = f"{base_name}.{fmt}"
        fig.savefig(file_path, format=fmt, dpi=300, bbox_inches="tight")
        print(f"Saved {fmt.upper()} format: {file_path}")


def set_axis_limits(ax, xlim=None, ylim=None, y_tick_interval=None):
    """
    Set axis limits and ticks

    Args:
        ax: Matplotlib axis object
        xlim: x-axis range (min, max)
        ylim: y-axis range (min, max)
        y_tick_interval: y-axis tick interval
    """
    if xlim:
        ax.set_xlim(*xlim)

    if ylim:
        ax.set_ylim(*ylim)

    if y_tick_interval:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


def set_legend_style(ax, loc="upper right"):
    """
    Set legend style for the given axis

    Args:
        ax: Matplotlib axis object
        loc: Legend location
    """
    legend = ax.legend(
        loc=loc,
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(1.0)
    return legend
