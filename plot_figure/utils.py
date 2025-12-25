#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘图工具函数集合
用于多个绘图文件共享的通用功能

作者: Auto-generated
日期: 2025-12-25
"""

import re

import matplotlib.ticker as ticker
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import rcParams


def setup_publication_style():
    """设置符合顶级期刊要求的绘图风格"""
    # 使用 Times New Roman 字体（Nature 等期刊常用）
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["DejaVu Serif"]
    rcParams["font.size"] = 11
    rcParams["axes.labelsize"] = 12
    rcParams["axes.titlesize"] = 12
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 10
    rcParams["legend.fontsize"] = 10
    rcParams["figure.titlesize"] = 13

    # 设置线条和标记
    rcParams["lines.linewidth"] = 1.5
    rcParams["lines.markersize"] = 4

    # 设置坐标轴
    rcParams["axes.linewidth"] = 1.0
    rcParams["xtick.major.width"] = 1.0
    rcParams["ytick.major.width"] = 1.0
    rcParams["xtick.minor.width"] = 0.8
    rcParams["ytick.minor.width"] = 0.8

    # 使用高质量输出
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = 0.05

    # 使用 PDF Type 42 字体（期刊要求）
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


def lighten_color(color: str, amount: float = 0.35) -> tuple:
    """
    将颜色向白色混合，让颜色"稍微亮一点"。
    amount ∈ [0, 1]，越大越亮。

    参数:
        color: 颜色字符串（如 "#006400"）
        amount: 混合比例

    返回:
        tuple: RGB颜色元组 (r, g, b)
    """
    r, g, b = mcolors.to_rgb(color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)


def parse_training_log_file(log_file, min_step=2000, max_step=6500):
    """
    解析训练日志文件，提取迭代步数和训练损失值

    参数:
        log_file: 日志文件路径
        min_step: 最小步数
        max_step: 最大步数

    返回:
        iterations: 迭代步数列表
        losses: 损失值列表
    """
    iterations = []
    losses = []

    # 正则表达式匹配训练日志行
    pattern = r"iteration\s+(\d+)/.*lm loss:\s+([\d.E+-]+)"

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                iteration = int(match.group(1))
                # 只保留指定范围内的数据
                if min_step <= iteration <= max_step:
                    loss = float(match.group(2))
                    iterations.append(iteration)
                    losses.append(loss)

    return np.array(iterations), np.array(losses)


def save_figure(fig, output_file, formats=["pdf", "png", "eps"]):
    """
    保存图形为多种格式

    参数:
        fig: matplotlib图形对象
        output_file: 输出文件路径（不含扩展名）
        formats: 保存格式列表
    """
    base_name = output_file.replace(".pdf", "")

    for fmt in formats:
        file_path = f"{base_name}.{fmt}"
        fig.savefig(file_path, format=fmt, dpi=300, bbox_inches="tight")
        print(f"已保存 {fmt.upper()} 格式: {file_path}")


def set_axis_limits(ax, xlim=None, ylim=None, y_tick_interval=None):
    """
    设置坐标轴范围和刻度

    参数:
        ax: matplotlib坐标轴对象
        xlim: x轴范围 (min, max)
        ylim: y轴范围 (min, max)
        y_tick_interval: y轴刻度间隔
    """
    if xlim:
        ax.set_xlim(*xlim)

    if ylim:
        ax.set_ylim(*ylim)

    if y_tick_interval:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_interval))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
