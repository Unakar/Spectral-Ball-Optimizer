#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制训练损失曲线（LM Loss）的专业科研论文级别绘图代码
适用于 Nature 等顶级期刊投稿

作者: Auto-generated
日期: 2025-12-24
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# 导入通用工具函数
from plot_figures.utils import (
    parse_training_log_file,
    save_figure,
    set_axis_limits,
    setup_publication_style,
)


def plot_training_loss_curves(
    log_files, labels, colors, output_file="lm_loss_comparison.pdf"
):
    """
    绘制训练损失曲线

    参数:
        log_files: 日志文件路径列表
        labels: 曲线标签列表
        colors: 颜色列表
        output_file: 输出文件路径
    """
    # 设置绘图风格
    setup_publication_style()

    # 解析所有日志文件
    all_data = []
    for log_file, label in zip(log_files, labels):
        print(f"正在解析日志文件: {log_file}")
        iterations, losses = parse_training_log_file(log_file)
        print(f"  - {label}: 找到 {len(iterations)} 个数据点 (步数范围: 1500-6500)")
        all_data.append((iterations, losses, label))

    # 创建图形 - 使用更适合论文的宽高比
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制主曲线
    lines = []
    for (iterations, losses, label), color in zip(all_data, colors):
        if len(iterations) > 0:
            (line,) = ax.plot(
                iterations,
                losses,
                color=color,
                linewidth=1.8,
                alpha=0.85,
                label=label,
                zorder=2,
            )
            lines.append(line)

    # 设置主图坐标轴
    ax.set_xlabel("Training Steps", fontsize=13, fontweight="bold")
    ax.set_ylabel("LM Loss", fontsize=13, fontweight="bold")

    # 设置y轴范围和刻度
    set_axis_limits(ax, xlim=(1500, 7000), ylim=(1.66, 2.11), y_tick_interval=0.05)

    # 添加网格
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    # 添加图例
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor="black",
    )
    legend.get_frame().set_linewidth(1.0)

    # 调整布局
    fig.set_constrained_layout(False)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)

    # 保存图形（多种格式）
    print("\n")
    save_figure(fig, output_file, formats=["pdf", "png", "eps"])

    print("\n绘图完成！")
    print(f"图形已保存为多种格式，可直接用于论文投稿。")


def main():
    """主函数"""
    # 输入文件路径
    log_files = [
        "/mnt2/data2/draw/new/adamw",
        "/mnt2/data2/draw/new/muon",
        "/mnt2/data2/draw/new/spectral_sphere",
    ]

    # 标签
    labels = ["AdamW", "Muon", "Spectral Sphere"]

    # 颜色方案
    colors = [
        "#C41E3A",  # 红色 - AdamW
        "#1E3A8A",  # 蓝色 - Muon
        "#166534",  # 绿色 - Spectral Sphere
    ]

    # 输出文件路径
    output_file = "/mnt2/data2/draw/new/lm_loss_comparison.pdf"

    # 绘制曲线
    plot_training_loss_curves(log_files, labels, colors, output_file)


if __name__ == "__main__":
    main()
