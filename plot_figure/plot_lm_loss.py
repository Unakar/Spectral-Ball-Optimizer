#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制训练损失曲线（LM Loss）的专业科研论文级别绘图代码
适用于 Nature 等顶级期刊投稿

作者: Auto-generated
日期: 2025-12-24
"""

import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# 设置论文级别的绘图参数
def setup_publication_style():
    """设置符合顶级期刊要求的绘图风格"""
    # 使用 Times New Roman 字体（Nature 等期刊常用）
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
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

    # 设置x轴范围为1000-7000（留白），数据点在1500-6500
    ax.set_xlim(1500, 7000)

    # 设置y轴范围为1.89到2.34，间隔0.05
    ax.set_ylim(1.66, 2.11)

    # 设置y轴刻度，间隔0.05
    import matplotlib.ticker as ticker

    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

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
    # PDF格式 - 用于论文投稿
    pdf_file = output_file.replace(".pdf", "") + ".pdf"
    plt.savefig(pdf_file, format="pdf", dpi=300, bbox_inches="tight")
    print(f"\n已保存 PDF 格式: {pdf_file}")

    # PNG格式 - 用于预览和演示
    png_file = output_file.replace(".pdf", "") + ".png"
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    print(f"已保存 PNG 格式: {png_file}")

    # EPS格式 - 某些期刊要求
    eps_file = output_file.replace(".pdf", "") + ".eps"
    plt.savefig(eps_file, format="eps", dpi=300, bbox_inches="tight")
    print(f"已保存 EPS 格式: {eps_file}")

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
