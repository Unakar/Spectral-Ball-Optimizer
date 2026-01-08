#!/usr/bin/env python3
"""绘制 muP (Maximal Update Parametrization) 图

横坐标: 学习率 (lr)
纵坐标: Loss
每组宽度 (hidden_size) 连成一条曲线
星星符号标记每条曲线的最低点
"""

import os
import sys

# 支持直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from utils import save_figure, set_legend_style, setup_plt_style

setup_plt_style()

# 所有学习率刻度值（排除 1e-3）
ALL_LR_TICKS = [3e-3, 5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2, 2e-2, 3e-2]

# 需要排除的学习率
EXCLUDE_LRS = [1e-3]

# 数据目录和输出目录
RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data"
OUTPUT_DIR = Path(__file__).parent / "results" / "mup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MUON_CSV = RAW_DATA_DIR / "mup" / "muon_results.csv"
SPBALL_CSV = RAW_DATA_DIR / "mup" / "spball_results.csv"

# 配色方案 - 使用鲜明的调色板
COLORS = {
    256: "#E63946",  # 红色
    512: "#F77F00",  # 橙色
    1024: "#2A9D8F",  # 青色
    2048: "#264653",  # 深蓝灰色
}

# 线条样式
LINE_STYLES = {
    256: "-",
    512: "-",
    1024: "-",
    2048: "-",
}


def format_lr_tick(x, pos):
    """格式化学习率刻度标签"""
    if x >= 0.01:
        # 1e-2 及以上，显示为 1e-2, 1.5e-2, 2e-2, 3e-2
        if x == 0.01:
            return "1e-2"
        elif x == 0.015:
            return "1.5e-2"
        elif x == 0.02:
            return "2e-2"
        elif x == 0.03:
            return "3e-2"
        else:
            return f"{x:.0e}"
    else:
        # 1e-3 到 9e-3
        coef = x / 1e-3
        if coef == int(coef):
            return f"{int(coef)}e-3"
        else:
            return f"{coef:.1f}e-3"


def set_lr_xticks(ax):
    """设置学习率 x 轴的细粒度刻度"""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ALL_LR_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(format_lr_tick))
    ax.tick_params(axis="x", rotation=45)


def load_and_process_data(csv_path: Path) -> pd.DataFrame:
    """加载 CSV 并处理学习率为数值"""
    df = pd.read_csv(csv_path)
    # 将 lr 字符串转换为浮点数
    df["lr_value"] = df["lr"].apply(lambda x: float(x))
    # 排除指定的学习率
    df = df[~df["lr_value"].isin(EXCLUDE_LRS)]
    # 按 hidden_size 和 lr_value 排序
    df = df.sort_values(["hidden_size", "lr_value"])
    return df


def plot_single_optimizer(
    ax, df: pd.DataFrame, title: str, show_legend: bool = True, show_label: bool = True
):
    """绘制单个优化器的 muP 图"""

    hidden_sizes = sorted(df["hidden_size"].unique())

    # 收集所有最低点信息，用于后续绘制垂直线
    min_points = []

    for hidden_size in hidden_sizes:
        subset = df[df["hidden_size"] == hidden_size].sort_values("lr_value")

        lr_values = subset["lr_value"].values
        losses = subset["loss"].values

        color = COLORS.get(hidden_size, "#333333")
        linestyle = LINE_STYLES.get(hidden_size, "-")

        # 绘制曲线
        (line,) = ax.plot(
            lr_values,
            losses,
            marker="o",
            markersize=6,
            linewidth=2,
            linestyle=linestyle,
            color=color,
            label=f"width={hidden_size}",
            alpha=0.85,
        )

        # 找到最低点
        min_idx = np.argmin(losses)
        min_lr = lr_values[min_idx]
        min_loss = losses[min_idx]

        # 保存最低点信息
        min_points.append((min_lr, min_loss, color))

        # 用星星标记最低点
        ax.scatter(
            [min_lr],
            [min_loss],
            marker="*",
            s=500,
            c=color,
            edgecolors="white",
            linewidths=2,
            zorder=10,
        )

        # 添加最低点标注
        ax.annotate(
            f"{min_loss:.4f}",
            xy=(min_lr, min_loss),
            xytext=(5, 10),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    # 设置细粒度的 x 轴刻度
    set_lr_xticks(ax)
    # x轴标签在 plot_mup_comparison 中统一设置
    if show_label:
        ax.set_ylabel("Val Loss", fontweight="bold")
    ax.set_title(title, pad=10, fontsize=18)
    if show_legend:
        set_legend_style(ax, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    # 加粗加黑左边 y 轴和下边 x 轴
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("black")
    # 隐藏右边和上边的轴线
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # 获取 y 轴范围，绘制从最低点到 x 轴的垂直虚线
    y_min, y_max = ax.get_ylim()
    for min_lr, min_loss, color in min_points:
        ax.vlines(
            x=min_lr,
            ymin=y_min,
            ymax=min_loss,
            colors=color,
            linestyles="dashed",
            linewidth=1.5,
            zorder=1,
        )
    # 恢复 y 轴范围（避免 vlines 扩展范围）
    ax.set_ylim(y_min, y_max)

    # 箭头（太丑了，不加比较好）
    # x_min, x_max = ax.get_xlim()
    # y_min, y_max = ax.get_ylim()

    # ax.annotate("", xy=(1, 0), xytext=(0.999, 0),
    #             arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2", facecolor="black", mutation_scale=20), xycoords="axes fraction")
    # ax.annotate("", xy=(0, 1), xytext=(0, 0.999),
    #             arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2", facecolor="black", mutation_scale=20), xycoords="axes fraction")

    # ax.spines["bottom"].set_bounds(x_min, x_min + (x_max - x_min) * 0.95)
    # ax.spines["left"].set_bounds(y_min-0.005, y_min + (y_max - y_min) * 0.98)


def plot_mup_comparison():
    """绘制 muon 和 spball 的 muP 对比图"""

    # 创建图形（稍矮一点）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 加载数据（不在子图中显示图例）
    if MUON_CSV.exists():
        df_muon = load_and_process_data(MUON_CSV)
        plot_single_optimizer(axes[0], df_muon, "Muon Optimizer", show_legend=False)
    else:
        axes[0].text(
            0.5,
            0.5,
            "muon_results.csv not found",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    if SPBALL_CSV.exists():
        df_spball = load_and_process_data(SPBALL_CSV)
        plot_single_optimizer(
            axes[1], df_spball, "Spectral Sphere Optimizer", show_legend=False
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "spball_results.csv not found",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    # 统一两个子图的 y 轴范围，都从 2.6 开始
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(2.6, y_max)

    # 获取第一个子图的 handles 和 labels 用于统一图例
    handles, labels = axes[0].get_legend_handles_labels()

    # 添加星星图例项表示 min loss
    star_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="gray",
        markersize=14,
        linestyle="None",
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="Min Loss",
    )
    handles.append(star_handle)
    labels.append("Min Loss")

    # 在图的底部添加统一的横向图例（缩小字体和 marker，往上移）
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),  # 横向排列成一行
        framealpha=0.95,
        edgecolor="black",
        fontsize=14,
    )
    legend.get_frame().set_linewidth(1.0)

    # 在两图下方中间添加共享的 x 轴标签（放在 x 轴线上）
    fig.text(0.5, 0.18, "LR", ha="center", va="bottom", fontsize=18, fontweight="bold")

    # 调整布局
    fig.set_constrained_layout(False)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.22)

    # 保存
    output_file = OUTPUT_DIR / "mup_comparison.pdf"
    save_figure(fig, str(output_file), formats=["pdf", "png", "eps"])


def plot_single_file(csv_path: Path, output_name: str = None):
    """绘制单个 CSV 文件的 muP 图"""

    if not csv_path.exists():
        print(f"错误: 文件不存在 {csv_path}")
        return

    df = load_and_process_data(csv_path)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 7))

    hidden_sizes = sorted(df["hidden_size"].unique())

    # 收集所有最低点信息，用于后续绘制垂直线
    min_points = []

    for hidden_size in hidden_sizes:
        subset = df[df["hidden_size"] == hidden_size].sort_values("lr_value")

        lr_values = subset["lr_value"].values
        losses = subset["loss"].values

        color = COLORS.get(hidden_size, "#333333")
        linestyle = LINE_STYLES.get(hidden_size, "-")

        # 绘制曲线
        ax.plot(
            lr_values,
            losses,
            marker="o",
            markersize=8,
            linewidth=2.5,
            linestyle=linestyle,
            color=color,
            label=f"width={hidden_size}",
            alpha=0.85,
        )

        # 找到最低点
        min_idx = np.argmin(losses)
        min_lr = lr_values[min_idx]
        min_loss = losses[min_idx]

        # 保存最低点信息
        min_points.append((min_lr, min_loss, color))

        # 用星星标记最低点
        ax.scatter(
            [min_lr],
            [min_loss],
            marker="*",
            s=800,
            c=color,
            edgecolors="white",
            linewidths=2.5,
            zorder=10,
        )

        # 添加最低点标注
        ax.annotate(
            f"{min_loss:.4f}",
            xy=(min_lr, min_loss),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            # bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # 设置细粒度的 x 轴刻度
    set_lr_xticks(ax)
    ax.set_xlabel("Learning Rate", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")

    # 从文件名推断标题
    optimizer_name = csv_path.stem.replace("_results", "").title()
    ax.set_title(
        f"μP Analysis - {optimizer_name} Optimizer",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # 设置图例（缩小字体）
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor="black",
        fontsize=14,
        markerscale=0.8,
    )
    legend.get_frame().set_linewidth(1.0)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    # 加粗加黑左边 y 轴和下边 x 轴
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("black")
    # 隐藏右边和上边的轴线
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # 获取 y 轴范围，绘制从最低点到 x 轴的垂直虚线
    y_min, y_max = ax.get_ylim()
    for min_lr, min_loss, color in min_points:
        ax.vlines(
            x=min_lr,
            ymin=y_min,
            ymax=min_loss,
            colors=color,
            linestyles="dashed",
            linewidth=1.5,
            alpha=0.6,
            zorder=1,
        )
    # 恢复 y 轴范围（避免 vlines 扩展范围）
    ax.set_ylim(y_min, y_max)

    # 调整布局
    fig.set_constrained_layout(False)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.12)

    # 保存图片
    if output_name is None:
        base_name = f"mup_{csv_path.stem}"
    else:
        base_name = output_name.replace(".png", "").replace(".pdf", "")

    output_file = OUTPUT_DIR / f"{base_name}.pdf"
    save_figure(fig, str(output_file), formats=["pdf", "png", "eps"])


def main():
    """主函数"""
    print("=" * 60)
    print("μP (Maximal Update Parametrization) 绘图工具")
    print("=" * 60)

    # 也可以生成普通版本
    print("\n生成普通对比图...")
    plot_mup_comparison()

    # 也可以单独绘制每个优化器
    print("\n生成单独的 Muon 图...")
    plot_single_file(MUON_CSV, "mup_muon.png")

    print("\n生成单独的 Spectral Ball 图...")
    plot_single_file(SPBALL_CSV, "mup_spball.png")

    print("\n完成！")


if __name__ == "__main__":
    main()
