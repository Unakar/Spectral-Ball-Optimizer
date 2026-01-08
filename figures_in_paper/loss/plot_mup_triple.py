#!/usr/bin/env python3
"""绘制 muP (Maximal Update Parametrization) 三优化器对比图

横坐标: 学习率 (lr)
纵坐标: Loss
每组宽度 (hidden_size) 连成一条曲线
星星符号标记每条曲线的最低点
包含: AdamW, Muon, Spectral Sphere 三个优化器
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
ALL_LR_TICKS = [1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2, 1.5e-2, 2e-2, 3e-2]

# 需要排除的学习率
EXCLUDE_LRS = []

# 数据目录和输出目录
RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data"
OUTPUT_DIR = Path(__file__).parent / "results" / "mup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADAMW_CSV = RAW_DATA_DIR / "mup" / "adamw_results.csv"
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
    ax.tick_params(axis="x", rotation=75)


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
    ax, df: pd.DataFrame, title: str, show_legend: bool = True, show_ylabel: bool = True
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
        ax.plot(
            lr_values,
            losses,
            marker="o",
            markersize=5,
            linewidth=1.8,
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
            s=400,
            c=color,
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
        )

        # 添加最低点标注
        ax.annotate(
            f"{min_loss:.4f}",
            xy=(min_lr, min_loss),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    # 设置细粒度的 x 轴刻度
    set_lr_xticks(ax)
    if show_ylabel:
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
            ymin=2.6,  # 保证最低点在所有图的最低点
            ymax=min_loss,
            colors=color,
            linestyles="dashed",
            linewidth=1.5,
            zorder=1,
        )
    # 恢复 y 轴范围（避免 vlines 扩展范围）
    ax.set_ylim(y_min, y_max)


def plot_mup_triple_comparison():
    """绘制 AdamW, Muon 和 Spectral Sphere 的 muP 三优化器对比图"""

    # 创建图形（三个子图）
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # 加载并绘制 AdamW
    if ADAMW_CSV.exists():
        df_adamw = load_and_process_data(ADAMW_CSV)
        plot_single_optimizer(
            axes[0], df_adamw, "AdamW", show_legend=False, show_ylabel=True
        )
    else:
        axes[0].text(
            0.5,
            0.5,
            "adamw_results.csv not found",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    # 加载并绘制 Muon
    if MUON_CSV.exists():
        df_muon = load_and_process_data(MUON_CSV)
        plot_single_optimizer(
            axes[1], df_muon, "Muon", show_legend=False, show_ylabel=False
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "muon_results.csv not found",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    # 加载并绘制 Spectral Sphere
    if SPBALL_CSV.exists():
        df_spball = load_and_process_data(SPBALL_CSV)
        plot_single_optimizer(
            axes[2], df_spball, "Spectral Sphere", show_legend=False, show_ylabel=False
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "spball_results.csv not found",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )

    # 统一三个子图的 y 轴范围
    y_mins = []
    y_maxs = []
    for ax in axes:
        y_min, y_max = ax.get_ylim()
        y_mins.append(y_min)
        y_maxs.append(y_max)

    # 统一设置 y 轴范围
    common_y_min = 2.6
    common_y_max = max(y_maxs)
    for ax in axes:
        ax.set_ylim(common_y_min, common_y_max)

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

    # 在图的底部添加统一的横向图例
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        framealpha=0.95,
        edgecolor="black",
        fontsize=20,
    )
    legend.get_frame().set_linewidth(1.0)

    # 在三图下方中间添加共享的 x 轴标签
    fig.text(0.355, 0.16, "LR", ha="center", va="bottom", fontsize=16, fontweight="bold")
    fig.text(0.665, 0.16, "LR", ha="center", va="bottom", fontsize=16, fontweight="bold")

    # 调整布局
    fig.set_constrained_layout(False)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.22)

    # 保存
    output_file = OUTPUT_DIR / "mup_triple_comparison.pdf"
    save_figure(fig, str(output_file), formats=["pdf", "png", "eps"])


def main():
    """主函数"""
    print("=" * 60)
    print("μP (Maximal Update Parametrization) 三优化器对比绘图工具")
    print("=" * 60)

    print("\n生成三优化器对比图 (AdamW, Muon, Spectral Sphere)...")
    plot_mup_triple_comparison()

    print("\n完成！")


if __name__ == "__main__":
    main()
