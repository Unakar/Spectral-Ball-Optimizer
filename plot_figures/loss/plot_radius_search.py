#!/usr/bin/env python3
"""绘制 Radius Scale 搜索实验图

横坐标: 学习率 (lr)
纵坐标: Final Loss
每组 radius_scale 连成一条曲线
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

# 所有学习率刻度值
ALL_LR_TICKS = [1e-3, 5e-3, 1e-2, 5e-2]

# 数据目录和输出目录
RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data"
OUTPUT_DIR = Path(__file__).parent / "results" / "radius_search"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = RAW_DATA_DIR / "radius" / "radius_search.csv"

# 配色方案 - 不同 radius scale 使用不同颜色
COLORS = {
    0.1: "#E63946",   # 红色
    0.5: "#F77F00",   # 橙色
    1: "#2A9D8F",     # 青色
    2: "#264653",     # 深蓝灰色
    10: "#7209B7",    # 紫色
}


def format_lr_tick(x, pos):
    """格式化学习率刻度标签"""
    if x >= 0.01:
        if x == 0.01:
            return "1e-2"
        elif x == 0.05:
            return "5e-2"
        else:
            return f"{x:.0e}"
    else:
        coef = x / 1e-3
        if coef == int(coef):
            return f"{int(coef)}e-3"
        else:
            return f"{coef:.1f}e-3"


def set_lr_xticks(ax, lr_values):
    """设置学习率 x 轴的刻度（均匀尺度）"""
    # 使用均匀尺度，不用 log scale
    sorted_lrs = sorted(set(lr_values))
    ax.set_xticks(range(len(sorted_lrs)))
    ax.set_xticklabels([format_lr_tick(lr, None) for lr in sorted_lrs])
    ax.tick_params(axis="x", rotation=45)
    return sorted_lrs


def load_and_process_data(csv_path: Path) -> pd.DataFrame:
    """加载 CSV 并处理数据"""
    df = pd.read_csv(csv_path)
    # 将 lr 字符串转换为浮点数
    df["lr_value"] = df["lr"].apply(lambda x: float(x))
    # 将 radius_scale 转换为浮点数
    df["radius_value"] = df["radius_scale"].apply(lambda x: float(x))
    # 按 radius_value 和 lr_value 排序
    df = df.sort_values(["radius_value", "lr_value"])
    return df


def plot_radius_search():
    """绘制 radius scale 搜索图"""

    if not CSV_PATH.exists():
        print(f"错误: 文件不存在 {CSV_PATH}")
        return

    df = load_and_process_data(CSV_PATH)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 7))

    radius_scales = sorted(df["radius_value"].unique())

    # 获取所有 lr 值并排序，建立 lr -> x 坐标的映射
    all_lrs = sorted(df["lr_value"].unique())
    lr_to_x = {lr: i for i, lr in enumerate(all_lrs)}

    for radius_scale in radius_scales:
        subset = df[df["radius_value"] == radius_scale].sort_values("lr_value")

        lr_values = subset["lr_value"].values
        losses = subset["final_loss"].values

        # 将 lr 转换为均匀的 x 坐标
        x_values = [lr_to_x[lr] for lr in lr_values]

        color = COLORS.get(radius_scale, "#333333")

        # 绘制曲线
        ax.plot(
            x_values,
            losses,
            marker="o",
            markersize=8,
            linewidth=2.5,
            linestyle="-",
            color=color,
            label=f"radius={radius_scale}",
            alpha=0.85,
        )


    # 设置 x 轴刻度（均匀尺度）
    set_lr_xticks(ax, all_lrs)
    ax.set_xlabel("Learning Rate", fontweight="bold", fontsize=14)
    ax.set_ylabel("Final Loss", fontweight="bold", fontsize=14)
    ax.set_title("Radius Scale Search", fontsize=18, fontweight="bold", pad=35)

    # 设置图例（放在中间上方，标题下方）
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(radius_scales),
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor="black",
        fontsize=12,
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

    # 调整布局
    fig.set_constrained_layout(False)
    plt.subplots_adjust(left=0.10, right=0.95, top=0.85, bottom=0.12)

    # 保存图片
    output_file = OUTPUT_DIR / "radius_search.pdf"
    save_figure(fig, str(output_file), formats=["pdf", "png", "eps"])


def main():
    """主函数"""
    print("=" * 60)
    print("Radius Scale 搜索实验绘图工具")
    print("=" * 60)

    plot_radius_search()

    print("\n完成！")


if __name__ == "__main__":
    main()

