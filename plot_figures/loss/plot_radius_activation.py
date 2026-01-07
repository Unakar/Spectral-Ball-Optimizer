#!/usr/bin/env python3
"""绘制不同 Radius Scale 下 MLP Activation 的 AbsMax 和 RMS 图

横坐标: Step
纵坐标: AbsMax 或 RMS
每个 radius scale 连成一条曲线
"""

import os
import sys

# 支持直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from utils import save_figure, setup_plt_style

setup_plt_style()

# 数据目录和输出目录
RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data"
OUTPUT_DIR = Path(__file__).parent / "results" / "radius_activation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ABSMAX_CSV = RAW_DATA_DIR / "radius" / "radius_mlp_absmax.csv"
RMS_CSV = RAW_DATA_DIR / "radius" / "radius_mlp_rms.csv"

# 配色方案 - 不同 radius scale 使用不同颜色
COLORS = {
    "radius0.1": "#E63946",   # 红色
    "radius0.5": "#F77F00",   # 橙色
    "radius1": "#2A9D8F",     # 青色
    "radius2": "#264653",     # 深蓝灰色
    "radius10": "#7209B7",    # 紫色
}

# 图例标签映射
LABELS = {
    "radius0.1": "radius=0.1",
    "radius0.5": "radius=0.5",
    "radius1": "radius=1",
    "radius2": "radius=2",
    "radius10": "radius=10",
}


def plot_activation(
    csv_path: Path,
    title: str,
    ylabel: str,
    output_name: str,
    exclude_radius: list = None,
    log_scale_y: bool = False,
    min_step: int = None,
    linewidth: float = 1.0,
    legend_linewidth: float = None,
    skip_step_interval: int = None,
):
    """绘制 activation 图

    Args:
        csv_path: CSV 文件路径
        title: 图标题
        ylabel: y轴标签
        output_name: 输出文件名
        exclude_radius: 要排除的 radius 列表
        log_scale_y: 是否使用 y轴 log scale
        min_step: 最小 step（从这个 step 开始画）
        linewidth: 曲线宽度
        legend_linewidth: 图例中线条宽度（默认与曲线相同）
        skip_step_interval: 跳过的 step 间隔（如 500 表示跳过 500, 1000, 1500...）
    """

    if not csv_path.exists():
        print(f"错误: 文件不存在 {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 过滤 step
    if min_step is not None:
        df = df[df["Step"] >= min_step]

    # 跳过特定间隔的 step（如 500 表示跳过 501, 1001, 1501...）
    if skip_step_interval is not None:
        df = df[df["Step"] % skip_step_interval != 1]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = df["Step"].values

    # 按照 radius 值排序绘制
    radius_order = ["radius0.1", "radius0.5", "radius1", "radius2", "radius10"]

    # 排除指定的 radius
    if exclude_radius:
        radius_order = [r for r in radius_order if r not in exclude_radius]

    for radius_col in radius_order:
        if radius_col not in df.columns:
            continue

        values = df[radius_col].values
        color = COLORS.get(radius_col, "#333333")
        label = LABELS.get(radius_col, radius_col)

        # 绘制曲线
        ax.plot(
            steps,
            values,
            linewidth=linewidth,
            color=color,
            label=label,
            alpha=0.85,
        )

    # 设置 x 轴范围，消除空白
    ax.set_xlim(steps.min(), steps.max())

    # 设置 y轴 log scale
    if log_scale_y:
        ax.set_yscale("log")

    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # 设置图例（放在中间上方）
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(radius_order),
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor="black",
        fontsize=11,
    )
    legend.get_frame().set_linewidth(1.0)

    # 单独设置图例中线条的粗细
    if legend_linewidth is not None:
        for line in legend.get_lines():
            line.set_linewidth(legend_linewidth)

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
    plt.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.12)

    # 保存图片
    output_file = OUTPUT_DIR / f"{output_name}.pdf"
    save_figure(fig, str(output_file), formats=["pdf", "png", "eps"])


def main():
    """主函数"""
    print("=" * 60)
    print("Radius Scale MLP Activation 绘图工具")
    print("=" * 60)

    print("\n生成 AbsMax 图...")
    plot_activation(
        ABSMAX_CSV,
        title="MLP Activation AbsMax",
        ylabel="AbsMax",
        output_name="radius_mlp_absmax",
        log_scale_y=True,
        min_step=2000,
        linewidth=0.4,
        legend_linewidth=2.0,
    )

    print("\n生成 RMS 图...")
    plot_activation(
        RMS_CSV,
        title="MLP Activation RMS",
        ylabel="RMS",
        output_name="radius_mlp_rms",
        log_scale_y=True,
        linewidth=1.5,
        skip_step_interval=500,
    )

    print("\n完成！")


if __name__ == "__main__":
    main()

