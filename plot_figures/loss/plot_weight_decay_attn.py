#!/usr/bin/env python3
"""绘制 Weight Decay 对 Attention Hidden State 的影响

两个子图：
- 左图：Attention Hidden State RMS
- 右图：Attention Hidden State Absmax

三条曲线：Muon without wd, Muon, Spectral Sphere
"""

import os
import sys

# 支持直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import save_figure, set_axis_limits, set_legend_style, setup_plt_style

setup_plt_style()

# 数据目录和输出目录
RAW_DATA_DIR = Path(__file__).parent.parent / "raw_data" / "details" / "weight_decay"
OUTPUT_DIR = Path(__file__).parent / "results" / "weight_decay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RMS_CSV = RAW_DATA_DIR / "wd_attn_rms.csv"
MAX_CSV = RAW_DATA_DIR / "wd_attn_max.csv"

# 配色方案 - 对齐utils.py的DARK_COLORS
# muon: #1e3ac4 (蓝色), spectral sphere: #2e9d18 (绿色)
COLORS = {
    "Muon without wd": "#8b5cf6",  # 紫色
    "Muon": "#1e3ac4",  # 蓝色 (对齐DARK_COLORS)
    "Spectral Sphere": "#2e9d18",  # 绿色 (对齐DARK_COLORS)
}

# 线条样式
LINE_STYLES = {
    "Muon without wd": "-",
    "Muon": "-",
    "Spectral Sphere": "-",
}

# 线宽
LINE_WIDTHS = {
    "Muon without wd": 2.0,
    "Muon": 2.0,
    "Spectral Sphere": 2.0,
}

# Smoothing参数
SMOOTH_WINDOW = 50


def load_and_smooth_data(csv_path: Path, smooth_window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """加载 CSV 并进行平滑处理"""
    df = pd.read_csv(csv_path)
    
    # 对每个优化器的数据进行平滑
    for col in df.columns:
        if col != "Step":
            df[f"{col}_smooth"] = (
                df[col]
                .rolling(window=smooth_window, center=True, min_periods=1)
                .mean()
            )
    
    return df


def plot_single_metric(ax, df: pd.DataFrame, title: str, ylabel: str, show_legend: bool = True):
    """绘制单个指标的图"""
    
    # 只显示Muon相关的曲线
    optimizers = ["Muon without wd", "Muon"]
    
    for optimizer in optimizers:
        steps = df["Step"].values
        values = df[f"{optimizer}_smooth"].values
        
        color = COLORS.get(optimizer, "#333333")
        linestyle = LINE_STYLES.get(optimizer, "-")
        linewidth = LINE_WIDTHS.get(optimizer, 2.0)
        
        ax.plot(
            steps,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=optimizer,
            alpha=0.9,
        )
    
    ax.set_xlabel("Training Steps", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, pad=10, fontsize=16)
    
    if show_legend:
        set_legend_style(ax, loc="upper right")
    
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # 加粗左边和下边轴线
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["bottom"].set_color("black")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_weight_decay_comparison():
    """绘制 Weight Decay 对比图"""
    
    # 创建图形（两个子图）
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 加载数据
    if RMS_CSV.exists():
        df_rms = load_and_smooth_data(RMS_CSV)
        plot_single_metric(
            axes[0], df_rms, 
            "Attention Hidden State RMS", 
            "RMS",
            show_legend=False
        )
    else:
        axes[0].text(
            0.5, 0.5, "wd_attn_rms.csv not found",
            ha="center", va="center", transform=axes[0].transAxes,
        )
    
    if MAX_CSV.exists():
        df_max = load_and_smooth_data(MAX_CSV)
        plot_single_metric(
            axes[1], df_max,
            "Attention Hidden State Absmax",
            "Absmax",
            show_legend=False
        )
    else:
        axes[1].text(
            0.5, 0.5, "wd_attn_max.csv not found",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    
    # 获取第一个子图的 handles 和 labels 用于统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    
    # 在图的底部添加统一的横向图例（大一点）
    legend = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(labels),
        framealpha=0.95,
        edgecolor="black",
        fontsize=14,
    )
    legend.get_frame().set_linewidth(1.0)
    
    # 调整布局
    fig.set_constrained_layout(False)
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.18, wspace=0.2)
    
    # 保存
    output_file = OUTPUT_DIR / "weight_decay_attn_comparison.pdf"
    save_figure(fig, str(output_file), formats=["pdf", "png", "eps"])


def main():
    """主函数"""
    print("=" * 60)
    print("Weight Decay 对 Attention Hidden State 影响绘图工具")
    print("=" * 60)
    
    print("\n生成对比图...")
    plot_weight_decay_comparison()
    
    print("\n完成！")


if __name__ == "__main__":
    main()

