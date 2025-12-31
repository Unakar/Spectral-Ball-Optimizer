#!/usr/bin/env python3
"""绘制 Weight Decay 对 Muon 的影响 - 三图对比

三个子图：
- 左图：Train Loss
- 中图：Attention Hidden State RMS
- 右图：Attention Hidden State Absmax

两条曲线：Muon without wd, Muon
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

LOSS_CSV = RAW_DATA_DIR / "muon_wd_loss.csv"
RMS_CSV = RAW_DATA_DIR / "wd_attn_rms.csv"
MAX_CSV = RAW_DATA_DIR / "wd_attn_max.csv"

# 配色方案 - 对齐utils.py的DARK_COLORS
# muon: #1e3ac4 (蓝色)
COLORS = {
    "Muon without wd": "#8b5cf6",  # 紫色
    "Muon": "#1e3ac4",  # 蓝色 (对齐DARK_COLORS)
}

# 线条样式
LINE_STYLES = {
    "Muon without wd": "-",
    "Muon": "-",
}

# 线宽
LINE_WIDTHS = {
    "Muon without wd": 2.0,
    "Muon": 2.0,
}

# Smoothing参数
SMOOTH_WINDOW = 50

# 起始step
MIN_STEP = 500


def load_and_smooth_data(csv_path: Path, smooth_window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """加载 CSV 并进行平滑处理"""
    df = pd.read_csv(csv_path)
    
    # 过滤step >= MIN_STEP
    df = df[df["Step"] >= MIN_STEP].copy()
    
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
        # 检查列是否存在
        smooth_col = f"{optimizer}_smooth"
        if smooth_col not in df.columns:
            continue
            
        steps = df["Step"].values
        values = df[smooth_col].values
        
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
    
    # 不显示x轴标签
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, pad=10, fontsize=14)
    
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


def plot_weight_decay_triple():
    """绘制 Weight Decay 三图对比"""
    
    # 创建图形（三个子图，稍窄一点）
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    
    # 左图：Loss
    if LOSS_CSV.exists():
        df_loss = load_and_smooth_data(LOSS_CSV)
        plot_single_metric(
            axes[0], df_loss, 
            "Train Loss", 
            "Loss",
            show_legend=False
        )
    else:
        axes[0].text(
            0.5, 0.5, "muon_wd_loss.csv not found",
            ha="center", va="center", transform=axes[0].transAxes,
        )
    
    # 中图：RMS
    if RMS_CSV.exists():
        df_rms = load_and_smooth_data(RMS_CSV)
        plot_single_metric(
            axes[1], df_rms, 
            "Attention Hidden State RMS", 
            "RMS",
            show_legend=False
        )
    else:
        axes[1].text(
            0.5, 0.5, "wd_attn_rms.csv not found",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    
    # 右图：Absmax
    if MAX_CSV.exists():
        df_max = load_and_smooth_data(MAX_CSV)
        plot_single_metric(
            axes[2], df_max,
            "Attention Hidden State Absmax",
            "Absmax",
            show_legend=False
        )
    else:
        axes[2].text(
            0.5, 0.5, "wd_attn_max.csv not found",
            ha="center", va="center", transform=axes[2].transAxes,
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
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.18, wspace=0.22)
    
    # 保存
    output_file = OUTPUT_DIR / "weight_decay_triple_comparison.pdf"
    save_figure(fig, str(output_file), formats=["pdf", "png", "eps"])


def main():
    """主函数"""
    print("=" * 60)
    print("Weight Decay 对 Muon 影响 - 三图对比绘图工具")
    print("=" * 60)
    
    print("\n生成三图对比...")
    plot_weight_decay_triple()
    
    print("\n完成！")


if __name__ == "__main__":
    main()

