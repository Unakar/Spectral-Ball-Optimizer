import os
import sys

# 支持直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import (
    ABLATION_COLORS,
    save_figure,
    set_axis_limits,
    set_legend_style,
    setup_plt_style,
)

setup_plt_style()

# 数据目录和输出目录
raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data")
output_dir = os.path.join(os.path.dirname(__file__), "results", "ablation")
os.makedirs(output_dir, exist_ok=True)

# 读取消融实验数据
data = pd.read_csv(os.path.join(raw_data_dir, "details", "module.csv"))

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 过滤数据：只保留step在5000到13500之间
data_filtered = data[(data["Step"] >= 5000) & (data["Step"] <= 13500)].copy()

# 定义三种配置的样式 - 全用实线，使用橙/紫/青三色
# all split是baseline，用较粗的线和较低透明度
ablation_styles = {
    "all split": {
        "color": ABLATION_COLORS[0],
        "linestyle": "-",
        "linewidth": 7,
        "alpha": 0.8,
        "label": "All Split (Standard)",
        "zorder": 1,
    },
    "split ffn only": {
        "color": ABLATION_COLORS[1],
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 1,
        "label": "Split FFN Only",
        "zorder": 2,
    },
    "split qkv only": {
        "color": ABLATION_COLORS[2],
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 1,
        "label": "Split QKV Only",
        "zorder": 3,
    },
}

# Smoothing参数
smooth_window = 50  # 滚动平均窗口大小

# 先画baseline（底层），再画其他线（上层）
configs = ["all split", "split qkv only", "split ffn only"]

for config in configs:
    style = ablation_styles[config]
    steps = data_filtered["Step"]
    series = data_filtered[config]

    # 跳过NaN值，创建有效数据的副本
    valid_mask = ~series.isna()
    valid_steps = steps[valid_mask].values
    valid_series = series[valid_mask].values

    # 对数据进行smoothing（滚动平均）
    smoothed_series = (
        pd.Series(valid_series)
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
        .values
    )

    # 主曲线
    ax.plot(
        valid_steps,
        smoothed_series,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        label=style["label"],
        alpha=style["alpha"],
        zorder=style["zorder"],
    )

# 设置坐标轴标签
ax.set_xlabel("Training Steps", fontweight="bold")
ax.set_ylabel("Train Loss", fontweight="bold")

# 设置坐标轴范围
set_axis_limits(ax, xlim=(5000, 13500))

# 添加网格
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# 设置图例
set_legend_style(ax, loc="upper right")

# 调整布局
fig.set_constrained_layout(False)
plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)

output_file = os.path.join(output_dir, "module_ablation_loss.pdf")
save_figure(fig, output_file, formats=["pdf", "png", "eps"])
