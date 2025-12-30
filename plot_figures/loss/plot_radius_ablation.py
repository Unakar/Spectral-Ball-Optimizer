import os
import sys

# 支持直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

# 读取radius消融实验数据
data = pd.read_csv(os.path.join(raw_data_dir, "details", "radius_v2.csv"))

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 过滤数据
data_filtered = data[(data["Step"] >= 1000) & (data["Step"] <= 23500)].copy()

# 定义四种radius配置的样式 - 红黄蓝绿
ablation_styles = {
    "radius 0.1": {
        "color": ABLATION_COLORS[2],
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 1.0,
        "label": "Radius 0.1",
        "zorder": 1,
    },
    "radius 0.5": {
        "color": ABLATION_COLORS[1],
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 1.0,
        "label": "Radius 0.5",
        "zorder": 2,
    },
    "radius 1": {
        "color": ABLATION_COLORS[3],
        "linestyle": "-",
        "linewidth": 1.0,
        "alpha": 1.0,
        "label": "Radius 1",
        "zorder": 3,
    },
    "radius 2": {
        "color": ABLATION_COLORS[0],
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 0.8,
        "label": "Radius 2",
        "zorder": 1,
    },
}

# Smoothing参数
smooth_window = 50  # 滚动平均窗口大小

# 绘制顺序：按radius从小到大
configs = ["radius 0.1", "radius 0.5", "radius 1", "radius 2"]

# 存储平滑后的数据用于放大图
smoothed_data = {}

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

    # 存储数据
    smoothed_data[config] = (valid_steps, smoothed_series)

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

# 创建放大图 (inset) - 往右下移动
axins = inset_axes(
    ax,
    width="40%",
    height="40%",
    loc="upper left",
    bbox_to_anchor=(0.5, -0.1, 1, 0.75),
    bbox_transform=ax.transAxes,
)

# 在放大图中绘制曲线
for config in configs:
    style = ablation_styles[config]
    valid_steps, smoothed_series = smoothed_data[config]

    # 只绘制15000-22500范围内的数据
    mask = (valid_steps >= 15000) & (valid_steps <= 20000)
    axins.plot(
        valid_steps[mask],
        smoothed_series[mask],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        zorder=style["zorder"],
    )

# 设置放大图的范围
axins.set_xlim(15000, 20000)
# 自动调整y轴范围
axins.set_ylim(2.58, 2.7)

# 添加放大图的网格，隐藏坐标轴刻度
axins.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.tick_params(length=0)  # 隐藏刻度线

# 标记放大区域
mark_inset(
    ax, axins, loc1=3, loc2=4, fc="none", ec="0.5", linestyle="--", linewidth=0.5, zorder=0
)

# 设置坐标轴标签
ax.set_xlabel("Training Steps", fontweight="bold")
ax.set_ylabel("Train Loss", fontweight="bold")

# 设置坐标轴范围
set_axis_limits(ax, xlim=(1000, 23500))

# 添加网格
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# 设置图例
set_legend_style(ax, loc="upper right")

# 调整布局
fig.set_constrained_layout(False)
plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)

output_file = os.path.join(output_dir, "radius_ablation_loss.pdf")
save_figure(fig, output_file, formats=["pdf", "png", "eps"])
