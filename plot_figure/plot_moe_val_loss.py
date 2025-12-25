import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
from utils import (
    lighten_color,
    save_figure,
    set_axis_limits,
    setup_publication_style,
)

setup_publication_style()

data = pd.read_csv("results/moe_lmloss.csv")

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 过滤数据：只保留6000-23000步的数据
data_filtered = data[(data["Step"] >= 5000) & (data["Step"] <= 23000)]

optimizer_styles = {
    "spectral sphere": {
        "color": "#166534",
        "linestyle": "-",
        "label": "Spectral Sphere",
    },  # 绿色，实线
    "muon": {"color": "#1E3A8A", "linestyle": "-", "label": "Muon"},  # 蓝色，实线
    "muon sphere": {
        "color": "#2E8B57",
        "linestyle": "--",
        "label": "Muon Sphere",
    },  # 海绿色，虚线
    "adamw": {"color": "#C41E3A", "linestyle": "-", "label": "AdamW"},  # 红色，实线
}


# 绘制四条曲线：每个数据点加同色圆圈标记；叠加长期平滑虚线；再加宽透明颜色带表示波动范围
band_window = 4  # 波动带窗口
band_q_low = 0.01
band_q_high = 0.99
optimizers = ["adamw", "muon", "muon sphere", "spectral sphere"]
for optimizer in optimizers:
    style = optimizer_styles[optimizer]
    steps = data_filtered["Step"]
    series = data_filtered[optimizer]

    # 宽透明颜色带：滚动分位数范围
    q_low = series.rolling(window=band_window, center=True, min_periods=1).quantile(
        band_q_low
    )
    q_high = series.rolling(window=band_window, center=True, min_periods=1).quantile(
        band_q_high
    )
    ax.fill_between(
        steps,
        q_low,
        q_high,
        color=lighten_color(style["color"], amount=0.50),
        alpha=0.14,
        linewidth=0,
        zorder=0,
    )

    # 主曲线
    ax.plot(
        steps,
        series,
        color=style["color"],
        linestyle=style["linestyle"],
        marker="o",
        label=style["label"],
        linewidth=1.8,
        alpha=0.85,
        zorder=2,
    )


# 设置坐标轴标签
ax.set_xlabel("Training Steps", fontsize=13, fontweight="bold")
ax.set_ylabel("Val Loss", fontsize=13, fontweight="bold")

# 设置标题（可选）
ax.set_title('MOE Validation Loss for Different Optimizers',
             fontsize=16, fontweight='bold')

# 设置坐标轴范围
set_axis_limits(ax, xlim=(6000, 24000), ylim=(2.4, 2.8))

# 添加网格
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
ax.set_axisbelow(True)

# 设置图例
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

output_file = "results/moe_val_loss_comparison.pdf"
save_figure(fig, output_file, formats=["pdf", "png", "eps"])
