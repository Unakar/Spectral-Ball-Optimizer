"""绘制 MLP/FFN Layer Output Hidden State RMS 随训练步数变化"""

import os
import sys

# 支持直接运行脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pandas as pd

from utils import (
    DARK_COLORS,
    lighten_color,
    save_figure,
    set_axis_limits,
    set_legend_style,
    setup_plt_style,
)

setup_plt_style()

# 数据目录和输出目录
raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data", "hidden_state_track")
output_dir = os.path.join(os.path.dirname(__file__), "results", "hidden_state")
os.makedirs(output_dir, exist_ok=True)

# 读取数据
data = pd.read_csv(os.path.join(raw_data_dir, "hsrms_mlp.csv"))

# 简化列名映射
col_mapping = {
    "spectral sphere": [c for c in data.columns if "spball" in c and "__MIN" not in c and "__MAX" not in c][0],
    "muon": [c for c in data.columns if "muonw" in c and "__MIN" not in c and "__MAX" not in c][0],
    "adamw": [c for c in data.columns if "adamw" in c and "__MIN" not in c and "__MAX" not in c][0],
}

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

optimizer_styles = {
    "spectral sphere": {"linestyle": "-", "label": "Spectral Sphere"},
    "muon": {"linestyle": "-", "label": "Muon"},
    "adamw": {"linestyle": "-", "label": "AdamW"},
}

# 绘制曲线（带波动带）
band_window = 50
band_q_low = 0.05
band_q_high = 0.95
optimizers = ["adamw", "muon", "spectral sphere"]

for optimizer in optimizers:
    style = optimizer_styles[optimizer]
    col_name = col_mapping[optimizer]
    
    # 过滤掉空值
    valid_data = data[["Step", col_name]].dropna()
    steps = valid_data["Step"]
    values = pd.to_numeric(valid_data[col_name], errors='coerce')
    
    # 波动带
    q_low = values.rolling(window=band_window, center=True, min_periods=1).quantile(band_q_low)
    q_high = values.rolling(window=band_window, center=True, min_periods=1).quantile(band_q_high)
    ax.fill_between(
        steps, q_low, q_high,
        color=lighten_color(DARK_COLORS[optimizer], amount=0.50),
        alpha=0.3, linewidth=0, zorder=0,
    )
    
    # 主曲线
    ax.plot(
        steps, values,
        color=DARK_COLORS[optimizer],
        linestyle=style["linestyle"],
        label=style["label"],
        alpha=0.95,
        linewidth=1.5,
        zorder=2,
    )

# 设置坐标轴
ax.set_xlabel("Training Steps", fontweight="bold")
ax.set_ylabel("FFN Hidden State RMS", fontweight="bold")

# 设置 y 轴为 log scale
ax.set_yscale('log')

# 添加网格
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
ax.set_axisbelow(True)

# 设置图例
set_legend_style(ax, loc="upper right")

# 调整布局
fig.set_constrained_layout(False)
plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.12)

output_file = os.path.join(output_dir, "hsrms_mlp.pdf")
save_figure(fig, output_file, formats=["pdf", "png", "eps"])

