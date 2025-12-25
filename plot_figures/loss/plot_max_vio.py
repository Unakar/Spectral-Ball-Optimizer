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
raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data")
output_dir = os.path.join(os.path.dirname(__file__), "results", "maxvio")
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(os.path.join(raw_data_dir, "triple.csv"))

# Rename columns to match DARK_COLORS keys
data = data.rename(
    columns={
        "steps": "Step",
        "adamw_value": "adamw",
        "muon_value": "muon",
        "muonball_value": "spectral sphere",  # muonball is actually spectral sphere
    }
)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Filter data: keep only steps 0-24000
data_filtered = data[(data["Step"] >= 0) & (data["Step"] <= 24000)]

optimizer_styles = {
    "spectral sphere": {
        "linestyle": "-",
        "label": "Spectral Sphere",
    },
    "muon": {"linestyle": "-", "label": "Muon"},
    "adamw": {"linestyle": "-", "label": "AdamW"},
}


# Plot three curves with volatility bands
band_window = 4
band_q_low = 0.01
band_q_high = 0.99
optimizers = ["adamw", "muon", "spectral sphere"]
for optimizer in optimizers:
    style = optimizer_styles[optimizer]
    steps = data_filtered["Step"]
    series = data_filtered[optimizer]

    # Wide transparent color band: rolling quantile range
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
        color=lighten_color(DARK_COLORS[optimizer], amount=0.50),
        alpha=0.3,
        linewidth=0,
        zorder=0,
    )

    # Main curve
    ax.plot(
        steps,
        series,
        color=DARK_COLORS[optimizer],
        linestyle=style["linestyle"],
        marker="o",
        label=style["label"],
        alpha=0.95,
        zorder=2,
    )


# Set axis labels
ax.set_xlabel("Training Steps", fontweight="bold")
ax.set_ylabel("Max Vio", fontweight="bold")

# Set axis limits
set_axis_limits(ax, xlim=(0, 24000), ylim=(0, 1.1))

# Add grid
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
ax.set_axisbelow(True)

# Set legend
set_legend_style(ax, loc="upper right")

# Adjust layout
fig.set_constrained_layout(False)
plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)

output_file = os.path.join(output_dir, "max_vio_comparison.pdf")
save_figure(fig, output_file, formats=["pdf", "png", "eps"])
