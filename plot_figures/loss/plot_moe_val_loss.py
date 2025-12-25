import os

import matplotlib.pyplot as plt
import pandas as pd

from ..utils import (
    lighten_color,
    save_figure,
    set_axis_limits,
    set_legend_style,
    setup_plt_style,
)

setup_plt_style()

result_dir = os.path.join(os.path.dirname(__file__), "results")
data = pd.read_csv(os.path.join(result_dir, "moe_lmloss.csv"))

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Filter data: keep only steps 5000-23000
data_filtered = data[(data["Step"] >= 5000) & (data["Step"] <= 23000)]

optimizer_styles = {
    "spectral sphere": {
        "color": "#166534",
        "linestyle": "-",
        "label": "Spectral Sphere",
    },
    "muon": {"color": "#1E3A8A", "linestyle": "-", "label": "Muon"},
    "muon sphere": {
        "color": "#2E8B57",
        "linestyle": "--",
        "label": "Muon Sphere",
    },
    "adamw": {"color": "#C41E3A", "linestyle": "-", "label": "AdamW"},
}


# Plot four curves with volatility bands
band_window = 4
band_q_low = 0.01
band_q_high = 0.99
optimizers = ["adamw", "muon", "muon sphere", "spectral sphere"]
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
        color=lighten_color(style["color"], amount=0.50),
        alpha=0.3,
        linewidth=0,
        zorder=0,
    )

    # Main curve
    ax.plot(
        steps,
        series,
        color=style["color"],
        linestyle=style["linestyle"],
        marker="o",
        label=style["label"],
        alpha=0.95,
        zorder=2,
    )


# Set axis labels
ax.set_xlabel("Training Steps", fontweight="bold")
ax.set_ylabel("Val Loss", fontweight="bold")

# Set title (optional)
# ax.set_title('MOE Validation Loss for Different Optimizers',
#              fontweight='bold')

# Set axis limits
set_axis_limits(ax, xlim=(6000, 24000), ylim=(2.4, 2.8))

# Add grid
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
ax.set_axisbelow(True)

# Set legend
set_legend_style(ax, loc="upper right")


# Adjust layout
fig.set_constrained_layout(False)
plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)

output_file = os.path.join(result_dir, "moe_val_loss_comparison.pdf")
save_figure(fig, output_file, formats=["pdf", "png", "eps"])
