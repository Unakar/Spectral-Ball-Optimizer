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
output_dir = os.path.join(os.path.dirname(__file__), "results", "dense")
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(os.path.join(raw_data_dir, "mup", "standard_baseline.csv"))

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Filter data: keep only steps 8500-23000
data_filtered = data[(data["Step"] >= 7000) & (data["Step"] <= 23000)]

optimizer_styles = {
    "spectral sphere": {
        "linestyle": "-",
        "label": "Spectral Sphere",
    },
    "muon": {"linestyle": "-", "label": "Muon"},
    "muon sphere": {
        "linestyle": "--",
        "label": "Muon Sphere",
    },
    "adamw": {"linestyle": "-", "label": "AdamW"},
}


# Plot four curves with volatility bands
band_window = 2
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
        color=lighten_color(DARK_COLORS[optimizer], amount=0.50),
        alpha=0.1,
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


# ===== Intersection analysis: calculate speedup compared to AdamW =====
# Get AdamW final data point
adamw_final_step = data_filtered["Step"].iloc[-1]
adamw_final_loss = data_filtered["adamw"].iloc[-1]


def find_intersection_step(steps, losses, target_loss):
    """Find the step when curve reaches target_loss via interpolation"""
    for i in range(len(losses) - 1):
        if losses.iloc[i] >= target_loss >= losses.iloc[i + 1]:
            # Linear interpolation to calculate precise step
            ratio = (target_loss - losses.iloc[i]) / (losses.iloc[i + 1] - losses.iloc[i])
            intersect_step = steps.iloc[i] + ratio * (steps.iloc[i + 1] - steps.iloc[i])
            return intersect_step
    return None


# Calculate intersections
spectral_intersect_step = find_intersection_step(
    data_filtered["Step"], data_filtered["spectral sphere"], adamw_final_loss
)
muon_intersect_step = find_intersection_step(
    data_filtered["Step"], data_filtered["muon"], adamw_final_loss
)

print(f"\n===== Efficiency Comparison =====")
print(f"AdamW final position: Step={adamw_final_step}, Loss={adamw_final_loss:.4f}")

# Draw intersection markers and faster annotations
arrow_y = adamw_final_loss

# Mark AdamW endpoint
ax.scatter(
    [adamw_final_step],
    [adamw_final_loss],
    color=DARK_COLORS["adamw"],
    s=100,
    zorder=6,
    edgecolors="white",
    linewidth=2,
    marker="o",
)

# Spectral Sphere intersection - draw arrow
if spectral_intersect_step:
    speedup_ss = adamw_final_step / spectral_intersect_step
    step_reduction_ss = (1 - spectral_intersect_step / adamw_final_step) * 100
    print(
        f"Spectral Sphere intersection: Step={spectral_intersect_step:.0f}, "
        f"speedup={speedup_ss:.2f}×, reduction={step_reduction_ss:.1f}%"
    )

    # Draw arrow from AdamW endpoint to intersection
    ax.annotate(
        "",
        xy=(spectral_intersect_step, arrow_y),
        xytext=(adamw_final_step - 100, arrow_y),
        arrowprops=dict(
            arrowstyle="->",
            color=DARK_COLORS["spectral sphere"],
            lw=2.0,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=4,
    )

    # Draw intersection marker
    ax.scatter(
        [spectral_intersect_step],
        [adamw_final_loss],
        color=DARK_COLORS["spectral sphere"],
        s=100,
        zorder=5,
        edgecolors="white",
        linewidth=2,
        marker="o",
    )

    # Add faster annotation
    mid_x = (muon_intersect_step + spectral_intersect_step) / 2
    ax.annotate(
        f"{speedup_ss:.2f}× faster",
        xy=(mid_x, arrow_y),
        xytext=(mid_x, arrow_y - 0.02),
        fontsize=11,
        fontweight="bold",
        color=DARK_COLORS["spectral sphere"],
        ha="center",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=DARK_COLORS["spectral sphere"],
            alpha=0.9,
        ),
    )

# Muon intersection - draw arrow
if muon_intersect_step:
    speedup_muon = adamw_final_step / muon_intersect_step
    step_reduction_muon = (1 - muon_intersect_step / adamw_final_step) * 100
    print(
        f"Muon intersection: Step={muon_intersect_step:.0f}, "
        f"speedup={speedup_muon:.2f}×, reduction={step_reduction_muon:.1f}%"
    )

    # Draw arrow from AdamW endpoint to intersection
    ax.annotate(
        "",
        xy=(muon_intersect_step, arrow_y),
        xytext=(adamw_final_step, arrow_y),
        arrowprops=dict(
            arrowstyle="->",
            color=DARK_COLORS["muon"],
            lw=2.0,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=4,
    )

    # Draw intersection marker
    ax.scatter(
        [muon_intersect_step],
        [adamw_final_loss],
        color=DARK_COLORS["muon"],
        s=100,
        zorder=5,
        edgecolors="white",
        linewidth=2,
        marker="o",
    )

    # Add faster annotation
    mid_x = (adamw_final_step + muon_intersect_step) / 2
    ax.annotate(
        f"{speedup_muon:.2f}× faster",
        xy=(mid_x, arrow_y),
        xytext=(mid_x, arrow_y + 0.018),
        fontsize=11,
        fontweight="bold",
        color=DARK_COLORS["muon"],
        ha="center",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=DARK_COLORS["muon"],
            alpha=0.8,
        ),
    )

print("=" * 30)

# Set axis labels
ax.set_xlabel("Training Steps", fontweight="bold")
ax.set_ylabel("Val Loss", fontweight="bold")

# Set axis limits
set_axis_limits(ax, xlim=(8000, 24000))
ax.set_xticks([8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000])

# Add grid
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8, zorder=1)
ax.set_axisbelow(True)

# Set legend
set_legend_style(ax, loc="upper right")

# Adjust layout
fig.set_constrained_layout(False)
plt.subplots_adjust(left=0.10, right=0.95, top=0.95, bottom=0.10)

output_file = os.path.join(output_dir, "dense_val_loss_comparison.pdf")
save_figure(fig, output_file, formats=["pdf", "png", "eps"])
