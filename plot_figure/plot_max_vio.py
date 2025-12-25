import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.font_manager import FontProperties

# 无显示环境时使用非交互后端，避免 plt.show() 报错
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# 设置科研图表风格
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["lines.linewidth"] = 2

# 读取合并后的数据
data = pd.read_csv('/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/triple.csv')

# 重命名列以匹配后续代码
data = data.rename(columns={
    'steps': 'Step',
    'adamw_value': 'AdamW',
    'muon_value': 'Muon',
    'muonball_value': 'Spectral_Sphere'  # muonball实际上是spectral sphere
})

# 过滤数据：只保留0-24000步的数据
data_filtered = data[(data["Step"] >= 0) & (data["Step"] <= 24000)]

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 定义颜色和线型（厚重配色方案）
optimizer_styles = {
    'AdamW': {'color': '#8B0000', 'linestyle': '-', 'label': 'AdamW'},                      # 深红色，实线
    'Muon': {'color': '#00008B', 'linestyle': '-', 'label': 'Muon'},                        # 深蓝色，实线
    'Spectral_Sphere': {'color': '#006400', 'linestyle': '-', 'label': 'Spectral Sphere'}  # 深绿色，实线
}


def lighten_color(color: str, amount: float = 0.35) -> tuple:
    """
    将颜色向白色混合，让颜色“稍微亮一点”。
    amount ∈ [0, 1]，越大越亮。
    """
    r, g, b = mcolors.to_rgb(color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)


# 绘制三条主曲线：每个数据点加同色空心圆圈标记；叠加长期平滑虚线；再加宽透明颜色带表示波动范围
smooth_window = 5  # 约 11*500=5500 steps 的长期平滑窗口（居中）
band_window = 4  # 波动带窗口（约 4500 steps）
band_q_low = 0.01
band_q_high = 0.99
for optimizer in ['AdamW', 'Muon', 'Spectral_Sphere']:
    style = optimizer_styles[optimizer]
    steps = data_filtered["Step"]
    series = data_filtered[optimizer]

    # 宽透明颜色带：滚动分位数范围（看起来像沿趋势“斜着”铺开的宽带）
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

    # 主曲线 + 空心圆点
    ax.plot(steps, series,
            color=style['color'],
            linestyle=style['linestyle'],
            label=style['label'],
            linewidth=2,
            alpha=0.95,
            marker='o',
            markersize=3.2,
            markerfacecolor='none',
            markeredgecolor=style['color'],
            markeredgewidth=0.9,
            zorder=3)

    # 长期平滑趋势（透明虚线、稍亮、与主曲线颜色一一对应）
    smooth_series = series.rolling(
        window=smooth_window, center=True, min_periods=1
    ).mean()
    ax.plot(
        steps,
        smooth_series,
        color=lighten_color(style["color"], amount=0.40),
        linestyle="--",
        linewidth=2.2,
        alpha=0.45,
        label=None,
        zorder=2,
    )

# 设置坐标轴标签
ax.set_xlabel("Training Steps", fontsize=14, fontweight="bold")
ax.set_ylabel("Max Vio", fontsize=14, fontweight="bold")

# 设置坐标轴范围
ax.set_xlim(0, 24000)
ax.set_ylim(0, 1.1)

# 添加网格
ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)

# 设置图例
legend_font = FontProperties(weight="bold", size=13)
ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=True,
    shadow=True,
    ncol=1,
    prop=legend_font,
    markerscale=1.3,
    handlelength=2.2,
    handletextpad=0.8,
    labelspacing=0.6,
)

# 设置刻度
ax.tick_params(axis="both", which="major", labelsize=11, direction="in", length=6)
ax.tick_params(axis="both", which="minor", direction="in", length=3)

# 启用次要刻度
ax.minorticks_on()

# 调整布局
plt.tight_layout()

# 保存图表（多种格式）
plt.savefig(
    "/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/max_vio_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/max_vio_comparison.pdf",
    bbox_inches="tight",
)

print("Max Violation图表已保存：")
print("  - max_vio_comparison.png (高分辨率PNG)")
print("  - max_vio_comparison.pdf (矢量图PDF)")

# 显示图表
plt.show()
