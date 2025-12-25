import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.font_manager import FontProperties
from scipy import interpolate

# 无显示环境时使用非交互后端，避免 plt.show() 报错
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# 设置科研图表风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2

# 读取数据
data = pd.read_csv('/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/standard_baseline.csv')

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 过滤数据：只保留6000-23000步的数据
data_filtered = data[(data['Step'] >= 8500) & (data['Step'] <= 23000)]

# 定义颜色和线型（厚重配色方案）
optimizer_styles = {
    'spectral sphere': {'color': '#006400', 'linestyle': '-', 'label': 'Spectral Sphere'},      # 深绿色，实线
    'muon': {'color': '#00008B', 'linestyle': '-', 'label': 'Muon'},                             # 深蓝色，实线
    'muon sphere': {'color': '#2E8B57', 'linestyle': '--', 'label': 'Muon Sphere'},             # 海绿色（厚重），虚线
    'adamw': {'color': '#8B0000', 'linestyle': '-', 'label': 'AdamW'}                            # 深红色，实线
}

def lighten_color(color: str, amount: float = 0.35) -> tuple:
    """
    将颜色向白色混合，让颜色"稍微亮一点"。
    amount ∈ [0, 1]，越大越亮。
    """
    r, g, b = mcolors.to_rgb(color)
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)

# 绘制四条曲线：每个数据点加同色空心圆圈标记；叠加长期平滑虚线；再加宽透明颜色带表示波动范围
smooth_window = 5  # 长期平滑窗口（居中）
band_window = 4    # 波动带窗口
band_q_low = 0.01
band_q_high = 0.99
optimizers = ['adamw', 'muon', 'muon sphere', 'spectral sphere']
for optimizer in optimizers:
    style = optimizer_styles[optimizer]
    steps = data_filtered['Step']
    series = data_filtered[optimizer]

    # 宽透明颜色带：滚动分位数范围
    q_low = series.rolling(window=band_window, center=True, min_periods=1).quantile(band_q_low)
    q_high = series.rolling(window=band_window, center=True, min_periods=1).quantile(band_q_high)
    ax.fill_between(
        steps,
        q_low,
        q_high,
        color=lighten_color(style['color'], amount=0.50),
        alpha=0.14,
        linewidth=0,
        zorder=0
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
    smooth_series = (
        series
        .rolling(window=smooth_window, center=True, min_periods=1)
        .mean()
    )
    ax.plot(
        steps,
        smooth_series,
        color=lighten_color(style['color'], amount=0.40),
        linestyle='--',
        linewidth=2.2,
        alpha=0.45,
        label=None,
        zorder=2
    )

# ===== 新增功能：计算并绘制AdamW水平线与其他优化器曲线的交点 =====
# 获取AdamW最后一个数据点
adamw_final_step = data_filtered['Step'].iloc[-1]
adamw_final_loss = data_filtered['adamw'].iloc[-1]

# 计算与spectral sphere和muon曲线的交点
def find_intersection_step(steps, losses, target_loss):
    """通过插值找到曲线达到target_loss时对应的step"""
    # 创建插值函数（loss -> step），需要确保loss是单调递减的
    # 找到第一个loss小于target_loss的点
    for i in range(len(losses) - 1):
        if losses.iloc[i] >= target_loss >= losses.iloc[i+1]:
            # 线性插值计算精确的step
            ratio = (target_loss - losses.iloc[i]) / (losses.iloc[i+1] - losses.iloc[i])
            intersect_step = steps.iloc[i] + ratio * (steps.iloc[i+1] - steps.iloc[i])
            return intersect_step
    return None

# 计算交点
spectral_intersect_step = find_intersection_step(
    data_filtered['Step'], data_filtered['spectral sphere'], adamw_final_loss)
muon_intersect_step = find_intersection_step(
    data_filtered['Step'], data_filtered['muon'], adamw_final_loss)

print(f"\n===== 效率对比分析 =====")
print(f"AdamW 最终位置: Step={adamw_final_step}, Loss={adamw_final_loss:.4f}")

# 绘制交点标记和faster标注
# 两条箭头在同一水平线上
arrow_y = adamw_final_loss

# 标记AdamW终点
ax.scatter([adamw_final_step], [adamw_final_loss], color='#8B0000', s=100, zorder=6,
           edgecolors='white', linewidth=2, marker='s')

# Spectral Sphere交点 - 画实线箭头
if spectral_intersect_step:
    speedup_ss = adamw_final_step / spectral_intersect_step
    step_reduction_ss = (1 - spectral_intersect_step / adamw_final_step) * 100
    print(f"Spectral Sphere 交点: Step={spectral_intersect_step:.0f}, 加速比={speedup_ss:.2f}×, 节省={step_reduction_ss:.1f}%")
    
    # 画从AdamW终点指向交点的实线箭头
    ax.annotate('', 
                xy=(spectral_intersect_step, arrow_y),  # 箭头指向的位置
                xytext=(adamw_final_step - 100, arrow_y),  # 箭头起始位置
                arrowprops=dict(arrowstyle='->', color='#006400', lw=2.0, 
                               shrinkA=0, shrinkB=0),
                zorder=4)
    
    # 画交点标记
    ax.scatter([spectral_intersect_step], [adamw_final_loss], color='#006400', s=100, zorder=5, 
               edgecolors='white', linewidth=2, marker='o')
    
    # 添加faster标注（在箭头下方）
    mid_x = (adamw_final_step + spectral_intersect_step) / 2
    ax.annotate(f'{speedup_ss:.2f}× faster', 
                xy=(mid_x, arrow_y),
                xytext=(mid_x - 3000, arrow_y - 0.02),
                fontsize=11, fontweight='bold', color='#006400',
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#006400', alpha=0.95))

# Muon交点 - 画实线箭头
if muon_intersect_step:
    speedup_muon = adamw_final_step / muon_intersect_step
    step_reduction_muon = (1 - muon_intersect_step / adamw_final_step) * 100
    print(f"Muon 交点: Step={muon_intersect_step:.0f}, 加速比={speedup_muon:.2f}×, 节省={step_reduction_muon:.1f}%")
    
    # 画从AdamW终点指向交点的实线箭头
    ax.annotate('', 
                xy=(muon_intersect_step, arrow_y),  # 箭头指向的位置
                xytext=(adamw_final_step - 100, arrow_y),  # 箭头起始位置
                arrowprops=dict(arrowstyle='->', color='#00008B', lw=2.0,
                               shrinkA=0, shrinkB=0),
                zorder=4)
    
    # 画交点标记
    ax.scatter([muon_intersect_step], [adamw_final_loss], color='#00008B', s=100, zorder=5,
               edgecolors='white', linewidth=2, marker='o')
    
    # 添加faster标注（在箭头上方）
    mid_x = (adamw_final_step + muon_intersect_step) / 2
    ax.annotate(f'{speedup_muon:.2f}× faster', 
                xy=(mid_x, arrow_y),
                xytext=(mid_x - 500, arrow_y + 0.02),
                fontsize=11, fontweight='bold', color='#00008B',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#00008B', alpha=0.95))

print("=" * 30)

# 设置坐标轴标签
ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Val Loss', fontsize=14, fontweight='bold')

# 设置标题（可选）
# ax.set_title('Validation Loss vs Training Steps for Different Optimizers',
#              fontsize=16, fontweight='bold', pad=20)

# 设置坐标轴范围（右侧留白）
ax.set_xlim(8000, 24000)


# 添加网格
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)

# 设置图例
legend_font = FontProperties(weight='bold', size=13)
ax.legend(
    loc='upper right',
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
ax.tick_params(axis='both', which='major', labelsize=11, direction='in', length=6)
ax.tick_params(axis='both', which='minor', direction='in', length=3)

# 启用次要刻度
ax.minorticks_on()

# 调整布局
plt.tight_layout()

# 保存图表（多种格式）
plt.savefig('/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/dense_val_loss_comparison.png',
            dpi=300, bbox_inches='tight')
plt.savefig('/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/dense_val_loss_comparison.pdf',
            bbox_inches='tight')

print("图表已保存：")
print("  - dense_val_loss_comparison.png (高分辨率PNG)")
print("  - dense_val_loss_comparison.pdf (矢量图PDF)")

# 显示图表
plt.show()
