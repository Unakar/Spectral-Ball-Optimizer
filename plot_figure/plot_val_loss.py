import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置科研图表风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2

# 读取数据
data = pd.read_csv('/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/moe_lmloss.csv')

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 过滤数据：只保留6000-23000步的数据
data_filtered = data[(data['Step'] >= 6000) & (data['Step'] <= 23000)]

# 定义颜色和线型（厚重配色方案）
optimizer_styles = {
    'spectral sphere': {'color': '#006400', 'linestyle': '-', 'label': 'Spectral Sphere'},      # 深绿色，实线
    'muon': {'color': '#00008B', 'linestyle': '-', 'label': 'Muon'},                             # 深橙色，实线
    'muon sphere': {'color': '#2E8B57', 'linestyle': '--', 'label': 'Muon Sphere'},             # 海绿色（厚重），虚线
    'adamw': {'color': '#8B0000', 'linestyle': '-', 'label': 'AdamW'}                            # 道奇蓝，实线
}
#深蓝色是#00008B
# 绘制四条曲线（不使用数据点标记）
optimizers = ['adamw', 'muon', 'muon sphere', 'spectral sphere']
for optimizer in optimizers:
    style = optimizer_styles[optimizer]
    ax.plot(data_filtered['Step'], data_filtered[optimizer],
            color=style['color'],
            linestyle=style['linestyle'],
            label=style['label'],
            linewidth=3,
            alpha=0.95)

# 设置坐标轴标签
ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax.set_ylabel('Val Loss', fontsize=14, fontweight='bold')

# 设置标题（可选）
# ax.set_title('Validation Loss vs Training Steps for Different Optimizers',
#              fontsize=16, fontweight='bold', pad=20)

# 设置坐标轴范围（右侧留白）
ax.set_xlim(6000, 24000)
#ax.set_ylim(top=2.7)

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
plt.savefig('/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/moe_val_loss_comparison.png',
            dpi=300, bbox_inches='tight')
plt.savefig('/home/t2vg-a100-G2-1/a_xietian/dev/numeric/run_data/moe_val_loss_comparison.pdf',
            bbox_inches='tight')

print("图表已保存：")
print("  - moe_val_loss_comparison.png (高分辨率PNG)")
print("  - moe_val_loss_comparison.pdf (矢量图PDF)")

# 显示图表
plt.show()
