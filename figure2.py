import numpy as np
import matplotlib.pyplot as plt
import os

# ======================== 全局字体设置 ========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# ======================== 用户可调参数 ========================
STRAIN_MIN = 0
STRAIN_MAX = 0.1
SIGMA_MIN = 0
SIGMA_MAX = 1600

XTICKS_STEP = 0.02
YTICKS_STEP = 400
XTICKS_ROTATION = 0

LINE_WIDTH = 2
AXIS_LABEL_FONTSIZE = 28
AXIS_TICK_FONTSIZE = 24
TITLE_FONTSIZE = 22

MARKER_SIZE = 16
MARKER_EDGEWIDTH = 0.5

LEGEND_FONTSIZE = 24
LEGEND_LABELSPACING = 1.2
LEGEND_HANDLETEXTPAD = 0.8
LEGEND_LOC = 'upper left'

# 标记形状和颜色序列（用于两个拟合数据）
MARKERS = ['o', 's']          # 两个拟合数据使用不同标记
COLORS = ['b', 'g']           # 蓝色、绿色，与实验曲线区分

OUTPUT_FILENAME = ' '
# ================================================================

# ------------------------ 数据加载 ------------------------
# 室温实验数据 (0)
try:
    rt_exp_eps = np.loadtxt('0_epsilon.txt')
    rt_exp_sig = np.loadtxt('0__sigma_MPa.txt')
    print("室温实验数据加载成功。")
except FileNotFoundError as e:
    print(f"错误：缺少室温实验数据文件。{e}")
    exit(1)

# 高温实验数据 (00)
try:
    ht_exp_eps = np.loadtxt('00_epsilon.txt')
    ht_exp_sig = np.loadtxt('00__sigma_MPa.txt')
    print("高温实验数据加载成功。")
except FileNotFoundError as e:
    print(f"错误：缺少高温实验数据文件。{e}")
    exit(1)

# 拟合数据：1_epsilon.txt (高温拟合) 和 11_epsilon.txt (室温拟合)
fit_files = [
    ('100', 'Fitted Room Temperature'),
    ('130', 'Fitted High Temperature')
]

fit_epsilons = []
fit_sigmas = []
fit_labels = []

for num, label in fit_files:
    eps_file = f"{num}_epsilon.txt"
    sig_file = f"{num}__sigma_MPa.txt"
    if os.path.exists(eps_file) and os.path.exists(sig_file):
        eps = np.loadtxt(eps_file)
        sig = np.loadtxt(sig_file)
        fit_epsilons.append(eps)
        fit_sigmas.append(sig)
        fit_labels.append(label)
        print(f"拟合数据 {eps_file} 加载成功。")
    else:
        print(f"警告：拟合数据文件 {eps_file} 或 {sig_file} 不存在，已跳过。")

# ------------------------ 绘图 ------------------------
plt.figure(figsize=(20, 12))

# 室温实验曲线 (黑色实线)
plt.plot(rt_exp_eps, rt_exp_sig, 'k-', linewidth=LINE_WIDTH + 1, label='Room Temperature (Exp.)')

# 高温实验曲线 (红色虚线)
plt.plot(ht_exp_eps, ht_exp_sig, 'r--', linewidth=LINE_WIDTH + 1, label='High Temperature (Exp.)')

# 拟合数据点（用点和不同标记/颜色）
for i, (eps, sig, label) in enumerate(zip(fit_epsilons, fit_sigmas, fit_labels)):
    marker = MARKERS[i % len(MARKERS)]
    color = COLORS[i % len(COLORS)]
    plt.plot(eps, sig,
             linestyle='',          # 只显示点
             marker=marker,
             markersize=MARKER_SIZE,
             markeredgewidth=MARKER_EDGEWIDTH,
             color=color,
             label=label)

# 坐标轴范围
plt.xlim(STRAIN_MIN, STRAIN_MAX)
plt.ylim(SIGMA_MIN, SIGMA_MAX)

# 刻度
xticks = np.arange(STRAIN_MIN, STRAIN_MAX + XTICKS_STEP, XTICKS_STEP)
yticks = np.arange(SIGMA_MIN, SIGMA_MAX + YTICKS_STEP, YTICKS_STEP)
plt.xticks(xticks, rotation=XTICKS_ROTATION, fontsize=AXIS_TICK_FONTSIZE)
plt.yticks(yticks, fontsize=AXIS_TICK_FONTSIZE)

# 标签
plt.xlabel('Strain', fontsize=AXIS_LABEL_FONTSIZE)
plt.ylabel('Stress (MPa)', fontsize=AXIS_LABEL_FONTSIZE)
# plt.title('Stress-Strain Curve', fontsize=TITLE_FONTSIZE, pad=15)   # 如需标题可取消注释

# 图例（横向，列数根据实际条目数调整：2条实验 + 2条拟合 = 4）
plt.legend(
    prop={'size': LEGEND_FONTSIZE},
    labelspacing=LEGEND_LABELSPACING,
    handletextpad=LEGEND_HANDLETEXTPAD,
    loc=LEGEND_LOC,
    ncol=2                     # 4条数据水平排列
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FILENAME, dpi=300, bbox_inches='tight')
plt.show()
