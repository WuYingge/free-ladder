"""生成波动率分位×因子切换的结论图片"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/root/.openclaw/workspace/opengouzi/output")

# ── 字体 ──
from matplotlib.font_manager import FontProperties
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp = FontProperties(fname=font_path, size=11)
fp_bold = FontProperties(fname=font_path, size=12, weight="bold")
fp_title = FontProperties(fname=font_path, size=16, weight="bold")
fp_small = FontProperties(fname=font_path, size=9)
fp_big = FontProperties(fname=font_path, size=14, weight="bold")

# ── 暗色主题 ──
BG = "#1a1a2e"
FG = "#e0e0e0"
HL = "#e94560"
ACCENT_GREEN = "#2ecc71"
ACCENT_YELLOW = "#f1c40f"
ACCENT_BLUE = "#3498db"
CARD_BG = "#16213e"
GRID_COLOR = "#2a2a4a"
RED_ALPHA = "#e74c3c"
MOM_GREEN = "#27ae60"
C2_ORANGE = "#e67e22"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD_BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG,
    "ytick.color": FG, "axes.edgecolor": GRID_COLOR,
    "grid.color": GRID_COLOR, "grid.alpha": 0.5,
})

# ============================================================
# 数据
# ============================================================

# 二维交叉表: fwd=10d, mom_IC / c2_IC
matrix_mom = np.array([
    [+0.0509, +0.1121, +0.1055],  # 低波: 下降, 平稳, 上升
    [+0.0058, -0.0998, -0.0986],  # 正常波
    [-0.2537, -0.2356, -0.1866],  # 高波
    [+0.1373, +0.4266, -0.2218],  # 极高波
])
matrix_c2 = np.array([
    [+0.0882, +0.0268, -0.0320],
    [+0.2770, +0.1387, +0.1007],
    [+0.2918, +0.1667, +0.1301],
    [+0.0982, -0.3190, +0.0932],
])

# 场景数据 (fwd=10d)
scenarios = [
    ("vol开始放大\n(低波+上升)", +0.1055, -0.0320, 56895, 0.015783, -0.001653),
    ("vol持续高位\n(高波+上升)", -0.1866, +0.1301, 15929, -0.028145, +0.010939),
    ("vol到达极致\n(极高波)", +0.1359, -0.1117, 19388, +0.015416, -0.013897),
    ("vol开始收敛\n(极高波+下降)", +0.1373, +0.0982, 2387, +0.019269, +0.009869),
    ("正常低波\n(低波+平稳)", +0.1121, +0.0268, 203383, +0.010474, +0.010552),
]

# 三状态机
states = [
    ("趋势形成", "低波 + vol上升", "动量 ✅ (+0.11)", MOM_GREEN, "vol刚从安静中抬头"),
    ("高波震荡", "高波 + vol在加速", "C2 ✅ (+0.13)", C2_ORANGE, "趋势策略绞肉机"),
    ("极端平台", "极高波 + vol见顶", "动量 ✅✅ (+0.43)", MOM_GREEN, "持续单向行情"),
]

# ============================================================
# 画图
# ============================================================
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(BG)

# ── 总标题 ──
fig.suptitle("波动率分位 x 因子切换 — 分桶验证结论", fontproperties=fp_title,
             color=FG, y=0.98, fontsize=18)

# ────────────────────────────────────────
# 左侧: 二维交叉热力图 (占2/3宽度)
# ────────────────────────────────────────
ax1 = fig.add_axes([0.04, 0.08, 0.52, 0.82])
ax1.set_facecolor(CARD_BG)

row_labels = ["低波\n(z<0)", "正常波\n(0≤z<1)", "高波\n(1≤z<2.5)", "极高波\n(z≥2.5)"]
col_labels = ["vol下降\n(tz<-0.5)", "vol平稳\n(-0.5~0.5)", "vol上升\n(tz≥0.5)"]

n_rows, n_cols = 4, 3
cell_w, cell_h = 1.0, 1.0

for i in range(n_rows):
    for j in range(n_cols):
        x, y = j * cell_w, (n_rows - 1 - i) * cell_h
        
        # 判断主导因子
        mom_v = matrix_mom[i, j]
        c2_v = matrix_c2[i, j]
        
        # 背景色: mom正=momentum绿, c2正=c2橙, 都负=灰色
        if mom_v > 0 and mom_v > c2_v:
            cell_bg = "#1a3a2a"
            border_color = MOM_GREEN
        elif c2_v > 0 and c2_v > mom_v:
            cell_bg = "#3a2a1a"
            border_color = C2_ORANGE
        elif mom_v > 0 and c2_v > 0:
            cell_bg = "#2a2a3a"
            border_color = ACCENT_YELLOW
        else:
            cell_bg = "#2a1a1a"
            border_color = RED_ALPHA
        
        rect = mpatches.FancyBboxPatch(
            (x + 0.05, y + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.05", facecolor=cell_bg,
            edgecolor=border_color, linewidth=2
        )
        ax1.add_patch(rect)
        
        # 因子IC值
        mom_color = MOM_GREEN if mom_v > 0 else RED_ALPHA
        c2_color = C2_ORANGE if c2_v > 0 else RED_ALPHA
        
        ax1.text(x + 0.5, y + 0.72, f"动量 {mom_v:+.4f}", ha="center", va="center",
                fontproperties=fp_small, color=mom_color, fontsize=10, fontweight="bold")
        ax1.text(x + 0.5, y + 0.40, f"C2   {c2_v:+.4f}", ha="center", va="center",
                fontproperties=fp_small, color=c2_color, fontsize=10, fontweight="bold")
        
        # 决定胜负标记
        if abs(mom_v - c2_v) > 0.02:
            winner = "动量胜" if mom_v > c2_v else "C2胜"
            w_color = MOM_GREEN if mom_v > c2_v else C2_ORANGE
            ax1.text(x + 0.5, y + 0.12, winner, ha="center", va="center",
                    fontproperties=fp_small, color=w_color, fontsize=8)

# 行标签
for i, label in enumerate(row_labels):
    ax1.text(-0.25, (n_rows - 1 - i) * cell_h + 0.5, label, ha="right", va="center",
            fontproperties=fp_bold, color=FG, fontsize=10)
# 列标签
for j, label in enumerate(col_labels):
    ax1.text(j * cell_w + 0.5, n_rows * cell_h + 0.3, label, ha="center", va="bottom",
            fontproperties=fp_bold, color=FG, fontsize=10)

ax1.text(1.5, n_rows * cell_h + 0.7, "波动率趋势 →", ha="center", va="bottom",
        fontproperties=fp_small, color="#888", fontsize=9)
ax1.text(-1.0, 2.0, "← 波动率水平", ha="center", va="center",
        fontproperties=fp_small, color="#888", fontsize=10, rotation=90)

ax1.set_xlim(-0.5, n_cols + 0.3)
ax1.set_ylim(-0.3, n_rows + 0.8)
ax1.set_aspect("equal")
ax1.axis("off")

# 图例
legend_y = -0.25
ax1.add_patch(mpatches.Rectangle((0,0),1,1, facecolor="#1a3a2a", edgecolor=MOM_GREEN, linewidth=1.5,
                              label="动量主导"))
ax1.add_patch(mpatches.Rectangle((0,0),1,1, facecolor="#3a2a1a", edgecolor=C2_ORANGE, linewidth=1.5,
                              label="C2主导"))
ax1.legend(loc="upper left", bbox_to_anchor=(0, legend_y - 0.05),
          fontsize=9, framealpha=0.8, facecolor=CARD_BG, edgecolor=GRID_COLOR,
          labelcolor=FG, prop=fp_small)

# ────────────────────────────────────────
# 右侧上半: 关键场景对比
# ────────────────────────────────────────
ax2 = fig.add_axes([0.58, 0.52, 0.40, 0.38])
ax2.set_facecolor(CARD_BG)

sc_labels = [s[0].replace("\n", " ") for s in scenarios]
sc_mom = [s[1] for s in scenarios]
sc_c2 = [s[2] for s in scenarios]
sc_n = [s[3] for s in scenarios]

y_pos = range(len(sc_labels))
bar_h = 0.35

bars_mom = ax2.barh([y + bar_h/2 for y in y_pos], sc_mom, bar_h,
                     color=MOM_GREEN, alpha=0.7, label="momentum IC", edgecolor=MOM_GREEN, linewidth=0.5)
bars_c2 = ax2.barh([y - bar_h/2 for y in y_pos], sc_c2, bar_h,
                    color=C2_ORANGE, alpha=0.7, label="C2 IC", edgecolor=C2_ORANGE, linewidth=0.5)

# 标注观测数
for i, (mom, c2, n) in enumerate(zip(sc_mom, sc_c2, sc_n)):
    ax2.text(max(mom, c2) + 0.01, i, f"n={n//1000}k", va="center",
            fontproperties=fp_small, color="#888", fontsize=8)

ax2.set_yticks(y_pos)
ax2.set_yticklabels([s[0] for s in scenarios], fontproperties=fp_small, fontsize=9, color=FG)
ax2.set_xlabel("Spearman Rank IC", fontproperties=fp_small, fontsize=9, color=FG)
ax2.axvline(0, color=GRID_COLOR, linewidth=0.8, linestyle="--")
ax2.set_title("关键场景 IC 对比 (fwd=10d)", fontproperties=fp_bold, fontsize=12, color=FG, pad=8)
ax2.legend(fontsize=8, loc="lower right", framealpha=0.8, facecolor=CARD_BG,
          edgecolor=GRID_COLOR, labelcolor=FG, prop=fp_small)
ax2.tick_params(colors=FG, labelsize=8)
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.grid(axis="x", alpha=0.3)

# ────────────────────────────────────────
# 右侧下半: 三状态机
# ────────────────────────────────────────
ax3 = fig.add_axes([0.58, 0.08, 0.40, 0.38])
ax3.set_facecolor(CARD_BG)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 8)
ax3.axis("off")
ax3.set_title("修正版三状态切换", fontproperties=fp_bold, fontsize=12, color=FG, pad=8)

state_colors = [MOM_GREEN, C2_ORANGE, "#27ae60"]
state_positions = [(2, 5.5), (5, 5.5), (8, 5.5)]

for idx, (name, condition, signal, color, desc) in enumerate(states):
    x, y = state_positions[idx]
    
    # 主框
    rect = mpatches.FancyBboxPatch(
        (x - 1.2, y - 1.0), 2.4, 2.2,
        boxstyle="round,pad=0.1", facecolor="#16213e" if idx != 1 else "#2a1a0a",
        edgecolor=color, linewidth=2.5
    )
    ax3.add_patch(rect)
    
    ax3.text(x, y + 0.5, name, ha="center", va="center",
            fontproperties=fp_bold, color=color, fontsize=13)
    ax3.text(x, y - 0.05, condition, ha="center", va="center",
            fontproperties=fp_small, color=FG, fontsize=9)
    ax3.text(x, y - 0.45, signal, ha="center", va="center",
            fontproperties=fp_small, color=color, fontsize=8, fontweight="bold")
    ax3.text(x, y - 0.85, desc, ha="center", va="center",
            fontproperties=fp_small, color="#999", fontsize=8)

# 箭头
arrow_props = dict(arrowstyle="->", color="#666", lw=2, connectionstyle="arc3,rad=0.2")
ax3.annotate("", xy=(3.8, 5.0), xytext=(3.2, 5.0), arrowprops=arrow_props)
ax3.annotate("", xy=(6.8, 5.0), xytext=(6.2, 5.0), arrowprops=arrow_props)
ax3.text(3.5, 4.4, "vol加速", ha="center", va="center",
        fontproperties=fp_small, color="#666", fontsize=7)
ax3.text(6.5, 4.4, "vol见顶", ha="center", va="center",
        fontproperties=fp_small, color="#666", fontsize=7)

# 底部总结
ax3.text(5, 1.0, "一刀切实操: vol_trend_zscore > 0 时切 C2, 否则用动量",
        ha="center", va="center", fontproperties=fp_bold, color=ACCENT_YELLOW, fontsize=11)
ax3.text(5, 0.3, "覆盖 95%+ 的市场状态, 避免极端obs不足问题",
        ha="center", va="center", fontproperties=fp_small, color="#999", fontsize=9)

# ── 保存 ──
out_path = OUTPUT_DIR / "vol_regime_switch_summary.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
print(f"已保存: {out_path}")
plt.close()
