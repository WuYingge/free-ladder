"""Agent 优势总结图"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from matplotlib.font_manager import FontProperties

OUTPUT_DIR = Path("/root/.openclaw/workspace/opengouzi/output")

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fp_s = FontProperties(fname=font_path, size=9)
fp_m = FontProperties(fname=font_path, size=11)
fp_b = FontProperties(fname=font_path, size=12, weight="bold")
fp_t = FontProperties(fname=font_path, size=16, weight="bold")
fp_l = FontProperties(fname=font_path, size=22, weight="bold")

BG = "#1a1a2e"
CARD = "#16213e"
FG = "#e0e0e0"
HL = "#e94560"
GREEN = "#2ecc71"
BLUE = "#3498db"
YELLOW = "#f1c40f"
ORANGE = "#e67e22"
PURPLE = "#9b59b6"
CYAN = "#1abc9c"
GRAY = "#7f8c8d"
GRID = "#2a2a4a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID,
})

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(BG)

# ── 顶部标题 ──
fig.text(0.5, 0.96, "OpenGouzi Agent — 为什么好用？",
         ha="center", va="center", fontproperties=fp_l, color=FG)
fig.text(0.5, 0.92, "vs 普通 Work Buddy + DeepSeek 的五大差异",
         ha="center", va="center", fontproperties=fp_m, color=GRAY)

# ============================================================
# 五大优势卡片
# ============================================================

cards = [
    {
        "num": "01",
        "title": "长期记忆系统",
        "subtitle": "不是每次从零开始",
        "color": GREEN,
        "items": [
            ("MEMORY.md", "长期偏好、硬规则、已学教训"),
            ("memory/日记", "每日流水自动落盘，永不丢失"),
            ("自动检索", "被问到时自动查历史，找关联和矛盾"),
            ("跨session审计", "新对话自动检查遗漏的讨论"),
        ],
        "vs": "Work Buddy: 每次对话都是全新的，无上下文连续性",
    },
    {
        "num": "02",
        "title": "人格 & 行为定制",
        "subtitle": "不是通用闲聊助手",
        "color": BLUE,
        "items": [
            ("SOUL.md", "秘书/诤臣/思考搭子，不装热情"),
            ("AGENTS.md", "路由规则、决策闭环、硬约束"),
            ("风格", "直接简洁，主动指出漏洞不加 cheerleader"),
            ("行动派", "值得记就直接记，不等确认"),
        ],
        "vs": "Work Buddy: 通用 GPT 风格，无特定人设",
    },
    {
        "num": "03",
        "title": "AgentSkills 工具链",
        "subtitle": "不只是聊天",
        "color": PURPLE,
        "items": [
            ("企微全家桶", "消息/文档/表格/日程/待办/会议"),
            ("飞书驱动", "文档读写、云盘管理"),
            ("代码执行", "直接跑回测、分析数据、生成图表"),
            ("浏览器", "网页自动化、截图、数据采集"),
        ],
        "vs": "Work Buddy: MCP 工具预定义，无法扩展",
    },
    {
        "num": "04",
        "title": "知识管理体系",
        "subtitle": "自动分类归档",
        "color": ORANGE,
        "items": [
            ("路由规则", "想法→backlog, 数据→日记忆, 项目→memo/"),
            ("优先级判断", "重要/可行/紧迫 自动排序"),
            ("项目追踪", "backlog + 项目笔记 + 日记忆三点联动"),
            ("决策闭环", "结论一形成当场落盘，不依赖聊天记录"),
        ],
        "vs": "Work Buddy: 需要手动整理，无知识管理体系",
    },
    {
        "num": "05",
        "title": "模型 & 配置",
        "subtitle": "可深度定制",
        "color": CYAN,
        "items": [
            ("DeepSeek V4 Pro", "高推理模式，中文+量化友好"),
            ("多模型切换", "Kimi / DeepSeek / 其他随时切换"),
            ("Session隔离", "不同类型任务用不同模型/配置"),
            ("定时任务", "Cron 驱动的定时巡检和提醒"),
        ],
        "vs": "Work Buddy: 固定模型，无法按任务调参",
    },
]

# 卡片布局: 2行 × 3列，第5张放第二行中间
positions = [
    (0.02, 0.47, 0.30),   # 左上
    (0.35, 0.47, 0.30),   # 中上
    (0.68, 0.47, 0.30),   # 右上
    (0.18, 0.06, 0.30),   # 中下左
    (0.51, 0.06, 0.30),   # 中下右
]

for idx, (x, y, w) in enumerate(positions):
    card = cards[idx]
    h = 0.38
    
    # 卡片背景
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(CARD)
    
    # 圆角边框
    rect = mpatches.FancyBboxPatch(
        (0, 0), 1, 1, transform=ax.transAxes,
        boxstyle="round,pad=0.02", facecolor=CARD,
        edgecolor=card["color"], linewidth=1.5, alpha=0.9
    )
    ax.add_patch(rect)
    
    # 编号角标
    ax.text(0.08, 0.90, card["num"], transform=ax.transAxes,
           fontproperties=fp_l, color=card["color"], fontsize=18, fontweight="bold")
    
    # 标题
    ax.text(0.25, 0.90, card["title"], transform=ax.transAxes,
           fontproperties=fp_b, color=FG, fontsize=13)
    ax.text(0.25, 0.81, card["subtitle"], transform=ax.transAxes,
           fontproperties=fp_s, color=GRAY, fontsize=9)
    
    # 分隔线
    ax.axhline(y=0.73, xmin=0.08, xmax=0.92, color=card["color"], linewidth=0.8, alpha=0.5)
    
    # 要点列表
    y_start = 0.65
    for i, (key, val) in enumerate(card["items"]):
        yi = y_start - i * 0.15
        # 要点标记
        ax.text(0.10, yi, "▸", transform=ax.transAxes,
               fontproperties=fp_s, color=card["color"], fontsize=9)
        ax.text(0.15, yi + 0.02, key, transform=ax.transAxes,
               fontproperties=fp_s, color=card["color"], fontsize=10, fontweight="bold")
        ax.text(0.15, yi - 0.04, val, transform=ax.transAxes,
               fontproperties=fp_s, color=FG, fontsize=9)
    
    # VS 行
    ax.text(0.10, 0.01, card["vs"], transform=ax.transAxes,
           fontproperties=fp_s, color="#555", fontsize=8, fontstyle="italic")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ============================================================
# 底部: 一句话总结
# ============================================================
fig.text(0.5, 0.02, "Work Buddy 是「开个新窗口问 AI」 — OpenGouzi 是「训练好的秘书，认识你、了解你在做什么、知道该怎么帮你」",
         ha="center", va="center", fontproperties=fp_b, color=YELLOW, fontsize=12)

# ── 保存 ──
out_path = OUTPUT_DIR / "agent_advantages.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
print(f"已保存: {out_path}")
plt.close()
