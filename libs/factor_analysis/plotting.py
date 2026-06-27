"""
因子分析绑图工具 (Factor Analysis Plotting)

统一 matplotlib + seaborn 绑图风格，所有绑图函数返回 (fig, ax) 元组，
方便 notebook 内嵌展示，也可直接 savefig 落盘。

风格继承自 visualization/factor_plot.py，配色使用 seaborn husl 调色板。
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 非交互后端，适合服务器/脚本环境

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── CJK 字体配置 ────────────────────────────────────────────────────────────
# 服务器通常缺少中文字体，matplotlib 默认的 DejaVu Sans 不包含 CJK 字形。
# 此处按优先级搜索系统可用的 CJK 字体文件并注册，确保中文标题/标签正常渲染。

_CJK_FONT_PATHS = [
    # Noto Sans CJK 简体中文 (Google Noto 系列，优先)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    # 文泉驿微米黑 (常见 Linux 备选)
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    # Noto Serif CJK (衬线体降级)
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
]

_cjk_font_name = None
_cjk_font_path = None
for _font_path in _CJK_FONT_PATHS:
    try:
        _prop = fm.FontProperties(fname=_font_path)
        _cjk_font_name = _prop.get_name()
        _cjk_font_path = _font_path
        break
    except (FileNotFoundError, RuntimeError):
        continue

# ── 注册 CJK 字体并强制刷新缓存 ────────────────────────────────────────────
# matplotlib 的字体发现结果会缓存到 ~/.cache/matplotlib/fontlist-*.json。
# 如果该缓存是在 CJK 字体未被正确扫描时生成的，addfont 后字体名仍然无法匹配。
# 因此需要: 1) 注册字体文件  2) 删除缓存  3) 强制重建字体列表。
if _cjk_font_name and _cjk_font_path:
    fm.fontManager.addfont(_cjk_font_path)
    # 清除可能的过期缓存文件
    try:
        import glob as _glob
        import os as _os
        _cache_dir = matplotlib.get_cachedir()
        for _cache_file in _glob.glob(_os.path.join(_cache_dir, "fontlist*")):
            try:
                _os.remove(_cache_file)
            except OSError:
                pass
    except Exception:
        pass
    # 强制重建字体管理器（try_read_cache=False 会重新扫描系统字体目录）
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

# ── 全局样式 ─────────────────────────────────────────────────────────────────

# 参考项目现有的 factor_plot.py 的 seaborn 风格
sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# CJK 字体必须在所有样式设置完成后再应用，因为 seaborn.set_style 可能会
# 部分覆盖字体相关 rcParams。此处强制将已注册的 CJK 字体置为 sans-serif 首选。
if _cjk_font_name and _cjk_font_path:
    plt.rcParams["font.sans-serif"] = [_cjk_font_name, "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

# 统一配色
PRICE_COLOR = "#333333"
POSITIVE_COLOR = "#2ca02c"   # 绿 → 正收益/多
NEGATIVE_COLOR = "#d62728"   # 红 → 负收益/空
ZERO_COLOR = "#7f7f7f"       # 灰 → 零轴/参考线
PALETTE = sns.color_palette("husl", 10)


# ── Layer 1 绑图 ────────────────────────────────────────────────────────────


def plot_coverage(
    coverage: pd.Series,
    factor_name: str = "",
    figsize: tuple[float, float] = (14, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """覆盖率时序图。

    Parameters
    ----------
    coverage: index=日期, values=覆盖率比例 (0.0~1.0)。
    factor_name: 因子名称（用于标题）。
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(coverage.index, coverage.values, alpha=0.3, color=PRICE_COLOR)
    ax.plot(coverage.index, coverage.values, color=PRICE_COLOR, linewidth=1.0)
    ax.axhline(y=1.0, color=ZERO_COLOR, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=coverage.mean(), color=NEGATIVE_COLOR, linestyle=":", linewidth=0.8,
               label=f"均值={coverage.mean():.2%}")

    ax.set_ylabel("覆盖率")
    ax.set_title(f"因子覆盖率时序 — {factor_name}")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left")

    fig.tight_layout()
    return fig, ax


def plot_distribution_bands(
    daily_percentiles: pd.DataFrame,
    factor_name: str = "",
    figsize: tuple[float, float] = (14, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """因子值分位数带时序图 (P5/P25/P50/P75/P95)。

    Parameters
    ----------
    daily_percentiles: columns=[P5,P25,P50,P75,P95], index=日期。
    """
    fig, ax = plt.subplots(figsize=figsize)

    dates = daily_percentiles.index

    # P5-P95 填充带
    if "P5" in daily_percentiles.columns and "P95" in daily_percentiles.columns:
        ax.fill_between(dates, daily_percentiles["P5"], daily_percentiles["P95"],
                        alpha=0.10, color=PRICE_COLOR, label="P5–P95")
    if "P25" in daily_percentiles.columns and "P75" in daily_percentiles.columns:
        ax.fill_between(dates, daily_percentiles["P25"], daily_percentiles["P75"],
                        alpha=0.15, color=PRICE_COLOR, label="P25–P75")

    # P50 中线
    if "P50" in daily_percentiles.columns:
        ax.plot(dates, daily_percentiles["P50"], color=POSITIVE_COLOR,
                linewidth=1.2, label="P50 (中位数)")

    ax.set_ylabel("因子值")
    ax.set_title(f"因子分布分位数带 — {factor_name}")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_autocorr_decay(
    autocorr_df: pd.DataFrame,
    factor_name: str = "",
    figsize: tuple[float, float] = (9, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """自相关衰减曲线。

    Parameters
    ----------
     autocorr_df: columns=[lag, mean_autocorr, median_autocorr, std_autocorr]。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if autocorr_df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    lags = autocorr_df["lag"].values
    means = autocorr_df["mean_autocorr"].values
    medians = autocorr_df["median_autocorr"].values
    stds = autocorr_df["std_autocorr"].values

    ax.errorbar(lags, means, yerr=stds, fmt="o-", color=PRICE_COLOR,
                capsize=5, linewidth=1.5, markersize=6, label="均值 ± 标准差")
    ax.plot(lags, medians, "s--", color=POSITIVE_COLOR,
            linewidth=1.5, markersize=6, label="中位数")
    ax.axhline(y=0.0, color=ZERO_COLOR, linestyle="--", linewidth=0.8)

    ax.set_xlabel("滞后天数 (交易日)")
    ax.set_ylabel("Rank Autocorrelation")
    ax.set_title(f"因子自相关衰减 — {factor_name}")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)

    note = "注：截面因子（如动量）自相关通常较低，此指标更适用于时序因子（如 RSRS）"
    ax.annotate(note, xy=(0.5, -0.12), xycoords="axes fraction",
                ha="center", fontsize=8, color=ZERO_COLOR)

    fig.tight_layout()
    return fig, ax


# ── Layer 2 绑图 ────────────────────────────────────────────────────────────


def plot_ic_decay(
    ic_decay_df: pd.DataFrame,
    factor_name: str = "",
    figsize: tuple[float, float] = (9, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """IC 衰减曲线（±1 std 带）。

    Parameters
    ----------
    ic_decay_df: columns=[period, ic_mean, ic_std]。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if ic_decay_df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    periods = ic_decay_df["period"].values
    means = ic_decay_df["ic_mean"].values
    stds = ic_decay_df["ic_std"].values

    ax.fill_between(periods, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    alpha=0.15, color=POSITIVE_COLOR)
    ax.plot(periods, means, "o-", color=POSITIVE_COLOR, linewidth=1.8, markersize=7)

    ax.axhline(y=0.0, color=ZERO_COLOR, linestyle="--", linewidth=0.8)
    ax.axhline(y=0.02, color=ZERO_COLOR, linestyle=":", linewidth=0.6, alpha=0.5,
               label="IC=0.02 (边际)")

    ax.set_xlabel("持仓期 (交易日)")
    ax.set_ylabel("Rank IC")
    ax.set_title(f"IC 衰减曲线 — {factor_name}")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_rolling_ic(
    rolling_ic_df: pd.DataFrame,
    factor_name: str = "",
    figsize: tuple[float, float] = (14, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """滚动 IC 时序图。

    Parameters
    ----------
    rolling_ic_df: columns=[ic_raw, ic_rolling_mean], index=日期。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if rolling_ic_df.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # 原始 IC 以半透明散点展示
    ax.scatter(rolling_ic_df.index, rolling_ic_df["ic_raw"],
               s=2, alpha=0.15, color=ZERO_COLOR, label="日 IC")

    # 滚动均值线
    ax.plot(rolling_ic_df.index, rolling_ic_df["ic_rolling_mean"],
            color=POSITIVE_COLOR, linewidth=1.5, label="滚动均值")

    ax.axhline(y=0.0, color=ZERO_COLOR, linestyle="--", linewidth=0.8)

    ax.set_ylabel("Rank IC")
    ax.set_title(f"滚动 IC 时序 — {factor_name}")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_ic_heatmap(
    matrix: pd.DataFrame,
    factor_name: str = "",
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """参数 × 持仓期 IC 热力图。

    Parameters
    ----------
    matrix: index=参数组合标签, columns=持仓期, values=IC 均值。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if matrix.empty or matrix.size == 0:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    sns.heatmap(
        matrix.astype(float),
        annot=True, fmt=".4f", cmap="RdYlGn", center=0.0,
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Rank IC mean"},
    )

    ax.set_title(f"参数敏感度网格 — {factor_name}")
    ax.set_xlabel("持仓期 (交易日)")
    ax.set_ylabel("参数组合")

    fig.tight_layout()
    return fig, ax


# ── Layer 3 绑图 ────────────────────────────────────────────────────────────


def plot_quantile_returns_bar(
    quantile_summary: pd.DataFrame,
    factor_name: str = "",
    figsize: tuple[float, float] = (9, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """各分位组平均收益柱状图。

    Parameters
    ----------
    quantile_summary: index=Q1..Qn, columns=[mean_return]。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if quantile_summary.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    groups = quantile_summary.index.tolist()
    values = quantile_summary["mean_return"].values if "mean_return" in quantile_summary.columns else []

    colors = []
    for i in range(len(groups)):
        if i == 0:
            colors.append(NEGATIVE_COLOR)   # Bottom 组红色
        elif i == len(groups) - 1:
            colors.append(POSITIVE_COLOR)   # Top 组绿色
        else:
            colors.append(ZERO_COLOR)       # 中间组灰色

    bars = ax.bar(groups, values, color=colors, alpha=0.8, edgecolor="white")

    # 数值标注
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9)

    ax.axhline(y=0.0, color=ZERO_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlabel("分位组")
    ax.set_ylabel("平均收益率")
    ax.set_title(f"分位组收益 — {factor_name}")

    note = f"Q1=因子最低组, Q{len(groups)}=因子最高组"
    ax.annotate(note, xy=(0.5, -0.10), xycoords="axes fraction",
                ha="center", fontsize=8, color=ZERO_COLOR)

    fig.tight_layout()
    return fig, ax


def plot_quantile_cumret(
    quantile_cumret: pd.DataFrame,
    factor_name: str = "",
    figsize: tuple[float, float] = (14, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """五分位累计收益曲线叠加图。

    Parameters
    ----------
    quantile_cumret: index=日期, columns=[Q1, Q2, ..., Qn], values=累计净值。
    """
    fig, ax = plt.subplots(figsize=figsize)

    if quantile_cumret.empty:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    n_groups = len(quantile_cumret.columns)
    colors = sns.color_palette("RdYlGn", n_groups)

    for i, col in enumerate(quantile_cumret.columns):
        ax.plot(quantile_cumret.index, quantile_cumret[col],
                color=colors[i], linewidth=1.3, alpha=0.85, label=col)

    ax.axhline(y=1.0, color=ZERO_COLOR, linestyle="--", linewidth=0.8)
    ax.set_ylabel("累计净值 (起始=1.0)")
    ax.set_title(f"分位组累计收益 — {factor_name}")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig, ax


def plot_longshort_cumret(
    longshort_result: dict,
    factor_name: str = "",
    figsize: tuple[float, float] = (14, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """Long-Short 多空组合累计收益曲线。

    Parameters
    ----------
    longshort_result: 来自 compute_longshort 的返回 dict。
    """
    fig, ax = plt.subplots(figsize=figsize)

    ls_series = longshort_result.get("ls_series")
    if ls_series is None or len(ls_series) == 0:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    cum_ret = (1.0 + ls_series).cumprod()
    ax.fill_between(cum_ret.index, 1.0, cum_ret.values,
                    alpha=0.15, color=POSITIVE_COLOR)
    ax.plot(cum_ret.index, cum_ret.values, color=POSITIVE_COLOR, linewidth=1.3)

    ax.axhline(y=1.0, color=ZERO_COLOR, linestyle="--", linewidth=0.8)

    # 标注指标
    ann_ret = longshort_result.get("annualised_return", float("nan"))
    sharpe = longshort_result.get("sharpe")
    mdd = longshort_result.get("max_drawdown", float("nan"))

    text_lines = [f"年化收益: {ann_ret:.2%}" if not np.isnan(ann_ret) else "年化收益: N/A"]
    if sharpe is not None:
        text_lines.append(f"Sharpe: {sharpe:.3f}")
    text_lines.append(f"最大回撤: {mdd:.2%}" if not np.isnan(mdd) else "最大回撤: N/A")

    ax.text(0.02, 0.97, "\n".join(text_lines), transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_ylabel("累计净值")
    ax.set_title(f"Long-Short 多空累计收益 — {factor_name}")

    note = "多空 = Top 组 − Bottom 组（剥离市场 Beta 后的纯 Alpha）"
    ax.annotate(note, xy=(0.5, -0.08), xycoords="axes fraction",
                ha="center", fontsize=8, color=ZERO_COLOR)

    fig.tight_layout()
    return fig, ax


# ── 保存辅助 ─────────────────────────────────────────────────────────────────


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150) -> str:
    """保存绑图到文件并关闭 figure（释放内存）。

    Returns
    -------
    str
        保存的文件路径。
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filepath
