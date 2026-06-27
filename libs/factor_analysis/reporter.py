"""
报告生成器 (Report Generator)

将因子分析结果输出为双格式报告：
1. JSON — 机器可读，所有统计量结构化存储
2. Markdown — 人类可读摘要，嵌入图表引用

同时负责将分析结果中的 DataFrame/Series 转换为可序列化格式。
"""

from __future__ import annotations

import json
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_analysis.config import FactorAnalysisConfig
from factor_analysis.panel import FactorPanel


# ── 序列化辅助 ───────────────────────────────────────────────────────────────


class _FactorReportEncoder(json.JSONEncoder):
    """自定义 JSON 编码器：处理 numpy/pandas/Timestamp 类型。"""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if isinstance(obj, (date,)):
            return obj.isoformat()
        if isinstance(obj, (pd.Series,)):
            return obj.to_dict()
        if isinstance(obj, (pd.DataFrame,)):
            return obj.to_dict(orient="records")
        return super().default(obj)


def _safe_json(obj: Any) -> Any:
    """将对象递归转换为 JSON 可序列化格式。"""
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (pd.Series,)):
        # Series → dict（保留 index→value 映射）
        d = obj.dropna().to_dict()
        return {str(k): (v if not isinstance(v, float) or not np.isnan(v) else None) for k, v in d.items()}
    if isinstance(obj, (pd.DataFrame,)):
        # DataFrame → list of dicts
        return obj.where(pd.notna(obj), None).to_dict(orient="records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (pd.Timestamp, date)):
        return obj.isoformat()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ── Markdown 生成 ────────────────────────────────────────────────────────────


def _generate_md_report(
    panel: FactorPanel,
    quality_results: dict,
    predictive_results: dict,
    grouping_results: dict,
    config: FactorAnalysisConfig,
    output_date: str,
) -> str:
    """生成 Markdown 格式的分析报告，每个指标附带「怎么算的」和「代表什么」解读。"""
    buf = StringIO()
    n_q = config.n_quantiles

    # ── 标题 ──
    buf.write(f"# 因子分析报告: {panel.factor_name}\n\n")
    buf.write(f"**分析日期**: {output_date} | **因子类型**: {type(config.factor).__name__}\n\n")
    buf.write(f"> 本报告评估因子 `{panel.factor_name}` 在 {panel.n_symbols} 个标的上"
              f"（{panel.date_range[0].date()} ~ {panel.date_range[1].date()}）的表现。"
              f"每个指标均附带「怎么算的」和「代表什么」解读。\n\n")

    # ── 面板概况 ──
    s = panel.summary()
    buf.write("## 数据概况\n\n")
    buf.write("| 项目 | 数值 |\n|---|---|\n")
    buf.write(f"| 有效标的数 | {s['n_symbols']} |\n")
    buf.write(f"| 有效日期数 | {s['n_dates']} |\n")
    buf.write(f"| 日期范围 | {s['start_date']} ~ {s['end_date']} |\n")
    buf.write(f"| 均值覆盖率 | {s['coverage_mean']:.2%} |\n")
    buf.write(f"| 加载失败 | {s['n_errors']} 个 |\n")
    buf.write(f"| min_bars 过滤 | {s['n_filtered']} 个 |\n")
    buf.write("\n> **怎么算的**: 均值覆盖率 = 每天有有效因子值的标的数 / 总标的数，再对所有交易日取平均。"
              "> 它回答了因子计算有没有系统性缺失——如果某天覆盖率骤降，说明数据源出了 bug 或大量标的进入 warmup 期。\n\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Layer 1: 因子质量
    # ═══════════════════════════════════════════════════════════════════════════
    if quality_results:
        buf.write("## Layer 1: 因子质量\n\n")
        buf.write("> **本层定位**: 只看因子自身，不涉及未来收益。回答「这个因子本身是否健康可用」。\n\n")

        # ── 1.1 覆盖率 ──
        cov = quality_results.get("coverage")
        if cov is not None and len(cov) > 0:
            buf.write("### 1.1 覆盖率\n\n")
            buf.write(f"![覆盖率时序](coverage.png)\n\n")
            buf.write("> **怎么算的**: 每个交易日统计「有有效因子值的标的数 ÷ 总标的数」，画成时间序列。\n")
            buf.write("> **代表什么**: 覆盖率随时间骤降 = 因子计算有 bug 或数据源退化。覆盖面过窄（如 < 30%）的因子无法做截面比较。\n\n")

        # ── 1.2 分布特征 ──
        dist = quality_results.get("distribution_stats", {})
        if dist:
            skew = dist.get("skewness", 0)
            kurt = dist.get("kurtosis", 0)
            p01 = dist.get("p01", 0)
            p99 = dist.get("p99", 0)

            buf.write("### 1.2 全样本分布统计\n\n")
            buf.write("| 指标 | 值 |\n|---|---|\n")
            for k, v in dist.items():
                buf.write(f"| {k} | {v:.6f} |\n")
            buf.write("\n")
            buf.write(f"![分布分位数带](distribution_bands.png)\n\n")
            buf.write("> **怎么算的**: 把所有标的、所有交易日的因子值混在一起，算均值/标准差/偏度/峰度/各分位数。")
            buf.write("图上是每天截面上的 P5/P25/P50/P75/P95 分位数随时间的变化。\n")
            buf.write(f"> **代表什么**: ")
            # 偏度解读
            if abs(skew) < 0.5:
                buf.write("偏度接近 0，分布大致对称。")
            elif skew > 0:
                buf.write(f"右偏 (skew={skew:.2f})——因子值向正方向拖着长尾巴，少数极端正值（如暴涨）拉高了均值。")
            else:
                buf.write(f"左偏 (skew={skew:.2f})——因子值向负方向拖着长尾巴，少数极端负值（如暴跌）拉低了均值。")
            # 峰度解读
            if abs(kurt) < 1:
                buf.write(f" 峰度接近 0（近似正态），极端值不频繁。")
            elif kurt > 3:
                buf.write(f" 峰度={kurt:.2f}（肥尾）——极端值远多于正态分布预期，因子可能被少数异常值驱动。")
            else:
                buf.write(f" 峰度={kurt:.2f}（轻微肥尾）。")
            # 百分位
            buf.write(f" P1={p01:.4f} 到 P99={p99:.4f} 覆盖了 98% 的因子值范围。")
            buf.write("极端偏态分布的因子不适合用 Pearson IC 评估（Spearman 更健壮）。\n\n")

        # ── 1.3 缺失模式 ──
        mv = quality_results.get("missing_by_volume", {})
        mb = quality_results.get("missing_by_bar_count", {})
        if mv or mb:
            buf.write("### 1.3 缺失模式分析\n\n")
            if mv:
                buf.write("**按成交额分档**:\n\n")
                buf.write("| 分档 | 缺失率 |\n|---|---|\n")
                for label, rate in mv.items():
                    buf.write(f"| {label} | {rate:.4%} |\n")
                buf.write("\n")
            if mb:
                buf.write("**按上市时长 (bar_count) 分档**:\n\n")
                buf.write("| 分档 | 缺失率 |\n|---|---|\n")
                for label, rate in mb.items():
                    buf.write(f"| {label} | {rate:.4%} |\n")
                buf.write("\n")
            buf.write("> **怎么算的**: 按标的的属性（日均成交额 / 有效交易日数）将它们分成 3 档，统计每档内因子 NaN 的比例。\n")
            buf.write("> **代表什么**: 如果缺失率在不同档次间差异很大（如低成交额 ETF 缺失率显著更高），说明因子在截面上系统性地偏向某一类标的，存在隐性偏差。\n\n")

        # ── 1.4 自相关 ──
        ac = quality_results.get("autocorr")
        if ac is not None and not ac.empty:
            lag1 = ac.iloc[0]["median_autocorr"]
            lag20 = ac.iloc[-1]["median_autocorr"]
            buf.write("### 1.4 自相关衰减\n\n")
            buf.write("| 滞后 (天) | 均值自相关 | 中位数自相关 | 标准差 |\n|---|---|---|---|\n")
            for _, row in ac.iterrows():
                buf.write(f"| {int(row['lag'])} | {row['mean_autocorr']:.4f} | {row['median_autocorr']:.4f} | {row['std_autocorr']:.4f} |\n")
            buf.write("\n")
            buf.write(f"![自相关衰减](autocorr_decay.png)\n\n")
            buf.write("> **怎么算的**: 对每个标的，计算 factor(t) 和 factor(t−lag) 的 Spearman 秩相关系数，取所有标的的截面均值和中位数。中位数比均值更稳健，不受少数极端标的影响。\n")
            buf.write(f"> **代表什么**: lag1 中位数={lag1:.2f}，说明一半以上标的今天的因子值和昨天{'几乎一样' if lag1 > 0.9 else '差别很大' if lag1 < 0.3 else '有一定相似性'}（自相关={lag1:.0%}）。")
            if abs(lag20) > 0.3:
                buf.write(f" lag20={lag20:.2f}，即使隔一个月仍有{lag20:.0%}相关性——因子变化缓慢，调仓频率不需要太高（如日频调仓意义不大，周频或月频更合理）。")
            buf.write("\n\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Layer 2: 预测力
    # ═══════════════════════════════════════════════════════════════════════════
    if predictive_results:
        buf.write("## Layer 2: 预测力\n\n")
        buf.write("> **本层定位**: 核心。回答「这个因子能否预测未来收益」——这是评估因子价值最关键的部分。\n\n")

        # 取默认持仓期（优先 20 日，fallback 取第一个可用）
        _rank_ic_map = predictive_results.get("rank_ic", {})
        _pearson_ic_map = predictive_results.get("pearson_ic", {})
        _periods = sorted(_rank_ic_map.keys())
        _default_period = 20 if 20 in _periods else (_periods[0] if _periods else None)
        rank = _rank_ic_map.get(_default_period, {}).get("summary", {})
        pearson = _pearson_ic_map.get(_default_period, {}).get("summary", {})

        if rank:
            mean_ic = rank.get("mean", float("nan"))
            ir = rank.get("ir", float("nan"))
            pos = rank.get("pos_ratio", float("nan"))
            t_stat = rank.get("t_stat", float("nan"))

            # ── 2.1 各持仓期 IC 一览表 ────────────────────────────────────
            buf.write("### 2.1 各持仓期 IC\n\n")
            if _periods:
                buf.write("| 持仓期 | Rank IC 均值 | IR | t 统计量 | IC>0 比例 | Pearson IC 均值 |\n")
                buf.write("|---|---|---|---|---|---|\n")
                for period in _periods:
                    rs = _rank_ic_map[period]["summary"]
                    ps = _pearson_ic_map.get(period, {}).get("summary", {})
                    buf.write(
                        f"| {period}d "
                        f"| {rs['mean']:.6f} "
                        f"| {rs['ir']:.4f} "
                        f"| {rs['t_stat']:.2f} "
                        f"| {rs['pos_ratio']:.1%} "
                        f"| {ps.get('mean', float('nan')):.6f} |\n"
                    )
                buf.write("\n")

            # ── 尺度感定级（以默认持仓期为例） ────────────────────────────
            if not np.isnan(mean_ic):
                if mean_ic > 0.10:
                    grade = "🔴 顶级 (> 0.10)"
                elif mean_ic > 0.05:
                    grade = "🟠 靠谱 (0.05~0.10)"
                elif mean_ic > 0.02:
                    grade = "🟡 可用 (0.02~0.05)"
                elif mean_ic > -0.02:
                    grade = "⚪ 接近零 (预测力很弱)"
                else:
                    grade = "🔵 负向（与预期方向相反）"
                buf.write(f"\n**IC 评级** (以 {_default_period}d 为例): {grade}\n\n")

            buf.write("> **怎么算的 (Rank IC)**: 每个交易日，在截面上计算「因子值」与「未来 N 日收益率」的 Spearman 秩相关系数。对所有交易日取均值。\n")
            buf.write("> **代表什么**: ")
            if not np.isnan(mean_ic):
                buf.write(f"（以 {_default_period}d 为例）Rank IC 均值 = {mean_ic:.4f}，")
                if mean_ic > 0.02:
                    buf.write("因子值与未来收益**正相关**——因子值越高，未来收益倾向于越高。这是一个正向预测因子。")
                elif mean_ic < -0.02:
                    buf.write("因子值与未来收益**负相关**——因子值越高，未来收益反而越低。可能是反转因子，或者你预期的方向反了。")
                else:
                    buf.write("因子值与未来收益的截面排序关系**非常弱**，接近随机。这个因子在这个样本上几乎没有预测力。")
            buf.write("\n")
            buf.write(f"> **Rank IC vs Pearson IC** ({_default_period}d): {'Rank IC ≈ Pearson IC，说明因子收益不是由少数极端值驱动的，分布较均匀。' if abs(mean_ic - pearson.get('mean', float('nan'))) < 0.02 else 'Rank IC 和 Pearson IC 差异较大——注意是否有肥尾效应影响线性评估。'}\n")
            buf.write(f"> **IR (信息比率)** = IC 均值 ÷ IC 标准差 = {ir:.4f}。IR 衡量的是 IC 的**稳定性**：IR > 0.5 意味着信号噪声比不错，信号比较稳定；IR < 0.2 意味着 IC 波动很大，每天忽正忽负。\n")
            buf.write(f"> **t 统计量** = {t_stat:.2f}：衡量 IC 均值是否统计上显著不等于 0。绝对值 > 2 通常认为显著。\n")
            buf.write(f"> **IC>0 比例** = {pos:.1%}：有多少天的 IC 是正数。接近 50% 说明因子方向随机，接近 60% 以上说明方向稳定。\n\n")

            # 各持仓期滚动 IC 图
            for period in sorted(predictive_results.get("rolling_ic", {}).keys()):
                buf.write(f"**{period}d 滚动 IC 时序**\n\n")
                buf.write(f"![滚动 IC 时序 ({period}d)](rolling_ic_{period}d.png)\n\n")
            buf.write("> 滚动 IC 图显示 IC 在不同时间段的表现。IC 均值好看但近几年归零 = 因子已经失效。'IC 在什么时间段有效'比'IC 均值多少'更重要。\n\n")

        # ── IC 衰减 ──
        decay = predictive_results.get("ic_decay")
        if decay is not None and not decay.empty:
            buf.write("### 2.3 IC 衰减曲线\n\n")
            buf.write("| 持仓期 (天) | IC 均值 | IC 标准差 | IC IR |\n|---|---|---|---|\n")
            for _, row in decay.iterrows():
                buf.write(f"| {int(row['period'])} | {row['ic_mean']:.6f} | {row['ic_std']:.6f} | {row['ic_ir']:.4f} |\n")
            buf.write("\n")
            buf.write(f"![IC 衰减曲线](ic_decay.png)\n\n")
            buf.write("> **怎么算的**: 分别对 T+5/10/20/60 日收益计算 Rank IC，画成衰减曲线。\n")
            buf.write("> **代表什么**: 看 IC 随持仓期延长怎么变化。衰减太快 = 信号太短命（需要高频交易才能抓住）；衰减太慢 = 可能只是捕捉了长期截面特征而非定价错误。好的因子应该有一个合理的半衰期（如 10~20 天衰减到一半）。\n\n")

        # ── 参数网格 ──
        pg = predictive_results.get("param_grid")
        if pg is not None and not pg.get("matrix", pd.DataFrame()).empty:
            buf.write(f"![参数敏感度热力图](ic_matrix.png)\n\n")
            buf.write("> 热力图展示了不同参数组合下 IC 的变化。如果只在某个窄参数区间有效 = 可能过拟合。在较宽参数区间都稳定有效 = 可靠的因子。\n\n")

    # ═══════════════════════════════════════════════════════════════════════════
    # Layer 3: 分组检验
    # ═══════════════════════════════════════════════════════════════════════════
    if grouping_results:
        buf.write("## Layer 3: 分组检验\n\n")
        buf.write(f"> **本层定位**: 从「因子能否排序」深化到「按因子分组能赚多少钱」。将 IC 的统计显著性转化为可交易的经济显著性。以下各表展示因子值最高组 (Top) 与最低组 (Bottom) 在各持仓期下的表现差异。\n\n")

        # 取默认持仓期（优先 20 日，fallback 取第一个可用）
        _gr_periods = sorted(grouping_results.keys())
        _gr_default = 20 if 20 in _gr_periods else (_gr_periods[0] if _gr_periods else None)

        if _gr_periods:
            # ── 3.1 各持仓期分位组收益 ─────────────────────────────────
            buf.write("### 3.1 各持仓期分位组收益\n\n")
            buf.write(f"> 每天把 {panel.n_symbols} 个标的按因子值从低到高分成 {n_q} 组，每组等权持有，计算未来 N 日收益的均值。\n\n")

            # 表头：先取第一个 period 看有哪些分位组
            _first_qs = grouping_results[_gr_periods[0]]["quantile_summary"]
            if not _first_qs.empty:
                _q_labels = _first_qs.index.tolist()
                _bottom_label = _q_labels[0]
                _top_label = _q_labels[-1]
                buf.write(f"| 持仓期 | {_bottom_label} (Bottom) | {_top_label} (Top) | Top−Bottom 差值 |\n")
                buf.write("|---|---|---|---|\n")
                for period in _gr_periods:
                    gr = grouping_results[period]
                    qs = gr["quantile_summary"]
                    if qs.empty:
                        continue
                    bottom_ret = qs.loc[_bottom_label, "mean_return"] if "mean_return" in qs.columns else float("nan")
                    top_ret = qs.loc[_top_label, "mean_return"] if "mean_return" in qs.columns else float("nan")
                    spread = top_ret - bottom_ret if not np.isnan(top_ret) and not np.isnan(bottom_ret) else float("nan")
                    buf.write(f"| {period}d | {bottom_ret:.6f} | {top_ret:.6f} | {spread:.6f} |\n")
                buf.write("\n")

            # ── 各持仓期图表 ──────────────────────────────────────────
            for period in _gr_periods:
                buf.write(f"**{period}d 分位组收益柱状图**\n\n")
                buf.write(f"![分位组收益柱状图](quantile_returns_bar_{period}d.png)\n\n")
                buf.write(f"**{period}d 分位组累计收益**\n\n")
                buf.write(f"![分位组累计收益](quantile_cumret_{period}d.png)\n\n")
            buf.write("\n")

            # ── 文字解读（以默认持仓期为例） ──────────────────────────
            if _gr_default is not None:
                gr_def = grouping_results[_gr_default]
                qs_def = gr_def["quantile_summary"]
                if not qs_def.empty:
                    _q_labels_def = qs_def.index.tolist()
                    _bl = _q_labels_def[0]
                    _tl = _q_labels_def[-1]
                    b_ret = qs_def.loc[_bl, "mean_return"] if "mean_return" in qs_def.columns else float("nan")
                    t_ret = qs_def.loc[_tl, "mean_return"] if "mean_return" in qs_def.columns else float("nan")
                    spread_def = t_ret - b_ret if not np.isnan(t_ret) and not np.isnan(b_ret) else float("nan")

                    buf.write(f"> **代表什么** (以 {_gr_default}d 为例): ")
                    if not np.isnan(spread_def):
                        if spread_def > 0.001:
                            buf.write(f"Top 组（{_tl}）的收益（{t_ret:.4%}）明显高于 Bottom 组（{_bl}）的收益（{b_ret:.4%}），差值为 {spread_def:.4%}。因子排序能力较好。\n")
                        elif spread_def < -0.001:
                            buf.write(f"Top 组的收益反而低于 Bottom 组——因子方向与预期相反。如果因子本意是\"追涨\"，那实际表现是\"反转\"。\n")
                        else:
                            buf.write("Top 组和 Bottom 组的收益差异很小，因子无法有效区分\"好标的\"和\"差标的\"。\n")
                    buf.write("> 理想情况：Q1 > Q2 > Q3 > Q4 > Q5 严格单调（或反过来，看因子方向）。如果中间组有穿插、交叉，说明因子只在极端值有效，中间不可靠。\n\n")

            # ── 3.2 各持仓期 Long-Short ────────────────────────────────
            buf.write("### 3.2 各持仓期 Long-Short 多空组合\n\n")
            buf.write("| 持仓期 | 年化收益 | Sharpe | 最大回撤 |\n")
            buf.write("|---|---|---|---|\n")
            for period in _gr_periods:
                gr = grouping_results[period]
                ls = gr.get("longshort", {})
                if ls:
                    ann_ret = ls.get("annualised_return", float("nan"))
                    sharpe = ls.get("sharpe")
                    mdd = ls.get("max_drawdown", float("nan"))
                    buf.write(f"| {period}d | {ann_ret:.4%} | {sharpe if sharpe is not None else 'N/A'} | {mdd:.4%} |\n")
            buf.write("\n")

            # Long-Short 各持仓期图表
            for period in _gr_periods:
                buf.write(f"**{period}d 多空累计收益曲线**\n\n")
                buf.write(f"![多空累计收益](longshort_cumret_{period}d.png)\n\n")
            buf.write("\n")

            # Long-Short 文字解读（以默认持仓期为例）
            if _gr_default is not None:
                ls_def = grouping_results[_gr_default].get("longshort", {})
                if ls_def:
                    buf.write("> **怎么算的**: 做多 Top 组 + 做空 Bottom 组，计算这个多空组合的每日收益序列，再算年化收益/波动/Sharpe/最大回撤。\n")
                    buf.write(f"> **代表什么** (以 {_gr_default}d 为例): 多空收益是因子**纯 alpha 的最直接度量**——剥离了市场整体涨跌（beta），只看因子排序本身能不能赚钱。\n")
                    ann_ret_def = ls_def.get("annualised_return", float("nan"))
                    sharpe_def = ls_def.get("sharpe")
                    mdd_def = ls_def.get("max_drawdown", float("nan"))
                    if not np.isnan(ann_ret_def):
                        if ann_ret_def > 0.05:
                            buf.write(f"这个因子的多空年化收益为 +{ann_ret_def:.1%}，说明按因子排序做多最强 + 做空最弱是能赚钱的。\n")
                        elif ann_ret_def < -0.05:
                            buf.write(f"这个因子的多空年化收益为 {ann_ret_def:.1%}（负），说明目前的排序方向是亏钱的——可能需要反向使用（做多低因子值 + 做空高因子值）。\n")
                        else:
                            buf.write(f"这个因子的多空年化收益接近零（{ann_ret_def:.1%}），纯 alpha 很弱。\n")
                    if sharpe_def is not None:
                        buf.write(f"Sharpe = {sharpe_def:.2f}，{'收益风险比不错' if sharpe_def > 0.5 else '收益风险比偏低' if sharpe_def > 0 else '负收益'}。\n")
                    buf.write(f"最大回撤 = {mdd_def:.1%}（{'风险很高' if mdd_def > 0.5 else '风险中等' if mdd_def > 0.2 else '风险较低'}）。\n\n")

            # ── 3.4 各持仓期单调性 ────────────────────────────────────
            buf.write("### 3.4 各持仓期单调性检验\n\n")
            buf.write("| 持仓期 | 严格单调比例 | 宽松单调比例 | 单调方向 |\n")
            buf.write("|---|---|---|---|\n")
            for period in _gr_periods:
                gr = grouping_results[period]
                mono = gr.get("monotonicity", {})
                if mono:
                    strict = mono.get("strict_monotonic_ratio", float("nan"))
                    loose = mono.get("loose_monotonic_ratio", float("nan"))
                    direction = mono.get("monotonic_direction", "N/A")
                    buf.write(f"| {period}d | {strict:.2%} | {loose:.2%} | {direction} |\n")
            buf.write("\n")

            # 单调性文字解读（以默认持仓期为例）
            if _gr_default is not None:
                mono_def = grouping_results[_gr_default].get("monotonicity", {})
                if mono_def:
                    strict_def = mono_def.get("strict_monotonic_ratio", float("nan"))
                    buf.write("> **怎么算的**: 对每天截面检查 Q1 > Q2 > ... > Qn 是否成立。严格单调 = 每个相邻组都满足大小关系。JT (Jonckheere-Terpstra) 检验评估是否存在统计显著的趋势。\n")
                    if not np.isnan(strict_def):
                        buf.write(f"> **代表什么** (以 {_gr_default}d 为例): 严格单调成立仅 {strict_def:.1%}——{'说明因子排序在不同分位组之间的一致性很强' if strict_def > 0.5 else '大多数交易日分组收益的排序并不完美，因子在中间组上的区分度有限' if strict_def > 0.1 else '因子排序非常不稳定，几乎每天的组间收益顺序都不一样'}。非单调的因子在极端值有效但中间不可靠，做分桶筛选时会有坑。\n\n")
    # ── 综合摘要 ──
    buf.write("---\n\n")
    buf.write("## 综合摘要\n\n")
    _write_summary_line(buf, quality_results, predictive_results, grouping_results, config)
    buf.write("\n---\n\n")
    buf.write(f"*报告由 factor_analysis 框架自动生成于 {output_date}*\n")

    return buf.getvalue()


def _write_summary_line(
    buf: StringIO,
    quality_results: dict | None,
    predictive_results: dict | None,
    grouping_results: dict | None,
    config: FactorAnalysisConfig,
) -> None:
    """生成报告末尾的综合定性摘要。"""
    points: list[str] = []

    # 数据质量
    if quality_results:
        cov_series = quality_results.get("coverage")
        if cov_series is not None and len(cov_series) > 0:
            mean_cov = float(cov_series.mean())
            if mean_cov < 0.5:
                points.append(f"⚠️ 因子覆盖率仅 {mean_cov:.0%}，大量缺失，数据质量堪忧")
            elif mean_cov < 0.9:
                points.append(f"因子覆盖率 {mean_cov:.0%}，中等水平，注意 warmup 期内的缺失")
            else:
                points.append(f"✅ 因子覆盖率 {mean_cov:.0%}，数据质量良好")

    # IC 评估
    if predictive_results:
        _rank_ic_map = predictive_results.get("rank_ic", {})
        _periods = sorted(_rank_ic_map.keys())
        _default_period = 20 if 20 in _periods else (_periods[0] if _periods else None)
        rank = _rank_ic_map.get(_default_period, {}).get("summary", {})
        mean_ic = rank.get("mean", float("nan"))
        ir = rank.get("ir", float("nan"))
        if not np.isnan(mean_ic):
            if mean_ic > 0.05:
                points.append(f"✅ Rank IC ({_default_period}d) = {mean_ic:.4f}（靠谱），IR = {ir:.3f}")
            elif mean_ic > 0.02:
                points.append(f"Rank IC ({_default_period}d) = {mean_ic:.4f}（可用但偏弱），IR = {ir:.3f}")
            elif mean_ic > -0.02:
                points.append(f"⚠️ Rank IC ({_default_period}d) = {mean_ic:.4f}（接近零，预测力很弱），IR = {ir:.3f}")
            else:
                points.append(f"🔴 Rank IC ({_default_period}d) = {mean_ic:.4f}（负值，方向与预期相反），IR = {ir:.3f}")

    # 分组
    if grouping_results:
        _gr_periods = sorted(grouping_results.keys())
        _gr_default = 20 if 20 in _gr_periods else (_gr_periods[0] if _gr_periods else None)
        gr_def = grouping_results.get(_gr_default, {}) if _gr_default is not None else {}
        ls = gr_def.get("longshort", {})
        ann_ret = ls.get("annualised_return", float("nan"))
        if not np.isnan(ann_ret):
            if ann_ret > 0.10:
                points.append(f"✅ Long-Short ({_gr_default}d) 年化收益 = {ann_ret:.1%}（优秀）")
            elif ann_ret > 0.03:
                points.append(f"Long-Short ({_gr_default}d) 年化收益 = {ann_ret:.1%}（中等）")
            elif ann_ret > -0.03:
                points.append(f"⚠️ Long-Short ({_gr_default}d) 年化收益 ≈ 零（{ann_ret:.1%}），分组后的 top/bottom 差异太小")
            else:
                points.append(f"🔴 Long-Short ({_gr_default}d) 年化收益 = {ann_ret:.1%}（负值）")

    if points:
        buf.write("| 维度 | 评价 |\n|---|---|\n")
        for p in points:
            parts = p.split("，", 1)
            if len(parts) == 2:
                buf.write(f"| {parts[0]} | {parts[1]} |\n")
            else:
                buf.write(f"| — | {p} |\n")
    else:
        buf.write("（无可用数据）\n")


# ── 报告保存 ─────────────────────────────────────────────────────────────────


def generate_and_save_reports(
    panel: FactorPanel,
    quality_results: dict | None,
    predictive_results: dict | None,
    grouping_results: dict | None,
    config: FactorAnalysisConfig,
    chart_paths: dict[str, str],
) -> tuple[Path, Path]:
    """生成 JSON 和 MD 报告，保存到 Linux 端输出目录。

    Parameters
    ----------
    panel: 因子面板。
    quality_results / predictive_results / grouping_results: 各层结果。
    config: 分析配置。
    chart_paths: {chart_name: filepath}，用于 MD 中的图片引用。

    Returns
    -------
    tuple[Path, Path]
        (json_path, md_path)
    """
    output_root = config.resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    output_date = config.resolve_output_date()

    # ── JSON 报告 ──
    json_data = _safe_json({
        "meta": {
            "factor_name": panel.factor_name,
            "factor_type": type(config.factor).__name__,
            "factor_params": config.factor.params,
            "analysis_date": output_date,
        },
        "panel_summary": panel.summary(),
        "layer1_quality": quality_results,
        "layer2_predictive": predictive_results,
        "layer3_grouping": grouping_results,
    })

    json_path = output_root / f"report_{output_date}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, cls=_FactorReportEncoder)

    # ── MD 报告 ──
    md_content = _generate_md_report(
        panel=panel,
        quality_results=quality_results or {},
        predictive_results=predictive_results or {},
        grouping_results=grouping_results or {},
        config=config,
        output_date=output_date,
    )

    md_path = output_root / f"report_{output_date}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


def copy_to_windows(
    linux_root: Path,
    windows_root: Path | None,
) -> bool:
    """将 Linux 端输出目录完整复制到 Windows 端。

    Returns
    -------
    bool
        True 如果复制成功，False 如果 Windows 目录为空或复制失败。
    """
    if windows_root is None:
        return False

    import shutil

    try:
        windows_root.mkdir(parents=True, exist_ok=True)
        # 复制所有文件
        for item in linux_root.iterdir():
            if item.is_file():
                shutil.copy2(item, windows_root / item.name)
        return True
    except Exception:
        return False