"""
因子分析编排器 (Factor Analysis Runner)

将面板构建、各层分析、绑图、报告生成串联为一条完整的分析流水线。

使用方式:
    >>> from factor_analysis.config import FactorAnalysisConfig
    >>> from factor_analysis.runner import run_factor_analysis
    >>> from factors.price_return import PriceReturn
    >>>
    >>> factor = PriceReturn(window=60)
    >>> config = FactorAnalysisConfig(factor=factor, layers=(1, 2, 3))
    >>> results = run_factor_analysis(config)

输出:
    - data/factors/{factor_name}/report_{date}.json
    - data/factors/{factor_name}/report_{date}.md
    - data/factors/{factor_name}/*.png（绑图）
    - 若 DEFAULT_WINDOWS_PATH 非空，同步复制到 Windows 端
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from factor_analysis.config import FactorAnalysisConfig
from factor_analysis.panel import build_factor_panel
from factor_analysis.forward_returns import compute_forward_returns
from factor_analysis.quality import run_quality_analysis
from factor_analysis.predictive import run_predictive_analysis
from factor_analysis.grouping import run_grouping_analysis
from factor_analysis.plotting import (
    plot_coverage,
    plot_distribution_bands,
    plot_autocorr_decay,
    plot_ic_decay,
    plot_rolling_ic,
    plot_ic_heatmap,
    plot_quantile_returns_bar,
    plot_quantile_cumret,
    plot_longshort_cumret,
    save_figure,
)
from factor_analysis.reporter import generate_and_save_reports, copy_to_windows


def run_factor_analysis(config: FactorAnalysisConfig) -> dict[str, Any]:
    """执行完整的单因子分析流水线。

    Parameters
    ----------
    config: 分析配置对象。至少需要设置 factor 字段。

    Returns
    -------
    dict
        {
            "panel": FactorPanel,
            "quality": dict | None,
            "predictive": dict | None,
            "grouping": dict | None,
            "output": {"linux_root": Path, "windows_root": Path | None, "files": list[str]},
        }

    Raises
    ------
    ValueError
        如果面板构建失败或配置无效。
    """
    # ── 0. 配置解析 ──────────────────────────────────────────────────────
    symbols = config.resolve_symbols()
    output_root = config.resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"因子: {config.factor.get_output_name()}")
    print(f"标的数: {len(symbols)}")
    print(f"输出目录: {output_root}")
    print(f"分析层: {config.layers}")

    # ── 1. 构建因子面板 ──────────────────────────────────────────────────
    print("\n[1/5] 构建截面因子面板...")
    panel = build_factor_panel(
        factor=config.factor,
        symbols=symbols,
        min_bars=config.min_bars,
        start_date=config.start_date,
        end_date=config.end_date,
        max_workers=config.max_workers,
    )
    print(f"  → {panel.n_symbols} 个有效标的, {panel.n_dates} 个交易日")
    print(f"  → 日期范围: {panel.date_range[0].date()} ~ {panel.date_range[1].date()}")
    print(f"  → 均值覆盖率: {panel.summary()['coverage_mean']:.2%}")
    if panel.errors:
        print(f"  → {len(panel.errors)} 个标的加载/计算失败")
    if panel.filtered_symbols:
        print(f"  → {len(panel.filtered_symbols)} 个标的因 min_bars 被过滤")

    # ── 2. 计算前向收益 ──────────────────────────────────────────────────
    need_layer_23 = 2 in config.layers or 3 in config.layers

    if need_layer_23:
        print("\n[2/5] 计算前向收益...")
        fwd_map = compute_forward_returns(panel.close_prices, config.forward_periods)
        for period in sorted(fwd_map.keys()):
            n_days = fwd_map[period].notna().sum().sum()
            print(f"  → {period}日持仓期: {n_days} 个有效收益观测")
    else:
        fwd_map = {}

    quality_results = None
    predictive_results = None
    grouping_results = None

    chart_paths: dict[str, str] = {}

    # ── 3. Layer 1: 因子质量 ─────────────────────────────────────────────
    if 1 in config.layers:
        print("\n[3/5] Layer 1: 因子质量分析...")
        quality_results = run_quality_analysis(panel)

        # 覆盖率
        cov = quality_results["coverage"]
        fig, _ = plot_coverage(cov, panel.factor_name)
        chart_paths["coverage.png"] = save_figure(fig, str(output_root / "coverage.png"))

        # 分位数带
        dp = quality_results["daily_percentiles"]
        fig, _ = plot_distribution_bands(dp, panel.factor_name)
        chart_paths["distribution_bands.png"] = save_figure(fig, str(output_root / "distribution_bands.png"))

        # 自相关
        ac = quality_results["autocorr"]
        fig, _ = plot_autocorr_decay(ac, panel.factor_name)
        chart_paths["autocorr_decay.png"] = save_figure(fig, str(output_root / "autocorr_decay.png"))

        print(f"  → 分布: mean={quality_results['distribution_stats'].get('mean', float('nan')):.4f}")
        if not ac.empty:
            print(f"  → 自相关 lag1: {ac.iloc[0]['mean_autocorr']:.4f}")

    # ── 4. Layer 2: 预测力 ───────────────────────────────────────────────
    if 2 in config.layers:
        print("\n[4/5] Layer 2: 预测力分析...")
        predictive_results = run_predictive_analysis(
            panel=panel,
            fwd_returns_map=fwd_map,
            rolling_ic_window=config.rolling_ic_window,
            param_grid=config.param_grid,
            factor_cls=type(config.factor) if config.param_grid else None,
            symbols=symbols if config.param_grid else None,
            forward_periods=config.forward_periods,
            min_bars=config.min_bars,
            max_workers=config.max_workers,
        )

        rank_ic_map = predictive_results["rank_ic"]
        for period in sorted(rank_ic_map.keys()):
            s = rank_ic_map[period]["summary"]
            print(f"  → Rank IC ({period}d): mean={s['mean']:.6f}, std={s['std']:.6f}, IR={s['ir']:.4f}")

        # IC 衰减
        decay = predictive_results["ic_decay"]
        fig, _ = plot_ic_decay(decay, panel.factor_name)
        chart_paths["ic_decay.png"] = save_figure(fig, str(output_root / "ic_decay.png"))

        # 滚动 IC — 每个持仓期各生成一张
        for period in sorted(predictive_results["rolling_ic"].keys()):
            ric = predictive_results["rolling_ic"][period]
            title = f"{panel.factor_name} ({period}d)"
            fig, _ = plot_rolling_ic(ric, title)
            key = f"rolling_ic_{period}d.png"
            chart_paths[key] = save_figure(fig, str(output_root / key))

        # 参数网格热力图
        pg = predictive_results.get("param_grid")
        if pg and not pg.get("matrix", pd.DataFrame()).empty:
            fig, _ = plot_ic_heatmap(pg["matrix"], panel.factor_name)
            chart_paths["ic_matrix.png"] = save_figure(fig, str(output_root / "ic_matrix.png"))
            # 同时保存 CSV
            pg["matrix"].to_csv(output_root / "ic_matrix.csv")

    # ── 5. Layer 3: 分组检验 ─────────────────────────────────────────────
    if 3 in config.layers:
        print("\n[5/5] Layer 3: 分组检验...")
        grouping_results = run_grouping_analysis(
            panel=panel,
            fwd_returns_map=fwd_map,
            n_quantiles=config.n_quantiles,
        )

        # 取默认持仓期（优先 20 日，fallback 取第一个可用）
        _gr_periods = sorted(grouping_results.keys())
        _gr_default = 20 if 20 in _gr_periods else (_gr_periods[0] if _gr_periods else None)

        # 分位组收益 — 遍历所有 period
        for period in _gr_periods:
            gr = grouping_results[period]
            q_summary = gr["quantile_summary"]
            if q_summary.empty:
                continue
            q_labels = q_summary.index.tolist()
            top_label = q_labels[-1]
            bottom_label = q_labels[0]
            top_ret = q_summary.loc[top_label, "mean_return"] if "mean_return" in q_summary.columns else float("nan")
            bottom_ret = q_summary.loc[bottom_label, "mean_return"] if "mean_return" in q_summary.columns else float("nan")
            print(f"  → 分位组 ({period}d): {bottom_label}={bottom_ret:.6f}, {top_label}={top_ret:.6f}, spread={top_ret - bottom_ret:.6f}")

        # Long-Short — 遍历所有 period
        for period in _gr_periods:
            gr = grouping_results[period]
            ls = gr.get("longshort", {})
            if ls:
                print(f"  → Long-Short ({period}d): ann_ret={ls.get('annualised_return', float('nan')):.4%}, sharpe={ls.get('sharpe')}")

        # 单调性 — 遍历所有 period
        for period in _gr_periods:
            gr = grouping_results[period]
            mono = gr.get("monotonicity", {})
            if mono:
                print(f"  → 严格单调 ({period}d): {mono.get('strict_monotonic_ratio', float('nan')):.2%}")

        # 图表和 CSV：每个持仓期各生成一套
        for period in _gr_periods:
            gr_period = grouping_results[period]
            q_summary = gr_period["quantile_summary"]
            q_cumret = gr_period["quantile_cumret"]
            ls = gr_period.get("longshort", {})

            suffix = f"_{period}d"
            title = f"{panel.factor_name} ({period}d)"

            # 柱状图
            fig, _ = plot_quantile_returns_bar(q_summary, title)
            key = f"quantile_returns_bar{suffix}.png"
            chart_paths[key] = save_figure(fig, str(output_root / key))

            # 累计收益
            fig, _ = plot_quantile_cumret(q_cumret, title)
            key = f"quantile_cumret{suffix}.png"
            chart_paths[key] = save_figure(fig, str(output_root / key))

            # Long-Short 曲线
            if ls:
                fig, _ = plot_longshort_cumret(ls, title)
                key = f"longshort_cumret{suffix}.png"
                chart_paths[key] = save_figure(fig, str(output_root / key))

            # 保存 CSV
            gr_period["quantile_returns"].to_csv(output_root / f"quantile_returns{suffix}.csv")
            if ls:
                ls_series = ls.get("ls_series")
                if ls_series is not None and hasattr(ls_series, "to_csv"):
                    ls_series.to_csv(output_root / f"longshort_returns{suffix}.csv")

    # ── 6. 生成报告 ──────────────────────────────────────────────────────
    print("\n生成报告...")
    json_path, md_path = generate_and_save_reports(
        panel=panel,
        quality_results=quality_results,
        predictive_results=predictive_results,
        grouping_results=grouping_results,
        config=config,
        chart_paths=chart_paths,
    )
    print(f"  → JSON: {json_path}")
    print(f"  → MD:   {md_path}")

    # ── 7. Windows 端同步 ────────────────────────────────────────────────
    windows_root = config.resolve_windows_output_root()
    if windows_root:
        ok = copy_to_windows(output_root, windows_root)
        print(f"  → Windows 同步: {'成功' if ok else '失败'} ({windows_root})")

    # 收集输出文件列表
    output_files = sorted([
        f.name for f in output_root.iterdir() if f.is_file()
    ])

    return {
        "panel": panel,
        "quality": quality_results,
        "predictive": predictive_results,
        "grouping": grouping_results,
        "output": {
            "linux_root": output_root,
            "windows_root": windows_root,
            "files": output_files,
        },
    }
