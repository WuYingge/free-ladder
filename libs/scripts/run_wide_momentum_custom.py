"""宽动量基线回测 — 通用引擎。

通过 --config 指定配置文件，引擎只负责执行，不包含业务配置。

用法：
    cd /home/gouzi/projects/invest
    uv run python libs/scripts/run_wide_momentum_custom.py [--config <module>]
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))           # 让 libs 作为包可见（配置模块需要）
sys.path.insert(0, str(PROJECT_ROOT / "libs"))  # 直接导入 backtesting 等

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    WideMomentumBaselineConfig,
    prepare_wide_momentum_universe,
    run_wide_momentum_baseline_from_prepared,
    save_wide_momentum_baseline_result,
)
from backtesting.html_report import generate_wide_momentum_html_report
from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP


# ====================================================================
# 配置变量占位（由 main() 从配置文件加载后赋值，fork 子进程继承）
# ====================================================================
GROUPS: list = []
GRID_TOP_N: tuple = ()
GRID_MIN_MOMENTUM: tuple = ()
GRID_CLUSTER_MAX_PER_GROUP: tuple = ()
GRID_REBALANCE_INTERVAL: tuple = ()
GRID_EXCLUDE_BONDS: tuple = ()
GRID_HOLD_OVERLAP: tuple = ()
GRID_WEIGHT_ALLOCATOR: tuple = ()
GRID_MAX_WORKERS: int | None = None
PERIOD_FREQ: str | None = None
CUSTOM_PERIODS: tuple | None = None
SHARED_PIPELINE: tuple = ()
_output_base_dir: str = ""
_title: str = ""
_start_date: str = ""
_end_date: str = ""

# 运行时变量（fork 继承用）
prepared = None
_output_root: Path | None = None
_current_ranking: object = None
_current_pipeline: tuple = ()
_current_filters: tuple[ThresholdFilter, ...] = ()


def _build_output_basename() -> str:
    """根据当前 GRID 参数生成输出目录名称。"""
    parts: list[str] = ["wide_momentum"]

    # 组数
    parts.append(f"{len(GROUPS)}g")

    # top_n：紧凑表示
    tops = sorted(GRID_TOP_N)
    if len(tops) == 1:
        parts.append(f"top{tops[0]}")
    elif tops == list(range(tops[0], tops[-1] + 1)):
        parts.append(f"top{tops[0]}-{tops[-1]}")
    else:
        parts.append("top" + "_".join(str(t) for t in tops))

    # rebalance
    rebals = sorted(GRID_REBALANCE_INTERVAL)
    if len(rebals) == 1:
        parts.append(f"rebal{rebals[0]}")
    elif rebals == list(range(rebals[0], rebals[-1] + 1)):
        parts.append(f"rebal{rebals[0]}-{rebals[-1]}")
    else:
        parts.append("rebal" + "_".join(str(r) for r in rebals))

    # 权重方案
    wt_names = [getattr(a, "__name__", "?") for a in GRID_WEIGHT_ALLOCATOR]
    if len(wt_names) <= 3:
        parts.append("wt_" + "_".join(wt_names))
    else:
        parts.append(f"wt{len(wt_names)}")

    # 债券剔除：仅当全部一致时才标记
    bonds_vals = set(GRID_EXCLUDE_BONDS)
    if bonds_vals == {True}:
        parts.append("nobond")
    elif bonds_vals == {False}:
        parts.append("withbond")

    # hold_overlap
    hold_vals = set(GRID_HOLD_OVERLAP)
    if hold_vals == {True}:
        parts.append("hold")

    # min_momentum（若非全部为 None）
    mom_vals = set(GRID_MIN_MOMENTUM)
    if mom_vals != {None}:
        parts.append("mom" + "_".join(str(m) for m in sorted(mom_vals, key=lambda x: (x is None, x))))

    # cluster_max
    cl_vals = set(GRID_CLUSTER_MAX_PER_GROUP)
    if cl_vals != {0}:
        parts.append("cl" + "_".join(str(c) for c in sorted(cl_vals)))

    # 日期
    parts.append(datetime.now().strftime("%Y%m%d"))

    return "_".join(parts)


def main(config_module: str) -> None:
    """主流程：逐组 prepare → grid search → 落盘。"""
    global \
        prepared, _output_root, _current_ranking, _current_pipeline, _current_filters, \
        GROUPS, GRID_TOP_N, GRID_MIN_MOMENTUM, GRID_CLUSTER_MAX_PER_GROUP, \
        GRID_REBALANCE_INTERVAL, GRID_EXCLUDE_BONDS, GRID_HOLD_OVERLAP, \
        GRID_WEIGHT_ALLOCATOR, GRID_MAX_WORKERS, PERIOD_FREQ, CUSTOM_PERIODS, \
        SHARED_PIPELINE, _output_base_dir, _title, _start_date, _end_date

    # ── 加载配置 ──
    cfg = importlib.import_module(config_module)

    GROUPS                  = cfg.GROUPS
    GRID_TOP_N              = cfg.GRID_TOP_N
    GRID_MIN_MOMENTUM       = cfg.GRID_MIN_MOMENTUM
    GRID_CLUSTER_MAX_PER_GROUP = cfg.GRID_CLUSTER_MAX_PER_GROUP
    GRID_REBALANCE_INTERVAL = cfg.GRID_REBALANCE_INTERVAL
    GRID_EXCLUDE_BONDS      = cfg.GRID_EXCLUDE_BONDS
    GRID_HOLD_OVERLAP       = cfg.GRID_HOLD_OVERLAP
    GRID_WEIGHT_ALLOCATOR   = cfg.WEIGHT_ALLOCATORS
    GRID_MAX_WORKERS        = getattr(cfg, "MAX_WORKERS", None)
    PERIOD_FREQ             = getattr(cfg, "PERIOD_FREQ", None)
    CUSTOM_PERIODS          = getattr(cfg, "CUSTOM_PERIODS", None)
    SHARED_PIPELINE         = cfg.SHARED_PIPELINE
    _output_base_dir        = cfg.OUTPUT_BASE_DIR
    _title                  = cfg.TITLE
    _start_date             = cfg.START_DATE
    _end_date               = cfg.END_DATE

    # ── 执行 ──
    symbols = ETF_INDEX_MAP.get_all_symbols()
    group_count = len(GROUPS)
    print("=" * 60)
    print(f"{_title}（{group_count} 组）")
    print("=" * 60)
    print(f"  标的池:         {len(symbols)} 标的 (ETF_INDEX_MAP)")
    print(f"  回测区间:       {_start_date} → {_end_date}")
    print(f"  持仓数量:       Top {GRID_TOP_N}")
    print(f"  权重方案:       {[getattr(a, '__name__', str(a)) for a in GRID_WEIGHT_ALLOCATOR]}")
    print(f"  调仓频率:       {GRID_REBALANCE_INTERVAL} 日")
    print(f"  债券剔除:       {GRID_EXCLUDE_BONDS}")
    print(f"  集群约束:       {GRID_CLUSTER_MAX_PER_GROUP}")
    print(f"  hold_overlap:   {GRID_HOLD_OVERLAP}")
    print()

    output_base = Path(_output_base_dir) / _build_output_basename()

    all_summaries: list[dict] = []

    for group_idx, (group_label, ranking_factor, builtin_filters) in enumerate(GROUPS, 1):
        _current_ranking = ranking_factor
        _current_pipeline = (ranking_factor,) + SHARED_PIPELINE
        _current_filters = builtin_filters

        _output_root = output_base / f"wide_momentum_{group_label}"
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'=' * 60}")
        print(f">>> [{group_idx}/{len(GROUPS)}] {group_label}")
        print(f"    排名因子: {ranking_factor}")
        print(f"    过滤器:   {[(b.field, b.operator, b.value) for b in builtin_filters]}")
        print(f"    输出目录: {_output_root}")
        print(f"{'=' * 60}")

        # ── 阶段 1: 准备 universe ──
        _bootstrap_config = WideMomentumBaselineConfig(
            ranking_factor=ranking_factor,
            factor_pipeline=SHARED_PIPELINE,
            builtin_filters=builtin_filters,
            start_date=_start_date,
            end_date=_end_date,
        )

        print(f"\n[阶段 1] 准备 shared universe ...", end=" ", flush=True)
        prepared = prepare_wide_momentum_universe(config=_bootstrap_config, symbols=symbols)
        print(
            f"完成 ({len(prepared.symbol_data_map)} 标的, "
            f"{prepared.start_date.date()} → {prepared.end_date.date()})"
        )

        # ── 阶段 2: Grid 变体并行 ──
        _grid_combos = list(
            itertools.product(
                GRID_TOP_N,
                GRID_MIN_MOMENTUM,
                GRID_CLUSTER_MAX_PER_GROUP,
                GRID_REBALANCE_INTERVAL,
                GRID_EXCLUDE_BONDS,
                GRID_HOLD_OVERLAP,
                range(len(GRID_WEIGHT_ALLOCATOR)),
            )
        )

        print(f"\n[阶段 2] Grid 变体: {len(_grid_combos)} 组合")
        print(f"    top_n:         {GRID_TOP_N}")
        print(f"    min_momentum:  {GRID_MIN_MOMENTUM}")
        print(f"    cluster_max:   {GRID_CLUSTER_MAX_PER_GROUP}")
        print(f"    rebalance:     {GRID_REBALANCE_INTERVAL}")
        print(f"    exclude_bonds: {GRID_EXCLUDE_BONDS}")
        print(f"    hold_overlap:  {GRID_HOLD_OVERLAP}")
        if CUSTOM_PERIODS:
            print(f"    custom_periods: {CUSTOM_PERIODS}")
        elif PERIOD_FREQ:
            print(f"    period_freq:   {PERIOD_FREQ}")
        print("=" * 60)

        # 打包轻量参数供 worker 使用
        _shared_args = (ranking_factor, SHARED_PIPELINE, builtin_filters, False)

        group_summaries: list[dict] = []
        futures_map: dict = {}

        _mp_ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=GRID_MAX_WORKERS, mp_context=_mp_ctx) as executor:
            for combo in _grid_combos:
                future = executor.submit(_run_single_combo, _shared_args + (combo,))
                futures_map[future] = combo

            for idx, future in enumerate(as_completed(futures_map), start=1):
                combo = futures_map[future]
                grid_label, summaries = future.result()
                group_summaries.extend(summaries)
                all_summaries.extend(summaries)

                # 补上组标签，方便后续 HTML 报告识别
                for s in summaries:
                    s["group_label"] = group_label

                s0 = summaries[0] if summaries else {}
                print(
                    f"[{idx}/{len(_grid_combos)}] {grid_label}: "
                    f"cum={s0.get('cumulative_return_pct', '?')}%, "
                    f"sharpe={s0.get('sharpe', '?')}, "
                    f"mdd={s0.get('max_drawdown_pct', '?')}%"
                )

        # ── 保存本组 grid_summary ──
        if group_summaries:
            import pandas as pd
            summary_df = pd.DataFrame(group_summaries)
            cols = ["grid_label"] + [c for c in summary_df.columns if c != "grid_label"]
            summary_df = summary_df[cols]
            csv_path = _output_root / f"grid_summary_{_ts}.csv"
            summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"\n[{group_label}] Grid 汇总: {csv_path}")
            print(f"  共 {len(_grid_combos)} 组合，{len(group_summaries)} 行")

    # ── 全部完成 ──
    print(f"\n{'=' * 60}")
    print(f"全部完成，共 {len(all_summaries)} 行汇总")
    print(f"输出根目录: {output_base}")
    print(f"{'=' * 60}")

    # ── 生成 HTML 报告 ──
    if all_summaries:
        _ts_report = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_base / f"wide_momentum_report_{_ts_report}.html"
        print(f"\n生成 HTML 报告 → {report_path} ...", end=" ", flush=True)

        grid_params = {
            "标的池": f"{len(symbols)} 标的",
            "回测区间": f"{_start_date} → {_end_date}",
            "Top N": GRID_TOP_N,
            "调仓频率(日)": GRID_REBALANCE_INTERVAL,
            "债券剔除": GRID_EXCLUDE_BONDS,
            "hold_overlap": GRID_HOLD_OVERLAP,
            "集群约束": GRID_CLUSTER_MAX_PER_GROUP,
            "min_momentum": GRID_MIN_MOMENTUM,
            "权重方案": [getattr(a, "__name__", str(a)) for a in GRID_WEIGHT_ALLOCATOR],
        }

        html = generate_wide_momentum_html_report(
            groups=GROUPS,
            all_summaries=all_summaries,
            output_base=output_base,
            grid_params=grid_params,
            title=_title,
        )
        report_path.write_text(html, encoding="utf-8")
        print(f"完成 ({report_path.stat().st_size // 1024} KB)")


def _run_single_combo(args):
    """单个 grid 组合的完整回测（供进程池并行）。

    只接收轻量参数。prepared / _output_root 等大对象由 fork 继承，不通过 pickle 传输。
    """
    import os as _os

    ranking_factor, factor_pipeline, builtin_filters, cluster_limit_enabled, combo = args
    top_n, min_mom, cluster_max, rebal, exclude_bonds, hold_overlap, alloc_idx = combo
    weight_allocator = GRID_WEIGHT_ALLOCATOR[alloc_idx]

    # 子目录标签
    grid_parts = [f"top{top_n}"]
    if min_mom is not None:
        grid_parts.append(f"mom{min_mom}".replace(".", "p"))
    if cluster_max > 0:
        grid_parts.append(f"cl{cluster_max}")
    if rebal != 5:
        grid_parts.append(f"rebal{rebal}")
    if exclude_bonds:
        grid_parts.append("no_bond")
    if hold_overlap:
        grid_parts.append("hold")
    _alloc_name = getattr(weight_allocator, "__name__", "")
    if _alloc_name and _alloc_name != "equal_weight_allocator":
        grid_parts.append(f"wt_{_alloc_name}")
    grid_label = "_".join(grid_parts)

    _BOND_CLUSTERS = (43, 44)

    config = WideMomentumBaselineConfig(
        top_n_values=(top_n,),
        ranking_factor=ranking_factor,
        factor_pipeline=factor_pipeline,
        builtin_filters=builtin_filters,
        min_momentum_value=min_mom,
        rebalance_interval=rebal,
        cluster_limit_enabled=(cluster_limit_enabled and cluster_max > 0),
        cluster_max_per_group=cluster_max if cluster_max > 0 else 3,
        exclude_clusters=_BOND_CLUSTERS if exclude_bonds else (),
        hold_overlap=hold_overlap,
        period_freq=PERIOD_FREQ,
        custom_periods=CUSTOM_PERIODS,
        weight_allocator=weight_allocator,
    )

    output_dir = _output_root / grid_label

    print(
        f"  [pid={_os.getpid()}] {grid_label} 开始...",
        flush=True,
    )
    result = run_wide_momentum_baseline_from_prepared(
        prepared=prepared, config=config,
    )
    save_wide_momentum_baseline_result(result=result, output_dir=output_dir)

    total_rebalances = 0
    for _tn, vr in sorted(result.variant_results.items(), key=lambda kv: kv[0]):
        total_rebalances += int(vr.summary.get("rebalance_count", 0))

    print(
        f"  [pid={_os.getpid()}] {grid_label} 完成 "
        f"({len(result.variant_results)} 变体, 共 {total_rebalances} 次调仓)",
        flush=True,
    )

    summaries = []
    for _tn, vr in sorted(result.variant_results.items(), key=lambda kv: kv[0]):
        s = vr.summary
        s["grid_label"] = grid_label
        s["top_n"] = int(_tn)
        summaries.append(s)

    return grid_label, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="宽动量基线回测通用引擎")
    parser.add_argument(
        "--config",
        default="libs.scripts.wide_momentum_configs.pr20_replacement",
        help="配置模块路径（默认: pr20_replacement）",
    )
    args = parser.parse_args()
    main(args.config)
