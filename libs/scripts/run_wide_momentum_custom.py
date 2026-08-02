"""宽动量基线回测 — 通用引擎。

通过 --config 指定配置文件，引擎只负责执行，不包含业务配置。

用法：
    cd /home/gouzi/projects/invest
    uv run python libs/scripts/run_wide_momentum_custom.py [--config <module>]
"""
from __future__ import annotations

import argparse
import hashlib
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
    RankFilter,
    StopRuleSpec,
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
SHARED_STOP_RULES: tuple[StopRuleSpec, ...] = ()
RANKING_FACTOR_CANDIDATES: tuple = ()
IC_WINDOW: int = 120
IC_SELECTION_MODE: str = "icir"
_output_base_dir: str = ""
_title: str = ""
_basename_tag: str = ""
_start_date: str = ""
_end_date: str = ""

# 运行时变量（fork 继承用）
prepared = None
_output_root: Path | None = None
_current_ranking: object = None
_current_pipeline: tuple = ()
_current_filters: tuple[ThresholdFilter, ...] = ()

# 跨组并行（fork 继承用）
_cross_group_parallel: bool = False
_prepared_map: dict[str, object] = {}       # group_label → prepared universe
_output_root_map: dict[str, Path] = {}       # group_label → output root dir
_group_ts_map: dict[str, str] = {}           # group_label → timestamp


def _build_output_basename(tag: str = "") -> str:
    """根据当前 GRID 参数生成输出目录名称。

    tag 非空时用 tag 替代组数标记；否则回退到 "{n}g"。
    """
    parts: list[str] = ["wide_momentum"]

    # 标识：优先使用 tag，否则组数
    if tag:
        parts.append(tag)
    else:
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

    # stop_rules
    if SHARED_STOP_RULES:
        sl_names = [s.name for s in SHARED_STOP_RULES if s.name]
        if sl_names:
            parts.append("sl_" + "_".join(sl_names))

    # 日期
    parts.append(datetime.now().strftime("%Y%m%d"))

    return "_".join(parts)


def _make_prepare_key(
    ranking_factor: object,
    factor_pipeline: tuple,
    ranking_factor_candidates: tuple,
    ic_window: int,
    ic_selection_mode: str,
    start_date: str,
    end_date: str,
) -> str:
    """为 prepare 阶段生成去重 key，相同 key 的组可共享 universe。

    仅包含实际影响 prepare_universe 结果的字段，builtin_filters 和
    cross_sectional_filters 不在其中（它们只在信号生成阶段被消费）。
    """
    parts = [
        "rf:" + str(id(ranking_factor)),
        "fp:" + ",".join(str(id(f)) for f in factor_pipeline),
        "rc:" + ",".join(str(id(f)) for f in ranking_factor_candidates),
        f"icw:{ic_window}",
        f"icsm:{ic_selection_mode}",
        f"sd:{start_date}",
        f"ed:{end_date}",
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _resolve_date_pairs(
    cfg_start_dates: tuple[str, ...] | None,
    cfg_end_dates: tuple[str, ...] | None,
    cfg_start_date: str,
    cfg_end_date: str,
    cli_start_dates: list[str] | None,
    cli_end_dates: list[str] | None,
) -> list[tuple[str, str]]:
    """解析最终的 (start_date, end_date) 对列表。

    优先级: 命令行 > 配置文件多日期列表 > 配置文件单日期
    """
    if cli_start_dates is not None:
        starts = cli_start_dates
        ends = cli_end_dates if cli_end_dates is not None else [cfg_end_date] * len(starts)
    elif cfg_start_dates is not None:
        starts = list(cfg_start_dates)
        ends = list(cfg_end_dates) if cfg_end_dates is not None else [cfg_end_date] * len(starts)
    else:
        return [(cfg_start_date, cfg_end_date)]

    if len(starts) != len(ends):
        raise ValueError(
            f"起始日期数量 ({len(starts)}) 与结束日期数量 ({len(ends)}) 不一致"
        )
    return list(zip(starts, ends))

def main(config_module: str, cli_start_dates: list[str] | None = None, cli_end_dates: list[str] | None = None) -> None:
    """主流程：逐组 prepare → grid search → 落盘。"""
    global \
        prepared, _output_root, _current_ranking, _current_pipeline, _current_filters, \
        _cross_group_parallel, _prepared_map, _output_root_map, _group_ts_map, \
        GROUPS, GRID_TOP_N, GRID_MIN_MOMENTUM, GRID_CLUSTER_MAX_PER_GROUP, \
        GRID_REBALANCE_INTERVAL, GRID_EXCLUDE_BONDS, GRID_HOLD_OVERLAP, \
        GRID_WEIGHT_ALLOCATOR, GRID_MAX_WORKERS, PERIOD_FREQ, CUSTOM_PERIODS, \
        SHARED_PIPELINE, SHARED_STOP_RULES, RANKING_FACTOR_CANDIDATES, IC_WINDOW, IC_SELECTION_MODE, \
        _output_base_dir, _title, _basename_tag, _start_date, _end_date

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
    _cross_group_parallel   = getattr(cfg, "CROSS_GROUP_PARALLEL", False)
    PERIOD_FREQ             = getattr(cfg, "PERIOD_FREQ", None)
    CUSTOM_PERIODS          = getattr(cfg, "CUSTOM_PERIODS", None)
    SHARED_PIPELINE         = cfg.SHARED_PIPELINE
    SHARED_STOP_RULES       = getattr(cfg, "SHARED_STOP_RULES", ())
    RANKING_FACTOR_CANDIDATES = getattr(cfg, "RANKING_FACTOR_CANDIDATES", ())
    IC_WINDOW               = getattr(cfg, "IC_WINDOW", 120)
    IC_SELECTION_MODE       = getattr(cfg, "IC_SELECTION_MODE", "icir")
    _output_base_dir        = cfg.OUTPUT_BASE_DIR
    _title                  = cfg.TITLE
    _basename_tag           = getattr(cfg, "BASENAME_TAG", "")
    _start_date             = cfg.START_DATE
    _end_date               = cfg.END_DATE

    # ── 解析日期对（命令行 > 配置文件多日期 > 单日期）──
    _cfg_start_dates = getattr(cfg, "START_DATES", None)
    _cfg_end_dates   = getattr(cfg, "END_DATES", None)
    date_pairs = _resolve_date_pairs(
        cfg_start_dates=_cfg_start_dates,
        cfg_end_dates=_cfg_end_dates,
        cfg_start_date=cfg.START_DATE,
        cfg_end_date=cfg.END_DATE,
        cli_start_dates=cli_start_dates,
        cli_end_dates=cli_end_dates,
    )
    _multi_date = len(date_pairs) > 1

    # ── 执行 ──
    symbols = getattr(cfg, "SYMBOLS", None) or ETF_INDEX_MAP.get_all_symbols()
    group_count = len(GROUPS)
    print("=" * 60)
    print(f"{_title}（{group_count} 组）")
    print("=" * 60)
    print(f"  标的池:         {len(symbols)} 标的 (ETF_INDEX_MAP)")
    if _multi_date:
        print(f"  回测区间:       {len(date_pairs)} 组日期对:")
        for _sd, _ed in date_pairs:
            print(f"                  {_sd} → {_ed}")
    else:
        print(f"  回测区间:       {date_pairs[0][0]} → {date_pairs[0][1]}")
    print(f"  持仓数量:       Top {GRID_TOP_N}")
    print(f"  权重方案:       {[getattr(a, '__name__', str(a)) for a in GRID_WEIGHT_ALLOCATOR]}")
    print(f"  调仓频率:       {GRID_REBALANCE_INTERVAL} 日")
    print(f"  债券剔除:       {GRID_EXCLUDE_BONDS}")
    print(f"  集群约束:       {GRID_CLUSTER_MAX_PER_GROUP}")
    print(f"  hold_overlap:   {GRID_HOLD_OVERLAP}")
    print()

    output_base = Path(_output_base_dir) / _build_output_basename(tag=_basename_tag)

    all_summaries: list[dict] = []

    for _date_idx, (_sd, _ed) in enumerate(date_pairs):
        _start_date = _sd
        _end_date   = _ed

        # ── 重置每轮日期对的状态 ──
        _prepared_map.clear()
        _output_root_map.clear()
        _group_ts_map.clear()

        print(f"\n{'#' * 60}")
        print(f"### 日期对 [{_date_idx + 1}/{len(date_pairs)}]: {_sd} → {_ed}")
        print(f"{'#' * 60}")

        if _cross_group_parallel:
            # ════════════════════════════════════════════════════════════════
            # 跨组并行模式
            # ════════════════════════════════════════════════════════════════
            print(f"\n{'=' * 60}")
            print(f"跨组并行模式: {len(GROUPS)} 组 × grid 变体统一调度")
            print(f"{'=' * 60}")

            # ── 阶段 1: 串行准备所有组的 universe（按 prepare-key 去重）──
            _all_jobs: list[tuple[str, object, tuple, tuple, tuple, tuple]] = []
            _grid_combos_count = 0
            _prepare_cache: dict[str, object] = {}          # prepare_key → prepared universe
            _prepare_source: dict[str, str] = {}             # prepare_key → 首次计算的 group_label
            _prepare_count = 0                                # 实际 prepare 次数统计

            for group_idx, group_entry in enumerate(GROUPS, 1):
                group_label = group_entry[0]
                ranking_factor = group_entry[1]
                builtin_filters = group_entry[2]
                cross_sectional_filters = (
                    group_entry[3] if len(group_entry) >= 4 else ()
                )

                _output_root = output_base / f"wide_momentum_{group_label}"
                _output_root_map[group_label] = _output_root
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _group_ts_map[group_label] = _ts

                print(f"\n{'=' * 60}")
                print(f">>> [{group_idx}/{len(GROUPS)}] {group_label}")
                print(f"    排名因子: {ranking_factor}")
                print(f"    硬性过滤器: {[(b.field, b.operator, b.value) for b in builtin_filters]}")
                if cross_sectional_filters:
                    print(f"    横截面过滤器: {[rf.name or rf.factor.get_output_name() for rf in cross_sectional_filters]}")
                print(f"{'=' * 60}")

                # ── 去重：相同 prepare-key 的组共享 universe ──
                prepare_key = _make_prepare_key(
                    ranking_factor=ranking_factor,
                    factor_pipeline=SHARED_PIPELINE,
                    ranking_factor_candidates=RANKING_FACTOR_CANDIDATES,
                    ic_window=IC_WINDOW,
                    ic_selection_mode=IC_SELECTION_MODE,
                    start_date=_start_date,
                    end_date=_end_date,
                )

                if prepare_key in _prepare_cache:
                    _prepared_map[group_label] = _prepare_cache[prepare_key]
                    print(
                        f"[阶段 1] 复用 universe（与 {_prepare_source[prepare_key]} 相同），跳过 prepare"
                    )
                else:
                    _bootstrap_config = WideMomentumBaselineConfig(
                        ranking_factor=ranking_factor,
                        factor_pipeline=SHARED_PIPELINE,
                        builtin_filters=builtin_filters,
                        cross_sectional_filters=cross_sectional_filters,
                        start_date=_start_date,
                        end_date=_end_date,
                        ranking_factor_candidates=RANKING_FACTOR_CANDIDATES,
                        ic_window=IC_WINDOW,
                        ic_selection_mode=IC_SELECTION_MODE,
                    )

                    print(f"[阶段 1] 准备 shared universe ...", end=" ", flush=True)
                    pg = prepare_wide_momentum_universe(config=_bootstrap_config, symbols=symbols)
                    _prepare_cache[prepare_key] = pg
                    _prepare_source[prepare_key] = group_label
                    _prepared_map[group_label] = pg
                    _prepare_count += 1
                    print(
                        f"完成 ({len(pg.symbol_data_map)} 标的, "
                        f"{pg.start_date.date()} → {pg.end_date.date()})"
                    )

                # 收集该组的 grid 变体
                grid_combos = list(
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
                _grid_combos_count += len(grid_combos)
                pipeline = (ranking_factor,) + SHARED_PIPELINE
                for combo in grid_combos:
                    _all_jobs.append(
                        (group_label, ranking_factor, pipeline, builtin_filters, cross_sectional_filters, combo)
                    )

            # ── 阶段 2: 所有组的 grid 变体统一并行 ──
            print(f"\n{'=' * 60}")
            print(
                f"[阶段 1 完成] 实际 prepare {_prepare_count} 次，"
                f"复用 {len(GROUPS) - _prepare_count} 次（共 {len(GROUPS)} 组）"
            )
            print(f"[阶段 2] 跨组并行: {len(GROUPS)} 组 × grid 变体 = {len(_all_jobs)} 个任务")
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
            print(f"{'=' * 60}")

            group_summaries_map: dict[str, list[dict]] = {group_entry[0]: [] for group_entry in GROUPS}
            completed_count = 0

            _mp_ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=GRID_MAX_WORKERS, mp_context=_mp_ctx) as executor:
                futures_map = {}
                for job in _all_jobs:
                    group_label, rf, pipeline, bf, csf, combo = job
                    # args: (group_label, rf, pipeline, bf, csf, cluster_enabled, combo)
                    future = executor.submit(
                        _run_single_combo, (group_label, rf, pipeline, bf, csf, False, combo)
                    )
                    futures_map[future] = group_label

                for idx, future in enumerate(as_completed(futures_map), start=1):
                    group_label = futures_map[future]
                    grid_label, summaries = future.result()
                    group_summaries_map[group_label].extend(summaries)
                    all_summaries.extend(summaries)

                    for s in summaries:
                        s["group_label"] = group_label

                    completed_count += 1
                    s0 = summaries[0] if summaries else {}
                    print(
                        f"[{completed_count}/{len(_all_jobs)}] [{group_label}] {grid_label}: "
                        f"cum={s0.get('cumulative_return_pct', '?')}%, "
                        f"sharpe={s0.get('sharpe', '?')}, "
                        f"mdd={s0.get('max_drawdown_pct', '?')}%"
                    )

            # ── 按组保存 grid_summary ──
            for group_entry in GROUPS:
                group_label = group_entry[0]
                summaries = group_summaries_map[group_label]
                if summaries:
                    import pandas as pd
                    summary_df = pd.DataFrame(summaries)
                    cols = ["grid_label"] + [c for c in summary_df.columns if c != "grid_label"]
                    summary_df = summary_df[cols]
                    _root = _output_root_map[group_label]
                    _ts = _group_ts_map[group_label]
                    csv_path = _root / f"grid_summary_{_ts}.csv"
                    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    print(f"\n[{group_label}] Grid 汇总: {csv_path}")
                    print(f"  共 {len(summaries)} 行")

        else:
            # ════════════════════════════════════════════════════════════════
            # 逐组串行模式（原有行为，增加 prepare 去重）
            # ════════════════════════════════════════════════════════════════
            _serial_prepare_cache: dict[str, object] = {}   # prepare_key → prepared universe
            _serial_source: dict[str, str] = {}             # prepare_key → 首次计算的 group_label

            for group_idx, group_entry in enumerate(GROUPS, 1):
                # 向后兼容: 3 元素 → (label, ranking, builtins)
                #            4 元素 → (label, ranking, builtins, cross_sectional)
                group_label = group_entry[0]
                ranking_factor = group_entry[1]
                builtin_filters = group_entry[2]
                cross_sectional_filters: tuple[RankFilter, ...] = (
                    group_entry[3] if len(group_entry) >= 4 else ()
                )

                _current_ranking = ranking_factor
                _current_pipeline = (ranking_factor,) + SHARED_PIPELINE
                _current_filters = builtin_filters

                _output_root = output_base / f"wide_momentum_{group_label}"
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                print(f"\n{'=' * 60}")
                print(f">>> [{group_idx}/{len(GROUPS)}] {group_label}")
                print(f"    排名因子: {ranking_factor}")
                print(f"    硬性过滤器: {[(b.field, b.operator, b.value) for b in builtin_filters]}")
                if cross_sectional_filters:
                    print(f"    横截面过滤器: {[rf.name or rf.factor.get_output_name() for rf in cross_sectional_filters]}")
                print(f"    输出目录: {_output_root}")
                print(f"{'=' * 60}")

                # ── 阶段 1: 准备 universe（去重）──
                prepare_key = _make_prepare_key(
                    ranking_factor=ranking_factor,
                    factor_pipeline=SHARED_PIPELINE,
                    ranking_factor_candidates=RANKING_FACTOR_CANDIDATES,
                    ic_window=IC_WINDOW,
                    ic_selection_mode=IC_SELECTION_MODE,
                    start_date=_start_date,
                    end_date=_end_date,
                )

                if prepare_key in _serial_prepare_cache:
                    prepared = _serial_prepare_cache[prepare_key]
                    print(
                        f"\n[阶段 1] 复用 universe（与 {_serial_source[prepare_key]} 相同），跳过 prepare"
                    )
                else:
                    _bootstrap_config = WideMomentumBaselineConfig(
                        ranking_factor=ranking_factor,
                        factor_pipeline=SHARED_PIPELINE,
                        builtin_filters=builtin_filters,
                        cross_sectional_filters=cross_sectional_filters,
                        start_date=_start_date,
                        end_date=_end_date,
                        ranking_factor_candidates=RANKING_FACTOR_CANDIDATES,
                        ic_window=IC_WINDOW,
                        ic_selection_mode=IC_SELECTION_MODE,
                    )

                    print(f"\n[阶段 1] 准备 shared universe ...", end=" ", flush=True)
                    prepared = prepare_wide_momentum_universe(config=_bootstrap_config, symbols=symbols)
                    _serial_prepare_cache[prepare_key] = prepared
                    _serial_source[prepare_key] = group_label
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
                _shared_args = (ranking_factor, SHARED_PIPELINE, builtin_filters, cross_sectional_filters, False)

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
    _date_desc = f"覆盖 {len(date_pairs)} 组日期对" if _multi_date else ""
    print(f"全部完成，共 {len(all_summaries)} 行汇总 {_date_desc}")
    print(f"输出根目录: {output_base}")
    print(f"{'=' * 60}")

    # ── 生成 HTML 报告 ──
    if all_summaries:
        _ts_report = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_base / f"wide_momentum_report_{_ts_report}.html"
        print(f"\n生成 HTML 报告 → {report_path} ...", end=" ", flush=True)

        if _multi_date:
            _date_display = f"{len(date_pairs)} 组日期对 ({date_pairs[0][0]}~{date_pairs[-1][1]})"
        else:
            _date_display = f"{date_pairs[0][0]} → {date_pairs[0][1]}"

        grid_params = {
            "标的池": f"{len(symbols)} 标的",
            "回测区间": _date_display,
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

    两种调用模式（通过 args 长度自动区分）：
      逐组模式（6 元素）:
        (ranking_factor, factor_pipeline, builtin_filters, cross_sectional_filters,
         cluster_limit_enabled, combo)
      跨组模式（7 元素）:
        (group_label, ranking_factor, factor_pipeline, builtin_filters,
         cross_sectional_filters, cluster_limit_enabled, combo)
    """
    import os as _os

    if len(args) == 7:
        group_label, ranking_factor, factor_pipeline, builtin_filters, cross_sectional_filters, cluster_limit_enabled, combo = args
        _pg = _prepared_map[group_label]
        _out_root = _output_root_map[group_label]
    else:
        group_label = None
        ranking_factor, factor_pipeline, builtin_filters, cross_sectional_filters, cluster_limit_enabled, combo = args
        _pg = prepared
        _out_root = _output_root
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
    grid_parts.append(f"sd{_start_date.replace('-', '')}")
    grid_label = "_".join(grid_parts)

    _BOND_CLUSTERS = (43, 44)

    config = WideMomentumBaselineConfig(
        top_n_values=(top_n,),
        ranking_factor=ranking_factor,
        factor_pipeline=factor_pipeline,
        builtin_filters=builtin_filters,
        cross_sectional_filters=cross_sectional_filters,
        min_momentum_value=min_mom,
        rebalance_interval=rebal,
        cluster_limit_enabled=(cluster_limit_enabled and cluster_max > 0),
        cluster_max_per_group=cluster_max if cluster_max > 0 else 3,
        exclude_clusters=_BOND_CLUSTERS if exclude_bonds else (),
        hold_overlap=hold_overlap,
        period_freq=PERIOD_FREQ,
        custom_periods=CUSTOM_PERIODS,
        weight_allocator=weight_allocator,
        ranking_factor_candidates=RANKING_FACTOR_CANDIDATES,
        ic_window=IC_WINDOW,
        ic_selection_mode=IC_SELECTION_MODE,
        stop_rules=SHARED_STOP_RULES,
    )

    output_dir = _out_root / grid_label

    label_prefix = f"[{group_label}] " if group_label else ""
    print(
        f"  [pid={_os.getpid()}] {label_prefix}{grid_label} 开始...",
        flush=True,
    )
    result = run_wide_momentum_baseline_from_prepared(
        prepared=_pg, config=config,
    )
    save_wide_momentum_baseline_result(result=result, output_dir=output_dir)

    total_rebalances = 0
    for _tn, vr in sorted(result.variant_results.items(), key=lambda kv: kv[0]):
        total_rebalances += int(vr.summary.get("rebalance_count", 0))

    print(
        f"  [pid={_os.getpid()}] {label_prefix}{grid_label} 完成 "
        f"({len(result.variant_results)} 变体, 共 {total_rebalances} 次调仓)",
        flush=True,
    )

    summaries = []
    for _tn, vr in sorted(result.variant_results.items(), key=lambda kv: kv[0]):
        s = vr.summary
        s["grid_label"] = grid_label
        s["top_n"] = int(_tn)
        s["start_date"] = _start_date
        s["end_date"] = _end_date
        summaries.append(s)

    return grid_label, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="宽动量基线回测通用引擎")
    parser.add_argument(
        "--config",
        default="libs.scripts.wide_momentum_configs.pr20_replacement",
        help="配置模块路径（默认: pr20_replacement）",
    )
    parser.add_argument(
        "--start-dates",
        type=str,
        default=None,
        help="逗号分隔的多个起始日期，如 2018-01-01,2020-01-01（覆盖配置文件中的 START_DATE / START_DATES）",
    )
    parser.add_argument(
        "--end-dates",
        type=str,
        default=None,
        help="逗号分隔的多个结束日期，数量须与 --start-dates 一致（覆盖配置文件中的 END_DATE / END_DATES）",
    )
    args = parser.parse_args()

    _cli_starts = (
        [d.strip() for d in args.start_dates.split(",") if d.strip()]
        if args.start_dates
        else None
    )
    _cli_ends = (
        [d.strip() for d in args.end_dates.split(",") if d.strip()]
        if args.end_dates
        else None
    )
    main(args.config, cli_start_dates=_cli_starts, cli_end_dates=_cli_ends)