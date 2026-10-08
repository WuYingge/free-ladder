"""股票动量基线回测 — 通用引擎（股票池版）。

模仿 run_wide_momentum_custom.py，仅替换数据源为 A 股股票：
  - 数据加载: get_stock_data_by_symbol (data/stock_data/*.csv, 后复权价)
  - 默认标的池: STOCK_LIST.get_all_symbol()（约 5300+ 只）
  - grid 维度不含 cluster/债券（股票无 cluster 概念）
  - 新股随数据起点自动加入候选池（min_listing_days 由配置控制，默认 0）

内存治理（全量股票池必须，否则 fork 后按 CPU 核数开 worker 必 OOM）：
  - worker 数按可用内存自动收缩，配置 MAX_WORKERS 仅作上限
  - 无 RankFilter/自定义过滤器时释放每标的的 etf_data 冗余副本
  - fork 前 gc + malloc_trim 归还 prepare 阶段的碎片页

用法：
    cd /home/gouzi/projects/invest
    uv run python libs/scripts/run_stock_momentum_custom.py [--config <module>]
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib
import itertools
import multiprocessing
import os
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
    prepare_wide_momentum_universe_from_etf_data_map,
    run_wide_momentum_baseline_from_prepared,
    save_wide_momentum_baseline_result,
)
from backtesting.html_report import generate_wide_momentum_html_report
from data_manager.providers.stock_list_provider import STOCK_LIST
from data_manager.stock_data_manager import get_stock_data_by_symbol


# ====================================================================
# 配置变量占位（由 main() 从配置文件加载后赋值，fork 子进程继承）
# ====================================================================
GROUPS: list = []
GRID_TOP_N: tuple = ()
GRID_MIN_MOMENTUM: tuple = ()
GRID_REBALANCE_INTERVAL: tuple = ()
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
MIN_LISTING_DAYS: int = 0
COMMISSION: float = 0.00025
CASH: float = 100000.0
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
    parts: list[str] = ["stock_momentum"]

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

    # hold_overlap
    hold_vals = set(GRID_HOLD_OVERLAP)
    if hold_vals == {True}:
        parts.append("hold")

    # min_momentum（若非全部为 None）
    mom_vals = set(GRID_MIN_MOMENTUM)
    if mom_vals != {None}:
        parts.append("mom" + "_".join(str(m) for m in sorted(mom_vals, key=lambda x: (x is None, x))))

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


def _prepare_stock_universe(config: WideMomentumBaselineConfig, symbols: list[str]):
    """从 data/stock_data 加载股票数据并准备共享 universe。

    加载失败的标的（文件缺失/损坏）静默跳过；cluster 全部标记为 -1
    （未分类，不参与集群约束，避免依赖 ETF 的 cluster 表）。
    """
    stock_data_map = {}
    load_errors = 0
    for symbol in symbols:
        try:
            stock_data_map[symbol] = get_stock_data_by_symbol(symbol)
        except Exception:
            load_errors += 1

    print(f"  数据加载: {len(stock_data_map)} 标的成功, {load_errors} 失败")

    return prepare_wide_momentum_universe_from_etf_data_map(
        etf_data_map=stock_data_map,
        config=config,
        cluster_lookup=lambda s: -1,
    )


# ====================================================================
# 内存治理（全量 5300+ 只股票的 universe 可达 2.5GB+，fork 后每个
# worker 各持一份，默认按 CPU 核数开 worker 必然 OOM，这里做三件事）：
#   1. 按可用内存自动收缩 grid worker 数（配置 MAX_WORKERS 仅作上限）
#   2. 安全前提下释放 SymbolBaselineData.etf_data（每标的第二份完整
#      DataFrame，仅 RankFilter/自定义候选过滤器需要它）
#   3. fork 进程池前 gc + malloc_trim，把 prepare 阶段的碎片页还给 OS
# ====================================================================


def _groups_need_etf_data(groups: list) -> bool:
    """判断任一组的配置是否需要 SymbolBaselineData.etf_data。

    需要的情况：
      - cross_sectional_filters（RankFilter 等）非空
      - builtin_filters 中有非 ThresholdFilter 的自定义过滤器
    （WideMomentumBaselineConfig.candidate_filters 由脚本构造 config 时
    始终传空，不在此判断；仍保留 getattr 检查兜底。）
    """
    for group_entry in groups:
        if len(group_entry) >= 4 and group_entry[3]:
            return True
        for f in group_entry[2]:
            if not isinstance(f, ThresholdFilter):
                return True
    return False


def _estimate_universe_bytes(prepared: object) -> int:
    """统计 universe 内 frame + etf_data.data 的总字节数（近似 RSS 贡献）。

    5300 只标的全量遍历约 0.5-1s，仅在 prepare 后调用一次，可接受。
    """
    total = 0
    for sd in prepared.symbol_data_map.values():
        total += int(sd.frame.memory_usage(deep=False).sum())
        if sd.etf_data is not None:
            total += int(sd.etf_data.data.memory_usage(deep=False).sum())
    return total


def _release_unused_etf_data(prepared: object) -> int:
    """把 sd.etf_data 置 None，释放每标的第二份完整 DataFrame（幂等）。

    仅当全部组的配置都不需要 etf_data 时由调用方触发；若后续代码
    真的用到它会得到明确的 ValueError（库内有断言兜底）。
    """
    released = 0
    for sd in prepared.symbol_data_map.values():
        if sd.etf_data is not None:
            sd.etf_data = None
            released += 1
    return released


def _malloc_trim() -> None:
    """把 glibc 的空闲堆页还给 OS（Linux only），降低 fork 继承的 RSS。"""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _compute_grid_workers(
    cfg_max_workers: int | None,
    universe_bytes: int,
    mem_fraction: float = 0.75,
    headroom_bytes: int = 1 << 30,
) -> tuple[int, str]:
    """按可用内存计算安全的 grid worker 数。

    每个 fork worker 需要一份 universe（回测逐日读取会破坏 COW，
    按整份计）+ 回测临时对象（candidates / 调仓日志等，headroom）。
    配置的 MAX_WORKERS 只作为上限，超过内存预算时自动收缩。
    """
    import psutil

    avail = psutil.virtual_memory().available
    budget = int(avail * mem_fraction)
    per_worker = universe_bytes + headroom_bytes
    auto = max(1, budget // per_worker) if per_worker > 0 else 1
    requested = cfg_max_workers or (os.cpu_count() or 1)
    workers = max(1, min(int(requested), auto))
    note = (
        f"内存预算 {budget / 1e9:.1f}GB（可用 {avail / 1e9:.1f}GB × {mem_fraction}），"
        f"每 worker 约 {(universe_bytes + headroom_bytes) / 1e9:.1f}GB"
        f"（universe {universe_bytes / 1e9:.1f}GB + 临时 {headroom_bytes / 1e9:.1f}GB）"
    )
    if auto < requested:
        note += f"，已从配置/核数 {requested} 收缩至 {workers}"
    return workers, note


def _prepare_worker_pool(
    universe_bytes: int,
    cfg_max_workers: int | None,
) -> tuple[ProcessPoolExecutor, int]:
    """fork 进程池前的统一收尾：gc → malloc_trim → 按内存算 worker 数。

    返回 (executor, workers)。
    """
    gc.collect()
    _malloc_trim()
    workers, note = _compute_grid_workers(
        cfg_max_workers=cfg_max_workers, universe_bytes=universe_bytes
    )
    print(f"  worker 数: {workers}（{note}）", flush=True)
    _mp_ctx = multiprocessing.get_context("fork")
    return (
        ProcessPoolExecutor(max_workers=workers, mp_context=_mp_ctx),
        workers,
    )


def main(config_module: str, cli_start_dates: list[str] | None = None, cli_end_dates: list[str] | None = None) -> None:
    """主流程：逐组 prepare → grid search → 落盘。"""
    global \
        prepared, _output_root, _current_ranking, _current_pipeline, _current_filters, \
        _cross_group_parallel, _prepared_map, _output_root_map, _group_ts_map, \
        GROUPS, GRID_TOP_N, GRID_MIN_MOMENTUM, GRID_REBALANCE_INTERVAL, \
        GRID_HOLD_OVERLAP, GRID_WEIGHT_ALLOCATOR, GRID_MAX_WORKERS, PERIOD_FREQ, \
        CUSTOM_PERIODS, SHARED_PIPELINE, SHARED_STOP_RULES, RANKING_FACTOR_CANDIDATES, \
        IC_WINDOW, IC_SELECTION_MODE, MIN_LISTING_DAYS, COMMISSION, CASH, \
        _output_base_dir, _title, _basename_tag, _start_date, _end_date

    # ── 加载配置 ──
    cfg = importlib.import_module(config_module)

    GROUPS                  = cfg.GROUPS
    GRID_TOP_N              = cfg.GRID_TOP_N
    GRID_MIN_MOMENTUM       = cfg.GRID_MIN_MOMENTUM
    GRID_REBALANCE_INTERVAL = cfg.GRID_REBALANCE_INTERVAL
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
    MIN_LISTING_DAYS        = getattr(cfg, "MIN_LISTING_DAYS", 0)
    COMMISSION              = getattr(cfg, "COMMISSION", 0.00025)
    CASH                    = getattr(cfg, "CASH", 100000.0)
    _output_base_dir        = cfg.OUTPUT_BASE_DIR
    _title                  = cfg.TITLE
    _basename_tag           = getattr(cfg, "BASENAME_TAG", "")
    _start_date             = cfg.START_DATE
    _end_date               = cfg.END_DATE

    # ── 内存治理：任一组的配置需要 etf_data（RankFilter/自定义过滤器）？──
    _needs_etf_data = (
        _groups_need_etf_data(GROUPS)
        or bool(getattr(cfg, "CANDIDATE_FILTERS", None))
    )

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
    symbols = getattr(cfg, "SYMBOLS", None) or STOCK_LIST.get_all_symbol()
    group_count = len(GROUPS)
    print("=" * 60)
    print(f"{_title}（{group_count} 组）")
    print("=" * 60)
    print(f"  标的池:         {len(symbols)} 标的 (STOCK_LIST)")
    print(f"  min_listing_days: {MIN_LISTING_DAYS}")
    if _multi_date:
        print(f"  回测区间:       {len(date_pairs)} 组日期对:")
        for _sd, _ed in date_pairs:
            print(f"                  {_sd} → {_ed}")
    else:
        print(f"  回测区间:       {date_pairs[0][0]} → {date_pairs[0][1]}")
    print(f"  持仓数量:       Top {GRID_TOP_N}")
    print(f"  权重方案:       {[getattr(a, '__name__', str(a)) for a in GRID_WEIGHT_ALLOCATOR]}")
    print(f"  调仓频率:       {GRID_REBALANCE_INTERVAL} 日")
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
            _all_jobs: list[tuple[str, object, tuple, tuple, tuple]] = []
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

                _output_root = output_base / f"stock_momentum_{group_label}"
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
                        min_listing_days=MIN_LISTING_DAYS,
                        ranking_factor_candidates=RANKING_FACTOR_CANDIDATES,
                        ic_window=IC_WINDOW,
                        ic_selection_mode=IC_SELECTION_MODE,
                    )

                    print(f"[阶段 1] 准备 shared universe ...", end=" ", flush=True)
                    pg = _prepare_stock_universe(config=_bootstrap_config, symbols=symbols)
                    _prepare_cache[prepare_key] = pg
                    _prepare_source[prepare_key] = group_label
                    _prepared_map[group_label] = pg
                    _prepare_count += 1
                    print(
                        f"完成 ({len(pg.symbol_data_map)} 标的, "
                        f"{pg.start_date.date()} → {pg.end_date.date()})"
                    )
                    _universe_bytes = _estimate_universe_bytes(pg)
                    if not _needs_etf_data:
                        _released = _release_unused_etf_data(pg)
                        print(
                            f"  内存: universe ≈ {_universe_bytes/1e9:.2f} GB"
                            f"（释放 etf_data 冗余 {_released} 只 → "
                            f"仅 frame ≈ {_estimate_universe_bytes(pg)/1e9:.2f} GB）"
                        )
                    else:
                        print(
                            f"  内存: universe ≈ {_universe_bytes/1e9:.2f} GB"
                            f"（保留 etf_data，供 RankFilter/自定义过滤器使用）"
                        )

                # 收集该组的 grid 变体
                grid_combos = list(
                    itertools.product(
                        GRID_TOP_N,
                        GRID_MIN_MOMENTUM,
                        GRID_REBALANCE_INTERVAL,
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
            print(f"    rebalance:     {GRID_REBALANCE_INTERVAL}")
            print(f"    hold_overlap:  {GRID_HOLD_OVERLAP}")
            if CUSTOM_PERIODS:
                print(f"    custom_periods: {CUSTOM_PERIODS}")
            elif PERIOD_FREQ:
                print(f"    period_freq:   {PERIOD_FREQ}")
            print(f"{'=' * 60}")

            group_summaries_map: dict[str, list[dict]] = {group_entry[0]: [] for group_entry in GROUPS}
            completed_count = 0

            # 阶段 2 前按内存预算计算 worker 数（fork 前 gc + malloc_trim）
            _universe_bytes = _estimate_universe_bytes(next(iter(_prepare_cache.values())))
            executor, _workers = _prepare_worker_pool(
                universe_bytes=_universe_bytes, cfg_max_workers=GRID_MAX_WORKERS,
            )
            with executor:
                futures_map = {}
                for job in _all_jobs:
                    group_label, rf, pipeline, bf, csf, combo = job
                    future = executor.submit(
                        _run_single_combo, (group_label, rf, pipeline, bf, csf, combo)
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

                _output_root = output_base / f"stock_momentum_{group_label}"
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
                    _universe_bytes = _estimate_universe_bytes(prepared)
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
                        min_listing_days=MIN_LISTING_DAYS,
                        ranking_factor_candidates=RANKING_FACTOR_CANDIDATES,
                        ic_window=IC_WINDOW,
                        ic_selection_mode=IC_SELECTION_MODE,
                    )

                    print(f"\n[阶段 1] 准备 shared universe ...", end=" ", flush=True)
                    prepared = _prepare_stock_universe(config=_bootstrap_config, symbols=symbols)
                    _serial_prepare_cache[prepare_key] = prepared
                    _serial_source[prepare_key] = group_label
                    print(
                        f"完成 ({len(prepared.symbol_data_map)} 标的, "
                        f"{prepared.start_date.date()} → {prepared.end_date.date()})"
                    )
                    _universe_bytes = _estimate_universe_bytes(prepared)
                    if not _needs_etf_data:
                        _released = _release_unused_etf_data(prepared)
                        print(
                            f"  内存: universe ≈ {_universe_bytes/1e9:.2f} GB"
                            f"（释放 etf_data 冗余 {_released} 只 → "
                            f"仅 frame ≈ {_estimate_universe_bytes(prepared)/1e9:.2f} GB）"
                        )
                    else:
                        print(
                            f"  内存: universe ≈ {_universe_bytes/1e9:.2f} GB"
                            f"（保留 etf_data，供 RankFilter/自定义过滤器使用）"
                        )

                # ── 阶段 2: Grid 变体并行 ──
                _grid_combos = list(
                    itertools.product(
                        GRID_TOP_N,
                        GRID_MIN_MOMENTUM,
                        GRID_REBALANCE_INTERVAL,
                        GRID_HOLD_OVERLAP,
                        range(len(GRID_WEIGHT_ALLOCATOR)),
                    )
                )

                print(f"\n[阶段 2] Grid 变体: {len(_grid_combos)} 组合")
                print(f"    top_n:         {GRID_TOP_N}")
                print(f"    min_momentum:  {GRID_MIN_MOMENTUM}")
                print(f"    rebalance:     {GRID_REBALANCE_INTERVAL}")
                print(f"    hold_overlap:  {GRID_HOLD_OVERLAP}")
                if CUSTOM_PERIODS:
                    print(f"    custom_periods: {CUSTOM_PERIODS}")
                elif PERIOD_FREQ:
                    print(f"    period_freq:   {PERIOD_FREQ}")
                print("=" * 60)

                # 打包轻量参数供 worker 使用
                _shared_args = (ranking_factor, SHARED_PIPELINE, builtin_filters, cross_sectional_filters)

                group_summaries: list[dict] = []
                futures_map: dict = {}

                # 阶段 2 前按内存预算计算 worker 数（fork 前 gc + malloc_trim）
                executor, _workers = _prepare_worker_pool(
                    universe_bytes=_universe_bytes, cfg_max_workers=GRID_MAX_WORKERS,
                )
                with executor:
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
        report_path = output_base / f"stock_momentum_report_{_ts_report}.html"
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
            "hold_overlap": GRID_HOLD_OVERLAP,
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
      逐组模式（5 元素）:
        (ranking_factor, factor_pipeline, builtin_filters, cross_sectional_filters, combo)
      跨组模式（6 元素）:
        (group_label, ranking_factor, factor_pipeline, builtin_filters,
         cross_sectional_filters, combo)
    """
    import os as _os

    if len(args) == 6:
        group_label, ranking_factor, factor_pipeline, builtin_filters, cross_sectional_filters, combo = args
        _pg = _prepared_map[group_label]
        _out_root = _output_root_map[group_label]
    else:
        group_label = None
        ranking_factor, factor_pipeline, builtin_filters, cross_sectional_filters, combo = args
        _pg = prepared
        _out_root = _output_root
    top_n, min_mom, rebal, hold_overlap, alloc_idx = combo
    weight_allocator = GRID_WEIGHT_ALLOCATOR[alloc_idx]

    # 子目录标签
    grid_parts = [f"top{top_n}"]
    if min_mom is not None:
        grid_parts.append(f"mom{min_mom}".replace(".", "p"))
    if rebal != 5:
        grid_parts.append(f"rebal{rebal}")
    if hold_overlap:
        grid_parts.append("hold")
    _alloc_name = getattr(weight_allocator, "__name__", "")
    if _alloc_name and _alloc_name != "equal_weight_allocator":
        grid_parts.append(f"wt_{_alloc_name}")
    grid_parts.append(f"sd{_start_date.replace('-', '')}")
    grid_label = "_".join(grid_parts)

    config = WideMomentumBaselineConfig(
        top_n_values=(top_n,),
        ranking_factor=ranking_factor,
        factor_pipeline=factor_pipeline,
        builtin_filters=builtin_filters,
        cross_sectional_filters=cross_sectional_filters,
        min_momentum_value=min_mom,
        rebalance_interval=rebal,
        hold_overlap=hold_overlap,
        min_listing_days=MIN_LISTING_DAYS,
        commission=COMMISSION,
        cash=CASH,
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
    parser = argparse.ArgumentParser(description="股票动量基线回测通用引擎")
    parser.add_argument(
        "--config",
        default="libs.scripts.stock_momentum_configs.stock_template",
        help="配置模块路径（默认: stock_template）",
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
