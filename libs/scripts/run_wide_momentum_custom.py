"""宽动量基线回测 — 可配置版 + Grid Search。

改因子/过滤器/约束只需修改下面「基础配置」和「Grid Search」区域。
Grid 组合用进程池并行跑变体，universe 只准备一次。

用法：
    cd /home/gouzi/projects/invest
    uv run python libs/scripts/run_wide_momentum_custom.py
"""
from __future__ import annotations

import itertools
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "libs"))

from backtesting.wide_momentum_baseline import (
    ThresholdFilter,
    WideMomentumBaselineConfig,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    make_tiered_weight_allocator,
    prepare_wide_momentum_universe,
    run_wide_momentum_baseline_from_prepared,
    save_wide_momentum_baseline_result,
    score_proportional_allocator,
)
from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP
from factors.average_amount import AverageAmount
from factors.price_return import PriceReturn
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor
from factors.volatility import Volatility
from factors.ma import MAFactor, MAPosition


# ====================================================================
# 因子定义
# ====================================================================
pr_return = PriceReturn(window=20, skip_recent=1)
trend_r2 = TrendR2Factor(window=120, output="r2")
rsrs = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
ma200 = MAPosition(window=200, price_column="close")
vol20 = Volatility(window=20)                                      # 20 日波动率（反波动率加权用）
# avg_amount_20 = AverageAmount(window=20)                         # 20 日平均成交额


# ====================================================================
# 基础配置（所有 Grid 组合共享）
# ====================================================================
RANKING_FACTOR = pr_return  # None = 默认 PriceReturn(window=20, skip_recent=1)
FACTOR_PIPELINE: tuple = (
    rsrs,
    pr_return,
    trend_r2,
    ma200,
    vol20,
    # avg_amount_20,
)
BUILTIN_FILTERS: tuple[ThresholdFilter, ...] = (
    ThresholdFilter(field=ma200.get_output_name(), operator=">=", value=0),
    ThresholdFilter(field=rsrs.get_output_name(), operator=">=", value=0),
    ThresholdFilter(field=trend_r2.get_output_name(), operator=">=", value=0.5),
    # ThresholdFilter(field=avg_amount_20.get_output_name(), operator=">=", value=5_000_000),
)


# ====================================================================
# Grid Search 参数（元组 = 遍历）
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (5,)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)  # (2, 3, 5) = 探索三种
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (10, )
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (True,)  # True = 剔除 cluster 43/44 债类标的
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)  # True = 调仓时保留重叠标的

# ====================================================================
# 权重分配器定义
# ====================================================================
# 方案 0: 等权（基线）
alloc_equal = equal_weight_allocator

# 方案 1: 动量加权 — 权重 ∝ 20日收益（排序因子 score）
alloc_momentum = score_proportional_allocator
alloc_momentum.__name__ = "momentum"

# 方案 2: 反波动率加权 — 权重 ∝ 1/σ_20d
alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"

# 方案 3: 分档加权 — 自适应比例：前 40% 权重×1.5，后 60% 权重×1.0
#   Top 2  → 前1名×1.5, 后1名×1.0
#   Top 5  → 前2名×1.5, 后3名×1.0
#   Top 10 → 前4名×1.5, 后6名×1.0
#   Top 20 → 前8名×1.5, 后12名×1.0
def _adaptive_tiered(candidates):
    """自适应分档：前 40% 权重 1.5，后 60% 权重 1.0。"""
    if not candidates:
        return {}
    n = len(candidates)
    top_count = max(1, round(n * 0.4))
    weights = {}
    for i, c in enumerate(candidates):
        weights[c.symbol] = 1.5 if i < top_count else 1.0
    return weights

_adaptive_tiered.__name__ = "tiered"
alloc_tiered = _adaptive_tiered

# 方案 4: RSRS 强度加权 — 权重 ∝ RSRS zscore
alloc_rsrs = make_factor_weighted_allocator(rsrs.get_output_name())
alloc_rsrs.__name__ = "rsrs"

# Grid: 选择要跑的分配器组合（逐个跑请只留一个）
GRID_WEIGHT_ALLOCATOR: tuple = (
    # alloc_equal,
    # alloc_momentum,
    alloc_inv_vol,
    # alloc_tiered,
    # alloc_rsrs,
)

# 集群约束开关
_CLUSTER_DEFAULT = any(v > 0 for v in GRID_CLUSTER_MAX_PER_GROUP)
CLUSTER_LIMIT_ENABLED = _CLUSTER_DEFAULT

# Grid 并行 worker 数（None = CPU 核数）
GRID_MAX_WORKERS: int | None = None

# 分段统计频率（None = 不启用分段统计）
# 常见取值: 'YE' (年), 'QE' (季度), 'ME' (月), 'W' (周)
PERIOD_FREQ: str | None = None

# 自定义分段统计的时间段列表（优先级高于 PERIOD_FREQ）
# 示例: (("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"))
CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None


# ====================================================================
# 基础标签构造
# ====================================================================
_base_parts: list[str] = ["wide_momentum"]
if RANKING_FACTOR is not None:
    _label_ranking = str(RANKING_FACTOR).replace(" ", "").replace("(", "_").replace(")", "")
    _base_parts.append(_label_ranking)
for bf in BUILTIN_FILTERS:
    _field = bf.field.replace("_", "")
    _op = bf.operator.replace(">=", "ge").replace("<=", "le").replace(">", "gt").replace("<", "lt")
    _base_parts.append(f"{_field}{_op}{bf.value}".replace(".", "p"))
if CLUSTER_LIMIT_ENABLED:
    _base_parts.append("cluster")
if any(GRID_EXCLUDE_BONDS):
    _base_parts.append("no_bond")

_base_label = "_".join(_base_parts)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_output_root = Path("/mnt/c/Users/wyg/Documents/invest/backtest/top5_rebal10_no_bond_wt_invvol") / _base_label

# 模块级占位：main() 中通过 global 赋值，子进程 fork 后继承。
prepared = None


def main() -> None:
    """主流程：阶段 1 universe 准备 + 阶段 2 Grid 并行。"""
    global prepared

    # ====================================================================
    # 阶段 1：准备共享 universe（串行，最重）
    # ====================================================================
    _bootstrap_config = WideMomentumBaselineConfig(
        ranking_factor=RANKING_FACTOR,
        factor_pipeline=FACTOR_PIPELINE,
        builtin_filters=BUILTIN_FILTERS,
        start_date = "2020-01-01",
        end_date = "2026-05-29"
    )

    print("=" * 60)
    print("宽动量基线回测 — Grid Search")
    print("=" * 60)
    print(f"  基础标签:       {_base_label}")
    print(f"  排序因子:       {RANKING_FACTOR or 'PriceReturn(window=20, skip_recent=1) (默认)'}")
    print(f"  因子管线:       {[f.get_output_name() for f in FACTOR_PIPELINE] or '(无)'}")
    print(f"  过滤器:         {[(b.field, b.operator, b.value) for b in BUILTIN_FILTERS] or '(无)'}")
    print(f"  集群约束:       {'启用' if CLUSTER_LIMIT_ENABLED else '禁用'}")
    print()

    print("[阶段 1] 准备 shared universe（加载数据 + 因子计算）...", end=" ", flush=True)
    symbols = ETF_INDEX_MAP.get_all_symbols()
    print(f"({len(symbols)} 标的 from ETF_INDEX_MAP)", end=" ", flush=True)
    prepared = prepare_wide_momentum_universe(config=_bootstrap_config, symbols=symbols)
    print(
        f"完成 ({len(prepared.symbol_data_map)} 标的, "
        f"{prepared.start_date.date()} → {prepared.end_date.date()})"
    )

    # ====================================================================
    # 阶段 2：Grid 变体并行
    # ====================================================================
    _grid_combos = list(
        itertools.product(
            GRID_TOP_N,
            GRID_MIN_MOMENTUM,
            GRID_CLUSTER_MAX_PER_GROUP,
            GRID_REBALANCE_INTERVAL,
            GRID_EXCLUDE_BONDS,
            GRID_HOLD_OVERLAP,
            range(len(GRID_WEIGHT_ALLOCATOR)),  # 传索引，避免 pickle 闭包
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

    all_summaries: list[dict] = []
    futures_map: dict = {}

    # 打包轻量参数供 picklable worker 使用（prepared/_output_root 由 fork 继承）
    _shared_args = (RANKING_FACTOR, FACTOR_PIPELINE, BUILTIN_FILTERS, CLUSTER_LIMIT_ENABLED)

    # 强制 fork 模式：prepared 由 fork 继承内存，无需 pickle 692 个 DataFrame
    _mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=GRID_MAX_WORKERS, mp_context=_mp_ctx) as executor:
        for combo in _grid_combos:
            future = executor.submit(_run_single_combo, _shared_args + (combo,))
            futures_map[future] = combo

        for idx, future in enumerate(as_completed(futures_map), start=1):
            combo = futures_map[future]
            grid_label, summaries = future.result()
            all_summaries.extend(summaries)

            s0 = summaries[0] if summaries else {}
            print(
                f"[{idx}/{len(_grid_combos)}] {grid_label}: "
                f"cum={s0.get('cumulative_return_pct', '?')}%, "
                f"sharpe={s0.get('sharpe', '?')}, "
                f"mdd={s0.get('max_drawdown_pct', '?')}%"
            )

    # 汇总
    import pandas as pd

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        cols = ["grid_label"] + [c for c in summary_df.columns if c != "grid_label"]
        summary_df = summary_df[cols]
        csv_path = _output_root / f"grid_summary_{_ts}.csv"
        summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\nGrid 汇总: {csv_path}")
        print(f"共 {len(_grid_combos)} 组合，{len(all_summaries)} 行")


def _run_single_combo(args):
    """单个 grid 组合的完整回测（供进程池并行）。

    只接收轻量参数。prepared / _output_root 等大对象由 fork 继承，不通过 pickle 传输。

    args = (RANKING_FACTOR, FACTOR_PIPELINE, BUILTIN_FILTERS,
            CLUSTER_LIMIT_ENABLED, combo)
    """
    import os as _os

    ranking_factor, factor_pipeline, builtin_filters, cluster_limit_enabled, combo = args
    top_n, min_mom, cluster_max, rebal, exclude_bonds, hold_overlap, alloc_idx = combo
    # allocator 通过 fork 继承的模块级变量获取（闭包不可 pickle）
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
    # 权重分配器标签（equal_weight 不标注，其他加 wt_ 前缀）
    _alloc_name = getattr(weight_allocator, "__name__", "")
    if _alloc_name and _alloc_name != "equal_weight_allocator":
        grid_parts.append(f"wt_{_alloc_name}")
    grid_label = "_".join(grid_parts)

    # 债券过滤通过 config.exclude_clusters 实现，由 backtesting 内部处理
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
    main()