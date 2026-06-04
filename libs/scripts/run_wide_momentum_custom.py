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
    prepare_wide_momentum_universe,
    run_wide_momentum_baseline_from_prepared,
    save_wide_momentum_baseline_result,
)
from factors.price_return import PriceReturn
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor
from factors.ma import MAFactor, MAPosition


# ====================================================================
# 因子定义
# ====================================================================
pr_return = PriceReturn(window=20, skip_recent=1)
trend_r2 = TrendR2Factor(window=120, output="r2")
rsrs = RsrsFactor(regression_window=14, zscore_window=600, output="zscore")
ma200 = MAPosition(window=200, price_column="close")


# ====================================================================
# 基础配置（所有 Grid 组合共享）
# ====================================================================
RANKING_FACTOR = pr_return  # None = 默认 PriceReturn(window=20, skip_recent=1)
FACTOR_PIPELINE: tuple = (
    rsrs,
    pr_return,
    trend_r2,
    ma200,
)
BUILTIN_FILTERS: tuple[ThresholdFilter, ...] = (
    ThresholdFilter(field=ma200.get_output_name(), operator=">=", value=0),
    ThresholdFilter(field=rsrs.get_output_name(), operator=">=", value=0),
    ThresholdFilter(field=trend_r2.get_output_name(), operator=">=", value=0.5),
)


# ====================================================================
# Grid Search 参数（元组 = 遍历）
# ====================================================================
GRID_TOP_N: tuple[int, ...] = (5, 10, 20)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,1,2,5)  # (2, 3, 5) = 探索三种
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5,10,20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False, True)  # True = 剔除 cluster 43/44 债类标的

# 集群约束开关
_CLUSTER_DEFAULT = any(v > 0 for v in GRID_CLUSTER_MAX_PER_GROUP)
CLUSTER_LIMIT_ENABLED = _CLUSTER_DEFAULT

# Grid 并行 worker 数（None = CPU 核数）
GRID_MAX_WORKERS: int | None = None


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
_output_root = Path("/mnt/c/Users/wyg/Documents/invest/backtest/全资产轮动") / _base_label

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
    prepared = prepare_wide_momentum_universe(config=_bootstrap_config)
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
        )
    )

    print(f"\n[阶段 2] Grid 变体: {len(_grid_combos)} 组合")
    print(f"    top_n:         {GRID_TOP_N}")
    print(f"    min_momentum:  {GRID_MIN_MOMENTUM}")
    print(f"    cluster_max:   {GRID_CLUSTER_MAX_PER_GROUP}")
    print(f"    rebalance:     {GRID_REBALANCE_INTERVAL}")
    print(f"    exclude_bonds: {GRID_EXCLUDE_BONDS}")
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
    top_n, min_mom, cluster_max, rebal, exclude_bonds = combo

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
