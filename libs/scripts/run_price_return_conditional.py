#!/usr/bin/env python3
"""
PriceReturn_20 条件因子：三条件 AND 过滤后的动量因子。

因子定义:
    主体信号: PriceReturn(window=20)
    条件1: TrendR2(window=120, output="r2") > 0.5       (趋势拟合优度)
    条件2: MAPosition(window=200) > 0                   (收盘价在 200 日均线之上)
    条件3: Rsrs(regression_window=14, zscore_window=600, output="zscore") > 0

三个条件全部满足（AND）时，因子值 = PriceReturn_20；否则为 NaN。

用法:
    # 仅因子分析 (Layer 1-3)
    uv run python libs/scripts/run_price_return_conditional.py --analysis

    # 因子分析 + wide_momentum 轮动回测
    uv run python libs/scripts/run_price_return_conditional.py --analysis --backtest

    # 只跑回测（跳过分析）
    uv run python libs/scripts/run_price_return_conditional.py --backtest

    # 指定标的子集
    uv run python libs/scripts/run_price_return_conditional.py --analysis --symbols 510300 159915
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from factors.meta_factor import ConditionSpec, MultiConditionalFactor
from factors.ma import MAPosition
from factors.price_return import PriceReturn
from factors.rsrs import RsrsFactor
from factors.trend_r2 import TrendR2Factor


# ═══════════════════════════════════════════════════════════════════════════════
# 因子构建
# ═══════════════════════════════════════════════════════════════════════════════

def build_factor() -> MultiConditionalFactor:
    """构建 PriceReturn_20 三条件 AND 因子。"""
    signal = PriceReturn(window=20)

    conditions = [
        ConditionSpec(
            condition=TrendR2Factor(window=120, output="r2"),
            op="gt",
            threshold=0.5,
        ),
        ConditionSpec(
            condition=MAPosition(window=200, price_column="close"),
            op="gt",
            threshold=0.0,
        ),
        ConditionSpec(
            condition=RsrsFactor(
                regression_window=14,
                zscore_window=600,
                output="zscore",
            ),
            op="gt",
            threshold=0.0,
        ),
    ]

    factor = MultiConditionalFactor(
        signal=signal,
        conditions=conditions,
        logic="and",
        false_value="nan",
    )

    print(f"因子名称: {factor.get_output_name()}")
    print(f"warmup: {factor.get_max_warmup_period()} 根 bar")
    print(f"条件数: {len(conditions)} (AND)")
    for i, cond in enumerate(conditions):
        print(f"  条件{i+1}: {cond.condition.get_output_name()} {cond.op} {cond.threshold}")
    print()
    return factor


# ═══════════════════════════════════════════════════════════════════════════════
# 因子分析
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(
    factor: MultiConditionalFactor,
    symbols: list[str] | None = None,
    layers: tuple[int, ...] = (1, 2, 3),
    forward_periods: tuple[int, ...] = (5, 10, 20, 60),
    start_date: str | None = None,
    end_date: str | None = None,
    max_workers: int | None = None,
) -> None:
    """运行单因子分析流水线 (Layer 1-3)。"""
    from factor_analysis.config import FactorAnalysisConfig
    from factor_analysis.runner import run_factor_analysis

    config = FactorAnalysisConfig(
        factor=factor,
        symbols=symbols,
        layers=layers,
        forward_periods=forward_periods,
        start_date=start_date,
        end_date=end_date,
        max_workers=max_workers,
    )
    results = run_factor_analysis(config)

    output = results.get("output", {})
    print("\n" + "=" * 60)
    print("因子分析完成！")
    if output.get("linux_root"):
        print(f"  输出目录: {output['linux_root']}")
    files = output.get("files", [])
    if files:
        print(f"  输出文件 ({len(files)} 个):")
        for f in files:
            print(f"    {f}")
    print("=" * 60)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Wide Momentum 轮动回测
# ═══════════════════════════════════════════════════════════════════════════════

def run_wide_momentum_backtest(
    factor: MultiConditionalFactor,
    top_n_values: tuple[int, ...] = (5, 10, 20),
    rebalance_intervals: tuple[int, ...] = (5, 10, 20),
    start_date: str = "2020-01-01",
    end_date: str = "2026-05-29",
    output_base_dir: str | None = None,
    max_workers: int | None = None,
) -> None:
    """运行 wide_momentum 轮动回测。

    将条件因子作为排名因子，在全 ETF 池中按因子值排序选 top-N。
    条件不满足（NaN）的标的自动排在末尾，不会被选中。
    """
    import itertools
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from backtesting.wide_momentum_baseline import (
        ThresholdFilter,
        WideMomentumBaselineConfig,
        equal_weight_allocator,
        prepare_wide_momentum_universe,
        run_wide_momentum_baseline_from_prepared,
        save_wide_momentum_baseline_result,
    )
    from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP

    symbols = ETF_INDEX_MAP.get_all_symbols()

    # 准备 universe
    print(f"\n{'=' * 60}")
    print("Wide Momentum 轮动回测")
    print(f"{'=' * 60}")
    print(f"  排名因子: {factor.get_output_name()}")
    print(f"  标的池:   {len(symbols)} 标的")
    print(f"  回测区间: {start_date} → {end_date}")
    print(f"  Top-N:    {top_n_values}")
    print(f"  调仓间隔: {rebalance_intervals} 日")
    print(f"  权重方案: 等权 (equal_weight)")
    print()

    config = WideMomentumBaselineConfig(
        ranking_factor=factor,
        factor_pipeline=(factor,),
        builtin_filters=(),  # 条件已在因子内处理
        start_date=start_date,
        end_date=end_date,
        weight_allocator=equal_weight_allocator,
    )
    prepared = prepare_wide_momentum_universe(config=config, symbols=symbols)
    print(f"  Universe 准备完成: {len(prepared.symbol_data_map)} 合格标的")

    # Grid Search
    output_base = Path(
        output_base_dir or (REPO_ROOT / "data" / "backtest_results")
    )
    output_dir = output_base / f"price_return_conditional_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_combos = list(itertools.product(top_n_values, rebalance_intervals))
    print(f"\nGrid 组合数: {len(grid_combos)}")
    print("-" * 60)

    mp_ctx = __import__("multiprocessing").get_context("spawn") if max_workers and max_workers > 1 else None

    def _run_one(top_n: int, rebalance: int) -> dict:
        variant_config = WideMomentumBaselineConfig(
            ranking_factor=factor,
            factor_pipeline=(factor,),
            top_n_values=(top_n,),
            rebalance_interval=rebalance,
            start_date=start_date,
            end_date=end_date,
            weight_allocator=equal_weight_allocator,
        )
        result = run_wide_momentum_baseline_from_prepared(
            prepared=prepared,
            config=variant_config,
        )
        summary = result.variant_results[top_n].summary
        return {
            "top_n": top_n,
            "rebalance": rebalance,
            "cum_return_pct": summary["cumulative_return_pct"],
            "ann_return_pct": summary["annualised_return_pct"],
            "sharpe": summary["sharpe"],
            "max_dd_pct": summary["max_drawdown_pct"],
            "calmar": summary.get("calmar_ratio", None),
            "turnover": summary.get("avg_turnover", None),
        }

    # 串行执行（简单可靠），除非指定了 max_workers > 1
    if max_workers and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
            futures = {
                pool.submit(_run_one, tn, rb): (tn, rb)
                for tn, rb in grid_combos
            }
            results = []
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        results = []
        for tn, rb in grid_combos:
            results.append(_run_one(tn, rb))

    # 输出汇总
    import pandas as pd
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print("\n回测结果汇总 (按 Sharpe 降序):")
    print("=" * 80)
    header = f"{'Top-N':>6} {'调仓':>5} {'累计收益%':>10} {'年化收益%':>10} {'Sharpe':>8} {'最大回撤%':>10} {'Calmar':>8}"
    print(header)
    print("-" * 80)
    for _, row in df.iterrows():
        print(
            f"{int(row.top_n):>6} {int(row.rebalance):>5} "
            f"{row.cum_return_pct:>10.2f} {row.ann_return_pct:>10.2f} "
            f"{row.sharpe:>8.3f} {row.max_dd_pct:>10.2f} "
            f"{row.calmar or 0:>8.3f}"
        )

    # 保存 CSV
    csv_path = output_dir / "grid_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存: {csv_path}")

    # 最佳参数
    best = df.iloc[0]
    print(f"\n最佳参数: Top-{int(best.top_n)}, 调仓{int(best.rebalance)}日")
    print(f"  累计收益: {best.cum_return_pct:.2f}%")
    print(f"  年化收益: {best.ann_return_pct:.2f}%")
    print(f"  Sharpe:   {best.sharpe:.3f}")
    print(f"  最大回撤: {best.max_dd_pct:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PriceReturn_20 条件因子：三条件 AND + 因子分析 + WideMomentum 回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--analysis", action="store_true", default=True,
                        help="运行因子分析 Layer 1-3（默认开启）")
    parser.add_argument("--no-analysis", action="store_true",
                        help="跳过因子分析")
    parser.add_argument("--backtest", action="store_true",
                        help="运行 wide_momentum 轮动回测")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="因子分析标的列表（空格分隔），默认全量 ETF_INDEX_MAP")
    parser.add_argument("--forward-periods", nargs="+", type=int,
                        default=[5, 10, 20, 60],
                        help="前向持仓期（默认 5 10 20 60）")
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3],
                        help="分析层（默认 1 2 3）")
    parser.add_argument("--start-date", type=str, default=None,
                        help="起始日期 YYYY-MM-DD（分析用）")
    parser.add_argument("--end-date", type=str, default=None,
                        help="结束日期 YYYY-MM-DD（分析用）")
    parser.add_argument("--bt-start", type=str, default="2020-01-01",
                        help="回测起始日期（默认 2020-01-01）")
    parser.add_argument("--bt-end", type=str, default="2026-05-29",
                        help="回测结束日期（默认 2026-05-29）")
    parser.add_argument("--top-n", nargs="+", type=int, default=[5, 10, 20],
                        help="Top-N 持仓数（默认 5 10 20）")
    parser.add_argument("--rebalance", nargs="+", type=int, default=[5, 10, 20],
                        help="调仓间隔（默认 5 10 20 日）")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="多进程 worker 数（默认 CPU 数）")
    parser.add_argument("--bt-output", type=str, default=None,
                        help="回测输出目录（默认 data/backtest_results/）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. 构建因子
    factor = build_factor()

    # 2. 因子分析
    if not args.no_analysis:
        print("启动因子分析...")
        run_analysis(
            factor=factor,
            symbols=args.symbols if args.symbols else None,
            layers=tuple(args.layers),
            forward_periods=tuple(args.forward_periods),
            start_date=args.start_date,
            end_date=args.end_date,
            max_workers=args.max_workers,
        )

    # 3. Wide Momentum 回测
    if args.backtest:
        run_wide_momentum_backtest(
            factor=factor,
            top_n_values=tuple(args.top_n),
            rebalance_intervals=tuple(args.rebalance),
            start_date=args.bt_start,
            end_date=args.bt_end,
            output_base_dir=args.bt_output,
            max_workers=args.max_workers,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
