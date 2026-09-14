#!/usr/bin/env python3
"""
Alpha101 扫描器 CLI (Alpha101 Scan CLI)

对若干 Alpha101 公式逐一构建横截面面板，跑 Layer1/2/3 分析并输出报告，
复用现有 factor_analysis 的 IC / 分组 / 多空 / 质量分析。

用法:
    # 默认：注册表内全部 101 个因子（按数据可用性自动跳过 cap/vwap/行业缺失的），ETF 池
    python libs/scripts/run_alpha101_scan.py

    # 只跑不需要 vwap、不需要行业的子集（ETF 池常用）
    python libs/scripts/run_alpha101_scan.py --only-plain

    # 股票池全量（含 18 个行业中性化公式；需先备好行业分类与 adj_factor）
    python libs/scripts/run_alpha101_scan.py --universe stock --alpha 001 002 003

    # 指定因子 + 股票池
    python libs/scripts/run_alpha101_scan.py --alpha 001 101 --universe stock

    # 指定标的 + 完整参数
    python libs/scripts/run_alpha101_scan.py \\
        --alpha 003 --symbols 510300 510500 159915 \\
        --layers 1 2 3 --min-bars 200 \\
        --forward-periods 5 10 20 --n-quantiles 5 --max-workers 4

参数:
    --alpha           指定 alpha id（空格分隔，默认全部可计算）
    --universe        数据源: etf / stock（默认 etf）
    --symbols         可选标的列表（空格分隔，默认使用数据源默认池）
    --layers          分析层 1 2 3（空格分隔，默认 1 2 3）
    --forward-periods 前向持仓期（交易日，空格分隔，默认 5 10 20 60）
    --min-bars        最少交易日数（默认 252）
    --start-date      起始日期 YYYY-MM-DD
    --end-date        结束日期 YYYY-MM-DD
    --n-quantiles     分位数分组数（默认 5）
    --rolling-ic-window 滚动 IC 窗口（交易日，默认 120）
    --max-workers     数据加载并行度（默认 CPU 数）
    --output-root     输出根目录（默认 data/factors/{factor_name}/）
    --output-date     报告日期标签（默认当天日期）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from factor_analysis.config import FactorAnalysisConfig
from factor_analysis.forward_returns import compute_forward_returns
from factor_analysis.grouping import run_grouping_analysis
from factor_analysis.panel import FactorPanel
from factor_analysis.predictive import run_predictive_analysis
from factor_analysis.quality import run_quality_analysis
from factor_analysis.reporter import generate_and_save_reports
from factors.alpha101 import (
    ALPHA101_REGISTRY,
    Alpha101Factor,
    build_alpha101_inputs,
    build_alpha101_panel,
    data_requirements_summary,
    get_alpha_spec,
    get_computable_alpha_ids,
)


def build_universe(universe_kind: str, symbols: list[str] | None):
    """按类型构造数据源实例。"""
    if universe_kind == "stock":
        from factors.alpha101.universe import StockAlpha101Universe

        return StockAlpha101Universe(symbols=symbols)
    from factors.alpha101.universe import EtfAlpha101Universe

    return EtfAlpha101Universe(symbols=symbols)


def _cap_available(universe) -> bool:
    """探测 daily_basic 是否已落盘（抽样前 20 只，避免全量加载）。"""
    from data_manager.daily_basic_manager import get_fp

    sample = universe.list_symbols()[:20]
    if not sample:
        return False
    return any(Path(get_fp(s)).exists() for s in sample)


def _industry_available(universe) -> bool:
    """探测申万行业分类是否覆盖该池（抽样前 20 只）——ETF 池天然无行业分类。"""
    from data_manager.providers.sw_industry_provider import SW_INDUSTRY

    mapping = SW_INDUSTRY.get_mapping()
    if mapping.empty:
        return False
    sample = {str(s).zfill(6) for s in universe.list_symbols()[:20]}
    return bool(sample & set(mapping["symbol"]))


def _vwap_available(universe) -> bool:
    """探测 data/adj_factor 是否已覆盖该池（抽样前 20 只）——没有因子就算不出 vwap。"""
    from data_manager.adj_factor_manager import get_fp

    sample = universe.list_symbols()[:20]
    if not sample:
        return False
    return any(Path(get_fp(s)).exists() for s in sample)


def run_alpha101_scan(
    alpha_ids: list[str] | None = None,
    universe_kind: str = "etf",
    symbols: list[str] | None = None,
    *,
    layers: tuple[int, ...] = (1, 2, 3),
    forward_periods: tuple[int, ...] = (5, 10, 20, 60),
    min_bars: int = 252,
    start_date: str | None = None,
    end_date: str | None = None,
    n_quantiles: int = 5,
    rolling_ic_window: int = 120,
    max_workers: int | None = None,
    output_root: Path | None = None,
    output_date: str | None = None,
) -> dict[str, Any]:
    """逐一扫描 alpha，构建面板并跑分析，返回结果字典。"""
    if alpha_ids is None:
        alpha_ids = get_computable_alpha_ids(exclude_cap=False)
    if not alpha_ids:
        print("没有可计算的 alpha（全部依赖 cap？）。")
        return {}

    universe = build_universe(universe_kind, symbols)
    cap_available = _cap_available(universe)
    vwap_available = _vwap_available(universe)
    industry_available = _industry_available(universe)
    if alpha_ids and len(alpha_ids) > 20:
        summary = data_requirements_summary()
        print(f"注册表: {summary['total']} 个公式（需 vwap {summary['needs_vwap']} / "
              f"需行业 {summary['needs_industry']} / 需 cap {summary['uses_cap']} / "
              f"无附加依赖 {summary['plain']}）")
    # 面板只构建一次（101 个 alpha 复用同一份全池输入，避免重复加载）
    pending = [
        aid for aid in alpha_ids
        if not (get_alpha_spec(aid).uses_cap and not cap_available)
        and not (get_alpha_spec(aid).needs_vwap and not vwap_available)
        and not (get_alpha_spec(aid).needs_industry and not industry_available)
    ]
    shared_inputs = None
    if pending:
        need_ind = any(get_alpha_spec(aid).needs_industry for aid in pending)
        print(f"\n构建共享面板（{len(pending)} 个 alpha 复用; 行业标签={'是' if need_ind else '否'}）...")
        shared_inputs = build_alpha101_inputs(
            universe, start_date=start_date, end_date=end_date,
            max_workers=max_workers, needs_industry=need_ind,
        )
        print(f"  → {shared_inputs.close.shape[1]} 标的, {shared_inputs.close.shape[0]} 交易日")

    all_results: dict[str, Any] = {}

    for alpha_id in alpha_ids:
        spec = get_alpha_spec(alpha_id)
        factor = Alpha101Factor(alpha_id, universe_kind=universe_kind)

        if spec.uses_cap and not cap_available:
            print(f"跳过 Alpha#{alpha_id}：依赖 cap 但 daily_basic 市值数据不可用。")
            continue
        if spec.needs_vwap and not vwap_available:
            print(f"跳过 Alpha#{alpha_id}：依赖 vwap 但 data/adj_factor 复权因子不可用"
                  f"（先跑 libs/scripts/update_adj_factor.py）。")
            continue
        if spec.needs_industry and not industry_available:
            print(f"跳过 Alpha#{alpha_id}：需要申万 level-{spec.needs_industry} 行业中性化，"
                  f"但该池无行业分类（股票池先跑 libs/scripts/update_sw_industry_clf.py）。")
            continue

        print(f"\n===== Alpha#{alpha_id} ({factor.get_output_name()}) =====")
        config = FactorAnalysisConfig(
            factor=factor,
            layers=list(layers),
            forward_periods=forward_periods,
            min_bars=min_bars,
            n_quantiles=n_quantiles,
            rolling_ic_window=rolling_ic_window,
            output_root=output_root,
            output_date=output_date,
        )

        try:
            panel: FactorPanel = build_alpha101_panel(
                spec.func,
                universe,
                factor_name=factor.get_output_name(),
                min_bars=min_bars,
                start_date=start_date,
                end_date=end_date,
                max_workers=max_workers,
                needs_industry=bool(spec.needs_industry),
                inputs=shared_inputs,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ 面板构建失败: {type(exc).__name__}: {exc}")
            all_results[alpha_id] = {"success": False, "error": str(exc)}
            continue

        print(f"  → {panel.n_symbols} 标的, {panel.n_dates} 交易日, "
              f"覆盖 {panel.summary()['coverage_mean']:.1%}")

        fwd_map = compute_forward_returns(panel.close_prices, forward_periods) if 2 in layers or 3 in layers else {}

        quality_results = run_quality_analysis(panel) if 1 in layers else None
        predictive_results = None
        grouping_results = None

        if 2 in layers:
            predictive_results = run_predictive_analysis(
                panel=panel,
                fwd_returns_map=fwd_map,
                rolling_ic_window=rolling_ic_window,
            )
            rank_ic = predictive_results.get("rank_ic", {})
            for period in sorted(rank_ic):
                s = rank_ic[period]["summary"]
                print(f"  → Rank IC ({period}d): mean={s['mean']:.6f} IR={s['ir']:.4f}")

        if 3 in layers:
            grouping_results = run_grouping_analysis(
                panel=panel, fwd_returns_map=fwd_map, n_quantiles=n_quantiles,
            )
            for period in sorted(grouping_results):
                gr = grouping_results[period]
                ls = gr.get("longshort", {})
                if ls:
                    print(f"  → Long-Short ({period}d): ann_ret={ls.get('annualised_return', float('nan')):.4%} "
                          f"sharpe={ls.get('sharpe')}")

        # 保存分组 CSV（与 run_factor_analysis 一致的落盘约定）
        out_root = config.resolve_output_root()
        out_root.mkdir(parents=True, exist_ok=True)
        if grouping_results:
            for period in sorted(grouping_results):
                gr = grouping_results[period]
                suffix = f"_{period}d"
                gr["quantile_returns"].to_csv(out_root / f"quantile_returns{suffix}.csv")
                ls = gr.get("longshort", {})
                ls_series = ls.get("ls_series")
                if ls_series is not None and hasattr(ls_series, "to_csv"):
                    ls_series.to_csv(out_root / f"longshort_returns{suffix}.csv")

        json_path = generate_and_save_reports(
            panel=panel,
            quality_results=quality_results,
            predictive_results=predictive_results,
            grouping_results=grouping_results,
            config=config,
        )
        print(f"  → 报告: {json_path}")

        all_results[alpha_id] = {
            "success": True,
            "output_dir": str(out_root),
            "n_symbols": panel.n_symbols,
            "n_dates": panel.n_dates,
        }

    return all_results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpha101 扫描器 — 横截面 alpha 因子分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--alpha", nargs="*", default=None, help="alpha id 列表（默认全部 101 个，按数据可用性跳过）")
    parser.add_argument("--only-plain", action="store_true",
                        help="只跑无附加依赖（不用 vwap/行业/cap）的公式子集")
    parser.add_argument("--universe", type=str, default="etf", choices=["etf", "stock"])
    parser.add_argument("--symbols", nargs="*", default=None, help="标的列表（默认默认池）")
    parser.add_argument("--layers", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--forward-periods", nargs="*", type=int, default=[5, 10, 20, 60])
    parser.add_argument("--min-bars", type=int, default=252)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--n-quantiles", type=int, default=5)
    parser.add_argument("--rolling-ic-window", type=int, default=120)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--output-date", type=str, default=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    alpha_ids = args.alpha
    if args.only_plain and alpha_ids is None:
        alpha_ids = [
            aid
            for aid, spec in ALPHA101_REGISTRY.items()
            if not (spec.needs_vwap or spec.needs_industry or spec.uses_cap)
        ]
    run_alpha101_scan(
        alpha_ids=alpha_ids,
        universe_kind=args.universe,
        symbols=args.symbols,
        layers=tuple(args.layers),
        forward_periods=tuple(args.forward_periods),
        min_bars=args.min_bars,
        start_date=args.start_date,
        end_date=args.end_date,
        n_quantiles=args.n_quantiles,
        rolling_ic_window=args.rolling_ic_window,
        max_workers=args.max_workers,
        output_root=args.output_root,
        output_date=args.output_date,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
