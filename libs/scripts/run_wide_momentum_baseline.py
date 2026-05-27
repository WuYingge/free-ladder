from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from backtesting import (  # noqa: E402
    WideMomentumBaselineConfig,
    run_wide_momentum_baseline,
    save_wide_momentum_baseline_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the wide-universe momentum rotation baseline backtest."
    )
    parser.add_argument(
        "--top-n",
        nargs="+",
        type=int,
        default=[5, 10],
        help="Top-N portfolio sizes to run, for example --top-n 5 10.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional explicit ETF symbol subset. Defaults to the ETF master list.",
    )
    parser.add_argument(
        "--min-listing-days",
        type=int,
        default=1200,
        help="Minimum natural-day listing age proxy before a symbol can enter the pool.",
    )
    parser.add_argument(
        "--momentum-window",
        type=int,
        default=20,
        help="Lookback window for the raw momentum factor.",
    )
    parser.add_argument(
        "--momentum-skip-recent",
        type=int,
        default=1,
        help="Lag bars skipped to enforce shift(1) anti-lookahead.",
    )
    parser.add_argument(
        "--rebalance-interval",
        type=int,
        default=20,
        help="Rebalance interval in trading days.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=100000.0,
        help="Initial capital.",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0005,
        help="Single-side commission estimate in decimal form.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.02,
        help="Annual risk-free rate used in Sharpe.",
    )
    parser.add_argument(
        "--stable-pool-size",
        type=int,
        default=100,
        help="Monthly eligible-pool count threshold used to flag the stable start month.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional manual lower bound for the backtest start date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional manual upper bound for the backtest end date.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "backtest_results",
        help="Root directory where the timestamped output folder will be created.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = WideMomentumBaselineConfig(
        top_n_values=tuple(int(value) for value in args.top_n),
        min_listing_days=int(args.min_listing_days),
        momentum_window=int(args.momentum_window),
        momentum_skip_recent=int(args.momentum_skip_recent),
        rebalance_interval=int(args.rebalance_interval),
        cash=float(args.cash),
        commission=float(args.commission),
        risk_free_rate=float(args.risk_free_rate),
        stable_pool_size=int(args.stable_pool_size),
        start_date=args.start_date,
        end_date=args.end_date,
    )

    result = run_wide_momentum_baseline(
        config=config,
        symbols=args.symbols,
    )
    output_dir = args.output_root / (
        f"wide_momentum_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_wide_momentum_baseline_result(result=result, output_dir=output_dir)

    print(f"Result directory: {output_dir}")
    print(
        "Prepared universe: "
        f"{len(result.prepared_universe.symbol_data_map)} eligible symbols "
        f"from {result.prepared_universe.source_symbol_count} source symbols"
    )
    print(
        "Backtest window: "
        f"{result.prepared_universe.start_date.date()} -> {result.prepared_universe.end_date.date()}"
    )
    if result.prepared_universe.stable_start_month is not None:
        print(
            "Pool stable start month: "
            f"{result.prepared_universe.stable_start_month.date()}"
        )
    else:
        print("Pool stable start month: not reached")

    for top_n, variant_result in sorted(result.variant_results.items(), key=lambda item: item[0]):
        summary = variant_result.summary
        print(
            f"Top {top_n}: cumulative={summary['cumulative_return_pct']}% "
            f"annualised={summary['annualised_return_pct']}% "
            f"sharpe={summary['sharpe']} "
            f"mdd={summary['max_drawdown_pct']}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())