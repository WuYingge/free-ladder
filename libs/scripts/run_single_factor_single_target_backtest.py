from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = PROJECT_ROOT / "libs"
# Allow running from project root without installing libs as a package.
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from backtesting.engine import (
    SingleFactorSingleTargetBacktestConfig,
    run_single_factor_single_target_backtest,
)
from factors.new_high import NewHigh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETF backtest with Backtrader")
    parser.add_argument("--symbol", required=True, help="ETF symbol, e.g. 159915")
    parser.add_argument("--cash", type=float, default=100000.0, help="Initial cash")
    parser.add_argument("--commission", type=float, default=0.0005, help="Commission ratio")
    parser.add_argument("--stake", type=int, default=100, help="Order size for each buy")
    parser.add_argument("--high-window", type=int, default=50, help="NewHigh long window")
    parser.add_argument("--low-window", type=int, default=25, help="NewHigh short window")
    parser.add_argument("--data-dir", type=str, default="", help="CSV directory, default uses config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factor = NewHigh(high_window=args.high_window, low_window=args.low_window)
    config = SingleFactorSingleTargetBacktestConfig(
        symbol=args.symbol,
        factor=factor,
        cash=args.cash,
        commission=args.commission,
        stake=args.stake,
        data_dir=args.data_dir or None,
    )
    result = run_single_factor_single_target_backtest(config)
    print(
        json.dumps(
            {
                "symbol": result.symbol,
                "start_value": round(result.start_value, 2),
                "end_value": round(result.end_value, 2),
                "pnl": round(result.pnl, 2),
                "pnl_pct": round(result.pnl_pct * 100, 2),
                "sharpe": result.sharpe,
                "max_drawdown_pct": result.max_drawdown_pct,
                "trades_total": result.trades_total,
                "trades_won": result.trades_won,
                "trades_lost": result.trades_lost,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
