from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from backtesting.index_rotation import discover_index_rotation_candidates


DEFAULT_BASE_SYMBOLS = ["518880", "513100", "511010"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover low-correlation ETF extension combos for the three-asset rotation base pool."
    )
    parser.add_argument(
        "--base-symbols",
        nargs="+",
        default=DEFAULT_BASE_SYMBOLS,
        help="Base symbols that must be included in every combo.",
    )
    parser.add_argument(
        "--min-bar-count",
        type=int,
        default=600,
        help="Minimum required history length for keyword representative ETFs.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.3,
        help="Exclusive upper bound for pairwise correlation in the strict combo search.",
    )
    parser.add_argument(
        "--corr-lower-bound",
        type=float,
        default=-0.3,
        help="Exclusive lower bound for pairwise correlation in the strict combo search.",
    )
    parser.add_argument(
        "--correlation-source",
        choices=("close", "return"),
        default="return",
        help="Price level or daily return series used to build the correlation matrix.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of combos to keep in the output table.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "backtest_results",
        help="Directory where the timestamped result folder will be created.",
    )
    return parser.parse_args()


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _format_symbol_cell(symbol: str, metadata_lookup: pd.DataFrame) -> str:
    if symbol not in metadata_lookup.index:
        return symbol

    row = metadata_lookup.loc[symbol]
    name = str(row.get("name") or "").strip()
    tracked_index = str(row.get("tracked_index") or "").strip()
    if name and tracked_index:
        return f"{symbol} {name}（{tracked_index}）"
    if name:
        return f"{symbol} {name}"
    return symbol


def _build_combo_summary_frame(discovery: Any) -> pd.DataFrame:
    combo_frame = discovery.combo_frame.copy()
    if combo_frame.empty:
        return pd.DataFrame(
            columns=[
                "组合编号",
                "标的数量",
                "基础标的",
                "扩展标的",
                "扩展关键词",
                "全部标的明细",
                "平均两两相关性",
                "最大两两相关性",
                "最小两两相关性",
                "扩展ETF平均成交额",
                "扩展ETF总成交额",
            ]
        )

    profile_lookup = discovery.etf_profile_frame.copy()
    profile_lookup["symbol"] = profile_lookup["symbol"].astype(str).str.zfill(6)
    profile_lookup = profile_lookup.drop_duplicates(subset=["symbol"], keep="first")
    profile_lookup = profile_lookup.set_index("symbol", drop=False)

    readable_rows: list[dict[str, Any]] = []
    for row in combo_frame.itertuples(index=False):
        base_symbols = [str(symbol) for symbol in row.base_symbols]
        extension_symbols = [str(symbol) for symbol in row.extension_symbols]
        all_symbols = [str(symbol) for symbol in row.symbols]

        readable_rows.append(
            {
                "组合编号": row.combo_label,
                "标的数量": int(row.symbol_count),
                "基础标的": "；".join(
                    _format_symbol_cell(symbol, profile_lookup)
                    for symbol in base_symbols
                ),
                "扩展标的": "；".join(
                    _format_symbol_cell(symbol, profile_lookup)
                    for symbol in extension_symbols
                ),
                "扩展关键词": str(row.source_keywords_text or ""),
                "全部标的明细": "；".join(
                    _format_symbol_cell(symbol, profile_lookup)
                    for symbol in all_symbols
                ),
                "平均两两相关性": round(float(row.avg_pairwise_correlation), 6),
                "最大两两相关性": round(float(row.max_pairwise_correlation), 6),
                "最小两两相关性": round(float(row.min_pairwise_correlation), 6),
                "扩展ETF平均成交额": round(float(row.extension_avg_value), 2),
                "扩展ETF总成交额": round(float(row.extension_total_avg_value), 2),
            }
        )

    return pd.DataFrame(readable_rows)


def _write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("无满足条件的组合。\n", encoding="utf-8")
        return

    markdown = frame.to_markdown(index=False)
    path.write_text(markdown + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def main() -> int:
    args = parse_args()
    base_symbols = [str(symbol).zfill(6) for symbol in args.base_symbols]

    discovery = discover_index_rotation_candidates(
        base_symbols=base_symbols,
        min_bar_count=args.min_bar_count,
        corr_threshold=args.corr_threshold,
        corr_lower_bound=args.corr_lower_bound,
        correlation_source=args.correlation_source,
        max_results=args.max_results,
    )

    output_dir = args.output_root / (
        f"three_asset_extension_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    _write_frame(discovery.keyword_selection_frame, output_dir / "keyword_selection.csv")
    _write_frame(discovery.extension_candidate_frame, output_dir / "extension_candidates.csv")
    _write_frame(discovery.base_conflict_frame, output_dir / "base_conflicts.csv")
    _write_frame(
        discovery.base_pairwise_correlation_frame,
        output_dir / "base_pairwise_correlation.csv",
    )
    _write_frame(discovery.combo_frame, output_dir / "combo_results.csv")
    combo_summary_frame = _build_combo_summary_frame(discovery)
    _write_frame(combo_summary_frame, output_dir / "combo_summary_readable.csv")
    _write_markdown_table(combo_summary_frame, output_dir / "combo_summary_readable.md")

    summary = {
        "base_symbols": base_symbols,
        "min_bar_count": int(args.min_bar_count),
        "correlation_source": args.correlation_source,
        "corr_lower_bound": float(args.corr_lower_bound),
        "corr_threshold": float(args.corr_threshold),
        "max_results": int(args.max_results),
        "base_is_feasible": bool(discovery.base_is_feasible),
        "keyword_count": int(len(discovery.keyword_selection_frame)),
        "selected_keyword_count": int(
            discovery.keyword_selection_frame["selected_symbol"].notna().sum()
        ),
        "extension_candidate_count": int(len(discovery.extension_candidate_frame)),
        "compatible_extension_count": int(
            discovery.base_conflict_frame["compatible_with_base"].fillna(False).sum()
        ) if not discovery.base_conflict_frame.empty else 0,
        "combo_count": int(len(discovery.combo_frame)),
        "correlation_start": discovery.correlation_start,
        "correlation_end": discovery.correlation_end,
        "correlation_bar_count": int(discovery.correlation_bar_count),
        "output_dir": output_dir,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    print(f"Result directory: {output_dir}")
    print(f"Base symbols: {base_symbols}")
    print(
        f"Correlation source: {args.correlation_source} "
        f"with bounds ({args.corr_lower_bound}, {args.corr_threshold})"
    )
    print(
        "Correlation window: "
        f"{discovery.correlation_start} -> {discovery.correlation_end} "
        f"({discovery.correlation_bar_count} bars)"
    )
    print(f"Extension ETF count: {len(discovery.extension_candidate_frame)}")
    print(f"Compatible extensions: {summary['compatible_extension_count']}")

    if discovery.base_pairwise_correlation_frame.empty:
        print("Base pairwise correlation frame is empty.")
    else:
        print("Base pairwise correlations:")
        print(discovery.base_pairwise_correlation_frame.to_string(index=False))

    if discovery.combo_frame.empty:
        print("No strict combos found under the current rules.")
    else:
        print("Readable combo summary:")
        print(combo_summary_frame.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())