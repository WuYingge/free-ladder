from __future__ import annotations

import argparse
import json
import math
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from config import DataPath
from core.models.etf_daily_data import EtfData
from factors.trend_r2 import compute_trend_r2


FORMATION_WINDOWS: tuple[int, ...] = (60, 120, 180, 240)
HOLDING_WINDOWS: tuple[int, ...] = (5, 10, 20)
DEFAULT_OUTPUT_PATH = Path(
    "/mnt/c/Users/wyg/Documents/invest/research/r2_trend/trend_r2_scan_results.csv"
)
DEFAULT_MIN_VALID_SAMPLES = 260
DEFAULT_TOP_FRACTION = 0.2
MIN_XS_SYMBOLS = 3


def minimum_required_rows(
    formation_windows: tuple[int, ...] = FORMATION_WINDOWS,
    holding_windows: tuple[int, ...] = HOLDING_WINDOWS,
    min_valid_samples: int = DEFAULT_MIN_VALID_SAMPLES,
) -> int:
    return int(min_valid_samples + min(formation_windows) - 1 + min(holding_windows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trend R2 ETF scan over local daily CSV data."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing local ETF csv files. Defaults to DataPath.DEFAULT_PATH or data/etf_data.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file path for the 4x3 scan summary.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(cpu_count(), 8)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on source csv files for smoke tests.",
    )
    parser.add_argument(
        "--min-valid-samples",
        type=int,
        default=DEFAULT_MIN_VALID_SAMPLES,
        help="Minimum valid paired observations required for an ETF/combo to be eligible.",
    )
    return parser.parse_args()


def resolve_source_dir(explicit_source_dir: Path | None) -> Path:
    if explicit_source_dir is not None:
        return explicit_source_dir

    configured = getattr(DataPath, "DEFAULT_PATH", None)
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return candidate

    return REPO_ROOT / "data" / "etf_data"


def _count_csv_rows(file_path: Path) -> int:
    with file_path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def collect_candidate_files(
    source_dir: Path,
    min_total_rows: int,
    limit: int | None = None,
) -> list[Path]:
    files = sorted(path for path in source_dir.glob("*.csv") if path.is_file())
    filtered_files = [path for path in files if _count_csv_rows(path) >= min_total_rows]
    if limit is not None:
        return filtered_files[:limit]
    return filtered_files


def normalize_symbol_frame(etf_data: EtfData) -> pd.DataFrame:
    data = etf_data.data.copy()
    if "close" not in data.columns:
        raise ValueError(f"ETF {etf_data.symbol} is missing close column")

    if "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
    else:
        dates = pd.to_datetime(data.index, errors="coerce")

    frame = pd.DataFrame(
        {
            "date": dates,
            "close": pd.to_numeric(data["close"], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame = frame.drop_duplicates(subset=["date"], keep="last")
    frame.reset_index(drop=True, inplace=True)
    return frame


def compute_forward_return(close: pd.Series, holding_window: int) -> pd.Series:
    if holding_window < 1:
        raise ValueError("holding_window must be at least 1")

    future_close = close.shift(-holding_window)
    safe_close = close.where(close != 0.0)
    result = ((future_close - close) / safe_close) * 100.0
    result.name = f"fwd_ret_{holding_window}"
    return result


def compute_top_group_return(
    r2: pd.Series,
    fwd_ret: pd.Series,
    top_fraction: float = DEFAULT_TOP_FRACTION,
) -> float:
    paired = pd.DataFrame({"r2": r2, "fwd_ret": fwd_ret}).dropna()
    if paired.empty:
        return float("nan")

    top_n = max(1, int(math.ceil(len(paired) * top_fraction)))
    top_slice = paired.sort_values("r2", ascending=False).head(top_n)
    return float(top_slice["fwd_ret"].mean())


def compute_direction_hit_rate(r2: pd.Series, fwd_ret: pd.Series) -> float:
    paired = pd.DataFrame({"r2": r2, "fwd_ret": fwd_ret}).dropna()
    if paired.empty:
        return float("nan")

    median_r2 = float(paired["r2"].median())
    hits = ((paired["r2"] > median_r2) & (paired["fwd_ret"] > 0.0)) | (
        (paired["r2"] < median_r2) & (paired["fwd_ret"] < 0.0)
    )
    return float(hits.mean())


def summarize_symbol_rows(
    symbol_rows: pd.DataFrame,
    formation_windows: tuple[int, ...],
    holding_windows: tuple[int, ...],
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []

    for formation_window in formation_windows:
        for holding_window in holding_windows:
            combo_rows = symbol_rows[
                (symbol_rows["formation_window"] == formation_window)
                & (symbol_rows["holding_window"] == holding_window)
            ]
            eligible_rows = combo_rows[combo_rows["eligible"]]
            valid_corr = eligible_rows["symbol_corr"].dropna()
            top_buy = eligible_rows["top_buy_return_pct"].dropna()
            hit_rate = eligible_rows["hit_rate"].dropna()

            summary_rows.append(
                {
                    "formation_window": formation_window,
                    "holding_window": holding_window,
                    "candidate_etf_count": int(combo_rows["symbol"].nunique()),
                    "eligible_etf_count": int(eligible_rows["symbol"].nunique()),
                    "skipped_etf_count": int((~combo_rows["eligible"]).sum()),
                    "valid_symbol_corr_count": int(valid_corr.shape[0]),
                    "valid_pair_count": int(eligible_rows["valid_pair_count"].sum()),
                    "mean_symbol_corr": float(valid_corr.mean()) if not valid_corr.empty else float("nan"),
                    "positive_symbol_share": float((valid_corr > 0.0).mean()) if not valid_corr.empty else float("nan"),
                    "mean_top_buy_return_pct": float(top_buy.mean()) if not top_buy.empty else float("nan"),
                    "mean_hit_rate": float(hit_rate.mean()) if not hit_rate.empty else float("nan"),
                }
            )

    return pd.DataFrame(summary_rows)


def accumulate_cross_sectional_stats(
    accumulator: dict[tuple[int, int], dict[int, np.ndarray]],
    combo_key: tuple[int, int],
    date_ns: np.ndarray,
    r2_values: np.ndarray,
    fwd_values: np.ndarray,
) -> None:
    combo_stats = accumulator.setdefault(combo_key, {})
    for current_date_ns, current_r2, current_fwd in zip(date_ns, r2_values, fwd_values):
        stats = combo_stats.get(int(current_date_ns))
        if stats is None:
            combo_stats[int(current_date_ns)] = np.array(
                [1.0, current_r2, current_fwd, current_r2 * current_r2, current_fwd * current_fwd, current_r2 * current_fwd],
                dtype=float,
            )
        else:
            stats[0] += 1.0
            stats[1] += current_r2
            stats[2] += current_fwd
            stats[3] += current_r2 * current_r2
            stats[4] += current_fwd * current_fwd
            stats[5] += current_r2 * current_fwd


def finalize_cross_sectional_ic(
    accumulator: dict[tuple[int, int], dict[int, np.ndarray]],
    formation_windows: tuple[int, ...],
    holding_windows: tuple[int, ...],
    min_symbols: int = MIN_XS_SYMBOLS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for formation_window in formation_windows:
        for holding_window in holding_windows:
            combo_key = (formation_window, holding_window)
            combo_stats = accumulator.get(combo_key, {})
            daily_corrs: list[float] = []
            xs_eligible_date_count = 0

            for stats in combo_stats.values():
                count, sum_x, sum_y, sum_x2, sum_y2, sum_xy = stats
                if count < float(min_symbols):
                    continue

                xs_eligible_date_count += 1
                numerator = count * sum_xy - sum_x * sum_y
                denom_left = count * sum_x2 - sum_x * sum_x
                denom_right = count * sum_y2 - sum_y * sum_y
                if denom_left <= 0.0 or denom_right <= 0.0:
                    continue

                daily_corrs.append(float(numerator / np.sqrt(denom_left * denom_right)))

            daily_corrs_series = pd.Series(daily_corrs, dtype=float)
            rows.append(
                {
                    "formation_window": formation_window,
                    "holding_window": holding_window,
                    "mean_daily_xs_ic": float(daily_corrs_series.mean()) if not daily_corrs_series.empty else float("nan"),
                    "positive_daily_xs_ic_share": float((daily_corrs_series > 0.0).mean()) if not daily_corrs_series.empty else float("nan"),
                    "valid_xs_ic_date_count": int(daily_corrs_series.shape[0]),
                    "xs_eligible_date_count": int(xs_eligible_date_count),
                }
            )

    return pd.DataFrame(rows)


def evaluate_symbol_frame(
    symbol: str,
    frame: pd.DataFrame,
    formation_windows: tuple[int, ...],
    holding_windows: tuple[int, ...],
    min_valid_samples: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], dict[str, np.ndarray]]]:
    close = frame["close"].astype(float)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    forward_returns = {
        holding_window: compute_forward_return(close=close, holding_window=holding_window)
        for holding_window in holding_windows
    }
    trend_metrics = {
        formation_window: compute_trend_r2(close=close, window=formation_window)
        for formation_window in formation_windows
    }

    summary_rows: list[dict[str, Any]] = []
    daily_payload: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    for formation_window in formation_windows:
        metrics = trend_metrics[formation_window]
        for holding_window in holding_windows:
            paired = pd.DataFrame(
                {
                    "date": dates,
                    "r2": metrics.r2,
                    "slope": metrics.slope,
                    "fwd_ret": forward_returns[holding_window],
                }
            ).dropna(subset=["r2", "fwd_ret"])
            valid_pair_count = int(paired.shape[0])
            eligible = valid_pair_count >= min_valid_samples

            summary_row = {
                "symbol": symbol,
                "formation_window": formation_window,
                "holding_window": holding_window,
                "valid_pair_count": valid_pair_count,
                "eligible": eligible,
                "symbol_corr": float("nan"),
                "top_buy_return_pct": float("nan"),
                "hit_rate": float("nan"),
            }

            if eligible:
                summary_row["symbol_corr"] = float(paired["r2"].corr(paired["fwd_ret"]))
                summary_row["top_buy_return_pct"] = compute_top_group_return(
                    r2=paired["r2"],
                    fwd_ret=paired["fwd_ret"],
                )
                summary_row["hit_rate"] = compute_direction_hit_rate(
                    r2=paired["r2"],
                    fwd_ret=paired["fwd_ret"],
                )
                daily_payload[(formation_window, holding_window)] = {
                    "date_ns": paired["date"].astype("int64").to_numpy(copy=False),
                    "r2": paired["r2"].to_numpy(dtype=float, copy=False),
                    "fwd_ret": paired["fwd_ret"].to_numpy(dtype=float, copy=False),
                }

            summary_rows.append(summary_row)

    return summary_rows, daily_payload


def process_one(
    task: tuple[str, tuple[int, ...], tuple[int, ...], int]
) -> dict[str, Any]:
    file_path_str, formation_windows, holding_windows, min_valid_samples = task
    file_path = Path(file_path_str)

    try:
        etf_data = EtfData.from_csv(str(file_path))
        frame = normalize_symbol_frame(etf_data)
        summary_rows, daily_payload = evaluate_symbol_frame(
            symbol=str(etf_data.symbol),
            frame=frame,
            formation_windows=formation_windows,
            holding_windows=holding_windows,
            min_valid_samples=min_valid_samples,
        )
        return {
            "symbol": str(etf_data.symbol),
            "status": "ok",
            "row_count": int(frame.shape[0]),
            "summary_rows": summary_rows,
            "daily_payload": daily_payload,
        }
    except Exception as err:
        return {
            "symbol": file_path.stem,
            "status": "error",
            "error": str(err),
        }


def formation_label(window: int) -> str:
    mapping = {
        60: "3个月 (≈60日)",
        120: "6个月 (≈120日)",
        180: "9个月 (≈180日)",
        240: "12个月 (≈240日)",
    }
    return mapping.get(window, f"{window}日")


def holding_label(window: int) -> str:
    return f"{window}日"


def _format_metric(value: float, prefix: str, suffix: str = "") -> str:
    if pd.isna(value):
        return f"{prefix}=nan"
    if suffix:
        return f"{prefix}={value:.1f}{suffix}"
    return f"{prefix}={value:.3f}"


def format_primary_matrix(summary_df: pd.DataFrame) -> str:
    cell_width = 15
    header = "形成期\\持仓期".ljust(16) + " | " + " | ".join(
        holding_label(window).ljust(cell_width) for window in HOLDING_WINDOWS
    )
    separator = "-" * len(header)
    lines = [header, separator]

    indexed = summary_df.set_index(["formation_window", "holding_window"])
    for formation_window in FORMATION_WINDOWS:
        label = formation_label(formation_window).ljust(16)
        row_entries = [indexed.loc[(formation_window, holding_window)] for holding_window in HOLDING_WINDOWS]
        lines.append(
            label + " | " + " | ".join(
                _format_metric(float(entry["mean_symbol_corr"]), "r").ljust(cell_width)
                for entry in row_entries
            )
        )
        lines.append(
            " " * 16 + " | " + " | ".join(
                _format_metric(float(entry["positive_symbol_share"]) * 100.0, "pos%", "%").ljust(cell_width)
                if not pd.isna(entry["positive_symbol_share"])
                else _format_metric(float("nan"), "pos%").ljust(cell_width)
                for entry in row_entries
            )
        )
        lines.append(
            " " * 16 + " | " + " | ".join(
                _format_metric(float(entry["mean_hit_rate"]) * 100.0, "hit", "%").ljust(cell_width)
                if not pd.isna(entry["mean_hit_rate"])
                else _format_metric(float("nan"), "hit").ljust(cell_width)
                for entry in row_entries
            )
        )
        lines.append(
            " " * 16 + " | " + " | ".join(
                _format_metric(float(entry["mean_top_buy_return_pct"]), "topBuy", "%").ljust(cell_width)
                if not pd.isna(entry["mean_top_buy_return_pct"])
                else _format_metric(float("nan"), "topBuy").ljust(cell_width)
                for entry in row_entries
            )
        )
        lines.append(separator)

    return "\n".join(lines)


def format_xs_ic_matrix(summary_df: pd.DataFrame) -> str:
    cell_width = 16
    header = "形成期\\持仓期".ljust(16) + " | " + " | ".join(
        holding_label(window).ljust(cell_width) for window in HOLDING_WINDOWS
    )
    separator = "-" * len(header)
    lines = [header, separator]

    indexed = summary_df.set_index(["formation_window", "holding_window"])
    for formation_window in FORMATION_WINDOWS:
        label = formation_label(formation_window).ljust(16)
        row_entries = [indexed.loc[(formation_window, holding_window)] for holding_window in HOLDING_WINDOWS]
        lines.append(
            label + " | " + " | ".join(
                _format_metric(float(entry["mean_daily_xs_ic"]), "xsIC").ljust(cell_width)
                for entry in row_entries
            )
        )
        lines.append(
            " " * 16 + " | " + " | ".join(
                _format_metric(float(entry["positive_daily_xs_ic_share"]) * 100.0, "xsPos%", "%").ljust(cell_width)
                if not pd.isna(entry["positive_daily_xs_ic_share"])
                else _format_metric(float("nan"), "xsPos%").ljust(cell_width)
                for entry in row_entries
            )
        )
        lines.append(
            " " * 16 + " | " + " | ".join(
                f"xsN={int(entry['valid_xs_ic_date_count'])}".ljust(cell_width)
                for entry in row_entries
            )
        )
        lines.append(separator)

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()

    source_dir = resolve_source_dir(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"ETF source directory does not exist: {source_dir}")

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_name(output_path.stem + "_metadata.json")

    min_total_rows = minimum_required_rows(min_valid_samples=args.min_valid_samples)
    candidate_files = collect_candidate_files(
        source_dir=source_dir,
        min_total_rows=min_total_rows,
        limit=args.limit,
    )
    if not candidate_files:
        print("No ETF csv files meet the minimum row requirement.")
        return 0

    tasks = [
        (str(file_path), FORMATION_WINDOWS, HOLDING_WINDOWS, int(args.min_valid_samples))
        for file_path in candidate_files
    ]
    worker_count = max(1, min(args.workers, len(tasks)))

    symbol_rows: list[dict[str, Any]] = []
    xs_accumulator: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    errors: list[dict[str, Any]] = []
    skipped_symbols_by_combo: dict[str, list[str]] = {
        f"{formation_window}_{holding_window}": []
        for formation_window in FORMATION_WINDOWS
        for holding_window in HOLDING_WINDOWS
    }

    with Pool(worker_count) as pool:
        iterator = pool.imap_unordered(process_one, tasks, chunksize=8)
        for result in tqdm(iterator, total=len(tasks), desc="Trend R2 scan"):
            if result["status"] != "ok":
                errors.append(
                    {
                        "symbol": result["symbol"],
                        "error": result["error"],
                    }
                )
                continue

            symbol_rows.extend(result["summary_rows"])
            for row in result["summary_rows"]:
                if not row["eligible"]:
                    combo_name = f"{row['formation_window']}_{row['holding_window']}"
                    skipped_symbols_by_combo[combo_name].append(row["symbol"])

            for combo_key, payload in result["daily_payload"].items():
                accumulate_cross_sectional_stats(
                    accumulator=xs_accumulator,
                    combo_key=combo_key,
                    date_ns=payload["date_ns"],
                    r2_values=payload["r2"],
                    fwd_values=payload["fwd_ret"],
                )

    symbol_rows_df = pd.DataFrame(symbol_rows)
    if symbol_rows_df.empty:
        print("No symbol-level results were produced.")
        return 1

    summary_df = summarize_symbol_rows(
        symbol_rows=symbol_rows_df,
        formation_windows=FORMATION_WINDOWS,
        holding_windows=HOLDING_WINDOWS,
    )
    xs_summary_df = finalize_cross_sectional_ic(
        accumulator=xs_accumulator,
        formation_windows=FORMATION_WINDOWS,
        holding_windows=HOLDING_WINDOWS,
    )
    final_summary_df = summary_df.merge(
        xs_summary_df,
        on=["formation_window", "holding_window"],
        how="left",
    )
    final_summary_df.insert(
        1,
        "formation_label",
        final_summary_df["formation_window"].map(formation_label),
    )
    final_summary_df.insert(
        3,
        "holding_label",
        final_summary_df["holding_window"].map(holding_label),
    )
    final_summary_df.sort_values(["formation_window", "holding_window"], inplace=True)
    final_summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    runtime_seconds = time.perf_counter() - started_at
    metadata = {
        "source_dir": str(source_dir),
        "output_path": str(output_path),
        "formation_windows": list(FORMATION_WINDOWS),
        "holding_windows": list(HOLDING_WINDOWS),
        "min_valid_samples": int(args.min_valid_samples),
        "min_total_rows": int(min_total_rows),
        "top_fraction": DEFAULT_TOP_FRACTION,
        "min_xs_symbols": MIN_XS_SYMBOLS,
        "candidate_file_count": len(candidate_files),
        "processed_symbol_count": int(symbol_rows_df["symbol"].nunique()),
        "workers": worker_count,
        "limit": args.limit,
        "runtime_seconds": runtime_seconds,
        "errors": errors,
        "skipped_symbols_by_combo": skipped_symbols_by_combo,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Primary scan matrix:")
    print(format_primary_matrix(final_summary_df))
    print()
    print("Daily cross-sectional IC:")
    print(format_xs_ic_matrix(final_summary_df))
    print()
    print(f"Summary CSV: {output_path}")
    print(f"Metadata JSON: {metadata_path}")
    print(f"Processed ETFs: {symbol_rows_df['symbol'].nunique()}")
    print(f"Errors: {len(errors)}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())