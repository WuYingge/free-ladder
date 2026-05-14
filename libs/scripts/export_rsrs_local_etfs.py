from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from config import DataPath
from core.models.etf_daily_data import EtfData
from factors.rsrs import RsrsFactor


DEFAULT_OUTPUT_DIR = Path("/mnt/c/Users/wyg/Documents/invest/tmp")
DEFAULT_MIN_TRADING_DAYS = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export local ETF data with RSRS factor columns for ETFs with sufficient trading history."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where per-symbol CSV outputs will be written.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing local ETF csv files. Defaults to DataPath.DEFAULT_PATH or data/etf_data.",
    )
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=DEFAULT_MIN_TRADING_DAYS,
        help="Only export ETFs with trading-day count strictly greater than this threshold.",
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
        help="Optional cap on eligible symbols, useful for quick validation.",
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

    fallback = REPO_ROOT / "data" / "etf_data"
    return fallback


def collect_eligible_files(source_dir: Path, min_trading_days: int) -> list[Path]:
    files = sorted(path for path in source_dir.glob("*.csv") if path.is_file())
    eligible_files: list[Path] = []

    for file_path in tqdm(files, desc="Scanning ETF files"):
        try:
            etf_data = EtfData.from_csv(str(file_path))
        except Exception as err:
            print(f"Skip {file_path.name}: load failed due to {err}")
            continue

        if len(etf_data.data) > min_trading_days:
            eligible_files.append(file_path)

    return eligible_files


def build_rsrs_factor() -> RsrsFactor:
    return RsrsFactor(
        regression_window=25,
        zscore_window=600,
        buy_threshold=0.7,
        sell_threshold=-0.7,
        output="zscore",
    )


def process_one(task: tuple[str, str]) -> dict[str, Any]:
    file_path_str, output_dir_str = task
    file_path = Path(file_path_str)
    output_dir = Path(output_dir_str)

    try:
        etf_data = EtfData.from_csv(str(file_path))
        etf_data.add_factors(build_rsrs_factor())
        etf_data.calc_factors()
        etf_data.output_with_factors_to(str(output_dir))
        return {
            "symbol": etf_data.symbol,
            "status": "ok",
            "rows": int(len(etf_data.data)),
            "output": str(output_dir / f"{etf_data.symbol}.csv"),
        }
    except Exception as err:
        return {
            "symbol": file_path.stem,
            "status": "error",
            "error": str(err),
        }


def main() -> int:
    args = parse_args()

    source_dir = resolve_source_dir(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"ETF source directory does not exist: {source_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_files = collect_eligible_files(source_dir, args.min_trading_days)
    if args.limit is not None:
        eligible_files = eligible_files[: args.limit]

    if not eligible_files:
        print("No eligible ETF files found.")
        return 0

    tasks = [(str(file_path), str(output_dir)) for file_path in eligible_files]
    worker_count = max(1, min(args.workers, len(tasks)))
    results: list[dict[str, Any]] = []

    with Pool(worker_count) as pool:
        iterator = pool.imap_unordered(process_one, tasks, chunksize=8)
        for result in tqdm(iterator, total=len(tasks), desc="Exporting RSRS"):
            results.append(result)

    success_results = [result for result in results if result["status"] == "ok"]
    error_results = [result for result in results if result["status"] == "error"]

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "min_trading_days": args.min_trading_days,
        "eligible_count": len(eligible_files),
        "success_count": len(success_results),
        "error_count": len(error_results),
        "workers": worker_count,
    }

    summary_path = output_dir / "rsrs_export_summary.json"
    summary_path.write_text(
        json.dumps({"summary": summary, "errors": error_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if error_results:
        print(f"Errors written to {summary_path}")
    return 0 if not error_results else 1


if __name__ == "__main__":
    raise SystemExit(main())