from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from config import DataPath
from data_manager.utils import extract_tracked_index_name, normalize_tracked_index_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为每个跟踪指数选出最优 ETF（数据最早优先，平局选近 20 日成交额高者），输出映射表。"
    )
    parser.add_argument(
        "--etf-name-list",
        type=Path,
        default=None,
        help="ETF 名称列表 CSV 路径，默认使用 DataPath.ETF_NAME_LIST_DF 对应的 CSV。",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="ETF 数据 CSV 目录，默认使用 DataPath.DEFAULT_PATH。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径，默认 data/const/etf_index_map.csv。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(cpu_count(), 8)),
        help="并行工作进程数。",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=20,
        help="计算平均成交额所用的最近 N 个交易日，默认 20。",
    )
    return parser.parse_args()


def resolve_etf_name_list_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    # ETF_NAME_LIST_DF 指向 xlsx，这里用同名的 csv
    csv_path = Path(str(DataPath.ETF_NAME_LIST_DF).replace(".xlsx", ".csv"))
    if csv_path.exists():
        return csv_path
    # fallback
    return REPO_ROOT / "data" / "const" / "etf_name_list.csv"


def resolve_source_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    configured = getattr(DataPath, "DEFAULT_PATH", None)
    if configured:
        candidate = Path(configured)
        if candidate.exists():
            return candidate
    return REPO_ROOT / "data" / "etf_data"


def resolve_output_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return REPO_ROOT / "data" / "const" / "etf_index_map.csv"


def load_etf_name_list(path: Path) -> pd.DataFrame:
    """加载 ETF 名称列表，返回含 symbol/name/type 的 DataFrame。"""
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    # 列名标准化：去 BOM 并 strip
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    if "symbol" not in df.columns:
        raise ValueError(f"ETF 名称列表缺少 symbol 列，实际列: {list(df.columns)}")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.zfill(6)
    if "name" not in df.columns:
        raise ValueError(f"ETF 名称列表缺少 name 列")
    df["name"] = df["name"].astype(str).str.strip()
    return df


def extract_tracked_indices(df: pd.DataFrame) -> pd.DataFrame:
    """对每条 ETF 提取跟踪指数（原始 + 规范化）。"""
    df = df.copy()
    df["tracked_index_raw"] = df["name"].apply(extract_tracked_index_name)
    df["tracked_index"] = df["tracked_index_raw"].apply(normalize_tracked_index_name)
    # 过滤掉无法提取跟踪指数的记录
    df = df[df["tracked_index"].str.strip() != ""].copy()
    return df


def _read_one_etf_stats(task: tuple[str, Path, int]) -> dict[str, Any]:
    """读取单个 ETF CSV，返回 first_date / avg_value_20d / bar_count。"""
    symbol, source_dir, recent_days = task
    fp = source_dir / f"{symbol}.csv"
    result: dict[str, Any] = {
        "symbol": symbol,
        "first_date": None,
        "avg_value_recent": 0.0,
        "bar_count": 0,
        "error": None,
    }
    if not fp.exists():
        result["error"] = "file_not_found"
        return result

    try:
        df = pd.read_csv(fp, encoding="utf-8-sig")
    except Exception as exc:
        result["error"] = f"read_error: {exc}"
        return result

    if df.empty:
        result["error"] = "empty_file"
        return result

    # 标准化列名
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    if "date" not in df.columns:
        result["error"] = "missing_date_column"
        return result

    try:
        dates = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
        result["error"] = "invalid_date_format"
        return result

    valid_mask = dates.notna()
    if not valid_mask.any():
        result["error"] = "no_valid_dates"
        return result

    first_valid_date = dates[valid_mask].min()
    result["first_date"] = first_valid_date.strftime("%Y-%m-%d") if pd.notna(first_valid_date) else None
    result["bar_count"] = int(valid_mask.sum())

    # 最近 N 日平均成交额
    col_name = "value" if "value" in df.columns else ("成交额" if "成交额" in df.columns else None)
    if col_name:
        value_series = pd.to_numeric(df[col_name], errors="coerce")
        recent_values = value_series.iloc[-recent_days:].dropna()
        if len(recent_values) > 0:
            result["avg_value_recent"] = float(recent_values.mean())
        else:
            result["avg_value_recent"] = 0.0

    return result


def select_best_per_index(
    etf_df: pd.DataFrame,
    source_dir: Path,
    recent_days: int = 20,
    workers: int = 4,
) -> pd.DataFrame:
    """按跟踪指数分组，选最优 ETF。"""
    # 收集所有唯一的 symbol
    all_symbols = etf_df["symbol"].unique().tolist()

    # 并行读取所有 ETF 的统计信息
    tasks = [(s, source_dir, recent_days) for s in all_symbols]
    stats_map: dict[str, dict[str, Any]] = {}

    with Pool(processes=workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_read_one_etf_stats, tasks, chunksize=20),
                total=len(tasks),
                desc="读取 ETF 数据统计",
            )
        )

    for r in results:
        stats_map[r["symbol"]] = r

    # 按 tracked_index 分组
    groups: dict[str, list[str]] = defaultdict(list)
    for _, row in etf_df.iterrows():
        ti = row["tracked_index"]
        symbol = row["symbol"]
        groups[ti].append(symbol)

    # 对每组选最优
    rows: list[dict[str, Any]] = []
    for ti, candidates in tqdm(groups.items(), desc="按跟踪指数选优"):
        scored: list[tuple[str, str, Any, Any, float, int]] = []
        for symbol in candidates:
            stats = stats_map.get(symbol)
            if stats is None or stats.get("error"):
                continue
            name = etf_df.loc[etf_df["symbol"] == symbol, "name"].iloc[0]
            first_date = stats["first_date"]
            avg_value = stats["avg_value_recent"]
            bar_count = stats["bar_count"]
            if first_date is None:
                continue
            scored.append((symbol, name, first_date, avg_value, bar_count))

        if not scored:
            # 所有候选都无法读取，跳过
            continue

        # 排序：first_date 升序（越早越优先），avg_value 降序（越大越优先）
        scored.sort(key=lambda x: (x[2], -x[3]))

        best = scored[0]
        all_candidate_symbols = [s[0] for s in scored]

        rows.append({
            "tracked_index": ti,
            "tracked_index_raw": etf_df.loc[etf_df["symbol"] == best[0], "tracked_index_raw"].iloc[0],
            "selected_symbol": best[0],
            "selected_name": best[1],
            "candidate_count": len(scored),
            "candidates": ",".join(all_candidate_symbols),
            "first_date": best[2],
            "avg_value_20d": round(best[3], 2),
            "bar_count": best[4],
        })

    result = pd.DataFrame(rows)
    # 按 tracked_index 排序
    result = result.sort_values("tracked_index").reset_index(drop=True)
    return result


def main() -> int:
    args = parse_args()

    etf_name_list_path = resolve_etf_name_list_path(args.etf_name_list)
    source_dir = resolve_source_dir(args.source_dir)
    output_path = resolve_output_path(args.output)

    if not etf_name_list_path.exists():
        print(f"ETF 名称列表不存在: {etf_name_list_path}")
        return 1
    if not source_dir.exists():
        print(f"ETF 数据目录不存在: {source_dir}")
        return 1

    print(f"ETF 名称列表: {etf_name_list_path}")
    print(f"ETF 数据目录: {source_dir}")
    print(f"输出路径:     {output_path}")

    # 1. 加载 ETF 名称列表
    etf_df = load_etf_name_list(etf_name_list_path)
    print(f"加载 ETF 数量: {len(etf_df)}")

    # 2. 提取跟踪指数
    etf_df = extract_tracked_indices(etf_df)
    unique_indices = etf_df["tracked_index"].nunique()
    print(f"唯一跟踪指数: {unique_indices}")

    # 3. 选优
    result_df = select_best_per_index(
        etf_df,
        source_dir,
        recent_days=args.recent_days,
        workers=args.workers,
    )
    print(f"选出 ETF:     {len(result_df)}")

    # 4. 输出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"结果已写入: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
