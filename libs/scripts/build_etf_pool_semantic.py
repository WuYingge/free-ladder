"""构建语义精选 ETF 池 CSV：与 etf_index_map 对齐并修复数据列。

背景：用户手工精选的 ETF 池（含语义分组 group 列）可能在 Excel 中打开后，
candidates 列被千分位格式化（如 "588,000,588,080,..."）而无法解析；
first_date 格式为 2020/9/25、avg_value_20d 为四舍五入整数。
本脚本按 tracked_index 与 etf_index_map.csv 对齐，用 etf_index_map 的
权威数据恢复 candidates / first_date / avg_value_20d，仅保留精选池的 group 列。

用法：
    python libs/scripts/build_etf_pool_semantic.py \
        --source <精选池 CSV 路径> \
        --output <输出 CSV 路径>

默认输出 data/const/etf_pool_semantic_20260808.csv。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from config import DataPath  # noqa: E402 - 需要先插入 libs 到 sys.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建语义精选 ETF 池 CSV：按 tracked_index 与 etf_index_map 对齐，恢复被 Excel 破坏的列。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="精选池 CSV 路径（GBK 或 utf-8-sig），默认从 /mnt/c 下查找 20260808 版本。",
    )
    parser.add_argument(
        "--index-map",
        type=Path,
        default=None,
        help="etf_index_map CSV 路径，默认 DataPath.ETF_INDEX_MAP_CSV。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 CSV 路径，默认 data/const/etf_pool_semantic_20260808.csv。",
    )
    return parser.parse_args()


def load_semantic_pool(path: Path) -> pd.DataFrame:
    """加载精选池 CSV，容忍 GBK / utf-8-sig 编码，标准化列名。"""
    encodings = ["gbk", "utf-8-sig"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
    else:
        raise ValueError(f"无法解码 {path}: {last_err}")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    return df


def build_aligned_pool(
    pool: pd.DataFrame, index_map: pd.DataFrame, output: Path
) -> pd.DataFrame:
    """按 tracked_index 对齐，输出与 etf_index_map 一致的列 + group。"""
    map_cols = [
        "tracked_index",
        "tracked_index_raw",
        "selected_symbol",
        "selected_name",
        "candidate_count",
        "candidates",
        "first_date",
        "avg_value_20d",
        "bar_count",
    ]
    for col in map_cols:
        if col not in index_map.columns:
            raise ValueError(f"etf_index_map 缺少列 {col}，实际列: {list(index_map.columns)}")
    if "group" not in pool.columns:
        raise ValueError(f"精选池缺少 group 列，实际列: {list(pool.columns)}")

    map_by_ti = index_map.set_index("tracked_index")
    missing = [ti for ti in pool["tracked_index"] if ti not in map_by_ti.index]
    if missing:
        raise ValueError(f"以下 tracked_index 不在 etf_index_map 中: {missing[:20]}")

    aligned = pool[["tracked_index", "group"]].copy()
    for col in map_cols:
        if col == "tracked_index":
            continue  # 已作为 set_index 的键
        aligned[col] = aligned["tracked_index"].map(map_by_ti[col])

    # 输出列顺序与 etf_index_map 一致，group 放最后
    aligned = aligned[map_cols + ["group"]]
    aligned.to_csv(output, index=False, encoding="utf-8-sig")
    return aligned


def main() -> None:
    args = parse_args()
    source = args.source or Path(
        "/mnt/c/Users/wyg/Documents/invest/const/etf_pool_semantic_20260808.csv"
    )
    index_map_path = args.index_map or Path(DataPath.ETF_INDEX_MAP_CSV)
    output = args.output or (REPO_ROOT / "data" / "const" / "etf_pool_semantic_20260808.csv")

    if not source.exists():
        raise FileNotFoundError(f"精选池文件不存在: {source}")
    if not index_map_path.exists():
        raise FileNotFoundError(f"etf_index_map 文件不存在: {index_map_path}")

    pool = load_semantic_pool(source)
    index_map = pd.read_csv(index_map_path, dtype=str, encoding="utf-8-sig")
    index_map.columns = index_map.columns.str.strip().str.lstrip("\ufeff")

    aligned = build_aligned_pool(pool, index_map, output)
    print(f"输出 {len(aligned)} 行 -> {output}")
    print(f"列: {list(aligned.columns)}")


if __name__ == "__main__":
    main()
