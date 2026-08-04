"""基于相关性阈值对 etf_index_map 做第二步去重（合并疑似同指数 ETF）。

用法（项目根目录执行）::

    python libs/scripts/merge_duplicate_etfs_by_corr.py
    python libs/scripts/merge_duplicate_etfs_by_corr.py --threshold 0.9 --min-rows 60

流程：
- 输入为第一步（名称去重）产物 etf_index_map.csv，保持原样不覆盖
- 对全部选中 ETF 两两计算日收益率相关性（复用
  libs/data_manager/etf_correlation.py 的公共逻辑）
- 相关性 > --threshold 的对视为同指数，用并查集聚类成簇（处理传递性）
- 每簇保留上市时间最早（first_date 最小）的 ETF 作为代表，输出时
  每簇一行（含未参与合并的独立指数）——合并版 CSV 本身就是干净的
  标的池，selected_symbol 全局唯一，下游无需再去重
- 输出合并后的新 CSV（默认 data/const/etf_index_map_corr_merged.csv），
  另输出合并明细报告（默认 output/duplicate_merge_report.csv）

输出：
- 合并后标的池 CSV（每行一个代表 ETF；新增 merged_indices 列记录
  该簇覆盖的全部跟踪指数，便于追溯）
- 合并明细报告：每个多元素簇一行（代表、被合并成员、各自 first_date、簇内最大 corr）
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

from data_manager.etf_correlation import (
    build_merge_clusters,
    find_duplicate_pairs,
    load_close_matrix,
    pick_representative,
)

DEFAULT_INPUT = REPO_ROOT / "data" / "const" / "etf_index_map.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "const" / "etf_index_map_corr_merged.csv"
DEFAULT_REPORT = REPO_ROOT / "output" / "duplicate_merge_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按相关性阈值合并疑似同指数 ETF，生成更干净的 etf_index_map。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="第一步名称去重后的映射表 CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="合并后输出 CSV（默认不覆盖输入）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="日收益率相关性阈值，默认 0.9",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=60,
        help="共同交易日数下限，默认 60",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="合并明细报告 CSV 路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.input, dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    if "selected_symbol" not in df.columns or "tracked_index" not in df.columns:
        print(f"输入 CSV 缺少必要列（tracked_index/selected_symbol）: {args.input}")
        return 1
    df["selected_symbol"] = df["selected_symbol"].astype(str).str.strip()
    print(f"输入: {args.input}（{len(df)} 行, {df['tracked_index'].nunique()} 个指数）")

    symbols = list(dict.fromkeys(df["selected_symbol"]))
    close_matrix, ret_matrix, skipped = load_close_matrix(symbols)
    if skipped:
        print(f"加载失败跳过 {len(skipped)} 只: {[s for s, _ in skipped]}")
    if close_matrix.empty:
        print("无可用数据")
        return 1

    pairs = find_duplicate_pairs(close_matrix, ret_matrix, args.threshold, args.min_rows)
    print(f"疑似同指数对（corr>{args.threshold}, 共同行数>={args.min_rows}）: {len(pairs)} 对")

    # first_date 取该 symbol 所有行中最早者
    first_date_map: dict[str, str] = {}
    for _, row in df.iterrows():
        s, fd = row["selected_symbol"], str(row.get("first_date", "")).strip()
        if s not in first_date_map or (fd and fd < first_date_map[s]):
            first_date_map[s] = fd

    clusters = build_merge_clusters(pairs)
    if not clusters:
        print("未发现需要合并的簇，直接复制输入")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"已输出: {args.output}")
        return 0

    pair_corr = {(p["symbol_a"], p["symbol_b"]): p["corr"] for p in pairs}
    pair_corr.update({(p["symbol_b"], p["symbol_a"]): p["corr"] for p in pairs})

    # 每 symbol 一行完整信息（symbol → 其 tracked_index 在名称去重版中唯一）
    info: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        s = row["selected_symbol"]
        if s not in info:
            info[s] = {col: str(row.get(col, "")) for col in df.columns}

    # 参与合并的 symbol 归入簇；其余 symbol 作为单元素组保留
    merged_syms = {s for c in clusters for s in c}
    groups = clusters + [[s] for s in symbols if s not in merged_syms]

    report_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []
    for group in groups:
        rep = pick_representative(group, first_date_map)
        rep_info = info[rep]
        members = [m for m in group if m != rep]
        merged_indices = sorted(
            info[m]["tracked_index"] for m in group
        )
        output_rows.append(
            {
                "tracked_index": rep_info["tracked_index"],
                "tracked_index_raw": rep_info["tracked_index_raw"],
                "selected_symbol": rep,
                "selected_name": rep_info["selected_name"],
                "candidate_count": len(group),
                "candidates": ",".join(sorted(group)),
                "first_date": rep_info["first_date"],
                "avg_value_20d": rep_info["avg_value_20d"],
                "bar_count": rep_info["bar_count"],
                "merged_indices": ",".join(merged_indices),
            }
        )
        if members:
            max_corr = max(
                pair_corr[(a, b)]
                for i, a in enumerate(group)
                for b in group[i + 1 :]
                if (a, b) in pair_corr
            )
            report_rows.append(
                {
                    "representative": rep,
                    "representative_first_date": first_date_map.get(rep, ""),
                    "merged_symbols": ",".join(members),
                    "merged_first_dates": ",".join(first_date_map.get(m, "") for m in members),
                    "cluster_max_corr": max_corr,
                }
            )

    result = (
        pd.DataFrame(output_rows)
        .sort_values("tracked_index")
        .reset_index(drop=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"合并簇: {len(clusters)} 个，被合并 symbol: {len(merged_syms) - len(clusters)} 只；"
        f"已输出: {args.output}（{len(result)} 行, 唯一 symbol {result['selected_symbol'].nunique()}）"
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(args.report, index=False, encoding="utf-8-sig")
    print(f"合并明细: {args.report}")
    if report_rows:
        print(pd.DataFrame(report_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
