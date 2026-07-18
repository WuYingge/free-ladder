"""收集回测输出中的所有 rebalance_log.csv，重命名后集中到一个目录。

用法:
    uv run python libs/scripts/collect_rebalance_logs.py <输出根目录>

示例:
    uv run python libs/scripts/collect_rebalance_logs.py \\
        /mnt/c/Users/wyg/Documents/invest/backtest/c2_vol_trank_filter/wide_momentum_..._20260719

输出:
    在 <输出根目录>/rebalance_logs/ 下生成:
        {group_label}__{grid_label}__rebalance_log.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def collect_rebalance_logs(output_root: Path) -> None:
    """扫描所有 rebalance_log.csv 并复制到 rebalance_logs/ 子目录。"""
    out_dir = output_root / "rebalance_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for log_path in sorted(output_root.rglob("rebalance_log.csv")):
        # 路径结构: wide_momentum_{group_label}/{grid_label}/top_{n}/rebalance_log.csv
        parts = log_path.relative_to(output_root).parts
        if len(parts) < 4:
            print(f"  SKIP (unexpected path): {log_path}")
            continue

        group_dir = parts[0]  # wide_momentum_{group_label}
        grid_label = parts[1]  # top10_rebal20_wt_invvol

        group_label = group_dir.removeprefix("wide_momentum_")

        new_name = f"{group_label}__{grid_label}__rebalance_log.csv"
        dest = out_dir / new_name

        shutil.copy2(log_path, dest)
        count += 1
        print(f"  [{count}] {new_name}")

    print(f"\n完成: {count} 个 rebalance_log → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="收集并重命名 rebalance_log.csv")
    parser.add_argument(
        "output_root",
        type=Path,
        help="回测输出根目录",
    )
    args = parser.parse_args()
    collect_rebalance_logs(args.output_root)


if __name__ == "__main__":
    main()
