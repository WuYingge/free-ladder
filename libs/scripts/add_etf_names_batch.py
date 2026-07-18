"""批量对目录下所有 rebalance_log.csv 添加 ETF 中文名称。

用法:
    cd /home/gouzi/projects/invest
    uv run python libs/scripts/add_etf_names_batch.py <回测输出根目录>

    # 示例
    uv run python libs/scripts/add_etf_names_batch.py /mnt/c/Users/wyg/Documents/invest/backtest/c2_pool_backtest/wide_momentum_study2_top1_5_10_rebal5_10_20_wt_invvol_withbond_20260715
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = _PROJECT_ROOT / "libs" / "scripts"
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "libs"))
sys.path.insert(0, str(SCRIPTS_DIR))

# add_etf_names_to_log 同级目录，直接导入
from add_etf_names_to_log import add_etf_names_to_rebalance_log


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量处理回测输出目录下所有 rebalance_log.csv，添加 ETF 中文名称"
    )
    parser.add_argument("root_dir", type=Path, help="回测输出根目录")
    parser.add_argument(
        "--no-dict-keys",
        action="store_true",
        help="不转换 target_weights 等字典列的 key（仅转换 selected_symbols）",
    )
    args = parser.parse_args()

    root: Path = args.root_dir
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    files = sorted(root.rglob("rebalance_log.csv"))
    if not files:
        raise SystemExit(f"未在 {root} 下找到任何 rebalance_log.csv")

    print(f"找到 {len(files)} 个文件")
    for i, fp in enumerate(files, 1):
        try:
            df = add_etf_names_to_rebalance_log(
                fp, map_dict_keys=not args.no_dict_keys
            )
            out = fp.with_name("rebalance_log_named.csv")
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  [{i}/{len(files)}] {fp.relative_to(root)} → {out.name}  ({len(df)} 行)")
        except Exception as exc:
            print(f"  [{i}/{len(files)}] {fp.relative_to(root)} ✗ {exc}")

    print(f"\n完成，共处理 {len(files)} 个文件")


if __name__ == "__main__":
    main()