#!/usr/bin/env python3
"""
daily_basic 首次回填 CLI (Daily Basic Backfill CLI)

对全 A 股代码集逐只回填 data/daily_basic/<code>.csv:
  * 2018-01-02 起: 东财 RPT_VALUEANALYSIS_DET 真值 (单请求全量);
  * 2016-01-01 ~ 2017-12-31 及退市股: 由 stock_data 成交额/换手率估算。

用法:
    # 全量回填 (默认代码集 = data/stock_data 文件集, 15 并发 + 代理池)
    python libs/scripts/backfill_daily_basic.py

    # 指定代码子集 (逗号分隔) + 并发数 + 预演
    python libs/scripts/backfill_daily_basic.py --codes 000001,600519 --pool-size 4

    # 从文件读取代码列表 (每行一个 6 位代码)
    python libs/scripts/backfill_daily_basic.py --codes-file codes.txt --dry-run

    # 汇总输出: data/daily_basic/backfill_summary.txt (成功/失败/估算占比)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from multiprocessing import Pool

from config import DataPath  # noqa: E402
from data_manager.daily_basic_manager import (  # noqa: E402
    backfill_daily_basic_symbol,
    list_stock_data_symbols,
)
from data_manager.stock_data_manager import (  # noqa: E402
    STOCK_UPDATE_POOL_SIZE,
    _initialize_stock_update_worker,
)
from data_manager.providers.stock_list_provider import STOCK_LIST  # noqa: E402


def _resolve_codes(codes: str | None, codes_file: str | None) -> list[str]:
    if codes:
        return [c.strip().zfill(6) for c in codes.split(",") if c.strip()]
    if codes_file:
        path = Path(codes_file)
        if not path.exists():
            raise FileNotFoundError(f"codes 文件不存在: {path}")
        return [
            line.strip().zfill(6)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.strip()[:6].isdigit()
        ]
    # 默认: 与 stock_data 同代码集 (含历史退市股文件)
    symbols = list_stock_data_symbols()
    if not symbols:
        # 兜底在册列表
        symbols = STOCK_LIST.get_all_symbol()
    return symbols


def run_backfill(
    codes: list[str],
    *,
    pool_size: int,
    dry_run: bool,
) -> None:
    summary_path = Path(DataPath.DAILY_BASIC_PATH) / "backfill_summary.txt"
    Path(DataPath.DAILY_BASIC_PATH).mkdir(parents=True, exist_ok=True)
    print(f"回填 {len(codes)} 只 (pool={pool_size}, dry_run={dry_run}) ...")

    if dry_run:
        return None

    with Pool(pool_size, initializer=_initialize_stock_update_worker) as pool:
        results = pool.map(backfill_daily_basic_symbol, codes)

    with open(summary_path, "w", encoding="utf-8") as f:
        for code, ok in zip(codes, results):
            f.write(f"{code}: {'OK' if ok else 'FAIL'}\n")

    ok_count = sum(1 for r in results if r)
    print(f"完成: {ok_count}/{len(codes)} 成功, 失败 {len(codes) - ok_count} 个")
    print(f"汇总: {summary_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="daily_basic 首次回填 — 东财市值/股本 (2018+ 真值 + 2016-2017 估算)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--codes", type=str, default=None, help="逗号分隔的 6 位代码列表")
    parser.add_argument("--codes-file", type=str, default=None, help="每行一个代码的文本文件路径")
    parser.add_argument("--pool-size", type=int, default=STOCK_UPDATE_POOL_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="只打印数量, 不实际拉取")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    codes = _resolve_codes(args.codes, args.codes_file)
    if not codes:
        print("无代码可回填")
        return 1
    run_backfill(codes, pool_size=args.pool_size, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
