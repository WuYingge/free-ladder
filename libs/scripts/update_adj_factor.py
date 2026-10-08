#!/usr/bin/env python
"""更新 data/adj_factor/<code>.csv（不复权收盘 + 逐日复权因子）。

为什么需要：本地行情 CSV 的价格列是**后复权**（fetcher 默认 adjust="hfq"），而
`value`(成交额, 元) / `volume`(成交量, 手) 是**不复权**成交口径，两者相差一个
逐股、逐日漂移的复权因子。alpha101 的 vwap 必须按

    vwap = value / (volume * 100) * adj_factor

搬到后复权尺度，才能与 OHLC 同口径（详见 libs/data_manager/adj_factor_manager.py）。

数据源：东财 K 线不复权（fqt=0），走项目代理；与本地后复权收盘同日相除得因子。
本地没有行情文件的标的会被跳过（默认只处理 `list_local_symbols`）。

用法:
    # 首次建库（逐标的全历史回填，ETF 池）
    uv run python libs/scripts/update_adj_factor.py --universe etf --all-history

    # 日常增量（只重取 [末日期-3天, 今天]，除权日自动带出新因子）
    uv run python libs/scripts/update_adj_factor.py --universe etf

    # 指定标的 / 全市场股票回填
    uv run python libs/scripts/update_adj_factor.py --symbols 600519 000001 --all-history
    uv run python libs/scripts/update_adj_factor.py --universe stock --all-history --max-workers 15

落盘后可用 libs/scripts/check_adj_factor.py 体检 low <= vwap <= high 不变量。
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataPath  # noqa: E402
from data_manager.adj_factor_manager import (  # noqa: E402
    batch_update_adj_factor,
    list_local_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="更新 data/adj_factor（不复权收盘 + 后复权因子）。"
    )
    parser.add_argument("--universe", choices=["stock", "etf"], default="stock",
                        help="标的池类型（默认 stock= data/stock_data 代码集）")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="指定标的（默认本地已有行情的全部标的）")
    parser.add_argument("--all-history", action="store_true",
                        help="逐标的全历史回填（首次建库用；默认增量）")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="并发进程数（默认复用股票更新管线规模，1 表示串行）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.symbols if args.symbols is not None else list_local_symbols(args.universe)
    if not targets:
        print(f"没有可更新的标的：{DataPath.STOCK_PATH if args.universe == 'stock' else DataPath.DEFAULT_PATH} 为空")
        return 1

    mode = "全历史回填" if args.all_history else "增量更新"
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] adj_factor {mode}: "
          f"universe={args.universe}, {len(targets)} 个标的")

    results = batch_update_adj_factor(
        symbols=targets,
        universe=args.universe,
        all_history=args.all_history,
        max_workers=args.max_workers,
    )
    ok = [code for code, success in results if success]
    failed = [code for code, success in results if not success]
    print(f"完成: 成功 {len(ok)} / 失败 {len(failed)}")
    if failed:
        print(f"失败标的(前 20): {failed[:20]}")
    print(f"落盘目录: {DataPath.ADJ_FACTOR_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
