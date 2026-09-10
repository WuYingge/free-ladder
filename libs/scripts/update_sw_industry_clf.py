#!/usr/bin/env python
"""刷新 data/const/stock_sw_industry_clf.csv（申万全市场个股行业分类历史）。

数据源：申万宏源官网 StockClassifyUse_stock.xls，走项目代理（fetch 侧自动重试）。
落盘前自动按生效时代为每个 6 位行业代码附上一/二/三级中文名称
（2021-07-30 起 → sw_industry_standard_2021.csv；2014-02-21 起 →
sw_industry_standard_2014.csv；更早时代按代码跨版近似回退；均无则留空）。
落盘后由 libs/data_manager/providers/sw_industry_provider.py 的 SW_INDUSTRY 提供
symbol → 行业代码/名称的时点映射与 indneutralize 分组标签；
get_stock_data_by_symbol(..., with_industry=True) 按日并入行业列。

用法:
    uv run python libs/scripts/update_sw_industry_clf.py
    uv run python libs/scripts/update_sw_industry_clf.py --output /tmp/sw_clf.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataPath  # noqa: E402
from data_manager.sw_industry_manager import refresh_sw_industry_clf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="刷新申万行业分类历史 CSV（默认 data/const/stock_sw_industry_clf.csv）。"
    )
    parser.add_argument(
        "--output",
        default=None,
        help=f"输出 CSV 路径，默认 {DataPath.STOCK_SW_INDUSTRY_CLF_CSV}。",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    refresh_sw_industry_clf(output_path=args.output)
    print("Done.")
