#!/usr/bin/env python3
"""财务数据抓取/落盘 CLI (Financial Data Update CLI)

把 A 股财报数据抓取并落盘到 ``data/financial/`` (七张宽表 CSV), 供
``get_stock_data_by_symbol(..., with_financial=True)`` 按公告日时点合并。

数据源 (与 akshare 等价接口, 列名已实测):
  * income_q / balance_q / cashflow_q — 东财 F10 财务分析 (逐股, 一次取全历史)
  * forecast / dividend / holder_num   — 东财数据中心 (逐报告期, 全市场一次)
  * share_capital                      — 巨潮资讯 (逐股)

口径: 金额=元, 股本=股, 比率=小数, 金额为累计口径; ann_date = 首次公告日;
缺失一律留空; 每张表末尾带 currency/update_date/basis/update_time 方法论列。

用法:
    # 日更: 只刷新最近 2 个报告期 (逐股表) + 全市场逐期表
    uv run python libs/scripts/update_financials.py --recent-periods 2

    # 首次全量回填 (2014 起, 逐股表取全历史)
    uv run python libs/scripts/update_financials.py --all-history --start-year 2014

    # 只更某几张表 / 某几只股票 / 预演
    uv run python libs/scripts/update_financials.py --tables income_q,forecast
    uv run python libs/scripts/update_financials.py --symbols 000001,600519,300750 --dry-run

    # 并发与通道
    uv run python libs/scripts/update_financials.py --threads 8
    # (通道由环境变量控制: FINANCIAL_FETCH_MODE=direct_first|proxy_first|direct_only|proxy_only)

汇总写入: data/financial/update_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from data_manager.financial_manager import (  # noqa: E402
    DEFAULT_PERIOD_START_YEAR,
    FINANCIAL_FETCH_POOL_SIZE,
    batch_check_financials_updated,
    get_financial_dir,
    list_tables,
    update_financials,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取/落盘 A 股财报数据到 data/financial/ (默认日更口径)。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tables",
        default="all",
        help=f"逗号分隔的表名, 或 all (默认)。可选: {','.join(list_tables())}",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_PERIOD_START_YEAR,
        help=f"抓取起始年 (默认 {DEFAULT_PERIOD_START_YEAR})。",
    )
    parser.add_argument("--end-year", type=int, default=None, help="抓取结束年 (默认今年)。")
    parser.add_argument(
        "--recent-periods",
        type=int,
        default=None,
        help="逐股表只取最近 N 个报告期 (默认 2, 日更用); 全量回填请配 --all-history。",
    )
    parser.add_argument(
        "--all-history",
        action="store_true",
        help="逐股表取全历史 (首次回填用, 耗时更长)。",
    )
    parser.add_argument("--symbols", default=None, help="逗号分隔的 6 位股票代码 (默认全市场)。")
    parser.add_argument("--codes-file", default=None, help="代码清单文件 (每行一个 6 位代码)。")
    parser.add_argument(
        "--include-delisted",
        dest="include_delisted",
        action="store_true",
        default=True,
        help="包含退市股 (默认包含, 防幸存者偏差)。",
    )
    parser.add_argument(
        "--no-include-delisted",
        dest="include_delisted",
        action="store_false",
        help="只抓在市股票。",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=FINANCIAL_FETCH_POOL_SIZE,
        help=f"逐股抓取并发线程数 (默认 {FINANCIAL_FETCH_POOL_SIZE})。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只抓取不落盘。")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="整表替换 (全量重抓用, 丢弃既有文件的对应表); 默认与既有文件增量合并。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="只跑新鲜度体检 (不联网、不抓取)。",
    )
    return parser.parse_args(argv)


def _resolve_tables(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list_tables()
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in list_tables()]
    if unknown:
        raise SystemExit(f"未知表名: {unknown}; 可选: {list_tables()}")
    return requested


def _resolve_symbols(args: argparse.Namespace) -> list[str] | None:
    if args.symbols:
        return [code.strip().zfill(6) for code in args.symbols.split(",") if code.strip()]
    if args.codes_file:
        path = Path(args.codes_file)
        if not path.exists():
            raise SystemExit(f"代码文件不存在: {path}")
        return [
            line.strip().zfill(6)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and line.strip()[:6].isdigit()
        ]
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    financial_dir = get_financial_dir()

    if args.check:
        status = batch_check_financials_updated(target_date=None, tables=_resolve_tables(args.tables))
        print(status.to_string(index=False))
        return 0

    tables = _resolve_tables(args.tables)
    symbols = _resolve_symbols(args)
    recent_periods = None if args.all_history else (args.recent_periods or 2)

    print(
        f"财务数据更新: tables={tables} start_year={args.start_year} "
        f"recent_periods={recent_periods} symbols={'全市场' if symbols is None else len(symbols)} "
        f"include_delisted={args.include_delisted} threads={args.threads} dry_run={args.dry_run}"
    )
    print(f"落盘目录: {financial_dir}")

    summary = update_financials(
        tables=tables,
        start_year=args.start_year,
        end_year=args.end_year,
        symbols=symbols,
        include_delisted=args.include_delisted,
        recent_periods=recent_periods,
        threads=args.threads,
        dry_run=args.dry_run,
        replace=args.replace,
    )
    print("\n=== 汇总 ===")
    print(summary.to_string(index=False))

    if not args.dry_run:
        summary_fp = Path(financial_dir) / "update_summary.json"
        print(f"\n汇总 JSON: {summary_fp}")
        failed = summary[summary["status"].astype(str).str.startswith("failed")]
        if not failed.empty:
            print(f"有 {len(failed)} 张表更新失败 (见上方回溯)", file=sys.stderr)
            return 1
    payload = json.loads(summary.to_json(orient="records", force_ascii=False))
    return 0 if all(item.get("status") in {"ok", "dry-run"} for item in payload) else 1


if __name__ == "__main__":
    raise SystemExit(main())
