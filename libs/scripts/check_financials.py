#!/usr/bin/env python3
"""财务数据体检 CLI (Financial Data Health Check CLI)

对 ``data/financial/*.csv`` 跑机器可判定的质量与准确性校验, 产出
``data/financial/health_report.json`` 与终端摘要。**默认不联网**。

检查项 (与公司 2026-09-10 需求清单 §5 对应):
  1. schema_and_column_order  列序与取数单表头逐字一致
  2. missing_value_convention 缺失必须留空 (拒绝 "NULL"/-1/0 填充)
  3. update_time_integrity    落盘时间可解析且非未来
  4. field_missing_rates      逐字段缺失率 (核心字段 >5% → fail)
  5. date_axis_invariants     ann_date >= report_date 必须 100%; 超法定披露期 <0.5%;
                              4/8/10 月披露峰; 非交易日公告占比 (只记录)
  6. revision_quantification  basis=revised 分布与更正滞后天数 (数值是否被追溯调整)
  7. unit_dimension_checks    资产=负债+权益; 归母 ≤ 合并净利; eps×股本 ≈ 归母净利;
                              年度中位数 1000 倍跳变
  8. coverage_survivorship    按报告期覆盖率; 退市股抽查 (600696/600193/605081)
 10. extremes_and_tradability 极值/可交易性; 公告日→交易日映射; 孤儿代码
 11. cross_source_audit      跨源一致性: 股本双源、行情侧流通股本、联网抽样数值审计 (见 audit_financials.py)
 12. field_fillability       33 字段可填率 (对照需求清单 §③-A)

用法:
    # 全表体检 (默认; 不联网)
    uv run python libs/scripts/check_financials.py

    # 只看某几张表 / 指定输出
    uv run python libs/scripts/check_financials.py --tables income_q,balance_q
    uv run python libs/scripts/check_financials.py --output /tmp/health.json

    # 退出码策略: hard(默认, 有 fail 即 1) / warn(有 warn 也 1) / never(恒 0)
    uv run python libs/scripts/check_financials.py --fail-on warn

    # 一次性双源公告日对拍 (联网, 见 scripts/crosscheck_financials.py)
    uv run python libs/scripts/crosscheck_financials.py --period 20250630 --sample 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from config import DataPath  # noqa: E402
from data_manager.financial_health import build_report, print_report, write_report  # noqa: E402
from data_manager.financial_manager import list_tables  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="财务数据体检 (默认不联网)。")
    parser.add_argument(
        "--tables",
        default="all",
        help=f"逗号分隔的表名, 或 all (默认)。可选: {','.join(list_tables())}",
    )
    parser.add_argument("--data-dir", default=None, help=f"数据目录 (默认 {DataPath.FINANCIAL_DIR})。")
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出路径 (默认 <data-dir>/health_report.json)。",
    )
    parser.add_argument(
        "--fail-on",
        choices=("hard", "warn", "never"),
        default="hard",
        help="退出码策略: hard=有 fail 即 1 (默认); warn=有 warn 也 1; never=恒 0。",
    )
    parser.add_argument("--quiet", action="store_true", help="只写 JSON, 不打印摘要。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tables = None if args.tables.strip().lower() == "all" else [
        item.strip() for item in args.tables.split(",") if item.strip()
    ]
    unknown = [item for item in (tables or []) if item not in list_tables()]
    if unknown:
        raise SystemExit(f"未知表名: {unknown}; 可选: {list_tables()}")

    data_dir = args.data_dir or DataPath.FINANCIAL_DIR
    report = build_report(tables=tables, data_dir=data_dir)
    output = args.output or str(Path(data_dir) / "health_report.json")
    write_report(report, output)

    if not args.quiet:
        print_report(report)
        print(f"\n报告: {output}")

    verdict = report["verdict"]
    if args.fail_on == "never":
        return 0
    if args.fail_on == "warn":
        return 1 if verdict in {"warn", "fail"} else 0
    return 1 if verdict == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
