#!/usr/bin/env python
"""复权因子体检：校验 VWAP 的硬不变量 `low <= vwap <= high`（不联网）。

背景：alpha101 的 vwap = `value/(volume*100) * adj_factor`。成交均价必然落在当日
振幅内，且该不等式在乘上正的复权因子后依然成立 —— 因此它是"因子配错 / 日期错位 /
单位错（例如忘了 x100）"的硬探针，比任何抽样目测都可靠。

输出：控制台摘要 + JSON 报告（默认 data/adj_factor/health_report.json）。
退出码：0 = 全部通过（或缺数据的标的已跳过），1 = 存在越界行。

用法:
    uv run python libs/scripts/check_adj_factor.py --universe etf
    uv run python libs/scripts/check_adj_factor.py --symbols 600519 000001 --tol-rel 5e-4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DataPath  # noqa: E402
from data_manager.adj_factor_manager import (  # noqa: E402
    DEFAULT_TOL_REL,
    build_health_report,
    list_local_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验 vwap 是否落在当日 [low, high] 内（复权因子体检）。"
    )
    parser.add_argument("--universe", choices=["stock", "etf"], default="stock")
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="指定标的（默认本地已有行情的全部标的）")
    parser.add_argument("--tol-rel", type=float, default=DEFAULT_TOL_REL,
                        help=f"相对容差（默认 {DEFAULT_TOL_REL}，覆盖价格 2~3 位小数舍入）")
    parser.add_argument("--output", default=None,
                        help="JSON 报告路径（默认 data/adj_factor/health_report.json）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = args.symbols if args.symbols is not None else list_local_symbols(args.universe)
    if not symbols:
        print("没有可体检的标的（本地行情目录为空？）")
        return 1

    print(f"体检 {len(symbols)} 个标的（universe={args.universe}, tol_rel={args.tol_rel}）...")
    report = build_health_report(symbols, args.universe, tol_rel=args.tol_rel)

    output = Path(args.output) if args.output else Path(DataPath.ADJ_FACTOR_PATH) / "health_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"标的数            : {report['n_symbols']}")
    print(f"缺复权因子的标的  : {report['n_missing_factor']}")
    print(f"已校验行数        : {report['n_checked_rows']}")
    print(f"越界行数          : {report['n_violation_rows']} "
          f"(越界率 {report['violation_rate']:.4%})")
    for row in report["worst_symbols"]:
        print(f"  ! {row['symbol']}: {row['violations']}/{row['rows']} 行越界, "
              f"最大超出 {row['max_excess_rel']:.2%}（{row['first_date']}~{row['last_date']}）")
    if report["n_missing_factor"]:
        print(f"缺因子标的(前 20): {report['missing_factor_symbols'][:20]}")
    print(f"报告: {output}")

    if report["n_checked_rows"] == 0:
        print("没有可校验的行（先跑 update_adj_factor.py 建库）。")
        return 1
    return 1 if report["n_violation_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
