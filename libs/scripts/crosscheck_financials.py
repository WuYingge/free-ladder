#!/usr/bin/env python3
"""财务公告日一次性双源对拍 CLI (One-off Announcement-Date Cross Check)

**用途: 一次性验收**。用与入库源相互独立的第二信源 (东财业绩报表
``RPT_LICO_FN_CPD`` ≈ akshare ``stock_yjbb_em``) 对已落盘的
``data/financial/income_q.csv`` 做 ``ann_date`` 逐股对拍, 通过后归档, 不进日常流程。

为什么不用它入库: 该表的 "最新公告日期" 会随更正公告**整体前移**, 不能代表
首次公告日 (违反 PIT), 因此只作对拍基准。

默认抽样: 按公告日分层, 每层取 2 只, 覆盖面尽量广 (实测 2025H1: 96 个不同公告日)。

用法:
    # 对 2025-06-30 报告期抽样 150 只 (联网)
    uv run python libs/scripts/crosscheck_financials.py --period 20250630 --sample 150

    # 指定输出与容差
    uv run python libs/scripts/crosscheck_financials.py --period 20241231 --sample 200 \\
        --tolerance 0.005 --output data/financial/crosscheck_report.json

退出码: 一致率 >= 1 - tolerance → 0; 否则 1 (并打印全部不一致行, 交人工裁决)。
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

import pandas as pd  # noqa: E402

from data_manager.financial_manager import get_financial_dir, load_table  # noqa: E402
from fetcher.financial import (  # noqa: E402
    get_financial_statement_data,
    get_period_announce_crosscheck_rows,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一次性双源公告日对拍 (联网)。")
    parser.add_argument("--period", required=True, help="报告期, 如 20250630。")
    parser.add_argument("--table", default="income_q", help="被对拍的落盘表 (默认 income_q)。")
    parser.add_argument("--sample", type=int, default=150, help="抽样股票数上限 (默认 150)。")
    parser.add_argument(
        "--per-ann-date",
        type=int,
        default=2,
        help="按公告日分层时每层抽样数 (默认 2)。",
    )
    parser.add_argument("--tolerance", type=float, default=0.005, help="允许的不一致比例 (默认 0.5%%, argparse 转义)。")
    parser.add_argument(
        "--source",
        choices=("local", "live"),
        default="local",
        help="local=与已落盘数据对拍 (快); live=逐股现场抓 F10 (慢, 用于验证抓取管线)。",
    )
    parser.add_argument("--seed", type=int, default=7, help="抽样随机种子 (默认 7, 可复现)。")
    parser.add_argument("--output", default=None, help="报告输出路径。")
    return parser.parse_args(argv)


def _layer_sample(codes: list[str], ann_dates: pd.Series, per_layer: int, cap: int, seed: int) -> list[str]:
    """按公告日分层抽样: 每个公告日取 per_layer 只, 直到达到 cap。"""
    buckets: dict[str, list[str]] = defaultdict(list)
    for code, ann in zip(codes, ann_dates):
        buckets[str(ann)].append(code)
    rng = random.Random(seed)
    for value in buckets.values():
        rng.shuffle(value)
    ordered = [code for key in sorted(buckets) for code in buckets[key][:per_layer]]
    return ordered[:cap]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    period_date = pd.to_datetime(args.period, format="%Y%m%d")
    period_str = period_date.strftime("%Y-%m-%d")

    local, _ = load_table(args.table)
    if local.empty:
        print(f"落盘表 {args.table} 为空, 无法对拍; 请先跑 update_financials.py", file=sys.stderr)
        return 2
    local["symbol"] = local["symbol"].astype(str).str.zfill(6)
    target = local[pd.to_datetime(local["report_date"]) == period_date].copy()
    if target.empty:
        print(f"{args.table} 内没有报告期 {period_str} 的数据", file=sys.stderr)
        return 2
    print(f"落盘 {args.table} 报告期 {period_str}: {len(target)} 行, {target['symbol'].nunique()} 只股票")

    print("拉取第二信源 (东财业绩报表 RPT_LICO_FN_CPD) ...")
    reference = get_period_announce_crosscheck_rows(args.period)
    if reference.empty:
        print("第二信源返回空", file=sys.stderr)
        return 2
    ref = reference.drop_duplicates(subset=["symbol"], keep="first").set_index("symbol")
    print(f"第二信源: {len(ref)} 只股票")

    candidates = [code for code in target["symbol"].tolist() if code in ref.index]
    if not candidates:
        print("两源股票代码无交集", file=sys.stderr)
        return 2
    ann_series = target.set_index("symbol").loc[candidates, "ann_date"]
    sample = _layer_sample(candidates, ann_series, args.per_ann_date, args.sample, args.seed)
    print(f"分层抽样: {len(sample)} 只, 覆盖 {ann_series.loc[sample].nunique()} 个不同公告日")

    rows: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    missing = 0
    for code in sample:
        local_ann = pd.Timestamp(ann_series.loc[code])
        if args.source == "live":
            try:
                data = get_financial_statement_data(code, periods=None, statements=("income",))
                income = data.get("income")
                if income is None or income.empty:
                    missing += 1
                    continue
                pick = income[pd.to_datetime(income["report_date"]) == period_date]
                if pick.empty:
                    missing += 1
                    continue
                local_ann = pd.Timestamp(pick.iloc[0]["ann_date"])
            except Exception as err:  # noqa: BLE001
                print(f"  抓取失败 {code}: {type(err).__name__}: {err}")
                missing += 1
                continue
        ref_ann = pd.Timestamp(ref.loc[code, "ann_date"])
        match = bool(local_ann == ref_ann)
        rows.append(
            {
                "symbol": code,
                "local_ann_date": str(local_ann.date()) if pd.notna(local_ann) else None,
                "reference_ann_date": str(ref_ann.date()) if pd.notna(ref_ann) else None,
                "match": match,
            }
        )
        if not match:
            mismatches.append(rows[-1])

    checked = len(rows)
    agreement = (checked - len(mismatches)) / checked if checked else 0.0
    report = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": period_str,
        "table": args.table,
        "source_mode": args.source,
        "local_rows": int(len(target)),
        "local_symbols": int(target["symbol"].nunique()),
        "reference_symbols": int(len(ref)),
        "sampled": len(sample),
        "checked": checked,
        "missing_on_one_side": missing,
        "mismatches": len(mismatches),
        "agreement": round(agreement, 6),
        "tolerance": args.tolerance,
        "passed": agreement >= (1.0 - args.tolerance) and checked > 0,
        "mismatch_examples": mismatches[:50],
        "samples": rows[:200],
    }

    output = args.output or str(Path(get_financial_dir()) / "crosscheck_report.json")
    parent = Path(output).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print(
        f"\n对拍结果: 检查 {checked} 只, 一致 {checked - len(mismatches)}, "
        f"不一致 {len(mismatches)}, 单侧缺失 {missing} → 一致率 {agreement:.4%} "
        f"(阈值 {1 - args.tolerance:.2%})"
    )
    for item in mismatches[:20]:
        print(f"  ❌ {item['symbol']}: 落盘 {item['local_ann_date']} vs 信源 {item['reference_ann_date']}")
    print(f"报告: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
