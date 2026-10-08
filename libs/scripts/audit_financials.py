#!/usr/bin/env python3
"""财务数据跨源审计 CLI (Cross-Source Audit CLI)

体检 (``check_financials.py``) 查的是**单源自洽性**; 本脚本查**跨源一致性** ——
每一列都要有一个独立仲裁者, 否则源端的系统性偏移 (如东财 `ann_date` 错位一年) 永远发现不了。

三套审计:
  1. ``shares``  东财 F10 的 ``balance_q.total_share``  vs  巨潮 ``share_capital.total_share``
                 (按股本变动生效日 asof 对齐) —— **离线, 全量** (实测一致率 97.7%)
  2. ``quotes``  ``balance_q.total_share``  vs  行情侧 ``daily_basic.float_share``
                 (判据: 流通股本不得超过总股本) —— **离线, 全量** (实测违反占比 2.2%)
  3. ``values``  本地三表 vs **新浪**三表, 核心字段 (净利润/资产总计/负债合计/经营现金流) 逐格
                 —— **联网, 分层抽样** (板块 × 上市年代, 固定种子可复现)

任一套不过阈值 → 退出码 1 (``--fail-on gated``, 默认)。

用法:
    # 离线全量 (秒级, 建议每次回填后跑)
    uv run python libs/scripts/audit_financials.py

    # 联网抽样数值对拍 (约 100 只 / 4 分钟; 8 线程)
    uv run python libs/scripts/audit_financials.py --suites values --sample 100

    # 指定分层抽样规模与种子 / 只看溯源概览
    uv run python libs/scripts/audit_financials.py --suites values --sample 300 --seed 11
    uv run python libs/scripts/audit_financials.py --provenance

    # 退出码策略: gated(默认, 受判定字段低于下限即 1) / never
    uv run python libs/scripts/audit_financials.py --fail-on never
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import DataPath  # noqa: E402
from data_manager import financial_audit as audit  # noqa: E402
from data_manager import financial_provenance as prov  # noqa: E402

DEFAULT_SUITES = "shares,quotes"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="财务数据跨源审计 (shares/quotes 离线全量, values 联网抽样)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suites", default=DEFAULT_SUITES, help=f"逗号分隔, 默认 {DEFAULT_SUITES}; 可选 shares,quotes,values")
    parser.add_argument("--sample", type=int, default=100, help="values 套件的抽样股票数 (默认 100)")
    parser.add_argument("--seed", type=int, default=7, help="分层抽样随机种子 (默认 7, 同种子结果可复现)")
    parser.add_argument("--threads", type=int, default=None, help="values 套件并发线程数 (默认取 FINANCIAL_FETCH_THREADS)")
    parser.add_argument("--tolerance", type=float, default=audit.DEFAULT_TOLERANCE, help="相对容差 (默认 0.005 = 0.5%%)")
    parser.add_argument("--symbols", default="", help="只审计指定股票 (逗号分隔); 默认分层抽样")
    parser.add_argument("--codes-file", default="", help="从文件读股票代码 (每行一个); 与 --symbols 二选一")
    parser.add_argument("--output", default="", help=f"报告输出路径 (默认 {Path(DataPath.FINANCIAL_DIR) / 'audit_report.json'})")
    parser.add_argument("--provenance", action="store_true", help="只打印溯源概览 (默认来源 + 修正流水统计)")
    parser.add_argument("--fail-on", choices=("gated", "never"), default="gated", help="退出码策略 (默认 gated)")
    parser.add_argument("--quiet", action="store_true", help="只输出退出码")
    return parser.parse_args(argv)


def _resolve_sample(args: argparse.Namespace) -> list[str] | None:
    if args.codes_file:
        return [line.strip().zfill(6) for line in Path(args.codes_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.symbols:
        return [item.strip().zfill(6) for item in args.symbols.split(",") if item.strip()]
    return None


def _print_suite(name: str, payload: dict) -> None:
    if payload.get("status") != "ok":
        print(f"  [{name}] skip: {payload.get('reason', '')}")
        return
    if name in {"shares", "quotes"}:
        print(f"  [{name}] {payload.get('source_pair')}")
        for key in ("compared_rows", "mismatch_rows", "agreement", "float_exceeds_total", "float_exceeds_total_share", "fully_floated_share"):
            if key in payload:
                print(f"        {key} = {payload[key]}")
        for example in payload.get("examples", [])[:3]:
            print(f"        例: {example}")
        return
    print(f"  [{name}] {payload.get('source_pair')} | 抽样 {payload.get('sampled_symbols')} 只"
          f" | 取数失败 {payload.get('failure_count', 0)}")
    for field, info in (payload.get("fields") or {}).items():
        flag = "判定" if info.get("gated") else "仅记录"
        print(f"        {field:18s} 一致率 {info['agreement']:.4%} ({info['compared'] - info['mismatch']}/{info['compared']}, {flag})")
        for example in info.get("examples", [])[:2]:
            print(f"            不一致例: {example}")


def _gated_failures(payload: dict) -> list[str]:
    issues: list[str] = []
    shares = payload["suites"].get("shares") or {}
    if shares.get("status") == "ok" and float(shares.get("agreement") or 0) < 0.95:
        issues.append(f"shares 一致率 {shares['agreement']:.4%} < 95%")
    quotes = payload["suites"].get("quotes") or {}
    if quotes.get("status") == "ok" and float(quotes.get("float_exceeds_total_share") or 0) > 0.05:
        issues.append(f"quotes 流通>总股本占比 {quotes['float_exceeds_total_share']:.4%} > 5%")
    values = payload["suites"].get("values") or {}
    for field, info in (values.get("fields") or {}).items():
        if info.get("gated") and float(info.get("agreement") or 0) < 0.95:
            issues.append(f"values.{field} 一致率 {info['agreement']:.4%} < 95%")
    return issues


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.provenance:
        summary = prov.provenance_summary()
        if not args.quiet:
            print("财务数据溯源概览 (默认来源 + 修正流水)")
            print(summary.to_string(index=False))
        return 0

    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    unknown = [item for item in suites if item not in {"shares", "quotes", "values"}]
    if unknown:
        raise SystemExit(f"未知 suite: {unknown}; 可选 shares,quotes,values")

    payload = audit.run_audit(
        suites,
        n=args.sample,
        seed=args.seed,
        threads=args.threads,
        tolerance=args.tolerance,
        sample=_resolve_sample(args),
    )
    output = args.output or str(Path(DataPath.FINANCIAL_DIR) / "audit_report.json")
    audit.write_audit_report(payload) if not args.output else None
    if args.output:
        Path(output).write_text(pd.io.json.dumps(payload, indent=2, force_ascii=False), encoding="utf-8")

    if not args.quiet:
        print(f"财务数据跨源审计 · {payload['generated_at']} · 套件 {','.join(suites)}")
        for name in suites:
            _print_suite(name, payload["suites"].get(name) or {})

    issues = _gated_failures(payload)
    if not args.quiet:
        print(f"\n报告: {output}")
        if issues:
            print("判定: FAIL")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("判定: PASS (受判定字段均在阈值内)")
    if args.fail_on == "never":
        return 0
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
