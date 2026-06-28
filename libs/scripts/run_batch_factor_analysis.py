#!/usr/bin/env python3
"""
批量因子分析编排器 (Batch Factor Analysis Runner)

一次性对 libs/factors/ 下全部已注册因子执行 factor_analysis，
每个因子支持自定义参数和参数网格扫描，并通过子进程并行加速。

用法:
    # 快速模式：所有因子默认参数，只跑 Layer 1+2（不含分组检验）
    python libs/scripts/run_batch_factor_analysis.py --mode quick

    # 标准模式：所有因子默认参数，跑 Layer 1+2+3
    python libs/scripts/run_batch_factor_analysis.py --mode standard

    # 完整模式：有 window 参数的因子做参数网格扫描，跑 Layer 1+2+3
    python libs/scripts/run_batch_factor_analysis.py --mode full

    # 自定义：只分析特定因子族
    python libs/scripts/run_batch_factor_analysis.py \\
        --families 价格动量族 均线偏离族 \\
        --mode standard

    # 控制并行度
    python libs/scripts/run_batch_factor_analysis.py \\
        --mode quick --parallel 4 --max-workers 2

    # 断点续跑 / 强制重跑
    python libs/scripts/run_batch_factor_analysis.py --mode full --resume
    python libs/scripts/run_batch_factor_analysis.py --mode full --force

    # 只分析指定因子
    python libs/scripts/run_batch_factor_analysis.py \\
        --factors PriceReturn TrendR2 RSRS --mode standard

参数:
    --mode        分析模式: quick / standard / full
    --factors     指定因子列表（空格分隔，默认全量）
    --families    指定因子族（如 价格动量族，默认全部）
    --parallel    并行度：同时运行的因子分析进程数（默认 2）
    --max-workers 每个因子分析内部的多进程 worker 数（默认 2）
    --resume      断点续跑：跳过已有 report.json 的因子
    --force       强制重跑：即使已有报告也重新分析
    --dry-run     试运行：只打印会执行的分析任务，不实际运行
    --output-dir  汇总输出目录（默认 data/factors/_batch_summary/）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

# 复用 CLI 中的 FACTOR_REGISTRY
from scripts.run_factor_analysis import FACTOR_REGISTRY, _import_factor

# ═══════════════════════════════════════════════════════════════════════════════
# 因子族分类（用于 --families 筛选）
# ═══════════════════════════════════════════════════════════════════════════════

FACTOR_FAMILIES: dict[str, list[str]] = {
    "价格动量族": [
        "PriceReturn", "RiskAdjustedReturn", "IntradayMomentum",
        "OvernightReturn", "HighPointPosition", "LowPointPosition",
        "TimeSeriesMomentum",
    ],
    "反转族": ["ShortTermReversal", "ExtremeReversal", "VolumeReversal"],
    "成交量族": [
        "VolumeRatio", "VolumePriceCorrelation", "OBV", "VPT",
        "AmihudIlliquidity", "VolumeStd", "VolumeSkew", "AverageAmount",
    ],
    "波动率族": [
        "DownsideVolatility", "ParkinsonVolatility", "GarmanKlassVolatility",
        "VolOfVol", "MaxDrawdown", "AvgDrawdown",
    ],
    "趋势质量族": [
        "HurstExponent", "KaufmanER", "UpDownRatio",
        "ConsecutiveUpDays", "ConsecutiveDownDays", "ADX",
    ],
    "超买超卖族": [
        "RSI", "Stochastic", "CCI", "WilliamsR", "MFI", "UltimateOscillator",
    ],
    "均线偏离族": [
        "MAPosition", "MA", "BIAS", "BollingerBandPosition",
        "MAAlignment", "MASlope", "MADistance", "MADispersion",
    ],
    "分布形态族": [
        "ReturnSkew", "ReturnKurtosis", "HistoricalVaR", "CVaR",
        "MFE", "MAE", "ID",
    ],
    "突破族": [
        "NewHigh", "DailyRebound", "TrendR2", "RSRS", "ATR",
        "NewHighContinuous", "NewLowContinuous", "DonchianChannelPosition",
        "ATRRatio", "ChandelierExit",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# 参数网格定义
# ═══════════════════════════════════════════════════════════════════════════════
# 只对核心 window 类参数定义网格。每个因子跑默认参数一次（主力分析），
# 再可选配 param_grid 做参数敏感度扫描。
# 原则：避免组合爆炸，每个因子最多 3-4 个参数值。

def _window_grid(values: list[int]) -> dict[str, list]:
    """为 window 参数生成网格。"""
    return {"window": values}

# full 模式下使用的参数网格
FULL_MODE_PARAM_GRIDS: dict[str, dict[str, list]] = {
    # ── 价格动量族 ──
    "PriceReturn": _window_grid([10, 20, 60]),
    "RiskAdjustedReturn": _window_grid([20, 60, 120]),
    # HighPointPosition/LowPointPosition: window=20 足够，不扫网格
    # IntradayMomentum/OvernightReturn: 无参数
    "TimeSeriesMomentum": _window_grid([60, 120, 252]),
    # ── 反转族 ──
    "ShortTermReversal": _window_grid([1, 5, 10, 20]),
    # ── 成交量族 ──
    "VolumeRatio": _window_grid([5, 10, 20]),
    "VolumePriceCorrelation": _window_grid([10, 20, 60]),
    "VolumeStd": _window_grid([10, 20, 60]),
    "VolumeSkew": _window_grid([10, 20, 60]),
    "AverageAmount": _window_grid([10, 20, 60]),
    "AmihudIlliquidity": _window_grid([10, 20, 60]),
    # ── 波动率族 ──
    "DownsideVolatility": _window_grid([10, 20, 60]),
    "ParkinsonVolatility": _window_grid([10, 20, 60]),
    "GarmanKlassVolatility": _window_grid([10, 20, 60]),
    "MaxDrawdown": _window_grid([20, 60, 120]),
    "AvgDrawdown": _window_grid([20, 60, 120]),
    # ── 趋势质量族 ──
    "HurstExponent": _window_grid([60, 120, 240]),
    "KaufmanER": _window_grid([10, 20, 60]),
    "UpDownRatio": _window_grid([10, 20, 60]),
    # ── 超买超卖族 ──
    "RSI": _window_grid([7, 14, 21]),
    "CCI": _window_grid([10, 20, 40]),
    "WilliamsR": _window_grid([7, 14, 28]),
    "MFI": _window_grid([7, 14, 21]),
    # ── 均线偏离族 ──
    "MAPosition": _window_grid([60, 120, 200, 250]),
    "MA": _window_grid([10, 20, 60, 120]),
    "BIAS": _window_grid([10, 20, 60]),
    "MASlope": {"ma_window": [10, 20, 60], "slope_window": [3, 5, 10]},
    "MADistance": {"short_window": [5, 10, 20], "long_window": [60, 120, 200]},
    # ── 分布形态族 ──
    "ReturnSkew": _window_grid([20, 60, 120]),
    "ReturnKurtosis": _window_grid([20, 60, 120]),
    "HistoricalVaR": _window_grid([60, 120, 252]),
    "CVaR": _window_grid([60, 120, 252]),
    "MFE": _window_grid([10, 20, 60]),
    "MAE": _window_grid([10, 20, 60]),
    "ID": _window_grid([10, 20, 60]),
    # ── 突破族 ──
    "TrendR2": {"window": [60, 120, 240], "output": ["r2", "slope"]},
    "RSRS": {
        "regression_window": [10, 18, 30],
        "output": ["zscore"],
    },
    "ATR": _window_grid([14, 25, 50]),
    "NewHighContinuous": _window_grid([20, 50, 100]),
    "NewLowContinuous": _window_grid([20, 50, 100]),
    "DonchianChannelPosition": _window_grid([10, 20, 60]),
    "ATRRatio": _window_grid([14, 25, 50]),
    "ChandelierExit": {"n": [10, 22, 44], "atr_window": [10, 22, 44]},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 分析任务定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisTask:
    """单个因子分析任务的定义。"""
    factor_name: str          # FACTOR_REGISTRY 键名
    factor_cls: type          # 因子类
    default_params: dict      # 默认构造参数
    layers: tuple[int, ...]   # 分析层
    param_grid: dict | None   # 参数网格（None 表示不扫）
    extra_args: list[str] = field(default_factory=list)  # 额外 CLI 参数

    @property
    def cli_args(self) -> list[str]:
        """生成 subprocess 调用 CLI 的参数列表。"""
        args = [
            sys.executable,
            str(LIBS_DIR / "scripts" / "run_factor_analysis.py"),
            "--factor", self.factor_name,
            "--layers", *[str(l) for l in self.layers],
        ]
        # 默认参数通过 --param 传入
        for k, v in self.default_params.items():
            if isinstance(v, list):
                vals = ",".join(str(x) for x in v)
                args.extend(["--param", f"{k}=[{vals}]"])
            else:
                args.extend(["--param", f"{k}={v}"])
        # 参数网格
        if self.param_grid:
            args.extend(["--param-grid", json.dumps(self.param_grid)])
        # 额外参数
        args.extend(self.extra_args)
        return args

    @property
    def output_dir(self) -> Path:
        """预期的输出目录路径。"""
        # 与 FactorAnalysisConfig.resolve_output_root() 保持一致
        # LIBS_DIR 已在 sys.path 中，直接 import config
        from config import DataPath
        factor_name = self.factor_name
        # 用默认参数实例化因子来获取 output_name
        try:
            factor_inst = self.factor_cls(**self.default_params)
            sanitized = factor_inst.get_output_name().replace("/", "_").replace("\\", "_").replace(":", "_")
        except Exception:
            sanitized = factor_name
        return Path(DataPath.DATA_DIR) / "factors" / sanitized

    def _latest_report(self) -> Path | None:
        """返回 output_dir 中最新的 report_*.json 路径，不存在时返回 None。"""
        if not self.output_dir.exists():
            return None
        reports = sorted(self.output_dir.glob("report_*.json"))
        return reports[-1] if reports else None

    @property
    def report_json(self) -> Path | None:
        """最新的 report JSON 路径（向后兼容）。"""
        return self._latest_report()

    @property
    def report_end_date(self) -> str | None:
        """从最新报告中提取面板结束日期（YYYY-MM-DD），失败/不存在时返回 None。"""
        rp = self._latest_report()
        if rp is None:
            return None
        try:
            with open(rp) as f:
                report = json.load(f)
            return report.get("panel_summary", {}).get("end_date")
        except Exception:
            return None

    def is_fresh(self, max_age_days: int | None = None) -> bool:
        """报告是否存在且数据足够新鲜。

        Parameters
        ----------
        max_age_days: 报告 end_date 距今超过此天数视为过期。None = 永不视为过期。

        Returns
        -------
        bool: True = 报告存在且数据在 max_age_days 以内。
        """
        end_str = self.report_end_date
        if end_str is None:
            return False  # 报告不存在
        if max_age_days is None:
            return True   # 不检查数据新鲜度
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
            age = (date.today() - end_date).days
            return age <= max_age_days
        except ValueError:
            return False

    @property
    def is_done(self) -> bool:
        """检查是否已完成分析（output_dir 中存在任一 report_*.json）。

        注意：此方法不检查数据新鲜度。如需检查新鲜度请用 is_fresh()。
        """
        return self._latest_report() is not None

    @property
    def label(self) -> str:
        pg = " [网格]" if self.param_grid else ""
        extra = ""
        end_str = self.report_end_date
        if end_str:
            extra = f" (已有报告, 数据至 {end_str})"
        return f"{self.factor_name}{pg}{extra}"


# ═══════════════════════════════════════════════════════════════════════════════
# 批量编排器
# ═══════════════════════════════════════════════════════════════════════════════

def build_tasks(
    factor_names: list[str],
    mode: str,
    extra_args: list[str] | None = None,
) -> list[AnalysisTask]:
    """根据模式和因子列表构建分析任务。

    Parameters
    ----------
    factor_names: 要分析的因子名列表。
    mode: "quick" | "standard" | "full"
    extra_args: 传递给 CLI 的额外参数（如 --max-workers）。

    Returns
    -------
    list[AnalysisTask]
    """
    if extra_args is None:
        extra_args = []

    tasks = []

    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            print(f"警告: 未知因子 '{name}'，跳过")
            continue

        module_path, class_name, default_params = FACTOR_REGISTRY[name]
        factor_cls = _import_factor(name)

        if mode == "quick":
            # 快速模式：默认参数，只跑 Layer 1+2（因子质量 + IC）
            tasks.append(AnalysisTask(
                factor_name=name,
                factor_cls=factor_cls,
                default_params=dict(default_params),
                layers=(1, 2),
                param_grid=None,
                extra_args=list(extra_args),
            ))
        elif mode == "standard":
            # 标准模式：默认参数，跑完整 Layer 1+2+3
            tasks.append(AnalysisTask(
                factor_name=name,
                factor_cls=factor_cls,
                default_params=dict(default_params),
                layers=(1, 2, 3),
                param_grid=None,
                extra_args=list(extra_args),
            ))
        elif mode == "full":
            # 完整模式：默认参数 + 参数网格，跑 Layer 1+2+3
            pg = FULL_MODE_PARAM_GRIDS.get(name)
            tasks.append(AnalysisTask(
                factor_name=name,
                factor_cls=factor_cls,
                default_params=dict(default_params),
                layers=(1, 2, 3),
                param_grid=pg,
                extra_args=list(extra_args),
            ))

    return tasks


def run_task(task: AnalysisTask) -> dict[str, Any]:
    """在子进程中运行单个分析任务。

    Returns
    -------
    dict: {"factor": str, "success": bool, "output_dir": str, "runtime_sec": float, "error": str|None}
    """
    start = time.monotonic()
    result = {
        "factor": task.factor_name,
        "success": False,
        "output_dir": str(task.output_dir),
        "runtime_sec": 0.0,
        "error": None,
    }

    try:
        proc = subprocess.run(
            task.cli_args,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 单因子最长 2 小时
        )
        result["runtime_sec"] = time.monotonic() - start

        if proc.returncode == 0:
            result["success"] = True
            # 尝试解析输出获取文件列表
            stdout = proc.stdout
            result["stdout_tail"] = stdout[-500:] if len(stdout) > 500 else stdout
        else:
            result["error"] = (
                f"exit_code={proc.returncode}\n"
                f"STDERR: {proc.stderr[-500:]}\n"
                f"STDOUT: {proc.stdout[-200:]}"
            )
    except subprocess.TimeoutExpired:
        result["runtime_sec"] = 7200
        result["error"] = "超时（2小时）"
    except Exception as e:
        result["runtime_sec"] = time.monotonic() - start
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总报告生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary_csv(results: list[dict], output_dir: Path) -> Path:
    """生成汇总 CSV，包含每个因子分析的 key metrics。"""
    import pandas as pd

    rows = []
    for r in results:
        row = {
            "factor": r["factor"],
            "success": r["success"],
            "runtime_sec": round(r.get("runtime_sec", 0), 1),
            "output_dir": r.get("output_dir", ""),
            "error": (r.get("error", "") or "")[:200],
        }

        # 尝试从 report_*.json 提取关键指标（取最新一个）
        output_dir = Path(r.get("output_dir", ""))
        report_files = sorted(output_dir.glob("report_*.json")) if output_dir.exists() else []
        report_path = report_files[-1] if report_files else None
        if report_path and report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)

                panel = report.get("panel_summary", {})
                row["n_symbols"] = panel.get("n_symbols")
                row["n_dates"] = panel.get("n_dates")
                row["start_date"] = panel.get("start_date")
                row["end_date"] = panel.get("end_date")
                row["coverage_mean"] = panel.get("coverage_mean")

                # Layer 2: IC（取 20 日持仓期）
                predictive = report.get("layer2_predictive", {})
                rank_ic = predictive.get("rank_ic", {})
                for period_key, ic_data in rank_ic.items():
                    summary = ic_data.get("summary", {})
                    period_int = int(period_key)
                    row[f"rank_ic_mean_{period_int}d"] = summary.get("mean")
                    row[f"rank_ic_ir_{period_int}d"] = summary.get("ir")

                # 参数网格最佳参数
                param_grid = predictive.get("param_grid", {})
                best = param_grid.get("best_params", {})
                if best:
                    row["best_params"] = json.dumps(best, default=str)
                    row["best_ic_mean"] = best.get("ic_mean")
            except Exception as e:
                row["report_parse_error"] = str(e)[:100]

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "batch_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n汇总 CSV 已保存: {csv_path}")
    print(f"  总任务: {len(rows)} | 成功: {df['success'].sum()} | 失败: {(~df['success']).sum()}")
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量因子分析 — 一键跑完全部因子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "standard", "full"],
        help="分析模式: quick (Layer 1+2), standard (Layer 1+2+3), full (Layer 1+2+3 + 参数网格)",
    )
    parser.add_argument(
        "--factors", nargs="*", default=None,
        help="指定因子列表（空格分隔），默认全量",
    )
    parser.add_argument(
        "--families", nargs="*", default=None,
        help=f"指定因子族，可选: {list(FACTOR_FAMILIES.keys())}",
    )
    parser.add_argument(
        "--parallel", type=int, default=2,
        help="并行度：同时运行的因子分析进程数（默认 2）",
    )
    parser.add_argument(
        "--max-workers", type=int, default=2,
        help="每个因子内部的多进程 worker 数（默认 2）",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help="断点续跑：跳过已有报告的因子（可与 --max-age 配合用）",
    )
    parser.add_argument(
        "--max-age", type=int, default=0,
        metavar="DAYS",
        help="报告有效天数。与 --resume 配合：报告 end_date 距今超过此天数则视为过期、触发重跑。"
             " 0=仅检查文件是否存在；默认 0。"
             " 例：--resume --max-age 7 表示跳过 7 天内跑过的因子。",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="强制重跑：忽略所有已有报告，全量重跑",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="试运行：只打印任务列表，不实际运行",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="汇总输出目录（默认 data/factors/_batch_summary/）",
    )
    parser.add_argument(
        "--min-bars", type=int, default=252,
        help="最少交易日数，传递给 CLI",
    )
    parser.add_argument(
        "--forward-periods", nargs="+", type=int, default=[5, 10, 20, 60],
        help="前向持仓期，传递给 CLI",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── 1. 确定因子列表 ────────────────────────────────────────────────────
    if args.factors:
        factor_names = list(args.factors)
    elif args.families:
        factor_names = []
        for family in args.families:
            if family in FACTOR_FAMILIES:
                factor_names.extend(FACTOR_FAMILIES[family])
            else:
                print(f"警告: 未知因子族 '{family}'，可用: {list(FACTOR_FAMILIES.keys())}")
        factor_names = list(dict.fromkeys(factor_names))  # 去重保序
    else:
        factor_names = list(FACTOR_REGISTRY.keys())

    print(f"模式: {args.mode} | 因子数: {len(factor_names)} | 并行度: {args.parallel}")
    print()

    # ── 2. 构建任务 ────────────────────────────────────────────────────────
    extra_cli_args = [
        "--min-bars", str(args.min_bars),
        "--forward-periods", *[str(p) for p in args.forward_periods],
        "--max-workers", str(args.max_workers),
    ]

    tasks = build_tasks(factor_names, args.mode, extra_cli_args)

    # ── 3. 断点续跑 / 强制重跑 / 数据新鲜度逻辑 ───────────────────────────
    max_age = args.max_age if args.max_age > 0 else None

    if args.force:
        print("强制重跑: 将覆盖已有报告")
    elif args.resume:
        # 分层判断：先看 max_age（数据新鲜度），再看文件是否存在
        stale: list[AnalysisTask] = []
        skipped: list[AnalysisTask] = []

        for t in tasks:
            if t.is_fresh(max_age):
                skipped.append(t)
            elif t.is_done:
                # 有报告但数据过期
                stale.append(t)
            # else: 没有报告 → 留在 tasks 中

        tasks = [t for t in tasks if not t.is_done or t in stale]
        if skipped:
            print(f"断点续跑: 跳过 {len(skipped)} 个新鲜任务")
            for s in skipped:
                print(f"  ✓ {s.label}")
            print()
        if stale:
            print(f"数据过期: {len(stale)} 个任务将重跑（报告 end_date 超过 {max_age} 天前）")
            for s in stale:
                print(f"  ↻ {s.label}")
            print()
        if not skipped and not stale and not tasks:
            print("断点续跑: 所有任务均无报告，全量运行")
            print()

    if not tasks:
        print("没有待执行的任务。")
        return 0

    print(f"待执行任务: {len(tasks)} 个")
    for t in tasks:
        pg_info = ""
        if t.param_grid:
            total_combos = 1
            for v in t.param_grid.values():
                total_combos *= len(v)
            pg_info = f"  (参数组合: {total_combos})"
        print(f"  → {t.label}{pg_info}")
    print()

    # ── 4. 试运行 ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("试运行模式：不执行实际分析。")
        print(f"\n示例命令 (第一个任务):")
        print(f"  {' '.join(tasks[0].cli_args)}")
        return 0

    # ── 5. 并行执行 ────────────────────────────────────────────────────────
    start_time = time.monotonic()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_map = {
            executor.submit(run_task, task): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(future_map):
            task = future_map[future]
            completed += 1
            try:
                r = future.result()
                results.append(r)
                status = "✓" if r["success"] else "✗"
                elapsed = r.get("runtime_sec", 0)
                print(f"[{completed}/{len(tasks)}] {status} {task.label} ({elapsed:.0f}s)")
                if not r["success"]:
                    err = r.get("error", "未知错误")
                    print(f"      错误: {err[:200]}")
            except Exception as e:
                results.append({
                    "factor": task.factor_name,
                    "success": False,
                    "output_dir": str(task.output_dir),
                    "runtime_sec": 0,
                    "error": f"Future exception: {e}",
                })
                print(f"[{completed}/{len(tasks)}] ✗ {task.label} (异常: {e})")

    total_sec = time.monotonic() - start_time
    print(f"\n总耗时: {total_sec/60:.1f} 分钟 ({total_sec:.0f} 秒)")

    # ── 6. 汇总 ────────────────────────────────────────────────────────────
    from config import DataPath
    summary_dir = args.output_dir or (Path(DataPath.DATA_DIR) / "factors" / "_batch_summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整结果 JSON
    results_path = summary_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"完整结果 JSON: {results_path}")

    # 生成汇总 CSV
    generate_summary_csv(results, summary_dir)

    # ── 7. 总结 ────────────────────────────────────────────────────────────
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    print(f"\n{'='*60}")
    print(f"批量分析完成: {success_count} 成功, {fail_count} 失败")
    if fail_count > 0:
        print("失败清单:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['factor']}: {r.get('error', '未知')[:150]}")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
