#!/usr/bin/env python3
"""
单因子分析 CLI 入口 (Factor Analysis CLI)

用法:
    python libs/scripts/run_factor_analysis.py \\
        --factor PriceReturn --window 60 \\
        --layers 1 2 3 \\
        --symbols 510300 510500 159915 \\
        --min-bars 200

参数:
    --factor        因子类名（如 PriceReturn, TrendR2, MAPosition）
    --window        因子参数 window（如 60）
    --layers        分析层 1 2 3（空格分隔）
    --symbols       可选标的列表（空格分隔，默认全量 ETF_INDEX_MAP）
    --min-bars      最少交易日数（默认 252）
    --start-date    起始日期 YYYY-MM-DD
    --end-date      结束日期 YYYY-MM-DD
    --n-quantiles   分位数分组数（默认 5）
    --max-workers   多进程 worker 数（默认 CPU 数）
    --output-root   输出目录（默认 data/factors/{factor_name}/）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from factor_analysis.config import FactorAnalysisConfig
from factor_analysis.runner import run_factor_analysis


# ── 支持的因子类和它们的构造参数 ───────────────────────────────────────────
# 键 = CLI 中使用的因子名，值 = (模块路径, 类名, 默认构造参数)。
# 新增因子时在此注册即可享受 CLI 调用的便利。
FACTOR_REGISTRY: dict[str, tuple[str, str, dict]] = {
    "PriceReturn": ("factors.price_return", "PriceReturn", {"window": 60}),
    "TrendR2": ("factors.trend_r2", "TrendR2Factor", {"window": 120, "output": "r2"}),
    "MAPosition": ("factors.ma", "MAPosition", {"window": 200}),
    "MA": ("factors.ma", "MAFactor", {"window": 20}),
    "NewHigh": ("factors.new_high", "NewHigh", {"high_window": 50, "low_window": 25}),
    "ATR": ("factors.average_true_range", "AverageTrueRange", {"window": 25}),
    "DailyRebound": ("factors.daily_rebound", "DailyRebound", {}),
    # RSRS 因子需要 output="zscore" 以获得连续值（可选 output="signal" 但那是离散信号）
    "RSRS": ("factors.rsrs", "RsrsFactor", {"output": "zscore"}),
}


def _import_factor(name: str) -> type:
    """动态导入因子类。"""
    if name not in FACTOR_REGISTRY:
        raise ValueError(
            f"未知因子: {name}。可用: {list(FACTOR_REGISTRY.keys())}"
        )
    module_path, class_name, _ = FACTOR_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单因子分析 CLI — 一键跑通 Layer 1-3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--factor", type=str, required=True,
        help=f"因子类名。可用: {list(FACTOR_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="因子参数 window（如果因子有的话）。如 PriceReturn --window 60",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[1, 2, 3],
        help="要运行的分析层，如 --layers 1 2 3",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="可选标的列表（空格分隔）。默认全量 ETF_INDEX_MAP",
    )
    parser.add_argument(
        "--min-bars", type=int, default=252,
        help="最少交易日数（默认 252 ≈ 1年）",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="起始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="结束日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--n-quantiles", type=int, default=5,
        help="分位数分组数（默认 5）",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="多进程 worker 数（默认 CPU 数）",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="输出目录（默认 data/factors/{factor_name}/）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. 导入因子类
    factor_cls = _import_factor(args.factor)

    # 2. 构造参数
    _, _, default_params = FACTOR_REGISTRY[args.factor]
    params = dict(default_params)
    # CLI 传入的 window 覆盖默认值（仅对有 window 参数的因子）
    if args.window is not None and "window" in params:
        params["window"] = args.window

    # 3. 实例化因子
    factor = factor_cls(**params)

    # 4. 构建配置
    symbols: list[str] | None = None
    if args.symbols is not None:
        symbols = list(args.symbols)

    config = FactorAnalysisConfig(
        factor=factor,
        symbols=symbols,
        layers=tuple(args.layers),
        min_bars=args.min_bars,
        start_date=args.start_date,
        end_date=args.end_date,
        n_quantiles=args.n_quantiles,
        max_workers=args.max_workers,
        output_root=args.output_root,
    )

    # 5. 执行分析
    results = run_factor_analysis(config)

    # 6. 输出摘要
    print("\n" + "=" * 60)
    print("分析完成！输出文件:")
    for f in results["output"]["files"]:
        print(f"  {f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
