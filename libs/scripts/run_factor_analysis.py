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
    # ── 价格动量族 ──
    "PriceReturn": ("factors.price_return", "PriceReturn", {"window": 60}),
    "RiskAdjustedReturn": ("factors.price_momentum", "RiskAdjustedReturn", {"window": 20}),
    "IntradayMomentum": ("factors.price_momentum", "IntradayMomentum", {}),
    "OvernightReturn": ("factors.price_momentum", "OvernightReturn", {}),
    "HighPointPosition": ("factors.price_momentum", "HighPointPosition", {"window": 20}),
    "LowPointPosition": ("factors.price_momentum", "LowPointPosition", {"window": 20}),
    "TimeSeriesMomentum": ("factors.price_momentum", "TimeSeriesMomentum", {"window": 252}),
    # ── 反转族 ──
    "ShortTermReversal": ("factors.reversal", "ShortTermReversal", {"window": 1}),
    "ExtremeReversal": ("factors.reversal", "ExtremeReversal", {"window": 20, "tail_pct": 0.1}),
    "VolumeReversal": ("factors.reversal", "VolumeReversal", {"ret_window": 5, "vol_window": 20}),
    # ── 成交量/流动性族 ──
    "VolumeRatio": ("factors.volume_family", "VolumeRatio", {"window": 5}),
    "VolumePriceCorrelation": ("factors.volume_family", "VolumePriceCorrelation", {"window": 20}),
    "OBV": ("factors.volume_family", "OBV", {}),
    "VPT": ("factors.volume_family", "VPT", {}),
    "AmihudIlliquidity": ("factors.volume_family", "AmihudIlliquidity", {"window": 20}),
    "VolumeStd": ("factors.volume_family", "VolumeStd", {"window": 20}),
    "VolumeSkew": ("factors.volume_family", "VolumeSkew", {"window": 20}),
    "AverageAmount": ("factors.average_amount", "AverageAmount", {"window": 20}),
    # ── 波动率族 ──
    "DownsideVolatility": ("factors.volatility_family", "DownsideVolatility", {"window": 20}),
    "ParkinsonVolatility": ("factors.volatility_family", "ParkinsonVolatility", {"window": 20}),
    "GarmanKlassVolatility": ("factors.volatility_family", "GarmanKlassVolatility", {"window": 20}),
    "VolOfVol": ("factors.volatility_family", "VolOfVol", {"vol_window": 20, "std_window": 60}),
    "MaxDrawdown": ("factors.volatility_family", "MaxDrawdown", {"window": 60}),
    "AvgDrawdown": ("factors.volatility_family", "AvgDrawdown", {"window": 60}),
    # ── 趋势质量族 ──
    "HurstExponent": ("factors.trend_quality", "HurstExponent", {"window": 120}),
    "KaufmanER": ("factors.trend_quality", "KaufmanEfficiencyRatio", {"window": 20}),
    "UpDownRatio": ("factors.trend_quality", "UpDownRatio", {"window": 20}),
    "ConsecutiveUpDays": ("factors.trend_quality", "ConsecutiveUpDays", {}),
    "ConsecutiveDownDays": ("factors.trend_quality", "ConsecutiveDownDays", {}),
    "ADX": ("factors.trend_quality", "ADX", {"window": 14, "output": "adx"}),
    # ── 超买超卖族 ──
    "RSI": ("factors.oscillator", "RSI", {"window": 14}),
    "Stochastic": ("factors.oscillator", "Stochastic", {"n": 14, "m": 3, "output": "K"}),
    "CCI": ("factors.oscillator", "CCI", {"window": 20}),
    "WilliamsR": ("factors.oscillator", "WilliamsR", {"window": 14}),
    "MFI": ("factors.oscillator", "MFI", {"window": 14}),
    "UltimateOscillator": ("factors.oscillator", "UltimateOscillator", {"short": 7, "mid": 14, "long": 28}),
    # ── 均线与偏离族 ──
    "MAPosition": ("factors.ma", "MAPosition", {"window": 200}),
    "MA": ("factors.ma", "MAFactor", {"window": 20}),
    "BIAS": ("factors.ma", "BIAS", {"window": 20}),
    "BollingerBandPosition": ("factors.ma", "BollingerBandPosition", {"window": 20, "k": 2.0}),
    "MAAlignment": ("factors.ma", "MAAlignment", {"windows": [5, 20, 60]}),
    "MASlope": ("factors.ma", "MASlope", {"ma_window": 20, "slope_window": 5}),
    "MADistance": ("factors.ma", "MADistance", {"short_window": 5, "long_window": 60}),
    "MADispersion": ("factors.ma", "MADispersion", {"windows": [5, 10, 20, 60]}),
    # ── 分布形态族 ──
    "ReturnSkew": ("factors.distribution_family", "ReturnSkew", {"window": 60}),
    "ReturnKurtosis": ("factors.distribution_family", "ReturnKurtosis", {"window": 60}),
    "HistoricalVaR": ("factors.distribution_family", "HistoricalVaR", {"window": 252, "q": 0.05}),
    "CVaR": ("factors.distribution_family", "CVaR", {"window": 252, "q": 0.05}),
    "MFE": ("factors.distribution_family", "MaxFavorableExcursion", {"window": 20}),
    "MAE": ("factors.distribution_family", "MaxAdverseExcursion", {"window": 20}),
    "ID": ("factors.distribution_family", "InformationDiscreteness", {"window": 20}),
    # ── 结构性/突破族 ──
    "NewHigh": ("factors.new_high", "NewHigh", {"high_window": 50, "low_window": 25}),
    "DailyRebound": ("factors.daily_rebound", "DailyRebound", {}),
    "TrendR2": ("factors.trend_r2", "TrendR2Factor", {"window": 120, "output": "r2"}),
    # RSRS 因子需要 output="zscore" 以获得连续值（可选 output="signal" 但那是离散信号）
    "RSRS": ("factors.rsrs", "RsrsFactor", {"output": "zscore"}),
    "ATR": ("factors.average_true_range", "AverageTrueRange", {"window": 25}),
    "NewHighContinuous": ("factors.breakout_family", "NewHighContinuous", {"window": 50}),
    "NewLowContinuous": ("factors.breakout_family", "NewLowContinuous", {"window": 50}),
    "DonchianChannelPosition": ("factors.breakout_family", "DonchianChannelPosition", {"window": 20}),
    "ATRRatio": ("factors.breakout_family", "ATRRatio", {"window": 25}),
    "ChandelierExit": ("factors.breakout_family", "ChandelierExit", {"n": 22, "atr_window": 22}),
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
