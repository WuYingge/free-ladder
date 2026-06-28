#!/usr/bin/env python3
"""
单因子分析 CLI 入口 (Factor Analysis CLI)

用法:
    # 默认参数
    python libs/scripts/run_factor_analysis.py --factor PriceReturn

    # 自定义参数（--param key=value 可多次指定）
    python libs/scripts/run_factor_analysis.py \
        --factor PriceReturn --param window=120 --param skip_recent=20

    # 参数网格扫描
    python libs/scripts/run_factor_analysis.py \
        --factor PriceReturn \
        --param-grid '{"window": [20, 60, 120]}'

    # 完整参数控制
    python libs/scripts/run_factor_analysis.py \
        --factor TrendR2 --param window=60 --param output=r2 \
        --layers 1 2 \
        --symbols 510300 510500 159915 \
        --forward-periods 5 10 20 60 \
        --min-bars 200 \
        --n-quantiles 5 \
        --max-workers 8

参数:
    --factor           因子类名（如 PriceReturn, TrendR2, RSRS, RSI 等）
    --param            因子构造参数 key=value（可多次指定，覆盖默认值）
    --param-grid       参数网格扫描 JSON，如 '{"window": [20,60,120]}'
    --layers           分析层 1 2 3（空格分隔，默认 1 2 3）
    --symbols          可选标的列表（空格分隔，默认全量 ETF_INDEX_MAP）
    --forward-periods  前向持仓期（交易日，空格分隔，默认 5 10 20 60）
    --min-bars         最少交易日数（默认 252）
    --start-date       起始日期 YYYY-MM-DD
    --end-date         结束日期 YYYY-MM-DD
    --n-quantiles      分位数分组数（默认 5）
    --rolling-ic-window 滚动 IC 窗口（交易日，默认 120）
    --max-workers      多进程 worker 数（默认 CPU 数）
    --output-root      输出目录（默认 data/factors/{factor_name}/）
    --output-date      报告日期标签（默认当天日期）
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
            f"未知因子: {name}。\n可用因子 ({len(FACTOR_REGISTRY)} 个):\n  "
            + "\n  ".join(sorted(FACTOR_REGISTRY.keys()))
        )
    module_path, class_name, _ = FACTOR_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _parse_value(raw: str) -> int | float | str | bool | list:
    """将字符串值自动转换为合适的 Python 类型。"""
    raw = raw.strip()
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.startswith("[") and raw.endswith("]"):
        import ast
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            inner = raw[1:-1]
            if not inner.strip():
                return []
            return [_parse_value(v) for v in inner.split(",")]
    try:
        if "." not in raw and "e" not in raw.lower():
            return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_param(pair: str) -> tuple[str, int | float | str | bool | list]:
    """解析 key=value 字符串，自动推断类型。"""
    if "=" not in pair:
        raise argparse.ArgumentTypeError(f"参数格式应为 key=value，收到: {pair}")
    key, value = pair.split("=", 1)
    return key, _parse_value(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单因子分析 CLI — 一键跑通 Layer 1-3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--factor", type=str, required=True,
        help="因子类名。可用: 见 FACTOR_REGISTRY",
    )
    parser.add_argument(
        "--param", nargs="*", type=str, default=[],
        metavar="KEY=VALUE",
        help="因子构造参数（可多次指定），如 --param window=120 --param output=r2",
    )
    parser.add_argument(
        "--param-grid", type=str, default=None,
        metavar="JSON",
        help='参数网格扫描 JSON，如 \'{"window": [20, 60, 120]}\'',
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[1, 2, 3],
        help="要运行的分析层，如 --layers 1 2 3（默认全跑）",
    )
    parser.add_argument(
        "--symbols", nargs="*", default=None,
        help="可选标的列表（空格分隔）。默认全量 ETF_INDEX_MAP",
    )
    parser.add_argument(
        "--forward-periods", nargs="+", type=int, default=[5, 10, 20, 60],
        help="前向持仓期（交易日），默认 5 10 20 60",
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
        "--rolling-ic-window", type=int, default=120,
        help="滚动 IC 窗口（交易日，默认 120）",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="多进程 worker 数（默认 CPU 数）",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="输出目录（默认 data/factors/{factor_name}/）",
    )
    parser.add_argument(
        "--output-date", type=str, default=None,
        help="报告日期标签（默认当天日期）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. 导入因子类
    factor_cls = _import_factor(args.factor)

    # 2. 构造参数：默认值 + CLI 覆盖
    _, _, default_params = FACTOR_REGISTRY[args.factor]
    params = dict(default_params)
    for pair in args.param:
        key, value = _parse_param(pair)
        params[key] = value

    # 3. 实例化因子
    try:
        factor = factor_cls(**params)
    except TypeError as e:
        print(f"错误: 构造因子 {args.factor} 失败。参数: {params}")
        print(f"  原因: {e}")
        return 1

    # 4. 解析参数网格
    param_grid: dict[str, list] | None = None
    if args.param_grid:
        try:
            param_grid = json.loads(args.param_grid)
        except json.JSONDecodeError as e:
            print(f"错误: --param-grid JSON 解析失败: {e}")
            return 1
        if not isinstance(param_grid, dict):
            print("错误: --param-grid 必须是 JSON 对象")
            return 1
        for k, v in param_grid.items():
            if not isinstance(v, list):
                param_grid[k] = [v]

    # 5. 构建配置
    symbols: list[str] | None = None
    if args.symbols is not None and len(args.symbols) > 0:
        symbols = list(args.symbols)

    config = FactorAnalysisConfig(
        factor=factor,
        symbols=symbols,
        layers=tuple(args.layers),
        forward_periods=tuple(args.forward_periods),
        min_bars=args.min_bars,
        start_date=args.start_date,
        end_date=args.end_date,
        n_quantiles=args.n_quantiles,
        rolling_ic_window=args.rolling_ic_window,
        param_grid=param_grid,
        max_workers=args.max_workers,
        output_root=args.output_root,
        output_date=args.output_date,
    )

    # 6. 执行分析
    results = run_factor_analysis(config)

    # 7. 输出摘要
    output = results.get("output", {})
    print("\n" + "=" * 60)
    print("分析完成！")
    if output.get("linux_root"):
        print(f"  输出目录: {output['linux_root']}")
    files = output.get("files", [])
    if files:
        print(f"  输出文件 ({len(files)} 个):")
        for f in files:
            print(f"    {f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
