"""全量动量/趋势因子扫描 — 宽池 489，同口径对比 C2。

排名因子: 30 个动量/趋势类因子（价格动量/路径动量/突破/均线/震荡/趋势质量）
用法:
    uv run python libs/scripts/run_wide_momentum_custom.py \
        --config libs.scripts.wide_momentum_configs.all_momentum_scan
"""
from __future__ import annotations

from backtesting.wide_momentum_baseline import (
    StopRuleSpec,
    ThresholdFilter,
    factor_threshold_stop,
    equal_weight_allocator,
    make_factor_weighted_allocator,
    score_proportional_allocator,
)
from factors.price_return import PriceReturn
from factors.price_momentum import (
    RiskAdjustedReturn, TimeSeriesMomentum, IntradayMomentum,
    OvernightReturn, HighPointPosition, LowPointPosition,
)
from factors.breakout_family import (
    NewHighContinuous, NewLowContinuous, DonchianChannelPosition,
    ATRRatio, ChandelierExit,
)
from factors.ma import BIAS, BollingerBandPosition, MAAlignment, MASlope, MADistance
from factors.oscillator import RSI, Stochastic, CCI, MFI, UltimateOscillator
from factors.trend_quality import HurstExponent, KaufmanEfficiencyRatio, ADX
from factors.distribution_family import ReturnSkew, InformationDiscreteness
from factors.meta_factor import NegateFactor
from factors.volatility import Volatility

vol20 = Volatility(window=20)

SHARED_PIPELINE: tuple = (vol20,)

# ── 动量/趋势因子定义 ──
F = {
    "PR20": PriceReturn(window=20),
    "PR60": PriceReturn(window=60),
    "PR120": PriceReturn(window=120),
    "RAR20": RiskAdjustedReturn(window=20),
    "RAR60": RiskAdjustedReturn(window=60),
    "TSM252": TimeSeriesMomentum(window=252),
    "Intraday": IntradayMomentum(),
    "Overnight": OvernightReturn(),
    "HPP20": HighPointPosition(window=20),
    "LPP20_neg": NegateFactor(LowPointPosition(window=20)),
    "Donchian20": DonchianChannelPosition(window=20),
    "NHC50": NewHighContinuous(window=50),
    "NLC50": NewLowContinuous(window=50),
    "Chandelier": ChandelierExit(n=22, atr_window=22),
    "ATRRatio": ATRRatio(window=25),
    "BIAS20": BIAS(window=20),
    "BBPos20": BollingerBandPosition(window=20),
    "MAAlign": MAAlignment(),
    "MASlope": MASlope(ma_window=20, slope_window=5),
    "MADist": MADistance(short_window=5, long_window=60),
    "RSI14": RSI(window=14),
    "CCI20": CCI(window=20),
    "StochK": Stochastic(),
    "MFI14": MFI(window=14),
    "UO": UltimateOscillator(),
    "Hurst120": HurstExponent(window=120),
    "KER20": KaufmanEfficiencyRatio(window=20),
    "ADX": ADX(),
    "Skew60": ReturnSkew(window=60),
    "ID20": InformationDiscreteness(window=20),
}

NO_FILTERS = ()
GROUPS: list[tuple] = [(label, f, NO_FILTERS) for label, f in F.items()]

GRID_TOP_N: tuple[int, ...] = (1, 2, 3, 5)
GRID_MIN_MOMENTUM: tuple = (None,)
GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)
GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 15, 20)
GRID_EXCLUDE_BONDS: tuple[bool, ...] = (False,)
GRID_HOLD_OVERLAP: tuple[bool, ...] = (False,)

alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)
alloc_inv_vol.__name__ = "invvol"
WEIGHT_ALLOCATORS: tuple = (alloc_inv_vol,)

OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"
BASENAME_TAG: str = "all_momentum_scan"
TITLE: str = "全量动量/趋势因子扫描（30 因子 × 宽池 489）"
START_DATE: str = "2020-01-01"
END_DATE: str = "2026-07-17"
MAX_WORKERS: int | None = None
CROSS_GROUP_PARALLEL = True
