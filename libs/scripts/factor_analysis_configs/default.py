"""批量因子分析 — 默认配置。

包含因子族分类、full 模式参数网格、变换/阈值/排除规则、组合白名单。
编辑此文件即可改变分析行为，无需修改脚本。
"""
from __future__ import annotations

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
# 参数网格定义 — full 模式下展开的参数敏感度扫描
# ═══════════════════════════════════════════════════════════════════════════════
# 原则：每个因子 2-4 个参数值，避免组合爆炸。

FULL_MODE_PARAM_GRIDS: dict[str, dict[str, list]] = {
    # ── 价格动量族 ──
    "PriceReturn": {"window": [10, 20, 40, 60]},
    "RiskAdjustedReturn": {"window": [20, 40, 60]},
    "HighPointPosition": {"window": [10, 20, 40]},
    "LowPointPosition": {"window": [10, 20, 40]},
    "TimeSeriesMomentum": {"window": [60, 120, 240]},

    # ── 反转族 ──
    "ShortTermReversal": {"window": [1, 5, 10, 20]},
    "ExtremeReversal": {
        "window": [10, 20],
        "tail_pct": [0.05, 0.1, 0.15],
    },
    "VolumeReversal": {
        "ret_window": [5, 10, 20],
        "vol_window": [10, 20],
    },

    # ── 成交量族 ──
    "VolumeRatio": {"window": [3, 5, 10, 20]},
    "VolumePriceCorrelation": {"window": [10, 20, 40]},
    "AmihudIlliquidity": {"window": [10, 20, 40]},
    "VolumeStd": {"window": [10, 20, 40]},
    "VolumeSkew": {"window": [10, 20, 40]},
    "AverageAmount": {"window": [5, 10, 20]},

    # ── 波动率族 ──
    "DownsideVolatility": {"window": [10, 20, 40]},
    "ParkinsonVolatility": {"window": [10, 20, 40]},
    "GarmanKlassVolatility": {"window": [10, 20, 40]},
    "VolOfVol": {
        "vol_window": [10, 20],
        "std_window": [40, 60, 120],
    },
    "MaxDrawdown": {"window": [20, 40, 60, 120]},
    "AvgDrawdown": {"window": [20, 40, 60, 120]},

    # ── 趋势质量族 ──
    "HurstExponent": {"window": [60, 120, 240]},
    "KaufmanER": {"window": [10, 20, 40]},
    "UpDownRatio": {"window": [10, 20, 40]},
    "ADX": {"window": [7, 14, 21]},

    # ── 超买超卖族 ──
    "RSI": {"window": [7, 14, 21]},
    "Stochastic": {
        "n": [7, 14, 21],
        "m": [3, 5],
    },
    "CCI": {"window": [10, 20, 40]},
    "WilliamsR": {"window": [7, 14, 21]},
    "MFI": {"window": [7, 14, 21]},

    # ── 均线偏离族 ──
    "MAPosition": {"window": [60, 120, 200, 250]},
    "MA": {"window": [10, 20, 40, 60]},
    "BIAS": {"window": [10, 20, 40]},
    "BollingerBandPosition": {
        "window": [10, 20, 40],
        "k": [1.5, 2.0, 2.5],
    },
    "MASlope": {
        "ma_window": [10, 20, 40],
        "slope_window": [3, 5, 10],
    },
    "MADistance": {
        "short_window": [5, 10, 20],
        "long_window": [40, 60, 120],
    },

    # ── 分布形态族 ──
    "ReturnSkew": {"window": [20, 40, 60, 120]},
    "ReturnKurtosis": {"window": [20, 40, 60, 120]},
    "HistoricalVaR": {"window": [60, 120, 252]},
    "CVaR": {"window": [60, 120, 252]},
    "MFE": {"window": [10, 20, 40]},
    "MAE": {"window": [10, 20, 40]},
    "ID": {"window": [10, 20, 40]},

    # ── 突破族 ──
    "TrendR2": {
        "window": [60, 120, 240],
        "output": ["r2", "slope"],
    },
    "RSRS": {
        "regression_window": [14],
        "zscore_window": [200, 400, 600],
        "output": ["zscore"],
    },
    "ATR": {"window": [14, 20, 25]},
    "NewHighContinuous": {"window": [20, 50, 100]},
    "NewLowContinuous": {"window": [20, 50, 100]},
    "DonchianChannelPosition": {"window": [10, 20, 40]},
    "ATRRatio": {"window": [14, 20, 25]},
    "ChandelierExit": {
        "n": [10, 22, 40],
        "atr_window": [10, 22, 40],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# binarize_winrate 定制阈值
# ==============================================================================
# 未列出的因子默认 threshold=0.0。值 = [threshold, ...]。
CUSTOM_THRESHOLDS: dict[str, list[float]] = {
    "KaufmanER":         [0.3, 0.4],
    "TrendR2":           [0.5],
    "TrendR2_slope":     [0.0],
    "RSRS":              [0.0, 0.7],
    "ADX":               [20.0, 25.0],
    "HurstExponent":     [0.5],
    "NewHighContinuous": [10.0, 20.0],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 变换配置 — 每种变换支持多个 window
# ═══════════════════════════════════════════════════════════════════════════════

TRANSFORM_CONFIGS: dict[str, dict] = {
    "rolling_mean":     {"windows": [5, 10]},
    "rolling_std":      {"windows": [10]},
    "delta":            {"windows": [5, 10]},
    "pct_change":       {"windows": [5, 10]},
    "binarize_winrate": {"windows": [10, 20]},
    "zscore":           {"windows": [120]},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 排除规则 — 某些变换不应用于某些因子
# ═══════════════════════════════════════════════════════════════════════════════

EXCLUSIONS: dict[str, set[str]] = {
    "rolling_std": {
        "PriceReturn", "RiskAdjustedReturn", "IntradayMomentum",
        "OvernightReturn", "TimeSeriesMomentum", "HighPointPosition",
        "LowPointPosition", "ShortTermReversal", "ExtremeReversal",
        "VolumeReversal",
    },
    "zscore": {
        "RSRS", "ADX", "BollingerBandPosition",
        "RSI", "Stochastic", "WilliamsR", "MFI", "UltimateOscillator",
    },
    "pct_change": {
        "OBV", "VPT", "ConsecutiveUpDays", "ConsecutiveDownDays",
        "NewHigh", "DailyRebound", "NewHighContinuous", "NewLowContinuous",
    },
    "binarize_winrate": {
        "ConsecutiveUpDays", "ConsecutiveDownDays", "NewHigh",
        "NewLowContinuous", "DailyRebound",
        "ATR", "ATRRatio",
    },
    "delta": {
        "OBV", "VPT", "ConsecutiveUpDays", "ConsecutiveDownDays",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 因子组合白名单
# ==============================================================================
# 格式: {"a": "因子名", "a_params": {...}, "method": "...", "b": "因子名", "b_params": {...}}

COMBO_WHITELIST: list[dict] = [
    {"a": "PriceReturn", "a_params": {"window": 20},
     "method": "product",
     "b": "KaufmanER", "b_params": {"window": 20}},
    {"a": "PriceReturn", "a_params": {"window": 20},
     "method": "ratio",
     "b": "DownsideVolatility", "b_params": {"window": 20}},
    {"a": "PriceReturn", "a_params": {"window": 20},
     "method": "diff",
     "b": "PriceReturn", "b_params": {"window": 60}},
]


# ═══════════════════════════════════════════════════════════════════════════════
# 条件因子白名单
# ==============================================================================
# 格式: {"signal": "...", "signal_params": {...}, "condition": "...", "condition_params": {...}, "op": "...", "threshold": 0.5, "false_value": "nan"}

CONDITIONAL_WHITELIST: list[dict] = [
    {"signal": "PriceReturn", "signal_params": {"window": 20},
     "condition": "TrendR2", "condition_params": {"window": 120, "output": "r2"},
     "op": "gt", "threshold": 0.5, "false_value": "nan"},
]
