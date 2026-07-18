"""
批量因子分析 — combo_54 配置
基于 docs/factor-combo-build.md v2（2026-07-05 修订）

覆盖 57 个因子（META_SPECS 37 基础变换/negate + COMBO_WHITELIST 9 复合 + CONDITIONAL_WHITELIST 11 条件）。
跑什么完全由本配置决定 — META_SPECS 直读 + 白名单驱动，零笛卡尔积。

用法:
    python libs/scripts/run_batch_factor_analysis.py \
        --config libs.scripts.factor_analysis_configs.combo_54 \
        --mode standard \
        --generate-meta combos conditionals

说明:
     - META_SPECS: 37 条精确基础变换/negate 因子定义
    - COMBO_WHITELIST: 9 条复合因子（含链式 C5 + C11）
    - CONDITIONAL_WHITELIST: 11 条条件因子（含 SwitchFactor + MultiConditional）
    - 无需 --families / --factors，本配置自包含
    - 剩余未实现: C6/C7/C7b（ICIR 动态权重，需两轮分析）
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# ★ META_SPECS — 基础变换因子的精确定义（35 条）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 格式:
#   type="base":      基础因子 + 特定参数（无变换）
#     {"type": "base", "factor": "因子名", "params": {...}}
#   type="transform": 基础因子 + 变换衍生
#     {"type": "transform", "factor": "因子名", "params": {...},
#      "transform": "变换类型", "transform_window": N, "threshold": X}
#   type="negate": 对衍生因子取反（可内嵌 transform）
#     {"type": "negate", "factor": "因子名", "params": {...},
#      "transform": "变换类型", "transform_window": N}

META_SPECS: list[dict] = [
    # ═══════════════════════════════════════════════════════════════════════
    # 1.1 AvgDrawdown 系列（6 个）
    # ═══════════════════════════════════════════════════════════════════════
    {"type": "transform", "factor": "AvgDrawdown", "params": {"window": 120},
     "transform": "zscore", "transform_window": 252},
    {"type": "transform", "factor": "AvgDrawdown", "params": {"window": 120},
     "transform": "delta", "transform_window": 10},
    {"type": "transform", "factor": "AvgDrawdown", "params": {"window": 40},
     "transform": "rolling_mean", "transform_window": 5},
    {"type": "transform", "factor": "AvgDrawdown", "params": {"window": 120},
     "transform": "rolling_mean", "transform_window": 5},
    {"type": "transform", "factor": "AvgDrawdown", "params": {"window": 250},
     "transform": "rolling_mean", "transform_window": 5},
    {"type": "transform", "factor": "AvgDrawdown", "params": {"window": 120},
     "transform": "binarize_winrate", "transform_window": 10, "threshold": -2.0},

    # ═══════════════════════════════════════════════════════════════════════
    # 1.2 MAE 系列（8 个）
    # ═══════════════════════════════════════════════════════════════════════
    {"type": "transform", "factor": "MAE", "params": {"window": 20},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "MAE", "params": {"window": 60},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "MAE", "params": {"window": 40},
     "transform": "rolling_mean", "transform_window": 5},
    {"type": "transform", "factor": "MAE", "params": {"window": 40},
     "transform": "rolling_mean", "transform_window": 10},
    {"type": "transform", "factor": "MAE", "params": {"window": 40},
     "transform": "delta", "transform_window": 5},
    {"type": "transform", "factor": "MAE", "params": {"window": 40},
     "transform": "delta", "transform_window": 10},
    {"type": "transform", "factor": "MAE", "params": {"window": 40},
     "transform": "binarize_winrate", "transform_window": 10, "threshold": -1.0},
    {"type": "negate", "factor": "MAE", "params": {"window": 40},
     "transform": "zscore", "transform_window": 120},

    # ═══════════════════════════════════════════════════════════════════════
    # 1.3 MADistance 系列（5 个）
    # ═══════════════════════════════════════════════════════════════════════
    {"type": "transform", "factor": "MADistance",
     "params": {"short_window": 10, "long_window": 20},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "MADistance",
     "params": {"short_window": 20, "long_window": 120},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "MADistance",
     "params": {"short_window": 20, "long_window": 60},
     "transform": "delta", "transform_window": 5},
    {"type": "transform", "factor": "MADistance",
     "params": {"short_window": 20, "long_window": 60},
     "transform": "delta", "transform_window": 10},
    {"type": "transform", "factor": "MADistance",
     "params": {"short_window": 20, "long_window": 60},
     "transform": "rolling_mean", "transform_window": 5},

    # ═══════════════════════════════════════════════════════════════════════
    # 1.4 TimeSeriesMomentum 系列（5 个）
    # ═══════════════════════════════════════════════════════════════════════
    {"type": "base", "factor": "TimeSeriesMomentum", "params": {"window": 120}},
    {"type": "base", "factor": "TimeSeriesMomentum", "params": {"window": 60}},
    {"type": "transform", "factor": "TimeSeriesMomentum", "params": {"window": 120},
     "transform": "delta", "transform_window": 10},
    {"type": "transform", "factor": "TimeSeriesMomentum", "params": {"window": 252},
     "transform": "delta", "transform_window": 10},
    {"type": "transform", "factor": "TimeSeriesMomentum", "params": {"window": 252},
     "transform": "zscore", "transform_window": 252},

    # ═══════════════════════════════════════════════════════════════════════
    # 1.5 HighPointPosition 补充（2 个）
    # ═══════════════════════════════════════════════════════════════════════
    {"type": "transform", "factor": "HighPointPosition", "params": {"window": 20},
     "transform": "delta", "transform_window": 5},
    {"type": "transform", "factor": "HighPointPosition", "params": {"window": 20},
     "transform": "zscore", "transform_window": 252},

    # ═══════════════════════════════════════════════════════════════════════
    # 1.6 MFI 补充（2 个）
    # ═══════════════════════════════════════════════════════════════════════
    {"type": "transform", "factor": "MFI", "params": {"window": 14},
     "transform": "delta", "transform_window": 5},
    {"type": "transform", "factor": "MFI", "params": {"window": 14},
     "transform": "delta", "transform_window": 10},

    # ═══════════════════════════════════════════════════════════════════════
    # 1.7 未覆盖基础因子补充（10 个）
    # ═══════════════════════════════════════════════════════════════════════

    # DailyRebound (3):
    {"type": "transform", "factor": "DailyRebound", "params": {},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "DailyRebound", "params": {},
     "transform": "rolling_mean", "transform_window": 5},
    {"type": "transform", "factor": "DailyRebound", "params": {},
     "transform": "delta", "transform_window": 5},

    # BollingerBandPosition (2):
    {"type": "transform", "factor": "BollingerBandPosition",
     "params": {"window": 20, "k": 2.0},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "BollingerBandPosition",
     "params": {"window": 20, "k": 2.0},
     "transform": "delta", "transform_window": 5},

    # DonchianChannelPosition (2):
    {"type": "transform", "factor": "DonchianChannelPosition",
     "params": {"window": 20},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "DonchianChannelPosition",
     "params": {"window": 20},
     "transform": "delta", "transform_window": 10},

    # CCI (3):
    {"type": "transform", "factor": "CCI", "params": {"window": 20},
     "transform": "zscore", "transform_window": 120},
    {"type": "transform", "factor": "CCI", "params": {"window": 20},
     "transform": "rolling_mean", "transform_window": 5},

    # ═══════════════════════════════════════════════════════════════════════
    # 1.8 NegateFactor 系列（1 个）
    # ═══════════════════════════════════════════════════════════════════════
    {"type": "negate", "factor": "TrendR2_slope", "params": {"window": 60},
     "transform": "zscore", "transform_window": 120},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 因子族分类（保留兼容 — META_SPECS 模式下不使用）
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

FULL_MODE_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "PriceReturn": {"window": [10, 20, 40, 60]},
    "RiskAdjustedReturn": {"window": [20, 40, 60]},
    "HighPointPosition": {"window": [10, 20, 40]},
    "LowPointPosition": {"window": [10, 20, 40]},
    "TimeSeriesMomentum": {"window": [60, 120, 240]},
    "ShortTermReversal": {"window": [1, 5, 10, 20]},
    "ExtremeReversal": {"window": [10, 20], "tail_pct": [0.05, 0.1, 0.15]},
    "VolumeReversal": {"ret_window": [5, 10, 20], "vol_window": [10, 20]},
    "VolumeRatio": {"window": [3, 5, 10, 20]},
    "VolumePriceCorrelation": {"window": [10, 20, 40]},
    "AmihudIlliquidity": {"window": [10, 20, 40]},
    "VolumeStd": {"window": [10, 20, 40]},
    "VolumeSkew": {"window": [10, 20, 40]},
    "AverageAmount": {"window": [5, 10, 20]},
    "DownsideVolatility": {"window": [10, 20, 40]},
    "ParkinsonVolatility": {"window": [10, 20, 40]},
    "GarmanKlassVolatility": {"window": [10, 20, 40]},
    "VolOfVol": {"vol_window": [10, 20], "std_window": [40, 60, 120]},
    "MaxDrawdown": {"window": [20, 40, 60, 120]},
    "AvgDrawdown": {"window": [20, 40, 60, 120]},
    "HurstExponent": {"window": [60, 120, 240]},
    "KaufmanER": {"window": [10, 20, 40]},
    "UpDownRatio": {"window": [10, 20, 40]},
    "ADX": {"window": [7, 14, 21]},
    "RSI": {"window": [7, 14, 21]},
    "Stochastic": {"n": [7, 14, 21], "m": [3, 5]},
    "CCI": {"window": [10, 20, 40]},
    "WilliamsR": {"window": [7, 14, 21]},
    "MFI": {"window": [7, 14, 21]},
    "MAPosition": {"window": [60, 120, 200, 250]},
    "MA": {"window": [10, 20, 40, 60]},
    "BIAS": {"window": [10, 20, 40]},
    "BollingerBandPosition": {"window": [10, 20, 40], "k": [1.5, 2.0, 2.5]},
    "MASlope": {"ma_window": [10, 20, 40], "slope_window": [3, 5, 10]},
    "MADistance": {"short_window": [5, 10, 20], "long_window": [40, 60, 120]},
    "ReturnSkew": {"window": [20, 40, 60, 120]},
    "ReturnKurtosis": {"window": [20, 40, 60, 120]},
    "HistoricalVaR": {"window": [60, 120, 252]},
    "CVaR": {"window": [60, 120, 252]},
    "MFE": {"window": [10, 20, 40]},
    "MAE": {"window": [10, 20, 40]},
    "ID": {"window": [10, 20, 40]},
    "TrendR2": {"window": [60, 120, 240], "output": ["r2", "slope"]},
    "RSRS": {"regression_window": [14], "zscore_window": [200, 400, 600], "output": ["zscore"]},
    "ATR": {"window": [14, 20, 25]},
    "NewHighContinuous": {"window": [20, 50, 100]},
    "NewLowContinuous": {"window": [20, 50, 100]},
    "DonchianChannelPosition": {"window": [10, 20, 40]},
    "ATRRatio": {"window": [14, 20, 25]},
    "ChandelierExit": {"n": [10, 22, 40], "atr_window": [10, 22, 40]},
}

CUSTOM_THRESHOLDS: dict[str, list[float]] = {
    "KaufmanER": [0.3, 0.4], "TrendR2": [0.5], "TrendR2_slope": [0.0],
    "RSRS": [0.0, 0.7], "ADX": [20.0, 25.0], "HurstExponent": [0.5],
    "NewHighContinuous": [10.0, 20.0],
    "AvgDrawdown": [-2.0], "MAE": [-1.0],
}

TRANSFORM_CONFIGS: dict[str, dict] = {
    "rolling_mean": {"windows": [5, 10]},
    "rolling_std": {"windows": [10]},
    "delta": {"windows": [5, 10]},
    "pct_change": {"windows": [5, 10]},
    "binarize_winrate": {"windows": [10, 20]},
    "zscore": {"windows": [120, 252]},
}

EXCLUSIONS: dict[str, set[str]] = {
    "rolling_std": {
        "PriceReturn", "RiskAdjustedReturn", "IntradayMomentum",
        "OvernightReturn", "TimeSeriesMomentum", "HighPointPosition",
        "LowPointPosition", "ShortTermReversal", "ExtremeReversal", "VolumeReversal",
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
        "NewLowContinuous", "DailyRebound", "ATR", "ATRRatio",
    },
    "delta": {"OBV", "VPT", "ConsecutiveUpDays", "ConsecutiveDownDays"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# 因子组合白名单（9 个 — 文档第二节，含 C5 链式 + C11 熊市防守）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 文档 v2 统一量纲规则：输入先 rank 归一化到 [0,1] 再合成。
# 当前框架 CombineFactor 仅支持时序 zscore（normalize=True），
# rank 与 zscore 在 IC 分析中保序等价，过渡方案可用。

COMBO_WHITELIST: list[dict] = [
    # ── 等权复合 ──
    {"a": "MFI", "a_params": {"window": 14},
     "method": "weighted_sum", "weight_a": 0.5, "weight_b": 0.5,
     "normalize": True, "normalize_window": 252,
     "b": "TimeSeriesMomentum", "b_params": {"window": 252}},
    {"a": "HighPointPosition", "a_params": {"window": 20},
     "method": "weighted_sum", "weight_a": 0.5, "weight_b": -0.5,
     "normalize": True, "normalize_window": 252,
     "b": "MADistance", "b_params": {"short_window": 20, "long_window": 60}},
    {"a": "MFI", "a_params": {"window": 14},
     "method": "weighted_sum", "weight_a": 0.5, "weight_b": 0.5,
     "normalize": True, "normalize_window": 252,
     "b": "HighPointPosition", "b_params": {"window": 20}},
    {"a": "MASlope", "a_params": {"ma_window": 10, "slope_window": 5},
     "method": "weighted_sum", "weight_a": 0.5, "weight_b": -0.5,
     "normalize": True, "normalize_window": 252,
     "b": "MAE", "b_params": {"window": 40}},
    {"a": "HighPointPosition", "a_params": {"window": 20},
     "method": "weighted_sum", "weight_a": 0.5, "weight_b": 0.5,
     "normalize": True, "normalize_window": 252,
     "b": "TimeSeriesMomentum", "b_params": {"window": 252}},

    # C11: Composite_HPP_AvgDD — 熊市防守因子（D6 用）
    {"a": "HighPointPosition", "a_params": {"window": 20},
     "method": "weighted_sum", "weight_a": 0.5, "weight_b": -0.5,
     "normalize": True, "normalize_window": 252,
     "b": "AvgDrawdown", "b_params": {"window": 120}},

    # ── 乘积型 ──
    {"a": "MFI", "a_params": {"window": 14},
     "method": "product",
     "b": "TimeSeriesMomentum", "b_params": {"window": 252}},
    {"a": "MAE", "a_params": {"window": 40},
     "method": "product",
     "b": "TimeSeriesMomentum", "b_params": {"window": 252}},
    {"a": "HighPointPosition", "a_params": {"window": 20},
     "method": "product",
     "b": "TimeSeriesMomentum", "b_params": {"window": 252}},

    # 保留 default.py 示例
    {"a": "PriceReturn", "a_params": {"window": 20},
     "method": "product", "b": "KaufmanER", "b_params": {"window": 20}},
    {"a": "PriceReturn", "a_params": {"window": 20},
     "method": "ratio", "b": "DownsideVolatility", "b_params": {"window": 20}},
    {"a": "PriceReturn", "a_params": {"window": 20},
     "method": "diff", "b": "PriceReturn", "b_params": {"window": 60}},
]

# ═══════════════════════════════════════════════════════════════════════════════
# 条件因子白名单（11 个 — 文档第三节全部覆盖）
# ═══════════════════════════════════════════════════════════════════════════════

CONDITIONAL_WHITELIST: list[dict] = [
    # ── 3.1 状态切换型（SwitchFactor） ──

    # D1: Cond_TrendR2_HPP_TSM
    {"type": "switch",
     "signal_true": "HighPointPosition", "true_params": {"window": 20},
     "true_transform": {"transform": "zscore", "window": 252},
     "signal_false": "TimeSeriesMomentum", "false_params": {"window": 252},
     "condition": "TrendR2", "condition_params": {"window": 120, "output": "r2"},
     "op": "gt", "threshold": 0.5},

    # D2: Cond_TrendR2_HPP_AvgDD
    {"type": "switch",
     "signal_true": "HighPointPosition", "true_params": {"window": 20},
     "true_transform": {"transform": "zscore", "window": 252},
     "signal_false": "AvgDrawdown", "false_params": {"window": 120},
     "false_negate": True,
     "condition": "TrendR2", "condition_params": {"window": 120, "output": "r2"},
     "op": "gt", "threshold": 0.5},

    # D3: Cond_MADist_TSM_MFI
    {"type": "switch",
     "signal_true": "TimeSeriesMomentum", "true_params": {"window": 252},
     "true_transform": {"transform": "zscore", "window": 252},
     "signal_false": "MFI", "false_params": {"window": 14},
     "condition": "MADistance", "condition_params": {"short_window": 20, "long_window": 60},
     "condition_transform": {"transform": "zscore", "window": 120},
     "op": "lt", "threshold": 0.0},

    # D11: CondF_HPP_TSM_Switch
    {"type": "switch",
     "signal_true": "HighPointPosition", "true_params": {"window": 20},
     "true_transform": {"transform": "zscore", "window": 252},
     "signal_false": "TimeSeriesMomentum", "false_params": {"window": 252},
     "condition": "TrendR2", "condition_params": {"window": 120, "output": "r2"},
     "op": "gt", "threshold": 0.3},

    # D5: Cond2_HPP_Cond — 多条件 + SwitchFactor
    {"type": "switch",
     "signal_true": "MFI", "true_params": {"window": 14},
     "signal_false": "AvgDrawdown", "false_params": {"window": 120},
     "false_negate": True,
     "conditions": [
         {"factor": "HighPointPosition", "params": {"window": 20}, "op": "lt", "threshold": 0.8},
         {"factor": "TrendR2", "params": {"window": 120, "output": "r2"}, "op": "gt", "threshold": 0.3},
     ], "logic": "and"},

    # D6: Cond2_BullBear — 多条件 + 复合因子 signal
    {"type": "switch",
     "signal_true_combo": "Composite_HighPointPosition_window20_weighted_sum_TimeSeriesMomentum_window252",
     "signal_false_combo": "Composite_HighPointPosition_window20_weighted_sum_AvgDrawdown_window120",
     "conditions": [
         {"factor": "MAPosition", "params": {"window": 200}, "op": "gt", "threshold": 0},
         {"factor": "TrendR2", "params": {"window": 120, "output": "r2"}, "op": "gt", "threshold": 0.3},
     ], "logic": "and"},

    # ── 3.2 多条件排名（MultiConditionalFactor） ──

    # D4: Cond2_TrendR2_MAE
    {"signal": "TimeSeriesMomentum", "signal_params": {"window": 252},
     "signal_transform": {"transform": "zscore", "window": 252},
     "conditions": [
         {"factor": "TrendR2", "params": {"window": 120, "output": "r2"}, "op": "gt", "threshold": 0.5},
         {"factor": "MAE", "params": {"window": 40},
          "transform": {"transform": "zscore", "window": 120}, "op": "lt", "threshold": 1.0},
     ], "logic": "and", "false_value": "nan"},

    # D10: CondF_All3 — 多条件 + 复合因子 signal
    {"signal_combo": "Composite_HighPointPosition_window20_weighted_sum_TimeSeriesMomentum_window252",
     "conditions": [
         {"factor": "MAPosition", "params": {"window": 200}, "op": "gt", "threshold": 0},
         {"factor": "TrendR2", "params": {"window": 120, "output": "r2"}, "op": "gt", "threshold": 0.3},
         {"factor": "MAE", "params": {"window": 40},
          "transform": {"transform": "zscore", "window": 120}, "op": "lt", "threshold": 1.5},
     ], "logic": "and", "false_value": "nan"},

    # ── 3.3 条件过滤器（ConditionalFactor） ──

    # D7: CondF_MA200
    {"signal": "MFI", "signal_params": {"window": 14},
     "condition": "MAPosition", "condition_params": {"window": 200},
     "op": "gt", "threshold": 0.0, "false_value": "nan"},

    # D8: CondF_TrendR2
    {"signal": "TimeSeriesMomentum", "signal_params": {"window": 252},
     "condition": "TrendR2", "condition_params": {"window": 120, "output": "r2"},
     "op": "gt", "threshold": 0.3, "false_value": "nan"},

    # D9: CondF_MAE
    {"signal": "HighPointPosition", "signal_params": {"window": 20},
     "signal_transform": {"transform": "zscore", "window": 252},
     "condition": "MAE", "condition_params": {"window": 40},
     "condition_transform": {"transform": "zscore", "window": 120},
     "op": "lt", "threshold": 1.5, "false_value": "nan"},

    # 保留 default.py 示例
    {"signal": "PriceReturn", "signal_params": {"window": 20},
     "condition": "TrendR2", "condition_params": {"window": 120, "output": "r2"},
     "op": "gt", "threshold": 0.5, "false_value": "nan"},
]
