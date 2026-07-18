"""
因子注册表 — 因子名字符串 ↔ BaseFactor 实例的双向映射

支持两种解析路径：
1. 基本因子：factor_type + factor_params → 直接构造
2. 衍生因子：解析因子名中的变换链 → 递归构造
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from factors.base_factor import BaseFactor
from factors.ma import MAPosition, MASlope, MADistance, MADispersion, BIAS, BollingerBandPosition, MAAlignment
from factors.price_return import PriceReturn
from factors.price_momentum import (
    TimeSeriesMomentum, RiskAdjustedReturn, IntradayMomentum,
    OvernightReturn, HighPointPosition, LowPointPosition,
)
from factors.oscillator import RSI, Stochastic, CCI, WilliamsR, MFI
from factors.volatility import Volatility
from factors.volatility_family import AvgDrawdown, DownsideVolatility, MaxDrawdown
from factors.distribution_family import (
    ReturnSkew, ReturnKurtosis, HistoricalVaR, CVaR,
    MaxFavorableExcursion, MaxAdverseExcursion, InformationDiscreteness,
)
from factors.reversal import ShortTermReversal, ExtremeReversal, VolumeReversal
from factors.breakout_family import (
    NewHighContinuous, NewLowContinuous, DonchianChannelPosition,
    ATRRatio, ChandelierExit,
)
from factors.average_true_range import AverageTrueRange
from factors.trend_r2 import TrendR2Factor
from factors.trend_quality import ADX as ADXFactor
from factors.rolling_ols import RollingOLS
from factors.volume_family import (
    VolumeRatio, VolumePriceCorrelation, OBV, VPT,
)
from factors.daily_rebound import DailyRebound
from factors.change_since_new_high import ChangeSinceNewHigh
from factors.average_amount import AverageAmount
from factors.rsrs import RsrsFactor
from factors.meta_factor import NegateFactor


# ── 基本因子类型映射：factor_type → (class, param_mapping) ─────────────────

class FactorTypeMapping:
    """factor_type → class 及参数适配。"""

    def __init__(self, cls: type[BaseFactor], param_map: dict[str, str] | None = None):
        self.cls = cls
        # param_map: report.json 中的 key → 构造参数的 key
        self.param_map = param_map or {}


BASIC_FACTOR_MAP: dict[str, FactorTypeMapping] = {
    # 均线族
    "MAPosition": FactorTypeMapping(MAPosition),
    "MASlope": FactorTypeMapping(MASlope),
    "MADistance": FactorTypeMapping(MADistance, {"short_window": "short_window", "long_window": "long_window"}),
    "MADispersion": FactorTypeMapping(MADispersion),
    "BIAS": FactorTypeMapping(BIAS),
    "BollingerBandPosition": FactorTypeMapping(BollingerBandPosition),
    "MAAlignment": FactorTypeMapping(MAAlignment),
    "MAFactor": FactorTypeMapping(MASlope),  # 别名

    # 价格动量
    "PriceReturn": FactorTypeMapping(PriceReturn),
    "TimeSeriesMomentum": FactorTypeMapping(TimeSeriesMomentum),
    "RiskAdjustedReturn": FactorTypeMapping(RiskAdjustedReturn),
    "IntradayMomentum": FactorTypeMapping(IntradayMomentum),
    "OvernightReturn": FactorTypeMapping(OvernightReturn),
    "HighPointPosition": FactorTypeMapping(HighPointPosition),
    "LowPointPosition": FactorTypeMapping(LowPointPosition),

    # 超买超卖
    "RSI": FactorTypeMapping(RSI),
    "Stochastic": FactorTypeMapping(Stochastic, {"K_period": "k_period", "D_period": "d_period"}),
    "CCI": FactorTypeMapping(CCI),
    "WilliamsR": FactorTypeMapping(WilliamsR),
    "MFI": FactorTypeMapping(MFI),

    # 波动率
    "Volatility": FactorTypeMapping(Volatility),
    "AvgDrawdown": FactorTypeMapping(AvgDrawdown),
    "DownsideVolatility": FactorTypeMapping(DownsideVolatility),
    "MaxDrawdown": FactorTypeMapping(MaxDrawdown),

    # 分布族
    "ReturnSkew": FactorTypeMapping(ReturnSkew),
    "ReturnKurtosis": FactorTypeMapping(ReturnKurtosis),
    "HistoricalVaR": FactorTypeMapping(HistoricalVaR, {"q": "q"}),
    "CVaR": FactorTypeMapping(CVaR, {"q": "q"}),
    "MaxFavorableExcursion": FactorTypeMapping(MaxFavorableExcursion),
    "MaxAdverseExcursion": FactorTypeMapping(MaxAdverseExcursion),
    "InformationDiscreteness": FactorTypeMapping(InformationDiscreteness),

    # 反转
    "ShortTermReversal": FactorTypeMapping(ShortTermReversal),
    "ExtremeReversal": FactorTypeMapping(ExtremeReversal, {"tail_pct": "tail_pct"}),
    "VolumeReversal": FactorTypeMapping(VolumeReversal),

    # 突破
    "NewHighContinuous": FactorTypeMapping(NewHighContinuous),
    "NewLowContinuous": FactorTypeMapping(NewLowContinuous),
    "DonchianPosition": FactorTypeMapping(DonchianChannelPosition),
    "DonchianChannelPosition": FactorTypeMapping(DonchianChannelPosition),
    "ATRRatio": FactorTypeMapping(ATRRatio),
    "ChandelierExit": FactorTypeMapping(ChandelierExit, {"n": "n"}),

    # 其他
    "ATR": FactorTypeMapping(AverageTrueRange),
    "AverageTrueRange": FactorTypeMapping(AverageTrueRange),
    "ADX": FactorTypeMapping(ADXFactor, {"output": "output"}),
    "TrendR2": FactorTypeMapping(TrendR2Factor),
    "TrendR2Factor": FactorTypeMapping(TrendR2Factor),
    "RollingOLS": FactorTypeMapping(RollingOLS),
    "OBV": FactorTypeMapping(OBV),
    "VPT": FactorTypeMapping(VPT),
    "DailyRebound": FactorTypeMapping(DailyRebound),
    "ChangeSinceNewHigh": FactorTypeMapping(ChangeSinceNewHigh),
    "AverageAmount": FactorTypeMapping(AverageAmount),
    "RSRS": FactorTypeMapping(RsrsFactor),
    "VolumeRatio": FactorTypeMapping(VolumeRatio),
    "VolumePriceCorrelation": FactorTypeMapping(VolumePriceCorrelation, {"window": "window"}),
}


def build_factor_from_meta(factor_type: str, factor_params: dict[str, Any]) -> BaseFactor | None:
    """从 report.json 的 meta 信息构造因子实例。

    Parameters
    ----------
    factor_type: 因子类型名
    factor_params: 构造参数字典

    Returns
    -------
    BaseFactor 实例，或 None（无法解析）
    """
    mapping = BASIC_FACTOR_MAP.get(factor_type)
    if mapping is None:
        return None

    # 适配参数名
    adapted: dict[str, Any] = {}
    for k, v in factor_params.items():
        target_key = mapping.param_map.get(k, k) if mapping.param_map else k
        adapted[target_key] = v

    try:
        return mapping.cls(**adapted)
    except Exception:
        return None


def parse_factor_name(factor_name: str) -> tuple[str, str, dict[str, Any]] | None:
    """从因子名字符串解析出因子类型和参数（用于从 report.json 获取）。

    Returns
    -------
    (factor_type, base_name, params) 或 None
    """
    # 尝试从 report.json 获取 meta 信息
    from factor_analysis.config import FactorAnalysisConfig  # noqa
    import json
    from config import DataPath

    dir_path = Path(DataPath.DATA_DIR) / "factors" / factor_name
    reports = sorted(dir_path.glob("report_*.json"), reverse=True)
    if not reports:
        return None

    try:
        with open(reports[0], encoding="utf-8") as f:
            full = json.load(f)
    except Exception:
        return None

    meta = full.get("meta", {})
    return (
        meta.get("factor_type", ""),
        factor_name,
        meta.get("factor_params", {}),
    )


def get_factor_instance(factor_name: str) -> BaseFactor | None:
    """从因子名获取用于构建 FactorPanel 的因子实例。

    支持：
    1. 基本因子 — factor_type → class 映射（从 report.json 或因子名推断）
    2. TransformFactor / CombineFactor / ConditionalFactor / SwitchFactor / MultiConditionalFactor
    """
    parsed = parse_factor_name(factor_name)
    if parsed is not None:
        factor_type, _, params = parsed
        # 基本因子：直接映射
        base = build_factor_from_meta(factor_type, params)
        if base is not None:
            return base
        # meta 因子：根据因子名反向解析
        return _build_meta_factor(factor_name, factor_type, params)

    # 没有 report.json，尝试从因子名直接构造基本因子
    base = _build_from_name_only(factor_name)
    if base is not None:
        return base

    # 尝试解析无 report.json 的衍生因子名（TransformFactor / NegateFactor 等）
    return _build_derived_from_name_only(factor_name)


# ── Meta 因子反向解析 ──────────────────────────────────────────────────────────

def _find_meta_split(factor_name: str) -> int:
    """找到因子名中 meta 操作开始位置（第一个 __）。

    例如: "ADX_14_adx__delta_10" → 返回 "__delta_10" 的起始索引。
    """
    idx = factor_name.find("__")
    return idx if idx > 0 else -1


def _build_derived_from_name_only(factor_name: str) -> BaseFactor | None:
    """从因子名解析衍生因子（无 report.json 时的后备路径）。

    支持：
    - NegateFactor: {dep}__neg
    - TransformFactor: {dep}__{transform}_{window}[_{threshold}]
    """
    from factors.meta_factor import TransformFactor

    # NegateFactor: {dep}__neg
    if factor_name.endswith("__neg"):
        dep_name = factor_name[:-5]
        dep = get_factor_instance(dep_name)
        if dep is not None:
            try:
                return NegateFactor(dep)
            except Exception:
                pass
        return None

    # TransformFactor: 需要至少一个 __
    split = _find_meta_split(factor_name)
    if split < 0:
        return None

    dep_name = factor_name[:split]
    dep = get_factor_instance(dep_name)
    if dep is None:
        return None

    rest = factor_name[split + 2:]  # __ 后面的部分

    # 已知 transform 列表
    known_transforms = [
        "zscore", "delta", "pct_change", "rolling_mean", "rolling_std",
        "binarize_winrate",
    ]
    for transform in known_transforms:
        prefix = f"{transform}_"
        if rest.startswith(prefix):
            param_str = rest[len(prefix):]
            parts = param_str.split("_")
            if not parts:
                continue
            try:
                window = int(parts[0])
            except ValueError:
                continue

            threshold = 0.0
            if len(parts) > 1:
                try:
                    threshold = float(parts[1])
                except ValueError:
                    pass

            try:
                return TransformFactor(
                    dependency=dep,
                    transform=transform,
                    window=window,
                    threshold=threshold,
                )
            except Exception:
                continue

    return None


def _build_meta_factor(factor_name: str, factor_type: str, params: dict[str, Any]) -> BaseFactor | None:
    """根据因子名和 report.json 的 meta 信息反向构造任意 meta 因子。"""
    split = _find_meta_split(factor_name)
    if split < 0:
        return None

    operation = factor_name[split + 2:]  # __ 后面的部分

    # CombineFactor: {a}__product_{b} / {a}__ratio_{b} / {a}__weighted_sum_{b} / {a}__diff_{b}
    if factor_type == "CombineFactor":
        return _build_combine_factor(factor_name, params)

    # SwitchFactor: {t}__if_{cond}__else_{f}[_neg]
    if factor_type == "SwitchFactor":
        return _build_switch_factor(factor_name, params)

    # ConditionalFactor: {signal}__if_{cond}_{op}_{threshold}
    if factor_type == "ConditionalFactor":
        return _build_conditional_factor(factor_name, params)

    # MultiConditionalFactor: {signal}__if_{c1}_{logic}_{c2}...
    if factor_type == "MultiConditionalFactor":
        return _build_multicond_factor(factor_name, params)

    # TransformFactor: {dep}__{transform}_{window}[_{threshold}]
    if factor_type == "TransformFactor":
        return _build_transform_factor(factor_name, params)

    # NegateFactor: {dep}__neg
    if factor_type == "NegateFactor":
        return _build_negate_factor(factor_name)

    return None


# ── TransformFactor ─────────────────────────────────────────────────────────────

def _build_transform_factor(factor_name: str, params: dict[str, Any]) -> BaseFactor | None:
    from factors.meta_factor import TransformFactor

    transform = params.get("transform", "")
    window = int(params.get("window", 0))
    threshold = float(params.get("threshold", 0.0))

    if transform == "binarize_winrate":
        suffix = f"__{transform}_{window}_{threshold}"
    else:
        suffix = f"__{transform}_{window}"

    if not factor_name.endswith(suffix):
        suffix = f"__{transform}_"
        idx = factor_name.rfind(suffix)
        if idx == -1:
            return None
        base_name = factor_name[:idx]
    else:
        base_name = factor_name[:-len(suffix)]

    if not base_name:
        return None

    dep = get_factor_instance(base_name)
    if dep is None:
        return None

    try:
        return TransformFactor(
            dependency=dep, transform=transform,
            window=window if window > 0 else None, threshold=threshold,
        )
    except Exception:
        return None


# ── NegateFactor ────────────────────────────────────────────────────────────────

def _build_negate_factor(factor_name: str) -> BaseFactor | None:
    """解析 NegateFactor：{dep}__neg"""
    # 格式: {dep}__neg
    suffix = "__neg"
    if not factor_name.endswith(suffix):
        return None

    dep_name = factor_name[:-len(suffix)]
    if not dep_name:
        return None

    dep = get_factor_instance(dep_name)
    if dep is None:
        return None

    try:
        return NegateFactor(dep)
    except Exception:
        return None


# ── CombineFactor ───────────────────────────────────────────────────────────────

_COMBINE_METHODS = {"product", "ratio", "weighted_sum", "diff"}


def _build_combine_factor(factor_name: str, params: dict[str, Any]) -> BaseFactor | None:
    from factors.meta_factor import CombineFactor

    method = params.get("method", "")
    # 从因子名中定位 __{method}_
    delimiter = f"__{method}_"
    idx = factor_name.find(delimiter)
    if idx < 0:
        return None

    name_a = factor_name[:idx]
    name_b = factor_name[idx + len(delimiter):]

    dep_a = get_factor_instance(name_a)
    dep_b = get_factor_instance(name_b)
    if dep_a is None or dep_b is None:
        return None

    try:
        return CombineFactor(
            factor_a=dep_a, factor_b=dep_b, method=method,
            weight_a=float(params.get("weight_a", 0.5)),
            weight_b=float(params.get("weight_b", 0.5)),
            normalize=bool(params.get("normalize", True)),
            normalize_window=int(params.get("normalize_window", 252)),
        )
    except Exception:
        return None


# ── ConditionalFactor ───────────────────────────────────────────────────────────

def _build_conditional_factor(factor_name: str, params: dict[str, Any]) -> BaseFactor | None:
    from factors.meta_factor import ConditionalFactor

    # 格式: {signal}__if_{cond}_{op}_{threshold}
    delimiter = "__if_"
    idx = factor_name.find(delimiter)
    if idx < 0:
        return None

    signal_name = factor_name[:idx]
    rest = factor_name[idx + len(delimiter):]  # {cond}_{op}_{threshold}

    # 从末尾解析 op 和 threshold
    op = params.get("op", "gt")
    threshold = float(params.get("threshold", 0.0))
    # cond 部分是去掉末尾 _{op}_{threshold}
    suffix = f"_{op}_{threshold}"
    if not rest.endswith(suffix):
        return None
    cond_name = rest[:-len(suffix)]

    signal = get_factor_instance(signal_name)
    cond = get_factor_instance(cond_name)
    if signal is None or cond is None:
        return None

    try:
        return ConditionalFactor(
            signal=signal, condition=cond, op=op,
            threshold=threshold,
            false_value=params.get("false_value", "nan"),
        )
    except Exception:
        return None


# ── SwitchFactor ────────────────────────────────────────────────────────────────

def _build_switch_factor(factor_name: str, params: dict[str, Any]) -> BaseFactor | None:
    from factors.meta_factor import SwitchFactor

    # 格式: {t}__if_{cond}_{op}_{threshold}__else_{f}[_neg]
    idx_if = factor_name.find("__if_")
    idx_else = factor_name.find("__else_")
    if idx_if < 0 or idx_else < 0:
        return None

    name_true = factor_name[:idx_if]
    cond_part = factor_name[idx_if + 5:idx_else]  # {cond}_{op}_{threshold}
    name_false_raw = factor_name[idx_else + 7:]   # {f} or {f}_neg

    false_negate = name_false_raw.endswith("_neg")
    name_false = name_false_raw[:-4] if false_negate else name_false_raw

    op = params.get("op", "gt")
    threshold = float(params.get("threshold", 0.0))

    # 从 cond_part 中提取 cond_name
    suffix = f"_{op}_{threshold}"
    if not cond_part.endswith(suffix):
        return None
    cond_name = cond_part[:-len(suffix)]

    dep_t = get_factor_instance(name_true)
    dep_f = get_factor_instance(name_false)
    cond = get_factor_instance(cond_name)
    if dep_t is None or dep_f is None or cond is None:
        return None

    try:
        return SwitchFactor(
            signal_true=dep_t, signal_false=dep_f,
            condition=cond, op=op, threshold=threshold,
            false_negate=false_negate,
        )
    except Exception:
        return None


# ── MultiConditionalFactor ──────────────────────────────────────────────────────

def _build_multicond_factor(factor_name: str, params: dict[str, Any]) -> BaseFactor | None:
    from factors.meta_factor import MultiConditionalFactor, ConditionSpec

    # 格式: {signal}__if_{c1}_{logic}_{c2}_{logic}_{c3}...
    delimiter = "__if_"
    idx = factor_name.find(delimiter)
    if idx < 0:
        return None

    signal_name = factor_name[:idx]
    rest = factor_name[idx + len(delimiter):]

    logic = params.get("logic", "and")
    logic_sep = f"_{logic}_"

    # 按 logic 分隔符拆分条件段
    cond_parts = rest.split(logic_sep)
    if len(cond_parts) < 2:
        return None

    signal = get_factor_instance(signal_name)
    if signal is None:
        return None

    cond_specs: list[ConditionSpec] = []
    for cp in cond_parts:
        # 每个 cp 格式: {cond_name}_{op}_{threshold}
        op = ""
        for valid_op in ("gte", "lte", "gt", "lt"):
            marker = f"_{valid_op}_"
            if marker in cp:
                op = valid_op
                break
        if not op:
            return None
        parts = cp.rsplit(f"_{op}_", 1)
        if len(parts) != 2:
            return None
        cond_name = parts[0]
        try:
            thresh = float(parts[1])
        except ValueError:
            return None

        cond = get_factor_instance(cond_name)
        if cond is None:
            return None
        cond_specs.append(ConditionSpec(condition=cond, op=op, threshold=thresh))

    if len(cond_specs) < 2:
        return None

    try:
        return MultiConditionalFactor(
            signal=signal, conditions=cond_specs,
            logic=logic, false_value=params.get("false_value", "nan"),
        )
    except Exception:
        return None


# ── 无 report.json 时的名称推断 ────────────────────────────────────────────────

# 基本因子命名模式：{name}_{param1}_{param2}_{param3}...
# 当因子没有 report.json 时，通过命名规则直接构造
_NAME_PATTERNS: list[tuple[str, type[BaseFactor], list[str]]] = [
    # (type_prefix, class, [param_names])
    # 均线族
    ("MADistance", MADistance, ["price_column", "short_window", "long_window"]),
    ("MASlope", MASlope, ["price_column", "window", "window2"]),
    ("MADispersion", MADispersion, ["price_column"] + [f"w{i}" for i in range(4)]),
    ("MAPosition", MAPosition, ["price_column", "window"]),
    ("BIAS", BIAS, ["price_column", "window"]),
    ("BollingerBandPosition", BollingerBandPosition, ["price_column", "window", "k"]),
    # 价格动量
    ("HighPointPosition", HighPointPosition, ["window"]),
    ("LowPointPosition", LowPointPosition, ["window"]),
    ("TimeSeriesMomentum", TimeSeriesMomentum, ["window"]),
    ("PriceReturn", PriceReturn, ["window"]),
    # 振荡器
    ("RSI", RSI, ["window"]),
    ("CCI", CCI, ["window"]),
    ("WilliamsR", WilliamsR, ["window"]),
    ("MFI", MFI, ["window"]),
    # 波动率
    ("Volatility", Volatility, ["window"]),
    ("AvgDrawdown", AvgDrawdown, ["window"]),
    ("DownsideVolatility", DownsideVolatility, ["window"]),
    ("MaxDrawdown", MaxDrawdown, ["window"]),
    # 突破
    ("ATRRatio", ATRRatio, ["window"]),
    ("NewHighContinuous", NewHighContinuous, ["window"]),
    ("NewLowContinuous", NewLowContinuous, ["window"]),
    ("DonchianChannelPosition", DonchianChannelPosition, ["window"]),
    # 分布
    ("MAE", MaxAdverseExcursion, ["window"]),
    ("MFE", MaxFavorableExcursion, ["window"]),
    ("ReturnSkew", ReturnSkew, ["window"]),
    ("ReturnKurtosis", ReturnKurtosis, ["window"]),
    # 趋势
    ("TrendR2", TrendR2Factor, ["window", "value_column", "output"]),
    # 成交量
    ("VolumeRatio", VolumeRatio, ["window"]),
    ("VolumePriceCorrelation", VolumePriceCorrelation, ["window"]),
    # 其他
    ("AverageTrueRange", AverageTrueRange, ["window"]),
    ("AverageAmount", AverageAmount, ["window"]),
    ("DailyRebound", DailyRebound, []),
    ("OBV", OBV, []),
    ("VPT", VPT, []),
    # ADX（在 trend_quality 中）
    ("ADX", ADXFactor, ["window", "output"]),
]


def _build_from_name_only(factor_name: str) -> BaseFactor | None:
    """没有 report.json 时，从因子名字符串直接构造基本因子。

    根据命名模式 {Type}_{param1}_{param2}... 尝试匹配。
    """
    for prefix, cls, param_names in _NAME_PATTERNS:
        if not factor_name.startswith(prefix + "_"):
            continue
        # 提取参数字符串
        param_str = factor_name[len(prefix) + 1:]
        param_parts = param_str.split("_")

        if len(param_parts) != len(param_names):
            continue

        # 构造 kwargs
        kwargs: dict[str, Any] = {}
        valid = True
        for i, pname in enumerate(param_names):
            val = param_parts[i]
            # 尝试转数值
            try:
                if "." in val:
                    kwargs[pname] = float(val)
                else:
                    kwargs[pname] = int(val)
            except ValueError:
                kwargs[pname] = val  # 字符串参数（如 price_column="close"）

        try:
            return cls(**kwargs)
        except Exception:
            continue

    return None


# ── 已知因子名集合（用于因子走势 API 的因子建议） ─────────────────────────

def get_available_factors() -> list[str]:
    """返回所有可构建 FactorPanel 的因子名（即有 report.json 的因子）。"""
    from config import DataPath
    from pathlib import Path

    root = Path(DataPath.DATA_DIR) / "factors"
    factors = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith(".") and not entry.name.startswith("_"):
            reports = list(entry.glob("report_*.json"))
            if reports:
                factors.append(entry.name)
    return factors
