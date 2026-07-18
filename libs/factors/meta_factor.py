"""
元因子框架 (Meta-Factor Framework)

通过"变换 + 组合"将基础因子扩展为衍生因子，为因子轮动提供充足且多样化的因子池。

三层架构:
    TransformFactor    — 单因子后处理变换（6 种）
    CombineFactor      — 双因子二元运算（4 种）
    SwitchFactor       — 条件切换因子（True/False 双路径）
    ConditionalFactor       — 条件信号（信号因子仅在条件满足时生效）
    MultiConditionalFactor  — 多条件组合（AND/OR 逻辑，两个及以上条件）

所有元因子继承自 DerivedFactor，复用了它的依赖注入、DataFrame 组装和
warmup 链式计算。每个子类只需实现 compute_from_frame(frame)。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from factors.derived_factor import DerivedFactor
from factors.base_factor import BaseFactor


# ═══════════════════════════════════════════════════════════════════════════════
# 元因子配方 (Phase 4 — 批量集成)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetaFactorSpec:
    """描述一个衍生因子的生成配方。

    不是 BaseFactor 子类 — 它是一个序列化的配方对象，
    在 worker 进程中通过 build_meta_factor() 实例化为真正的因子。

    Attributes
    ----------
    base_factor_name: 基础因子在 FACTOR_REGISTRY 中的键。
    base_factor_module: 基础因子模块路径。
    base_factor_class: 基础因子类名。
    base_params: 基础因子的构造参数。
    meta_type: "transform" | "combine" | "conditional" | "multi_conditional"
    meta_params: 传给 meta factor 构造函数的参数。
    """
    base_factor_name: str
    base_factor_module: str
    base_factor_class: str
    base_params: dict[str, Any]
    meta_type: str
    meta_params: dict[str, Any]


def _resolve_factor(raw):
    """递归解析因子构造描述。

    MetaFactorSpec → build_meta_factor 递归构建
    dict → MetaFactorSpec(**dict) → build_meta_factor（JSON 反序列化恢复）
    tuple/list (module, class_name, params) → import + 实例化
    """
    import importlib
    if isinstance(raw, MetaFactorSpec):
        return build_meta_factor(raw)
    if isinstance(raw, dict):
        # JSON 反序列化后的嵌套 MetaFactorSpec 恢复
        return build_meta_factor(MetaFactorSpec(**raw))
    module, cls_name, params = raw
    mod = importlib.import_module(module)
    return getattr(mod, cls_name)(**params)


def build_meta_factor(spec: MetaFactorSpec) -> BaseFactor:
    """从 MetaFactorSpec 配方实例化衍生因子。

    在 worker 进程中调用，将序列化的配方还原为可调用的因子对象。
    支持递归嵌套：meta_params 中可携带 MetaFactorSpec 或 base-factor tuple。
    """
    import importlib

    # 深拷贝 meta_params，避免 pop 破坏原始 spec（对 dry-run/重试等场景至关重要）
    meta = deepcopy(spec.meta_params)

    # 1. 解析 signal/base 因子 — 优先使用 _signal_raw（嵌套），否则用 base_* 字段
    if "_signal_raw" in meta:
        signal_raw = meta.pop("_signal_raw")
        base = _resolve_factor(signal_raw)
    else:
        mod = importlib.import_module(spec.base_factor_module)
        base_cls = getattr(mod, spec.base_factor_class)
        base = base_cls(**spec.base_params)

    # 2. 根据 meta_type 构建衍生因子
    if spec.meta_type == "transform":
        if "_dependency_raw" in meta:
            dep_raw = meta.pop("_dependency_raw")
            dep = _resolve_factor(dep_raw)
        else:
            mod = importlib.import_module(spec.base_factor_module)
            base_cls = getattr(mod, spec.base_factor_class)
            dep = base_cls(**spec.base_params)
        return TransformFactor(dependency=dep, **meta)

    elif spec.meta_type == "switch":
        true_raw = meta.pop("_true_raw")
        false_raw = meta.pop("_false_raw")
        signal_true = _resolve_factor(true_raw)
        signal_false = _resolve_factor(false_raw)

        if "_conditions" in meta:
            # 多条件模式
            conditions_data = meta.pop("_conditions")
            conditions: list[ConditionSpec] = []
            for cd in conditions_data:
                cond_factor = _resolve_factor(cd["_raw"])
                conditions.append(
                    ConditionSpec(
                        condition=cond_factor,
                        op=cd["op"],
                        threshold=cd["threshold"],
                    )
                )
            return SwitchFactor(
                signal_true=signal_true,
                signal_false=signal_false,
                conditions=conditions,
                **meta,
            )
        else:
            cond_raw = meta.pop("_cond_raw")
            condition = _resolve_factor(cond_raw)
            return SwitchFactor(
                signal_true=signal_true,
                signal_false=signal_false,
                condition=condition,
                **meta,
            )

    elif spec.meta_type == "combine":
        b_raw = meta.pop("_b_raw")
        factor_b = _resolve_factor(b_raw)
        return CombineFactor(factor_a=base, factor_b=factor_b, **meta)

    elif spec.meta_type == "conditional":
        cond_raw = meta.pop("_cond_raw")
        condition = _resolve_factor(cond_raw)
        return ConditionalFactor(signal=base, condition=condition, **meta)

    elif spec.meta_type == "multi_conditional":
        conditions_data = meta.pop("_conditions")
        conditions: list[ConditionSpec] = []
        for cd in conditions_data:
            cond_raw = cd["_raw"]
            cond_factor = _resolve_factor(cond_raw)
            conditions.append(
                ConditionSpec(
                    condition=cond_factor,
                    op=cd["op"],
                    threshold=cd["threshold"],
                )
            )
        return MultiConditionalFactor(
            signal=base, conditions=conditions, **meta
        )

    elif spec.meta_type == "negate":
        if "_dependency_raw" in meta:
            dep_raw = meta.pop("_dependency_raw")
            dep = _resolve_factor(dep_raw)
            return NegateFactor(dep)
        return NegateFactor(base)

    else:
        raise ValueError(f"未知 meta_type: {spec.meta_type!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: TransformFactor — 单因子变换
# ═══════════════════════════════════════════════════════════════════════════════

class TransformFactor(DerivedFactor):
    """对任意基础因子施加后处理变换，生成衍生因子。

    接收一个依赖因子实例，对其输出的 Series 施加指定的变换。

    六种变换:
        rolling_mean     — N 日简单移动平均，降噪
        rolling_std      — N 日滚动标准差，度量因子稳定性
        delta            — N 日差值，因子的"加速度"
        pct_change       — N 日变化率，delta 的百分比版本
        binarize_winrate — 二值化 + 滚动胜率，信号一致性
        zscore           — 时序标准化（rolling z-score）

    Parameters
    ----------
    dependency : BaseFactor
        被变换的基础因子实例。
    transform : str
        变换类型。可选: rolling_mean, rolling_std, delta, pct_change,
        binarize_winrate, zscore。
    window : int
        变换窗口（交易日）。默认值因变换类型而异。
    threshold : float
        二值化阈值（仅 binarize_winrate 使用）。默认 0.0。

    Examples
    --------
    >>> from factors.price_return import PriceReturn
    >>> pr = PriceReturn(window=20)
    >>> ma_pr = TransformFactor(dependency=pr, transform="rolling_mean", window=10)
    >>> ma_pr.get_output_name()
    'PriceReturn_20__rolling_mean_10'
    """

    name = "TransformFactor"

    _DEFAULT_WINDOWS: dict[str, int] = {
        "rolling_mean": 10,
        "rolling_std": 20,
        "delta": 5,
        "pct_change": 5,
        "binarize_winrate": 20,
        "zscore": 252,
    }

    _VALID_TRANSFORMS = frozenset(_DEFAULT_WINDOWS.keys())

    def __init__(
        self,
        dependency: BaseFactor,
        transform: str,
        window: int | None = None,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if transform not in self._VALID_TRANSFORMS:
            raise ValueError(
                f"未知变换类型: {transform!r}。可选: {sorted(self._VALID_TRANSFORMS)}"
            )

        self.transform = transform
        self.window = int(window if window is not None else self._DEFAULT_WINDOWS[transform])
        self.threshold = float(threshold)

        if self.window < 1:
            raise ValueError(f"window 必须 >= 1，收到: {self.window}")

        # 注册依赖 — DerivedFactor 会自动处理注入和 warmup 链
        self.add_dependency(dependency)

        # warmup = 依赖因子的 warmup + 本变换所需的窗口 - 1
        # PriceReturn(20).warmup=21 (1-indexed: 第 21 根 bar 是第一个有效值)
        # rolling_mean(10) 需要 10 个有效值才能出第一个结果
        # → 第一个有效值在 1-indexed: 21 + (10 - 1) = 30
        self.warmup_period = self._dep_warmup() + self.window - 1

        self._set_params(
            transform=transform,
            window=self.window,
            threshold=self.threshold,
        )

    def _dep_warmup(self) -> int:
        """依赖因子的最大 warmup。"""
        if not self._dependencies:
            return 0
        return self._dependencies[0].get_max_warmup_period()

    def get_output_name(self) -> str:
        dep_name = self._dependencies[0].get_output_name()
        base = f"{dep_name}__{self.transform}_{self.window}"
        if self.transform == "binarize_winrate":
            # 不同 threshold 需要不同输出名，避免同名冲突
            base += f"_{self.threshold}"
        return base

    # ── compute_from_frame ──────────────────────────────────────────────────

    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        """从 build_input_frame 组装的 frame 中计算衍生因子。

        frame 中依赖因子的列名为其 get_output_name()。
        """
        dep_name = self._dependencies[0].get_output_name()
        if dep_name not in frame.columns:
            raise ValueError(
                f"依赖因子列 {dep_name!r} 不在 frame 中，"
                f"可用列: {list(frame.columns)}"
            )
        factor_value = frame[dep_name]

        dispatch = {
            "rolling_mean": self._rolling_mean,
            "rolling_std": self._rolling_std,
            "delta": self._delta,
            "pct_change": self._pct_change,
            "binarize_winrate": self._binarize_winrate,
            "zscore": self._zscore,
        }
        result = dispatch[self.transform](factor_value)
        result.name = self.get_output_name()
        return result

    # ── 6 种变换实现 ────────────────────────────────────────────────────────

    def _rolling_mean(self, series: pd.Series) -> pd.Series:
        """因子值的 N 日简单移动平均。"""
        return series.rolling(window=self.window, min_periods=self.window).mean()

    def _rolling_std(self, series: pd.Series) -> pd.Series:
        """因子值的 N 日滚动标准差。"""
        return series.rolling(window=self.window, min_periods=self.window).std()

    def _delta(self, series: pd.Series) -> pd.Series:
        """因子值在过去 N 日的差值（因子加速度）。"""
        return series - series.shift(self.window)

    def _pct_change(self, series: pd.Series) -> pd.Series:
        """因子值的 N 日变化率。

        分母接近 0 时结果可能为 inf，统一替换为 NaN。
        """
        result = series.pct_change(periods=self.window)
        return result.replace([np.inf, -np.inf], np.nan)

    def _binarize_winrate(self, series: pd.Series) -> pd.Series:
        """二值化 + 滚动胜率。

        将因子值二值化（> threshold = 1，否则 = 0），
        然后计算 N 日滚动均值作为"胜率"。
        """
        binary = (series > self.threshold).astype(float)
        return binary.rolling(window=self.window, min_periods=self.window).mean()

    def _zscore(self, series: pd.Series) -> pd.Series:
        """时序滚动 z-score 标准化。

        result = (factor - rolling_mean) / rolling_std
        当 rolling_std = 0 时（如长期横盘），结果为 NaN。
        """
        rm = series.rolling(window=self.window, min_periods=self.window).mean()
        rs = series.rolling(window=self.window, min_periods=self.window).std(ddof=1)
        # 避免除以 0
        rs_safe = rs.replace(0.0, np.nan)
        return (series - rm) / rs_safe


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1b: NegateFactor — 因子方向反转
# ═══════════════════════════════════════════════════════════════════════════════

class NegateFactor(DerivedFactor):
    """对依赖因子取反（乘 -1），用于反转因子方向。

    典型场景：将"越高越差"的因子（如最大不利偏移 MAE）转为"越高越好"，
    使其与"越高越好"的动量因子方向一致，便于统一 ranking。

    与 TransformFactor 不同：NegateFactor 不依赖窗口参数，是纯符号反转。

    Examples
    --------
    >>> from factors.distribution_family import MaxAdverseExcursion
    >>> mae = MaxAdverseExcursion(window=40)
    >>> anti_mae = NegateFactor(mae)
    >>> anti_mae.get_output_name()
    'MaxAdverseExcursion_40__neg'
    """

    name = "NegateFactor"

    def __init__(self, dependency: BaseFactor) -> None:
        super().__init__()
        self.add_dependency(dependency)
        self.warmup_period = dependency.get_max_warmup_period()

    def get_output_name(self) -> str:
        dep_name = self._dependencies[0].get_output_name()
        return f"{dep_name}__neg"

    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        dep_name = self._dependencies[0].get_output_name()
        result = -frame[dep_name]
        result.name = self.get_output_name()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: CombineFactor — 双因子运算
# ═══════════════════════════════════════════════════════════════════════════════

class CombineFactor(DerivedFactor):
    """对两个因子的输出做二元运算，捕获因子间的交互效应。

    四种运算:
        product     — 两因子相乘（共振信号）
        ratio       — 两因子相除（风险调整/效率度量）
        weighted_sum — 两因子加权求和（因子合成）
        diff        — 两因子差值（期限结构/分歧度量）

    Parameters
    ----------
    factor_a : BaseFactor
        第一个因子实例。
    factor_b : BaseFactor
        第二个因子实例。
    method : str
        运算方法。可选: product, ratio, weighted_sum, diff。
    weight_a : float
        因子 A 的权重（仅 weighted_sum）。默认 0.5。
    weight_b : float
        因子 B 的权重（仅 weighted_sum）。默认 0.5。
    normalize : bool
        weighted_sum 时是否先对两个因子做时序 z-score 标准化。默认 True。
    normalize_window : int
        normalize 时使用的滚动窗口。默认 252。

    Examples
    --------
    >>> from factors.price_return import PriceReturn
    >>> from factors.trend_quality import KaufmanEfficiencyRatio
    >>> pr = PriceReturn(window=20)
    >>> ker = KaufmanEfficiencyRatio(window=20)
    >>> combo = CombineFactor(factor_a=pr, factor_b=ker, method="product")
    >>> combo.get_output_name()
    'PriceReturn_20__product_KaufmanER_20'
    """

    name = "CombineFactor"

    _VALID_METHODS = frozenset({"product", "ratio", "weighted_sum", "diff"})

    def __init__(
        self,
        factor_a: BaseFactor,
        factor_b: BaseFactor,
        method: str,
        weight_a: float = 0.5,
        weight_b: float = 0.5,
        normalize: bool = True,
        normalize_window: int = 252,
    ) -> None:
        super().__init__()
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"未知运算方法: {method!r}。可选: {sorted(self._VALID_METHODS)}"
            )

        self.method = method
        self.weight_a = float(weight_a)
        self.weight_b = float(weight_b)
        self.normalize = bool(normalize)
        self.normalize_window = int(normalize_window)

        # 注册两个依赖
        self.add_dependency(factor_a)
        self.add_dependency(factor_b)

        # warmup: 取两个依赖 max warmup；如果 normalize，再加 normalize_window - 1
        base_warmup = max(
            self._dependencies[0].get_max_warmup_period(),
            self._dependencies[1].get_max_warmup_period(),
        )
        if self.normalize:
            base_warmup += self.normalize_window - 1
        self.warmup_period = base_warmup

        self._set_params(
            method=method,
            weight_a=weight_a,
            weight_b=weight_b,
            normalize=normalize,
            normalize_window=normalize_window,
        )

    def get_output_name(self) -> str:
        name_a = self._dependencies[0].get_output_name()
        name_b = self._dependencies[1].get_output_name()
        return f"{name_a}__{self.method}_{name_b}"

    # ── 依赖列映射：使用固定别名避免同名冲突 ──────────────────────────────

    def get_dependency_column_map(self) -> list[str]:
        """为两个依赖分配固定别名 '_a' 和 '_b'。"""
        return ["_a", "_b"]

    # ── compute_from_frame ──────────────────────────────────────────────────

    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        if "_a" not in frame.columns or "_b" not in frame.columns:
            raise ValueError(
                f"CombineFactor 需要列 '_a' 和 '_b'，可用列: {list(frame.columns)}"
            )
        series_a = frame["_a"]
        series_b = frame["_b"]

        dispatch = {
            "product": self._product,
            "ratio": self._ratio,
            "weighted_sum": self._weighted_sum,
            "diff": self._diff,
        }
        result = dispatch[self.method](series_a, series_b)
        result.name = self.get_output_name()
        return result

    # ── 4 种运算实现 ────────────────────────────────────────────────────────

    @staticmethod
    def _product(a: pd.Series, b: pd.Series) -> pd.Series:
        """两因子相乘 — 共振信号。"""
        return a * b

    @staticmethod
    def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
        """两因子相除 — 风险调整/效率度量。

        分母接近 0 时结果为 inf，统一替换为 NaN。
        """
        result = a / b
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _diff(a: pd.Series, b: pd.Series) -> pd.Series:
        """两因子差值 — 期限结构/分歧度量。"""
        return a - b

    def _weighted_sum(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """两因子加权求和 — 因子合成。

        若 normalize=True，先对 a、b 分别做时序 z-score 标准化，
        使量纲可比后再加权求和。
        """
        if self.normalize:
            a = self._rolling_zscore(a)
            b = self._rolling_zscore(b)
        return self.weight_a * a + self.weight_b * b

    def _rolling_zscore(self, series: pd.Series) -> pd.Series:
        """时序滚动 z-score（与 TransformFactor._zscore 相同逻辑）。"""
        w = self.normalize_window
        rm = series.rolling(window=w, min_periods=w).mean()
        rs = series.rolling(window=w, min_periods=w).std(ddof=1)
        rs_safe = rs.replace(0.0, np.nan)
        return (series - rm) / rs_safe


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: ConditionalFactor — 条件组合
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionalFactor(DerivedFactor):
    """只有条件因子满足阈值时，信号因子才生效；否则输出 NaN（或 0）。

    典型用途:
        - "只对趋势结构好的标的做动量排名" → 条件因子: TrendR²_r2 > 0.5
        - "只对高流动性标的使用某因子" → 条件因子: AverageAmount > 阈值

    Parameters
    ----------
    signal : BaseFactor
        信号因子实例。条件满足时取其值，不满足时按 false_value 处理。
    condition : BaseFactor
        条件因子实例。其值与 threshold 比较。
    op : str
        条件运算符: gt (>), lt (<), gte (>=), lte (<=)。
    threshold : float
        条件阈值。默认 0.0。
    false_value : str
        条件不满足时的填充值: "nan" (默认) 或 "zero"。

    Examples
    --------
    >>> from factors.price_return import PriceReturn
    >>> from factors.trend_r2 import TrendR2Factor
    >>> signal = PriceReturn(window=20)
    >>> cond = TrendR2Factor(window=120, output="r2")
    >>> cf = ConditionalFactor(
    ...     signal=signal, condition=cond,
    ...     op="gt", threshold=0.5, false_value="nan",
    ... )
    >>> cf.get_output_name()
    'PriceReturn_20__if_TrendR2_120_r2_gt_0.5'
    """

    name = "ConditionalFactor"

    _VALID_OPS = frozenset({"gt", "lt", "gte", "lte"})
    _VALID_FALSE = frozenset({"nan", "zero"})

    _OP_FUNCS = {
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
    }

    def __init__(
        self,
        signal: BaseFactor,
        condition: BaseFactor,
        op: str = "gt",
        threshold: float = 0.0,
        false_value: str = "nan",
    ) -> None:
        super().__init__()
        if op not in self._VALID_OPS:
            raise ValueError(
                f"未知运算符: {op!r}。可选: {sorted(self._VALID_OPS)}"
            )
        if false_value not in self._VALID_FALSE:
            raise ValueError(
                f"未知 false_value: {false_value!r}。可选: {sorted(self._VALID_FALSE)}"
            )

        self.op = op
        self.threshold = float(threshold)
        self.false_value = false_value

        # 检测 signal 和 condition 是否为等价因子（params 相同）。
        # 由于 BaseFactor.__eq__ 按 params 判等，在 dependency_results dict
        # 中它们会碰撞。self_conditional=True 时只注册一个依赖，
        # compute_from_frame 中自动复用为 signal 和 condition。
        self.self_conditional = signal == condition

        if self.self_conditional:
            # 只注册一个依赖 — 既是 signal 也是 condition
            self.add_dependency(signal)
            self.warmup_period = signal.get_max_warmup_period()
        else:
            self.add_dependency(signal)
            self.add_dependency(condition)
            self.warmup_period = max(
                self._dependencies[0].get_max_warmup_period(),
                self._dependencies[1].get_max_warmup_period(),
            )

        self._set_params(
            op=op,
            threshold=threshold,
            false_value=false_value,
        )

    def get_output_name(self) -> str:
        signal_name = self._dependencies[0].get_output_name()
        if self.self_conditional:
            cond_name = signal_name
        else:
            cond_name = self._dependencies[1].get_output_name()
        return f"{signal_name}__if_{cond_name}_{self.op}_{self.threshold}"

    # ── 依赖列映射：使用固定别名避免同名冲突 ──────────────────────────────

    def get_dependency_column_map(self) -> list[str]:
        """为 signal 和 condition 分配固定别名。"""
        if self.self_conditional:
            return ["_val"]
        return ["_signal", "_cond"]

    # ── compute_from_frame ──────────────────────────────────────────────────

    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        if self.self_conditional:
            if "_val" not in frame.columns:
                raise ValueError(
                    f"ConditionalFactor (self) 需要列 '_val'，"
                    f"可用列: {list(frame.columns)}"
                )
            signal_series = frame["_val"]
            cond_series = frame["_val"]
        else:
            if "_signal" not in frame.columns or "_cond" not in frame.columns:
                raise ValueError(
                    f"ConditionalFactor 需要列 '_signal' 和 '_cond'，"
                    f"可用列: {list(frame.columns)}"
                )
            signal_series = frame["_signal"]
            cond_series = frame["_cond"]

        op_func = self._OP_FUNCS[self.op]
        mask = op_func(cond_series, self.threshold)

        if self.false_value == "nan":
            fallback = np.nan
        else:
            fallback = 0.0

        result = signal_series.where(mask, other=fallback)
        result.name = self.get_output_name()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: MultiConditionalFactor — 多条件组合
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConditionSpec:
    """描述单个筛选条件，供 MultiConditionalFactor 使用。

    Attributes
    ----------
    condition : BaseFactor
        条件因子实例。其值与 threshold 比较。
    op : str
        条件运算符: gt (>), lt (<), gte (>=), lte (<=)。
    threshold : float
        条件阈值。
    """

    condition: BaseFactor
    op: str
    threshold: float

    def __post_init__(self) -> None:
        _VALID = frozenset({"gt", "lt", "gte", "lte"})
        if self.op not in _VALID:
            raise ValueError(
                f"未知运算符: {self.op!r}。可选: {sorted(_VALID)}"
            )


class MultiConditionalFactor(DerivedFactor):
    """多个条件因子按 AND/OR 逻辑组合后过滤信号因子。

    当所有条件满足（AND）或任一条件满足（OR）时，信号因子生效；
    否则输出 NaN（或 0）。

    典型用途:
        - "趋势 R² > 0.5 且 波动率 < 0.03 时使用动量信号"
        - "流动性 > 阈值 或 趋势强度 > 阈值 时生效"

    单个条件请使用 ConditionalFactor；两个及以上条件用本类。

    Parameters
    ----------
    signal : BaseFactor
        信号因子实例。
    conditions : list[ConditionSpec]
        条件列表，至少 2 个。每个元素描述一个条件因子 + 运算符 + 阈值。
    logic : str
        条件组合逻辑: "and"（全部满足，默认）或 "or"（任一满足）。
    false_value : str
        条件不满足时的填充值: "nan" (默认) 或 "zero"。

    Examples
    --------
    >>> from factors.price_return import PriceReturn
    >>> from factors.trend_r2 import TrendR2Factor
    >>> signal = PriceReturn(window=20)
    >>> conditions = [
    ...     ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.0),
    ...     ConditionSpec(condition=PriceReturn(window=120), op="gt", threshold=-0.01),
    ... ]
    >>> mcf = MultiConditionalFactor(signal=signal, conditions=conditions, logic="and")
    >>> mcf.get_output_name()
    'PriceReturn_20__if_PriceReturn_60_gt_0.0_and_PriceReturn_120_gt_-0.01'
    """

    name = "MultiConditionalFactor"

    _VALID_OPS = frozenset({"gt", "lt", "gte", "lte"})
    _VALID_FALSE = frozenset({"nan", "zero"})
    _VALID_LOGIC = frozenset({"and", "or"})

    _OP_FUNCS = {
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
    }

    def __init__(
        self,
        signal: BaseFactor,
        conditions: list[ConditionSpec],
        logic: str = "and",
        false_value: str = "nan",
    ) -> None:
        super().__init__()
        if len(conditions) < 2:
            raise ValueError(
                f"conditions 至少需要 2 个条件，收到 {len(conditions)} 个。"
                f" 单个条件请使用 ConditionalFactor。"
            )
        if logic not in self._VALID_LOGIC:
            raise ValueError(
                f"未知逻辑: {logic!r}。可选: {sorted(self._VALID_LOGIC)}"
            )
        if false_value not in self._VALID_FALSE:
            raise ValueError(
                f"未知 false_value: {false_value!r}。可选: {sorted(self._VALID_FALSE)}"
            )

        self.logic = logic
        self.false_value = false_value
        self._condition_specs = list(conditions)

        # 校验每个 condition spec 的 op
        for i, spec in enumerate(self._condition_specs):
            if spec.op not in self._VALID_OPS:
                raise ValueError(
                    f"conditions[{i}] 未知运算符: {spec.op!r}。"
                    f" 可选: {sorted(self._VALID_OPS)}"
                )

        # 注册依赖并去重:
        #   dependencies[0] = signal
        #   dependencies[1:] = 与 signal 不等的唯一 condition 因子
        # _cond_dep_index[i] 记录第 i 个 spec 对应的 dependencies 索引:
        #   0 表示复用 signal 列，1+ 表示 _unique_conds[j]。
        # 这样避免相同 params 的因子在 dependency_results / column_map
        # 中发生 key 碰撞（BaseFactor 的 __hash__/__eq__ 基于 params）。
        self.add_dependency(signal)

        _unique_conds: list[BaseFactor] = []
        _cond_dep_index: dict[int, int] = {}   # spec_index -> dep_index

        for i, spec in enumerate(self._condition_specs):
            # 先检查是否与 signal 相等
            if spec.condition == signal:
                _cond_dep_index[i] = 0
                continue
            # 再检查是否与已有唯一 condition 相等
            found = False
            for j, uc in enumerate(_unique_conds):
                if spec.condition == uc:
                    _cond_dep_index[i] = j + 1
                    found = True
                    break
            if not found:
                _cond_dep_index[i] = len(_unique_conds) + 1
                _unique_conds.append(spec.condition)
                self.add_dependency(spec.condition)

        self._unique_conds = _unique_conds
        self._cond_dep_index = _cond_dep_index

        # warmup = 所有依赖的最大 warmup
        self.warmup_period = max(
            dep.get_max_warmup_period() for dep in self._dependencies
        )

        self._set_params(
            logic=logic,
            false_value=false_value,
            conditions_count=len(self._condition_specs),
        )

    def get_output_name(self) -> str:
        """生成可读输出名称，包含所有条件的描述。"""
        signal_name = self._dependencies[0].get_output_name()
        cond_parts: list[str] = []
        for i, spec in enumerate(self._condition_specs):
            # _cond_dep_index[i] 直接给出 dependencies 索引
            dep_idx = self._cond_dep_index[i]
            cond_name = self._dependencies[dep_idx].get_output_name()
            cond_parts.append(f"{cond_name}_{spec.op}_{spec.threshold}")
        logic_sep = f"_{self.logic}_"
        return f"{signal_name}__if_{logic_sep.join(cond_parts)}"

    # ── 依赖列映射 ──────────────────────────────────────────────────────────

    def get_dependency_column_map(self) -> list[str]:
        """signal → '_signal'，后续依序为 '_cond_0', '_cond_1', ..."""
        cols = ["_signal"]
        for i in range(len(self._unique_conds)):
            cols.append(f"_cond_{i}")
        return cols

    # ── compute_from_frame ──────────────────────────────────────────────────

    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        """从 frame 中取出信号列和各条件列，按 logic 组合 mask 后过滤信号。"""
        if "_signal" not in frame.columns:
            raise ValueError(
                f"MultiConditionalFactor 需要列 '_signal'，"
                f"可用列: {list(frame.columns)}"
            )
        signal_series = frame["_signal"]

        # 逐条件计算 mask（通过 _cond_dep_index 找到正确的列）
        masks: list[pd.Series] = []
        for i, spec in enumerate(self._condition_specs):
            dep_idx = self._cond_dep_index[i]
            col = "_signal" if dep_idx == 0 else f"_cond_{dep_idx - 1}"
            if col not in frame.columns:
                raise ValueError(
                    f"MultiConditionalFactor 需要列 '{col}'，"
                    f"可用列: {list(frame.columns)}"
                )
            cond_series = frame[col]
            op_func = self._OP_FUNCS[spec.op]
            mask = op_func(cond_series, spec.threshold)
            masks.append(mask)

        # 按 logic 组合所有 mask
        combined = masks[0]
        if self.logic == "and":
            for m in masks[1:]:
                combined = combined & m
        else:  # "or"
            for m in masks[1:]:
                combined = combined | m

        # 应用 mask
        fallback: float = np.nan if self.false_value == "nan" else 0.0
        result = signal_series.where(combined, other=fallback)
        result.name = self.get_output_name()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: SwitchFactor — 条件切换因子
# ═══════════════════════════════════════════════════════════════════════════════

class SwitchFactor(DerivedFactor):
    """条件切换因子：条件满足 → signal_true，否则 → signal_false。

    与 ConditionalFactor 的区别：
        - ConditionalFactor 的 False 状态只能是 NaN 或 0
        - SwitchFactor 的 False 状态是另一个完整的因子

    对应场景：趋势强→动量因子，趋势弱→反转因子；牛市→HPP+TSM，熊市→HPP+AvgDD。

    Parameters
    ----------
    signal_true : BaseFactor
        条件满足时输出的因子。
    signal_false : BaseFactor
        条件不满足时输出的因子。
    condition : BaseFactor
        条件因子实例。其值与 threshold 比较。
    op : str
        条件运算符: gt (>), lt (<), gte (>=), lte (<=)。默认 "gt"。
    threshold : float
        条件阈值。默认 0.0。
    false_negate : bool
        True 时对 signal_false 的输出取负。默认 False。

    Examples
    --------
    >>> from factors.price_return import PriceReturn
    >>> from factors.trend_r2 import TrendR2Factor
    >>> signal_t = PriceReturn(window=20)
    >>> signal_f = PriceReturn(window=60)
    >>> cond = TrendR2Factor(window=120, output="r2")
    >>> sw = SwitchFactor(signal_t, signal_f, cond, "gt", 0.5)
    >>> sw.get_output_name()
    'PriceReturn_20__if_TrendR2_120_r2_gt_0.5__else_PriceReturn_60'
    """

    name = "SwitchFactor"

    _VALID_OPS = frozenset({"gt", "lt", "gte", "lte"})

    _OP_FUNCS = {
        "gt": lambda a, b: a > b,
        "lt": lambda a, b: a < b,
        "gte": lambda a, b: a >= b,
        "lte": lambda a, b: a <= b,
    }

    _VALID_LOGIC = frozenset({"and", "or"})

    def __init__(
        self,
        signal_true: BaseFactor,
        signal_false: BaseFactor,
        condition: BaseFactor | None = None,
        conditions: list[ConditionSpec] | None = None,
        logic: str = "and",
        op: str = "gt",
        threshold: float = 0.0,
        false_negate: bool = False,
    ) -> None:
        super().__init__()
        if condition is None and conditions is None:
            raise ValueError("SwitchFactor 需要 condition 或 conditions")
        if condition is not None and conditions is not None:
            raise ValueError("SwitchFactor 不能同时指定 condition 和 conditions")
        if conditions is not None and len(conditions) < 2:
            raise ValueError(f"conditions 至少需要 2 个条件，收到 {len(conditions)}。单个请用 condition 参数")
        if logic not in self._VALID_LOGIC:
            raise ValueError(f"未知 logic: {logic!r}。可选: {sorted(self._VALID_LOGIC)}")

        self.op = op
        self.threshold = float(threshold)
        self.false_negate = bool(false_negate)
        self.logic = logic
        self._condition_specs: list[ConditionSpec] = []

        # 注册依赖
        self.add_dependency(signal_true)   # _dependencies[0] = "_true"
        self.add_dependency(signal_false)  # _dependencies[1] = "_false"

        if condition is not None:
            if op not in self._VALID_OPS:
                raise ValueError(f"未知运算符: {op!r}。可选: {sorted(self._VALID_OPS)}")
            self.add_dependency(condition)  # _dependencies[2] = "_cond"
            self._cond_dep_index: dict[int, int] = {}
        else:
            # 多条件模式
            self._cond_dep_index = {}
            for i, spec in enumerate(conditions):
                if spec.op not in self._VALID_OPS:
                    raise ValueError(f"conditions[{i}] 未知运算符: {spec.op!r}")
                self._condition_specs.append(spec)
                self.add_dependency(spec.condition)
                self._cond_dep_index[i] = i + 2  # offset by _true(0) and _false(1)

        self.warmup_period = max(
            signal_true.get_max_warmup_period(),
            signal_false.get_max_warmup_period(),
            max((dep.get_max_warmup_period() for dep in self._dependencies[2:]), default=0),
        )

        self._set_params(op=op, threshold=threshold, false_negate=false_negate,
                         logic=logic, conditions_count=len(self._condition_specs))

    def get_dependency_column_map(self) -> list[str]:
        cols = ["_true", "_false"]
        if self._condition_specs:
            for i in range(len(self._condition_specs)):
                cols.append(f"_cond_{i}")
        else:
            cols.append("_cond")
        return cols

    def get_output_name(self) -> str:
        t_name = self._dependencies[0].get_output_name()
        f_name = self._dependencies[1].get_output_name()
        neg = "_neg" if self.false_negate else ""
        if self._condition_specs:
            parts = []
            for i, spec in enumerate(self._condition_specs):
                c_name = self._dependencies[i + 2].get_output_name()
                parts.append(f"{c_name}_{spec.op}_{spec.threshold}")
            c_name = f"_{self.logic}_".join(parts)
        else:
            c_name = f"{self._dependencies[2].get_output_name()}_{self.op}_{self.threshold}"
        return f"{t_name}__if_{c_name}__else_{f_name}{neg}"

    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        if "_true" not in frame.columns or "_false" not in frame.columns:
            raise ValueError(
                f"SwitchFactor 需要列 '_true', '_false'，"
                f"可用列: {list(frame.columns)}"
            )

        # 构造条件 mask
        if self._condition_specs:
            masks = []
            for i, spec in enumerate(self._condition_specs):
                col = f"_cond_{i}"
                if col not in frame.columns:
                    raise ValueError(f"SwitchFactor 需要列 '{col}'，可用列: {list(frame.columns)}")
                masks.append(self._OP_FUNCS[spec.op](frame[col], spec.threshold))
            combined = masks[0]
            if self.logic == "and":
                for m in masks[1:]:
                    combined = combined & m
            else:
                for m in masks[1:]:
                    combined = combined | m
            mask = combined
        else:
            if "_cond" not in frame.columns:
                raise ValueError(f"SwitchFactor 需要列 '_cond'，可用列: {list(frame.columns)}")
            mask = self._OP_FUNCS[self.op](frame["_cond"], self.threshold)

        true_vals = frame["_true"]
        false_vals = -frame["_false"] if self.false_negate else frame["_false"]

        result = pd.Series(
            np.where(mask.fillna(False), true_vals, false_vals),
            index=frame.index,
        )
        result.name = self.get_output_name()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: CompositeRankFactor — 横截面 rank 加权合成
# ═══════════════════════════════════════════════════════════════════════════════

class CompositeRankFactor:
    """横截面 rank 加权合成因子配方。

    不是一个 BaseFactor 子类 —— 它标记回测引擎在 symbol_data_map
    构建完成后，对多个子因子做横截面 rank 百分位归一化后加权求和。

    典型用法:
        ranking = CompositeRankFactor(
            factors=[
                (PriceReturn(window=60), 1.0),
                (PriceReturn(window=20), 0.5),
            ]
        )

    Attributes
    ----------
    factors : list[tuple[BaseFactor, float]]
        子因子及其权重的列表。每个子因子是一个 BaseFactor 实例。
    rank_method : str
        排名方法: "pct" (百分位归一化, 默认) 或 "minmax" (最小-最大归一化)。

    Examples
    --------
    >>> from factors.price_return import PriceReturn
    >>> ranking = CompositeRankFactor(
    ...     factors=[
    ...         (PriceReturn(window=60), 1.0),
    ...         (PriceReturn(window=20), 0.5),
    ...     ]
    ... )
    >>> ranking.get_output_name()
    'cs_rank_PriceReturn_60_w1.0__PriceReturn_20_w0.5'
    """

    def __init__(
        self,
        factors: list[tuple[BaseFactor, float]],
        rank_method: str = "pct",
    ) -> None:
        if not factors:
            raise ValueError("factors 不能为空")
        if rank_method not in ("pct", "minmax"):
            raise ValueError(f"不支持的 rank_method: {rank_method!r}，可选: 'pct', 'minmax'")

        self.factors = list(factors)
        self.rank_method = rank_method

    def get_output_name(self) -> str:
        """生成唯一的输出列名。"""
        parts = [f"{f.get_output_name()}_w{w}" for f, w in self.factors]
        return "cs_rank_" + "__".join(parts)

    def get_sub_factors(self) -> list[BaseFactor]:
        """返回所有子因子列表（供注入 factor_pipeline 使用）。"""
        return [f for f, _ in self.factors]

    def __repr__(self) -> str:
        return (
            f"CompositeRankFactor(factors={self.factors!r}, "
            f"rank_method={self.rank_method!r})"
        )
