"""
元因子框架 (Meta-Factor Framework)

通过"变换 + 组合"将基础因子扩展为衍生因子，为因子轮动提供充足且多样化的因子池。

三层架构:
    TransformFactor    — 单因子后处理变换（6 种）
    CombineFactor      — 双因子二元运算（4 种）
    ConditionalFactor  — 条件信号（信号因子仅在条件满足时生效）

所有元因子继承自 DerivedFactor，复用了它的依赖注入、DataFrame 组装和
warmup 链式计算。每个子类只需实现 compute_from_frame(frame)。
"""

from __future__ import annotations

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
    meta_type: "transform" | "combine" | "conditional"
    meta_params: 传给 meta factor 构造函数的参数。
    """
    base_factor_name: str
    base_factor_module: str
    base_factor_class: str
    base_params: dict[str, Any]
    meta_type: str
    meta_params: dict[str, Any]


def build_meta_factor(spec: MetaFactorSpec) -> BaseFactor:
    """从 MetaFactorSpec 配方实例化衍生因子。

    在 worker 进程中调用，将序列化的配方还原为可调用的因子对象。
    """
    import importlib

    # 1. 动态导入基础因子类
    mod = importlib.import_module(spec.base_factor_module)
    base_cls = getattr(mod, spec.base_factor_class)

    # 2. 实例化第一个基础因子
    base = base_cls(**spec.base_params)

    # 3. 根据 meta_type 构建衍生因子
    if spec.meta_type == "transform":
        return TransformFactor(dependency=base, **spec.meta_params)

    elif spec.meta_type == "combine":
        b_module = spec.meta_params.pop("_b_module")
        b_class = spec.meta_params.pop("_b_class")
        b_params = spec.meta_params.pop("_b_params")
        b_mod = importlib.import_module(b_module)
        b_cls = getattr(b_mod, b_class)
        factor_b = b_cls(**b_params)
        return CombineFactor(factor_a=base, factor_b=factor_b, **spec.meta_params)

    elif spec.meta_type == "conditional":
        c_module = spec.meta_params.pop("_cond_module")
        c_class = spec.meta_params.pop("_cond_class")
        c_params = spec.meta_params.pop("_cond_params")
        c_mod = importlib.import_module(c_module)
        c_cls = getattr(c_mod, c_class)
        condition = c_cls(**c_params)
        return ConditionalFactor(signal=base, condition=condition, **spec.meta_params)

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
        return f"{dep_name}__{self.transform}_{self.window}"

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

    def get_dependency_column_map(self) -> dict[BaseFactor, str]:
        """为两个依赖分配固定别名 '_a' 和 '_b'。

        DerivedFactor 默认用依赖因子的 get_output_name() 作为列名，
        当两个依赖同名时（如都用了 PriceReturn_20）会冲突。
        固定别名彻底避免此问题。
        """
        return {
            self._dependencies[0]: "_a",
            self._dependencies[1]: "_b",
        }

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

    def get_dependency_column_map(self) -> dict[BaseFactor, str]:
        """为 signal 和 condition 分配固定别名。

        当 self_conditional 时只有一个依赖，列名为 '_val'；
        否则分别为 '_signal' 和 '_cond'。
        """
        if self.self_conditional:
            return {self._dependencies[0]: "_val"}
        return {
            self._dependencies[0]: "_signal",
            self._dependencies[1]: "_cond",
        }

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
