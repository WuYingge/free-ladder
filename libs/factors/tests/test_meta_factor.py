"""TransformFactor 单元测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factors.price_return import PriceReturn
from factors.meta_factor import (
    TransformFactor,
    CombineFactor,
    ConditionalFactor,
    ConditionSpec,
    MultiConditionalFactor,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 测试夹具
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def ohlcv_data() -> pd.DataFrame:
    """构造 200 行的 OHLCV 模拟数据。

    价格从 10.0 开始，以对数正态随机游走方式生成，
    保证有足够的行数让任意 warmup 都能覆盖。
    """
    n = 200
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.normal(0.0005, 0.015, size=n)
    close = 10.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(rng.normal(0, 0.01, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, size=n)))
    open_ = close * (1 + rng.normal(0, 0.005, size=n))
    volume = rng.integers(1_000_000, 10_000_000, size=n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return df


@pytest.fixture
def pr20(ohlcv_data: pd.DataFrame) -> pd.Series:
    """基础因子：PriceReturn(window=20)。"""
    factor = PriceReturn(window=20)
    return factor(ohlcv_data)


# ═══════════════════════════════════════════════════════════════════════════════
# 输出名称
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputName:
    """get_output_name 格式校验。"""

    def _make(self, transform: str, window: int) -> TransformFactor:
        return TransformFactor(
            dependency=PriceReturn(window=20),
            transform=transform,
            window=window,
        )

    def test_rolling_mean(self):
        tf = self._make("rolling_mean", 10)
        assert tf.get_output_name() == "PriceReturn_20__rolling_mean_10"

    def test_rolling_std(self):
        tf = self._make("rolling_std", 20)
        assert tf.get_output_name() == "PriceReturn_20__rolling_std_20"

    def test_delta(self):
        tf = self._make("delta", 5)
        assert tf.get_output_name() == "PriceReturn_20__delta_5"

    def test_pct_change(self):
        tf = self._make("pct_change", 5)
        assert tf.get_output_name() == "PriceReturn_20__pct_change_5"

    def test_binarize_winrate(self):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="binarize_winrate",
            window=20,
            threshold=0.0,
        )
        assert tf.get_output_name() == "PriceReturn_20__binarize_winrate_20_0.0"

    def test_zscore(self):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="zscore",
            window=252,
        )
        assert tf.get_output_name() == "PriceReturn_20__zscore_252"

    def test_default_window(self):
        """不传 window 时使用该变换的默认窗口。"""
        tf_m = self._make("rolling_mean", None)  # type: ignore[arg-type]
        assert tf_m.window == 10
        assert tf_m.get_output_name() == "PriceReturn_20__rolling_mean_10"


# ═══════════════════════════════════════════════════════════════════════════════
# 变换正确性
# ═══════════════════════════════════════════════════════════════════════════════


class TestTransformCorrectness:
    """验证每种变换的数值正确性。"""

    def test_rolling_mean(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="rolling_mean",
            window=10,
        )
        result = tf(ohlcv_data)
        # 手工验算
        pr = PriceReturn(window=20)(ohlcv_data)
        expected = pr.rolling(window=10, min_periods=10).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_rolling_std(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="rolling_std",
            window=10,
        )
        result = tf(ohlcv_data)
        pr = PriceReturn(window=20)(ohlcv_data)
        expected = pr.rolling(window=10, min_periods=10).std()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_delta(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="delta",
            window=5,
        )
        result = tf(ohlcv_data)
        pr = PriceReturn(window=20)(ohlcv_data)
        expected = pr - pr.shift(5)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_pct_change(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="pct_change",
            window=5,
        )
        result = tf(ohlcv_data)
        pr = PriceReturn(window=20)(ohlcv_data)
        expected = pr.pct_change(periods=5)
        # pct_change 可能产生 inf，替换为 NaN
        expected = expected.replace([np.inf, -np.inf], np.nan)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_binarize_winrate(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="binarize_winrate",
            window=10,
            threshold=0.0,
        )
        result = tf(ohlcv_data)
        pr = PriceReturn(window=20)(ohlcv_data)
        binary = (pr > 0.0).astype(float)
        expected = binary.rolling(window=10, min_periods=10).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_zscore(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="zscore",
            window=60,
        )
        result = tf(ohlcv_data)
        pr = PriceReturn(window=20)(ohlcv_data)
        rm = pr.rolling(window=60, min_periods=60).mean()
        rs = pr.rolling(window=60, min_periods=60).std(ddof=1)
        rs_safe = rs.replace(0.0, np.nan)
        expected = (pr - rm) / rs_safe
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_result_name_matches_output_name(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="rolling_mean",
            window=10,
        )
        result = tf(ohlcv_data)
        assert result.name == tf.get_output_name()


# ═══════════════════════════════════════════════════════════════════════════════
# warmup 行为
# ═══════════════════════════════════════════════════════════════════════════════


class TestWarmup:
    """warmup 期内应产生 NaN，warmup 期后应有有效值。"""

    def test_warmup_period_gt_dependency(self):
        """TransformFactor 的 warmup 应大于依赖因子的 warmup。"""
        dep = PriceReturn(window=20)
        tf = TransformFactor(dependency=dep, transform="rolling_mean", window=10)
        assert tf.get_max_warmup_period() > dep.get_max_warmup_period()

    def test_warmup_period_equals_dep_plus_window_minus_1(self, ohlcv_data: pd.DataFrame):
        dep = PriceReturn(window=20)
        tf = TransformFactor(dependency=dep, transform="rolling_mean", window=10)
        # PriceReturn(window=20) warmup = 21 (1-indexed first valid)
        # rolling_mean(10) 需要 10 个有效值 → 21 + 10 - 1 = 30
        expected = dep.get_max_warmup_period() + 10 - 1
        assert tf.get_max_warmup_period() == expected

    def test_first_n_rows_nan(self, ohlcv_data: pd.DataFrame):
        tf = TransformFactor(
            dependency=PriceReturn(window=20),
            transform="rolling_mean",
            window=10,
        )
        result = tf(ohlcv_data)
        warmup = tf.get_max_warmup_period()
        # warmup 是 1-indexed（如 30 = 第 30 根 bar 是第一个有效值），
        # iloc[:warmup-1] = 前 warmup-1 根 bar（0-indexed 索引 0..warmup-2）都是 NaN
        assert warmup >= 2
        assert result.iloc[:warmup - 1].isna().all()
        # 第 warmup 根 bar（0-indexed 索引 warmup-1）应有有效值
        assert not pd.isna(result.iloc[warmup - 1])


# ═══════════════════════════════════════════════════════════════════════════════
# 错误处理
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrors:
    """非法输入应抛出明确的 ValueError。"""

    def test_invalid_transform(self):
        with pytest.raises(ValueError, match="未知变换类型"):
            TransformFactor(
                dependency=PriceReturn(window=20),
                transform="not_a_transform",  # type: ignore[arg-type]
                window=10,
            )

    def test_window_lt_1(self):
        with pytest.raises(ValueError, match="window 必须 >= 1"):
            TransformFactor(
                dependency=PriceReturn(window=20),
                transform="rolling_mean",
                window=0,
            )

    def test_nested_transform(self, ohlcv_data: pd.DataFrame):
        """验证对衍生因子再次施加变换是可行的（依赖链）。"""
        dep = PriceReturn(window=20)
        t1 = TransformFactor(dependency=dep, transform="rolling_mean", window=10)
        t2 = TransformFactor(dependency=t1, transform="delta", window=5)
        result = t2(ohlcv_data)
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_data)
        # warmup 应该更深
        assert t2.get_max_warmup_period() > t1.get_max_warmup_period()


# ═══════════════════════════════════════════════════════════════════════════════
# 依赖链正确性
# ═══════════════════════════════════════════════════════════════════════════════


class TestDependencyChain:
    """验证依赖因子被正确调用且结果被正确传入。"""

    def test_dependency_called_once_per_symbol(self, ohlcv_data: pd.DataFrame):
        """对同一 data 多次调用 TransformFactor，依赖因子不会重复调用？"""
        # 这里的行为由 BaseFactor 管理 — 本测试确保基本调用没问题
        dep = PriceReturn(window=20)
        tf = TransformFactor(dependency=dep, transform="rolling_mean", window=10)
        r1 = tf(ohlcv_data)
        r2 = tf(ohlcv_data)
        pd.testing.assert_series_equal(r1, r2)

    def test_different_dependency_different_output(self, ohlcv_data: pd.DataFrame):
        """不同依赖因子的 TransformFactor 输出不应相同。"""
        dep_a = PriceReturn(window=20)
        dep_b = PriceReturn(window=60)
        t_a = TransformFactor(dependency=dep_a, transform="rolling_mean", window=10)
        t_b = TransformFactor(dependency=dep_b, transform="rolling_mean", window=10)
        r_a = t_a(ohlcv_data)
        r_b = t_b(ohlcv_data)
        # 去掉 NaN 后应不同（不同窗口的 PriceReturn 输出不同）
        valid = r_a.notna() & r_b.notna()
        assert not r_a[valid].equals(r_b[valid])


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: CombineFactor 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestCombineOutputName:
    """CombineFactor 输出名称格式校验。"""

    def test_product(self):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="product",
        )
        assert c.get_output_name() == "PriceReturn_20__product_PriceReturn_60"

    def test_ratio(self):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="ratio",
        )
        assert c.get_output_name() == "PriceReturn_20__ratio_PriceReturn_60"

    def test_weighted_sum(self):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="weighted_sum",
            weight_a=0.3,
            weight_b=0.7,
        )
        assert c.get_output_name() == "PriceReturn_20__weighted_sum_PriceReturn_60"

    def test_diff(self):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="diff",
        )
        assert c.get_output_name() == "PriceReturn_20__diff_PriceReturn_60"


class TestCombineCorrectness:
    """验证每种运算的数值正确性。"""

    def test_product(self, ohlcv_data: pd.DataFrame):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="product",
        )
        result = c(ohlcv_data)
        a = PriceReturn(window=20)(ohlcv_data)
        b = PriceReturn(window=60)(ohlcv_data)
        expected = a * b
        expected.name = c.get_output_name()
        pd.testing.assert_series_equal(result, expected)

    def test_ratio(self, ohlcv_data: pd.DataFrame):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="ratio",
        )
        result = c(ohlcv_data)
        a = PriceReturn(window=20)(ohlcv_data)
        b = PriceReturn(window=60)(ohlcv_data)
        expected = (a / b).replace([np.inf, -np.inf], np.nan)
        expected.name = c.get_output_name()
        pd.testing.assert_series_equal(result, expected)

    def test_diff(self, ohlcv_data: pd.DataFrame):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="diff",
        )
        result = c(ohlcv_data)
        a = PriceReturn(window=20)(ohlcv_data)
        b = PriceReturn(window=60)(ohlcv_data)
        expected = a - b
        expected.name = c.get_output_name()
        pd.testing.assert_series_equal(result, expected)

    def test_weighted_sum_normalized(self, ohlcv_data: pd.DataFrame):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="weighted_sum",
            weight_a=0.5,
            weight_b=0.5,
            normalize=True,
            normalize_window=60,
        )
        result = c(ohlcv_data)
        a = PriceReturn(window=20)(ohlcv_data)
        b = PriceReturn(window=60)(ohlcv_data)

        # 手工做 zscore
        def _zscore(s: pd.Series, w: int) -> pd.Series:
            rm = s.rolling(window=w, min_periods=w).mean()
            rs = s.rolling(window=w, min_periods=w).std(ddof=1)
            return (s - rm) / rs.replace(0.0, np.nan)

        za = _zscore(a, 60)
        zb = _zscore(b, 60)
        expected = 0.5 * za + 0.5 * zb
        expected.name = c.get_output_name()
        pd.testing.assert_series_equal(result, expected)

    def test_result_name_matches(self, ohlcv_data: pd.DataFrame):
        c = CombineFactor(
            factor_a=PriceReturn(window=20),
            factor_b=PriceReturn(window=60),
            method="product",
        )
        result = c(ohlcv_data)
        assert result.name == c.get_output_name()


class TestCombineErrors:
    """非法输入应抛出明确错误。"""

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="未知运算方法"):
            CombineFactor(
                factor_a=PriceReturn(window=20),
                factor_b=PriceReturn(window=60),
                method="not_a_method",  # type: ignore[arg-type]
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: ConditionalFactor 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestConditionalOutputName:
    """ConditionalFactor 输出名称格式校验。"""

    def test_gt(self):
        cf = ConditionalFactor(
            signal=PriceReturn(window=20),
            condition=PriceReturn(window=60),
            op="gt",
            threshold=0.5,
        )
        assert cf.get_output_name() == "PriceReturn_20__if_PriceReturn_60_gt_0.5"

    def test_lte_zero(self):
        cf = ConditionalFactor(
            signal=PriceReturn(window=20),
            condition=PriceReturn(window=60),
            op="lte",
            threshold=0.0,
        )
        assert cf.get_output_name() == "PriceReturn_20__if_PriceReturn_60_lte_0.0"


class TestConditionalCorrectness:
    """验证条件运算的正确性。"""

    def test_gt_nan(self, ohlcv_data: pd.DataFrame):
        cf = ConditionalFactor(
            signal=PriceReturn(window=20),
            condition=PriceReturn(window=20),
            op="gt",
            threshold=0.0,
            false_value="nan",
        )
        result = cf(ohlcv_data)
        signal = PriceReturn(window=20)(ohlcv_data)
        mask = signal > 0.0
        expected = signal.where(mask, other=np.nan)
        expected.name = cf.get_output_name()
        pd.testing.assert_series_equal(result, expected)

    def test_gt_zero(self, ohlcv_data: pd.DataFrame):
        cf = ConditionalFactor(
            signal=PriceReturn(window=20),
            condition=PriceReturn(window=20),
            op="gt",
            threshold=0.0,
            false_value="zero",
        )
        result = cf(ohlcv_data)
        signal = PriceReturn(window=20)(ohlcv_data)
        mask = signal > 0.0
        expected = signal.where(mask, other=0.0)
        expected.name = cf.get_output_name()
        pd.testing.assert_series_equal(result, expected)

    def test_different_signal_and_condition(self, ohlcv_data: pd.DataFrame):
        """signal 和 condition 是不同的因子。"""
        cf = ConditionalFactor(
            signal=PriceReturn(window=20),
            condition=PriceReturn(window=60),
            op="gt",
            threshold=0.0,
        )
        result = cf(ohlcv_data)
        signal = PriceReturn(window=20)(ohlcv_data)
        cond = PriceReturn(window=60)(ohlcv_data)
        mask = cond > 0.0
        expected = signal.where(mask, other=np.nan)
        expected.name = cf.get_output_name()
        pd.testing.assert_series_equal(result, expected)

    def test_result_name_matches(self, ohlcv_data: pd.DataFrame):
        cf = ConditionalFactor(
            signal=PriceReturn(window=20),
            condition=PriceReturn(window=60),
            op="gt",
            threshold=0.0,
        )
        result = cf(ohlcv_data)
        assert result.name == cf.get_output_name()


class TestConditionalErrors:
    """非法输入应抛出明确错误。"""

    def test_invalid_op(self):
        with pytest.raises(ValueError, match="未知运算符"):
            ConditionalFactor(
                signal=PriceReturn(window=20),
                condition=PriceReturn(window=60),
                op="eq",  # type: ignore[arg-type]
            )

    def test_invalid_false_value(self):
        with pytest.raises(ValueError, match="未知 false_value"):
            ConditionalFactor(
                signal=PriceReturn(window=20),
                condition=PriceReturn(window=60),
                false_value="one",  # type: ignore[arg-type]
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: MultiConditionalFactor 测试
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiConditionalOutputName:
    """MultiConditionalFactor 输出名称格式校验。"""

    def test_and_two_conditions(self):
        conditions = [
            ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.5),
            ConditionSpec(condition=PriceReturn(window=120), op="lt", threshold=0.0),
        ]
        mcf = MultiConditionalFactor(
            signal=PriceReturn(window=20),
            conditions=conditions,
            logic="and",
        )
        assert (
            mcf.get_output_name()
            == "PriceReturn_20__if_PriceReturn_60_gt_0.5_and_PriceReturn_120_lt_0.0"
        )

    def test_or_three_conditions(self):
        conditions = [
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=0.0),
            ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.0),
            ConditionSpec(condition=PriceReturn(window=120), op="gte", threshold=-0.01),
        ]
        mcf = MultiConditionalFactor(
            signal=PriceReturn(window=20),
            conditions=conditions,
            logic="or",
        )
        name = mcf.get_output_name()
        assert "__if_" in name
        assert "_or_" in name
        assert "PriceReturn_20_gt_0.0" in name
        assert "PriceReturn_60_gt_0.0" in name
        assert "PriceReturn_120_gte_-0.01" in name


class TestMultiConditionalCorrectness:
    """验证多条件运算的正确性。"""

    def test_and_both_true(self, ohlcv_data: pd.DataFrame):
        """AND 逻辑：两个条件都满足时信号生效。"""
        signal = PriceReturn(window=20)
        # 用同一个因子的不同窗口作为两个条件，确保有足够变化
        cond1 = PriceReturn(window=20)
        cond2 = PriceReturn(window=60)
        conditions = [
            ConditionSpec(condition=cond1, op="gt", threshold=-999.0),  # 几乎总是 True
            ConditionSpec(condition=cond2, op="gt", threshold=-999.0),  # 几乎总是 True
        ]
        mcf = MultiConditionalFactor(
            signal=signal, conditions=conditions, logic="and"
        )
        result = mcf(ohlcv_data)
        expected_signal = signal(ohlcv_data)
        # 只比较所有条件都生效后的区域（MultiConditionalFactor 的 warmup 取
        # 所有依赖的最大值，这里 cond2(w=60) 有更长的 warmup）
        valid = result.notna()
        assert valid.sum() > 0, "至少应有部分数据通过所有条件"
        pd.testing.assert_series_equal(
            result[valid], expected_signal[valid], check_names=False
        )

    def test_and_one_false(self, ohlcv_data: pd.DataFrame):
        """AND 逻辑：任一条不满足 → 输出 NaN。"""
        signal = PriceReturn(window=20)
        conditions = [
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=999.0),  # 总是 False
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=-999.0),  # 总是 True
        ]
        mcf = MultiConditionalFactor(
            signal=signal, conditions=conditions, logic="and", false_value="nan"
        )
        result = mcf(ohlcv_data)
        # 条件1 不满足，全部应为 NaN（除 NaN 本身）
        valid = result.notna()
        assert valid.sum() == 0

    def test_or_one_true(self, ohlcv_data: pd.DataFrame):
        """OR 逻辑：任一条件满足 → 信号生效。"""
        signal = PriceReturn(window=20)
        conditions = [
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=999.0),   # 总是 False
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=-999.0),  # 总是 True
        ]
        mcf = MultiConditionalFactor(
            signal=signal, conditions=conditions, logic="or"
        )
        result = mcf(ohlcv_data)
        expected_signal = signal(ohlcv_data)
        valid = expected_signal.notna()
        pd.testing.assert_series_equal(result[valid], expected_signal[valid], check_names=False)

    def test_or_all_false(self, ohlcv_data: pd.DataFrame):
        """OR 逻辑：所有条件都不满足 → 全部 NaN。"""
        signal = PriceReturn(window=20)
        conditions = [
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=999.0),
            ConditionSpec(condition=PriceReturn(window=60), op="lt", threshold=-999.0),
        ]
        mcf = MultiConditionalFactor(
            signal=signal, conditions=conditions, logic="or", false_value="nan"
        )
        result = mcf(ohlcv_data)
        valid = result.notna()
        assert valid.sum() == 0

    def test_false_value_zero(self, ohlcv_data: pd.DataFrame):
        """false_value='zero' 时不满足条件的位置填充 0。"""
        signal = PriceReturn(window=20)
        conditions = [
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=999.0),
            ConditionSpec(condition=PriceReturn(window=20), op="gt", threshold=999.0),
        ]
        mcf = MultiConditionalFactor(
            signal=signal, conditions=conditions, logic="and", false_value="zero"
        )
        result = mcf(ohlcv_data)
        # 条件都不满足 → 输出 0
        expected_signal = signal(ohlcv_data)
        valid = expected_signal.notna()
        assert (result[valid] == 0.0).all()

    def test_result_name_matches(self, ohlcv_data: pd.DataFrame):
        """输出 Series 的 name 应与 get_output_name() 一致。"""
        conditions = [
            ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.0),
            ConditionSpec(condition=PriceReturn(window=120), op="lt", threshold=0.0),
        ]
        mcf = MultiConditionalFactor(
            signal=PriceReturn(window=20), conditions=conditions
        )
        result = mcf(ohlcv_data)
        assert result.name == mcf.get_output_name()

    def test_different_condition_factors(self, ohlcv_data: pd.DataFrame):
        """signal 和多个 condition 使用不同参数的因子。"""
        signal = PriceReturn(window=10)
        conditions = [
            ConditionSpec(condition=PriceReturn(window=30), op="gt", threshold=0.0),
            ConditionSpec(condition=PriceReturn(window=90), op="lt", threshold=0.01),
            ConditionSpec(condition=PriceReturn(window=180), op="gte", threshold=-0.02),
        ]
        mcf = MultiConditionalFactor(
            signal=signal, conditions=conditions, logic="and", false_value="nan"
        )
        result = mcf(ohlcv_data)
        # 验证返回的是 Series，且 name 正确
        assert isinstance(result, pd.Series)
        assert result.name == mcf.get_output_name()
        # 验证长度与输入一致
        assert len(result) == len(ohlcv_data)


class TestMultiConditionalErrors:
    """非法输入应抛出明确错误。"""

    def test_too_few_conditions(self):
        """少于 2 个条件应报错。"""
        with pytest.raises(ValueError, match="至少需要 2 个条件"):
            MultiConditionalFactor(
                signal=PriceReturn(window=20),
                conditions=[],
            )

        with pytest.raises(ValueError, match="至少需要 2 个条件"):
            MultiConditionalFactor(
                signal=PriceReturn(window=20),
                conditions=[
                    ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.0),
                ],
            )

    def test_invalid_logic(self):
        """非法 logic 值应报错。"""
        with pytest.raises(ValueError, match="未知逻辑"):
            MultiConditionalFactor(
                signal=PriceReturn(window=20),
                conditions=[
                    ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.0),
                    ConditionSpec(condition=PriceReturn(window=120), op="gt", threshold=0.0),
                ],
                logic="xor",  # type: ignore[arg-type]
            )

    def test_invalid_false_value(self):
        """非法 false_value 应报错。"""
        with pytest.raises(ValueError, match="未知 false_value"):
            MultiConditionalFactor(
                signal=PriceReturn(window=20),
                conditions=[
                    ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.0),
                    ConditionSpec(condition=PriceReturn(window=120), op="gt", threshold=0.0),
                ],
                false_value="one",  # type: ignore[arg-type]
            )

    def test_invalid_op_in_condition_spec(self):
        """ConditionSpec 中的非法 op 应报错。"""
        with pytest.raises(ValueError, match="未知运算符"):
            ConditionSpec(condition=PriceReturn(window=60), op="eq", threshold=0.0)  # type: ignore[arg-type]

    def test_invalid_op_in_conditions_list(self):
        """conditions 列表中某元素的非法 op 应报错。"""
        with pytest.raises(ValueError, match="未知运算符"):
            MultiConditionalFactor(
                signal=PriceReturn(window=20),
                conditions=[
                    ConditionSpec(condition=PriceReturn(window=60), op="gt", threshold=0.0),
                    ConditionSpec(condition=PriceReturn(window=120), op="eq", threshold=0.0),  # type: ignore[arg-type]
                ],
            )
