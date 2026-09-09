"""
Alpha101 模块测试 (Alpha101 Module Tests)

覆盖：
  * 算子库的正确性（横截面 rank/scale、时序 ts_rank/correlation/decay_linear/delta/adv 等）
  * 5 个代表因子在 (小样本) 真实数据上的端到端面板构建 + forward return + Rank IC 冒烟
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from factors.alpha101 import (
    ALPHA101_REGISTRY,
    Alpha101Factor,
    build_alpha101_panel,
    get_computable_alpha_ids,
)
from factors.alpha101 import operators as ops


# ── 合成面板工具 ──────────────────────────────────────────────────────────────


def _frame(data: dict[str, list[float]], index: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data, index=pd.DatetimeIndex(index), dtype=float)


# ── 算子单测 ──────────────────────────────────────────────────────────────────


def test_rank_is_cross_sectional_percentile():
    df = _frame({"a": [3.0, 1.0], "b": [1.0, 5.0], "c": [2.0, 3.0]}, ["2024-01-01", "2024-01-02"])
    r = ops.rank(df)
    # 第一行截面值 [3,1,2] -> 秩 3,1,2 归一化
    assert r.loc["2024-01-01", "a"] == pytest.approx(3 / 3)
    assert r.loc["2024-01-01", "b"] == pytest.approx(1 / 3)
    assert r.loc["2024-01-01", "c"] == pytest.approx(2 / 3)
    # 值域 (0,1]
    assert r.min().min() > 0 and r.max().max() <= 1.0


def test_scale_normalises_row_abs_sum():
    df = _frame({"a": [1.0, -2.0], "b": [3.0, 4.0]}, ["2024-01-01", "2024-01-02"])
    s = ops.scale(df, a=10.0)
    for row in s.index:
        assert s.loc[row].abs().sum() == pytest.approx(10.0)


def test_ts_rank_within_window():
    df = _frame({"a": [3.0, 1.0, 4.0, 2.0, 5.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    r = ops.ts_rank(df, 3)
    # index2 窗口 [3,1,4]，当前 4 -> count<=4 = 3 -> 1.0
    assert r.iloc[2, 0] == pytest.approx(1.0)
    # index3 窗口 [1,4,2]，当前 2 -> count<=2 = 2 -> 2/3
    assert r.iloc[3, 0] == pytest.approx(2 / 3)


def test_ts_mean_matches_rolling_mean():
    df = _frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    got = ops.ts_mean(df, 3)
    exp = df.rolling(3, min_periods=3).mean()
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_delta_and_delay():
    df = _frame({"a": [1.0, 2.0, 4.0, 7.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    d2 = ops.delta(df, 2)
    assert d2.iloc[2, 0] == pytest.approx(4.0 - 1.0)
    assert d2.iloc[3, 0] == pytest.approx(7.0 - 2.0)
    lag = ops.delay(df, 1)
    assert lag.iloc[1, 0] == pytest.approx(1.0)


def test_decay_linear_weights():
    df = _frame({"a": [1.0, 2.0, 3.0, 4.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    got = ops.decay_linear(df, 3)
    # index2 窗口 [1,2,3] 权重 [1,2,3]/6 -> (1+4+9)/6
    assert got.iloc[2, 0] == pytest.approx((1 + 4 + 9) / 6)
    assert got.iloc[3, 0] == pytest.approx((2 * 1 + 3 * 2 + 4 * 3) / 6)


def test_correlation_matches_pandas():
    x = _frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    y = _frame({"a": [2.0, 1.0, 4.0, 3.0, 6.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    got = ops.correlation(x, y, 3)
    exp = x.rolling(3, min_periods=3).corr(y)
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_signedpower():
    df = _frame({"a": [-2.0, 3.0]}, ["2024-01-01", "2024-01-02"])
    g = ops.signedpower(df, 2.0)
    assert g.iloc[0, 0] == pytest.approx(-4.0)
    assert g.iloc[1, 0] == pytest.approx(9.0)


def test_adv_is_rolling_mean_of_value():
    df = _frame({"a": [10.0, 20.0, 30.0, 40.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    got = ops.adv(df, 3)
    assert got.iloc[2, 0] == pytest.approx((10 + 20 + 30) / 3)


# ── 因子/面板端到端冒烟（真实小样本数据） ─────────────────────────────────────


@pytest.fixture(scope="module")
def sample_symbols() -> list[str]:
    return ["510300", "510500", "159915", "518880", "510050"]


@pytest.mark.skipif(not all(
    __import__("pathlib").Path(f"data/etf_data/{s}.csv").exists() for s in ["510300"]
), reason="ETF 数据文件不存在")
def test_alpha_101_panel_smoke(sample_symbols):
    from factors.alpha101.universe import EtfAlpha101Universe

    universe = EtfAlpha101Universe(symbols=sample_symbols)
    spec = ALPHA101_REGISTRY["101"]
    panel = build_alpha101_panel(
        spec.func, universe, factor_name="Alpha101_101",
        min_bars=100, max_workers=1,
    )
    assert panel.n_symbols >= 1
    assert panel.n_dates >= 100
    assert panel.factor_name == "Alpha101_101"
    # 部分列应有有效因子值（温暖期后）
    assert panel.factor_values.notna().values.any()


def test_alpha_panel_builds_for_all_computable(sample_symbols):
    from factor_analysis.forward_returns import compute_forward_returns
    from factors.alpha101.universe import EtfAlpha101Universe

    universe = EtfAlpha101Universe(symbols=sample_symbols)
    for alpha_id in get_computable_alpha_ids():
        spec = ALPHA101_REGISTRY[alpha_id]
        panel = build_alpha101_panel(
            spec.func, universe, factor_name=f"Alpha101_{alpha_id}",
            min_bars=100, max_workers=1,
        )
        assert panel.n_symbols >= 1, f"Alpha#{alpha_id} 面板无有效标的"
        fwd = compute_forward_returns(panel.close_prices, (5,))
        # 因子值与收益矩阵应能对齐（下游 IC 计算的先决条件）
        assert fwd[5].shape[1] == panel.n_symbols


def test_alpha_factor_adapter_naming():
    f = Alpha101Factor("001", universe_kind="etf")
    assert f.get_output_name() == "Alpha101_001"
    assert f.params == {"alpha": "001", "universe": "etf"}
