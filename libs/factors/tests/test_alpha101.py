"""
Alpha101 模块测试 (Alpha101 Module Tests)

覆盖：
  * 算子库正确性（横截面 rank/scale/indneutralize、时序 ts_rank/correlation/covariance/
    decay_linear/delta/adv、逐元素 min/max、序列指数幂、窗口缺失容忍语义）
  * 注册表完整性（101 个 id）与数据依赖标记（cap/vwap/adv/行业档位）与源码一致性
  * 101 个公式在合成面板上的冒烟（形状/有限值/覆盖率）+ 关键公式的定点数值校验
  * 真实小样本数据上的端到端面板构建（含 vwap 复权口径）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from factors.alpha101 import (
    ALPHA101_REGISTRY,
    FORMULAS,
    Alpha101Factor,
    build_alpha101_panel,
    data_requirements_summary,
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
    """无缺失时与 pandas 严格满窗结果一致（min_periods 容忍只在有空位时生效）。"""
    df = _frame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    got = ops.ts_mean(df, 3)
    exp = df.rolling(3, min_periods=3).mean()
    pd.testing.assert_frame_equal(got.iloc[2:], exp.iloc[2:], check_dtype=False)


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
    pd.testing.assert_frame_equal(got.iloc[2:], exp.iloc[2:], check_dtype=False)


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
    """101 个公式在同一合成面板上全部可算（不变量：形状/有限值/覆盖率）。"""
    from factors.alpha101.testing import make_synthetic_inputs

    inputs = make_synthetic_inputs(n_dates=400, n_symbols=25, seed=7)
    rows = []
    for alpha_id, spec in ALPHA101_REGISTRY.items():
        out = spec.func(inputs)
        assert isinstance(out, pd.DataFrame), f"Alpha#{alpha_id} 未返回 DataFrame"
        assert out.shape == inputs.close.shape, f"Alpha#{alpha_id} 形状不符"
        values = out.to_numpy(dtype="float64", na_value=np.nan)
        assert not np.isinf(values).any(), f"Alpha#{alpha_id} 出现 ±inf"
        coverage = float(np.isfinite(values).mean())
        assert coverage > 0.05, f"Alpha#{alpha_id} 覆盖率仅 {coverage:.1%}"
        rows.append((alpha_id, coverage))

    # 至少 8 成公式覆盖率 > 50%（长链条公式本身会有更多缺口）
    good = sum(1 for _, c in rows if c > 0.5)
    assert good >= 80, f"覆盖率 >50% 的公式只有 {good}/101"


def test_registry_is_complete_and_flags_match_sources():
    """注册表：101 个 id 齐全，且 cap/vwap/adv/行业档位标记与公式源码一致。"""
    import ast
    import inspect

    expected = {f"{i:03d}" for i in range(1, 102)}
    assert set(ALPHA101_REGISTRY) == expected
    assert set(FORMULAS) == expected

    def _max_industry_level(func) -> int:
        tree = ast.parse(inspect.getsource(func))
        levels = [
            int(node.args[2].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_ind"
            and len(node.args) >= 3
            and isinstance(node.args[2], ast.Constant)
        ]
        return max(levels, default=0)

    for alpha_id, spec in ALPHA101_REGISTRY.items():
        src = inspect.getsource(spec.func)
        assert spec.uses_cap == ("inp.cap" in src), f"#{alpha_id} cap 标记不符"
        assert spec.needs_vwap == ("inp.vwap" in src), f"#{alpha_id} vwap 标记不符"
        assert spec.needs_adv == ("_adv(inp" in src), f"#{alpha_id} adv 标记不符"
        assert spec.needs_industry == _max_industry_level(spec.func), (
            f"#{alpha_id} 行业档位标记不符"
        )
        assert spec.description, f"#{alpha_id} 缺公式描述"

    summary = data_requirements_summary()
    assert summary == {
        "total": 101, "needs_vwap": 43, "needs_adv": 45,
        "needs_industry": 18, "uses_cap": 1, "plain": 52,
    }


def test_alpha_factor_adapter_naming():
    f = Alpha101Factor("001", universe_kind="etf")
    assert f.get_output_name() == "Alpha101_001"
    assert f.params == {"alpha": "001", "universe": "etf"}


# ── vwap 口径（不复权成交额/成交量 × 复权因子 → 与后复权 OHLC 同尺度） ──────────


class _FakeUniverse:
    """最小 Universe 桩：直接返回内存里的后复权 OHLCV 帧。"""

    kind = "test"

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self._frames = frames

    def list_symbols(self) -> list[str]:
        return list(self._frames)

    def load(self, symbol: str) -> pd.DataFrame:
        return self._frames[symbol]


def _hfq_frame(dates: list[str], close: float) -> pd.DataFrame:
    """后复权行情：不复权均价 = close/1.2，成交额 = 均价 x 手数 x 100。"""
    factor = 1.2
    volume = 1e4
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.005, "low": close * 0.995, "close": close,
            "volume": volume, "value": close / factor * volume * 100.0,
        },
        index=pd.DatetimeIndex(dates),
    )


@pytest.fixture
def adj_factor_env(tmp_path, monkeypatch):
    """隔离 data/adj_factor 目录并注入 510300 的因子文件。"""
    import os

    from config import DataPath
    from data_manager.providers.adj_factor_provider import ADJ_FACTOR

    monkeypatch.setattr(DataPath, "ADJ_FACTOR_PATH", str(tmp_path / "adj_factor"))
    os.makedirs(DataPath.ADJ_FACTOR_PATH, exist_ok=True)
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="date")
    pd.DataFrame({"close_raw": [4.0, 4.1], "adj_factor": [1.2, 1.2]}, index=idx).to_csv(
        os.path.join(DataPath.ADJ_FACTOR_PATH, "510300.csv"),
        encoding="utf-8-sig", index=True,
    )
    ADJ_FACTOR.reload()
    yield tmp_path
    monkeypatch.undo()
    ADJ_FACTOR.reload()


def test_panel_vwap_is_adjusted_and_within_daily_range(adj_factor_env):
    """vwap 必须与后复权 OHLC 同尺度（乘因子），且落在当日 [low, high] 内。"""
    from factors.alpha101 import build_alpha101_inputs

    universe = _FakeUniverse({
        "510300": _hfq_frame(["2024-01-02", "2024-01-03"], close=4.8),
        "510500": _hfq_frame(["2024-01-02", "2024-01-03"], close=6.0),
    })
    inputs = build_alpha101_inputs(universe, max_workers=1)

    # 510300 有因子 → vwap == close (构造时即按此设计); 510500 无因子 → NaN 而非错尺度值
    assert inputs.vwap["510300"].tolist() == pytest.approx([4.8, 4.8])
    assert inputs.vwap["510500"].isna().all()

    vwap = inputs.vwap["510300"]
    assert (vwap <= inputs.high["510300"]).all()
    assert (vwap >= inputs.low["510300"]).all()
    # 旧口径 (成交额/成交量, 未 x100 未乘因子) 会比 close 大两个量级 —— 回归护栏
    legacy = inputs.value["510300"] / inputs.volume["510300"]
    assert (legacy > vwap * 10).all()


# ── 新增算子：covariance / 逐元素 min-max / 序列指数幂 / PIT 行业中性化 ────────


def test_covariance_matches_pandas():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    y = pd.DataFrame({"a": [2.0, 1.0, 4.0, 3.0]}, index=idx)
    got = ops.covariance(x, y, 3)
    exp = x.rolling(3, min_periods=2).cov(y)          # _min_obs(3) = 2
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_min_max_are_elementwise():
    idx = pd.DatetimeIndex(["2024-01-01"])
    x = pd.DataFrame({"a": [1.0], "b": [5.0]}, index=idx)
    y = pd.DataFrame({"a": [3.0], "b": [2.0]}, index=idx)
    assert ops.min_(x, y).iloc[0].tolist() == [1.0, 2.0]
    assert ops.max_(x, y).iloc[0].tolist() == [3.0, 5.0]
    # 标量广播（论文里的 min(x, 5) 走 ts_min，这里只验证逐元素语义）
    assert ops.min_(x, 4.0).iloc[0].tolist() == [1.0, 4.0]
    assert ops.max_(x, 4.0).iloc[0].tolist() == [4.0, 5.0]


def test_signedpower_accepts_series_exponent():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    x = pd.DataFrame({"a": [-2.0, 3.0]}, index=idx)
    exp = pd.DataFrame({"a": [2.0, 0.5]}, index=idx)
    got = ops.signedpower(x, exp)
    assert got.iloc[0, 0] == pytest.approx(-4.0)      # sign(-2)*|−2|^2
    assert got.iloc[1, 0] == pytest.approx(np.sqrt(3.0))  # sign(3)*|3|^0.5


def test_indneutralize_static_series_is_group_demean():
    idx = pd.DatetimeIndex(["2024-01-01"])
    x = pd.DataFrame({"a": [1.0], "b": [3.0], "c": [10.0]}, index=idx)
    group = pd.Series({"a": "bank", "b": "bank", "c": "tech"})
    out = ops.indneutralize(x, group)
    assert out.loc["2024-01-01", "a"] == pytest.approx(-1.0)
    assert out.loc["2024-01-01", "b"] == pytest.approx(1.0)
    assert out.loc["2024-01-01", "c"] == pytest.approx(0.0)


def test_indneutralize_point_in_time_dataframe_labels():
    """逐日标签：行业切换日按当日分组，且标签缺失的单元格保持原值。"""
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"])
    x = pd.DataFrame({"a": [1.0, 1.0], "b": [3.0, 3.0], "c": [5.0, 5.0]}, index=idx)
    labels = pd.DataFrame(
        {"a": ["g1", "g1"], "b": ["g1", "g2"], "c": [None, "g2"]}, index=idx
    )
    out = ops.indneutralize(x, labels)
    # 第 1 天 a/b 同组 → 去均值; c 无标签 → 原值
    assert out.loc["2024-01-01", "a"] == pytest.approx(-1.0)
    assert out.loc["2024-01-01", "b"] == pytest.approx(1.0)
    assert out.loc["2024-01-01", "c"] == pytest.approx(5.0)
    # 第 2 天 b/c 同组 → 去均值; a 独立成组 → 0
    assert out.loc["2024-01-02", "b"] == pytest.approx(-1.0)
    assert out.loc["2024-01-02", "c"] == pytest.approx(1.0)
    assert out.loc["2024-01-02", "a"] == pytest.approx(0.0)


def test_window_missing_tolerance_does_not_change_clean_data():
    """无缺失时 min_periods 容忍不生效：结果与严格满窗完全一致。"""
    idx = pd.bdate_range("2024-01-01", periods=12)
    x = pd.DataFrame({"a": np.arange(12, dtype=float)}, index=idx)
    assert ops.ts_mean(x, 5).iloc[4, 0] == pytest.approx(2.0)
    assert ops.ts_rank(x, 5).iloc[4, 0] == pytest.approx(1.0)
    assert ops.decay_linear(x, 5).iloc[4, 0] == pytest.approx(
        (0 * 1 + 1 * 2 + 2 * 3 + 3 * 4 + 4 * 5) / 15
    )
    # 单点缺失不会把后续长链条整段抹掉（容忍 20%）
    x_gap = x.copy()
    x_gap.iloc[6, 0] = np.nan
    assert not np.isnan(ops.ts_mean(x_gap, 5).iloc[7, 0])
    assert not np.isnan(ops.decay_linear(x_gap, 5).iloc[7, 0])


# ── 关键公式定点校验 ──────────────────────────────────────────────────────────


def _small_inputs():
    from factors.alpha101.testing import make_synthetic_inputs

    return make_synthetic_inputs(n_dates=60, n_symbols=8, seed=3)


def test_alpha_101_formula_exact():
    from factors.alpha101.testing import make_synthetic_inputs

    inp = make_synthetic_inputs(n_dates=20, n_symbols=4, seed=1)
    got = FORMULAS["101"](inp)
    exp = (inp.close - inp.open) / ((inp.high - inp.low) + 0.001)
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_alpha_041_and_042_formulas():
    inp = _small_inputs()
    got41 = FORMULAS["041"](inp)
    exp41 = np.sqrt(inp.high * inp.low) - inp.vwap
    pd.testing.assert_frame_equal(got41, exp41, check_dtype=False)

    got42 = FORMULAS["042"](inp)
    exp42 = ops.rank(inp.vwap - inp.close) / ops.rank(inp.vwap + inp.close)
    pd.testing.assert_frame_equal(got42, exp42, check_dtype=False)


def test_comparison_alphas_are_boolean_cast_to_float():
    """比较型公式输出必须是 0/±1（进入 IC/分组前的约定；带 *-1 的为 −1/0）。"""
    inp = _small_inputs()
    for alpha_id in ["061", "062", "064", "065", "068", "074", "075", "079", "081", "086", "095", "099"]:
        values = FORMULAS[alpha_id](inp).to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        assert set(np.unique(finite)) <= {-1.0, 0.0, 1.0}, f"#{alpha_id} 非 ±1/0 输出"


def test_formula_requiring_industry_raises_without_industry_labels():
    from factors.alpha101.testing import make_synthetic_inputs

    inp = make_synthetic_inputs(n_dates=30, n_symbols=5, with_industry=False)
    with pytest.raises(ValueError, match="行业分类"):
        FORMULAS["048"](inp)


def test_formula_requiring_cap_raises_without_cap():
    from factors.alpha101.testing import make_synthetic_inputs

    inp = make_synthetic_inputs(n_dates=30, n_symbols=5, with_cap=False)
    with pytest.raises(TypeError):
        FORMULAS["056"](inp)
