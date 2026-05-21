from __future__ import annotations

import pandas as pd
import pytest

from core.models.etf_daily_data import EtfData
from core.portfolios import portfolio as portfolio_module
from factors.base_factor import BaseFactor
from factors.rsrs import RsrsDerivedFactor, RsrsFactor
from factors.trend_r2 import TrendR2Factor


class ConstantNamedFactor(BaseFactor):
    name = "ConstantNamedFactor"
    params = {
        "fill_value": 1.0,
    }

    def __init__(self, fill_value: float) -> None:
        super().__init__()
        self.fill_value = float(fill_value)
        self._set_params(fill_value=fill_value)

    def get_output_name(self) -> str:
        return "duplicate"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(self.fill_value, index=data.index, dtype=float)
        result.name = self.get_output_name()
        return result


def _make_etf(
    symbol: str,
    close_values: list[float],
    high_values: list[float] | None = None,
    low_values: list[float] | None = None,
) -> EtfData:
    close = pd.Series(close_values, dtype=float)
    high = pd.Series(high_values if high_values is not None else [value + 1.0 for value in close_values], dtype=float)
    low = pd.Series(low_values if low_values is not None else [value - 1.0 for value in close_values], dtype=float)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": pd.Series([1_000] * len(close), dtype="int64"),
            "value": pd.Series([100_000] * len(close), dtype="int64"),
            "turnOver": pd.Series([0.01] * len(close), dtype=float),
            "gain": pd.Series([0.0] * len(close), dtype=float),
            "change": pd.Series([0.0] * len(close), dtype=float),
        }
    )
    return EtfData(frame, symbol=symbol)


def _mock_analysis_dependencies(monkeypatch: pytest.MonkeyPatch, etf_map: dict[str, EtfData]) -> None:
    monkeypatch.setattr(
        portfolio_module,
        "get_etf_data_by_symbol",
        lambda symbol: etf_map[symbol],
    )
    monkeypatch.setattr(
        portfolio_module.ClusterInfo,
        "get_cluster",
        lambda symbol: 1 if symbol in {"HOLD", "AAA"} else 2,
    )
    monkeypatch.setattr(
        portfolio_module.Portfolio,
        "calc_corrs_with_current_position",
        lambda self, symbols, name_dict=None: pd.DataFrame(
            {
                "average_correlation": [0.2, 0.1],
                "name": [
                    (name_dict or {}).get("AAA", "AAA"),
                    (name_dict or {}).get("BBB", "BBB"),
                ],
            },
            index=pd.Index(["AAA", "BBB"], name="symbol"),
        ),
    )
    monkeypatch.setattr(
        portfolio_module.Portfolio,
        "max_money_for_symbol_by_ATR",
        lambda self, symbol: {"AAA": 111.0, "BBB": 222.0}[symbol],
    )


def test_trend_r2_factor_returns_expected_r2_series() -> None:
    factor = TrendR2Factor(window=3, output="r2")
    frame = pd.DataFrame({"close": pd.Series([100.0, 110.0, 120.0], dtype=float)})

    result = factor(frame)

    assert result.name == "TrendR2_3_r2"
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(1.0)


def test_analyze_with_add_to_list_joins_requested_factor_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    etf_map = {
        "HOLD": _make_etf("HOLD", [100.0, 101.0, 102.0, 103.0, 104.0]),
        "AAA": _make_etf("AAA", [100.0, 105.0, 111.0, 118.0, 126.0]),
        "BBB": _make_etf("BBB", [100.0, 102.0, 103.0, 105.0, 106.0]),
    }
    _mock_analysis_dependencies(monkeypatch, etf_map)

    r2_factor = TrendR2Factor(window=3, output="r2")
    slope_factor = TrendR2Factor(window=3, output="slope")

    portfolio = portfolio_module.Portfolio(1_000)
    portfolio.add_position_with_symbol_quantities(("HOLD", 1))

    result = portfolio.analyze_with_add_to_list(
        ["AAA", "BBB"],
        name_dict={"AAA": "Alpha", "BBB": "Beta"},
        analysis_factors=[r2_factor, slope_factor],
    )

    assert list(result.index) == ["BBB", "AAA"]
    assert list(result.columns) == [
        "average_correlation",
        "name",
        "max_value",
        "TrendR2_3_r2",
        "TrendR2_3_slope",
        "cluster",
        "newCluster",
    ]
    assert "change_since_new_high" not in result.columns
    assert result.loc["AAA", "name"] == "Alpha"
    assert result.loc["BBB", "name"] == "Beta"
    assert result.loc["AAA", "max_value"] == pytest.approx(111.0)
    assert result.loc["BBB", "max_value"] == pytest.approx(222.0)
    assert result.loc["AAA", "TrendR2_3_r2"] == pytest.approx(r2_factor(etf_map["AAA"].data).iloc[-1])
    assert result.loc["BBB", "TrendR2_3_r2"] == pytest.approx(r2_factor(etf_map["BBB"].data).iloc[-1])
    assert result.loc["AAA", "TrendR2_3_slope"] == pytest.approx(slope_factor(etf_map["AAA"].data).iloc[-1])
    assert result.loc["BBB", "TrendR2_3_slope"] == pytest.approx(slope_factor(etf_map["BBB"].data).iloc[-1])
    assert result.loc["AAA", "newCluster"] == "No"
    assert result.loc["BBB", "newCluster"] == "Yes"


def test_analyze_with_add_to_list_raises_on_duplicate_factor_output_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    etf_map = {
        "HOLD": _make_etf("HOLD", [100.0, 101.0, 102.0]),
        "AAA": _make_etf("AAA", [100.0, 110.0, 120.0]),
        "BBB": _make_etf("BBB", [100.0, 105.0, 111.0]),
    }
    _mock_analysis_dependencies(monkeypatch, etf_map)

    portfolio = portfolio_module.Portfolio(1_000)
    portfolio.add_position_with_symbol_quantities(("HOLD", 1))

    with pytest.raises(ValueError, match="Duplicate factor output name"):
        portfolio.analyze_with_add_to_list(
            ["AAA", "BBB"],
            analysis_factors=[ConstantNamedFactor(1.0), ConstantNamedFactor(2.0)],
        )


def _mock_rsrs_dependency_outputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    signal_by_first_close: dict[float, list[int]],
    zscore_by_first_close: dict[float, list[float]],
) -> None:
    def _fake_rsrs_call(self: RsrsFactor, data: pd.DataFrame) -> pd.Series:
        marker = float(pd.to_numeric(data["close"], errors="coerce").iloc[0])
        if self.output == "signal":
            values = pd.Series(signal_by_first_close[marker], index=data.index, dtype="int64")
        elif self.output == "zscore":
            values = pd.Series(zscore_by_first_close[marker], index=data.index, dtype=float)
        else:
            raise AssertionError(f"Unexpected mocked RSRS output {self.output!r}")
        values.name = self.get_output_name()
        return values

    monkeypatch.setattr(RsrsFactor, "__call__", _fake_rsrs_call)


def test_rsrs_derived_factor_signal_duration_and_return_reset_after_sell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "close": pd.Series([10.0, 11.0, 12.0, 9.0, 10.0, 12.0, 15.0], dtype=float),
        }
    )
    _mock_rsrs_dependency_outputs(
        monkeypatch,
        signal_by_first_close={10.0: [0, 1, 0, -1, 0, 1, 0]},
        zscore_by_first_close={10.0: [float("nan"), 0.8, 0.6, -0.9, -0.2, 0.9, 1.1]},
    )

    duration_factor = RsrsDerivedFactor(
        regression_window=3,
        zscore_window=3,
        short_ema_span=2,
        long_ema_span=3,
        output="signal_duration",
    )
    return_factor = RsrsDerivedFactor(
        regression_window=3,
        zscore_window=3,
        short_ema_span=2,
        long_ema_span=3,
        output="return_since_buy",
    )

    duration_result = duration_factor(frame)
    return_result = return_factor(frame)

    expected_duration = pd.Series(
        [float("nan"), 0.0, 1.0, float("nan"), float("nan"), 0.0, 1.0],
        index=frame.index,
        name=duration_factor.get_output_name(),
        dtype=float,
    )
    expected_return = pd.Series(
        [float("nan"), 0.0, (12.0 / 11.0) - 1.0, float("nan"), float("nan"), 0.0, (15.0 / 12.0) - 1.0],
        index=frame.index,
        name=return_factor.get_output_name(),
        dtype=float,
    )

    pd.testing.assert_series_equal(duration_result, expected_duration)
    pd.testing.assert_series_equal(return_result, expected_return)


def test_rsrs_derived_factor_zscore_trend_matches_ema_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "close": pd.Series([20.0, 21.0, 22.0, 23.0, 24.0, 25.0], dtype=float),
        }
    )
    zscore = pd.Series([float("nan"), float("nan"), 0.0, 1.0, 2.0, 1.0], index=frame.index, dtype=float)
    _mock_rsrs_dependency_outputs(
        monkeypatch,
        signal_by_first_close={20.0: [0, 0, 1, 0, 0, 0]},
        zscore_by_first_close={20.0: zscore.tolist()},
    )

    factor = RsrsDerivedFactor(
        regression_window=3,
        zscore_window=3,
        short_ema_span=2,
        long_ema_span=3,
        output="zscore_trend",
    )

    result = factor(frame)
    expected = (
        zscore.ewm(span=2, adjust=False).mean()
        - zscore.ewm(span=3, adjust=False).mean()
    ).where(zscore.notna())
    expected.name = factor.get_output_name()

    pd.testing.assert_series_equal(result, expected)


def test_analyze_with_add_to_list_joins_rsrs_derived_factor_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    etf_map = {
        "HOLD": _make_etf("HOLD", [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0]),
        "AAA": _make_etf("AAA", [100.0, 101.0, 102.0, 99.0, 100.0, 102.0, 105.0]),
        "BBB": _make_etf("BBB", [200.0, 201.0, 203.0, 204.0, 206.0, 208.0, 210.0]),
    }
    _mock_analysis_dependencies(monkeypatch, etf_map)
    _mock_rsrs_dependency_outputs(
        monkeypatch,
        signal_by_first_close={
            100.0: [0, 1, 0, -1, 0, 1, 0],
            200.0: [0, 0, 1, 0, 0, 0, 0],
        },
        zscore_by_first_close={
            100.0: [float("nan"), 0.8, 0.6, -0.9, -0.2, 0.9, 1.1],
            200.0: [float("nan"), float("nan"), 0.1, 0.4, 0.8, 1.2, 1.5],
        },
    )

    duration_factor = RsrsDerivedFactor(
        regression_window=3,
        zscore_window=3,
        short_ema_span=2,
        long_ema_span=3,
        output="signal_duration",
    )
    trend_factor = RsrsDerivedFactor(
        regression_window=3,
        zscore_window=3,
        short_ema_span=2,
        long_ema_span=3,
        output="zscore_trend",
    )
    return_factor = RsrsDerivedFactor(
        regression_window=3,
        zscore_window=3,
        short_ema_span=2,
        long_ema_span=3,
        output="return_since_buy",
    )

    portfolio = portfolio_module.Portfolio(1_000)
    portfolio.add_position_with_symbol_quantities(("HOLD", 1))

    result = portfolio.analyze_with_add_to_list(
        ["AAA", "BBB"],
        name_dict={"AAA": "Alpha", "BBB": "Beta"},
        analysis_factors=[duration_factor, trend_factor, return_factor],
    )

    assert list(result.index) == ["BBB", "AAA"]
    assert list(result.columns) == [
        "average_correlation",
        "name",
        "max_value",
        duration_factor.get_output_name(),
        trend_factor.get_output_name(),
        return_factor.get_output_name(),
        "cluster",
        "newCluster",
    ]
    assert result.loc["AAA", duration_factor.get_output_name()] == pytest.approx(
        duration_factor(etf_map["AAA"].data).iloc[-1]
    )
    assert result.loc["BBB", duration_factor.get_output_name()] == pytest.approx(
        duration_factor(etf_map["BBB"].data).iloc[-1]
    )
    assert result.loc["AAA", trend_factor.get_output_name()] == pytest.approx(
        trend_factor(etf_map["AAA"].data).iloc[-1]
    )
    assert result.loc["BBB", trend_factor.get_output_name()] == pytest.approx(
        trend_factor(etf_map["BBB"].data).iloc[-1]
    )
    assert result.loc["AAA", return_factor.get_output_name()] == pytest.approx(
        return_factor(etf_map["AAA"].data).iloc[-1]
    )
    assert result.loc["BBB", return_factor.get_output_name()] == pytest.approx(
        return_factor(etf_map["BBB"].data).iloc[-1]
    )
