"""Regression tests for factor output naming.

Run from project root:
    PYTHONPATH=libs pytest libs/backtesting/tests/test_factor_output_names.py -v
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtesting.preprocessing import parallel_calc_factors_for_map
from core.models.etf_daily_data import EtfData
from data_manager.etf_data_manager import get_etf_data_by_symbol
from factors.base_factor import BaseFactor
from factors.price_return import PriceReturn
from factors.rolling_ols import RollingOLS


def _make_etf_frame(n: int = 260) -> pd.DataFrame:
    close = pd.Series([100.0 + i for i in range(n)], dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([1_000] * n, dtype="int64"),
            "value": pd.Series([100_000] * n, dtype="int64"),
            "turnOver": pd.Series([0.01] * n, dtype=float),
            "gain": pd.Series([0.0] * n, dtype=float),
            "change": pd.Series([0.0] * n, dtype=float),
        }
    )


def _make_etf(symbol: str = "TEST") -> EtfData:
    return EtfData(_make_etf_frame(), symbol=symbol)


def _price_return_columns() -> list[str]:
    return ["PriceReturn_20", "PriceReturn_60", "PriceReturn_120"]


def _rolling_ols_columns() -> list[str]:
    return ["RollingOLS_close_20_slope", "RollingOLS_close_20_r2"]


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
        return "conflict"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(self.fill_value, index=data.index, dtype=float)
        result.name = self.get_output_name()
        return result


def test_calc_factors_preserves_parameterized_output_names():
    etf = _make_etf()
    factors = [PriceReturn(window=20), PriceReturn(window=60), PriceReturn(window=120)]

    for factor in factors:
        etf.add_factors(factor)

    output = etf.calc_factors()

    for factor, column in zip(factors, _price_return_columns()):
        assert etf.factor_results[factor].name == column
        assert column in output.columns


def test_calc_factors_raises_on_duplicate_output_names():
    etf = _make_etf()
    etf.add_factors(ConstantNamedFactor(1.0))
    etf.add_factors(ConstantNamedFactor(2.0))

    with pytest.raises(ValueError, match="Duplicate factor output name"):
        etf.calc_factors()


def test_calc_factors_preserves_rolling_ols_output_names():
    etf = _make_etf()
    factors = [RollingOLS(window=20, output="slope"), RollingOLS(window=20, output="r2")]

    for factor in factors:
        etf.add_factors(factor)

    output = etf.calc_factors()

    for factor, column in zip(factors, _rolling_ols_columns()):
        assert etf.factor_results[factor].name == column
        assert column in output.columns


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[3] / "data" / "etf_data" / "159007.csv").exists(),
    reason="requires local ETF fixture data",
)
def test_parallel_calc_factors_preserves_parameterized_output_names():
    symbol = "159007"
    factors = [PriceReturn(window=20), PriceReturn(window=60), PriceReturn(window=120)]
    result = parallel_calc_factors_for_map(
        etf_data_map={symbol: get_etf_data_by_symbol(symbol)},
        factor_pipeline=factors,
        symbols=[symbol],
        max_workers=1,
        retry_serial_on_fail=False,
    )

    assert result.errors == []

    output = result.etf_data_map[symbol].output_with_factors()
    for column in _price_return_columns():
        assert column in output.columns