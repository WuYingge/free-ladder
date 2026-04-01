from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from data_manager.etf_data_manager import get_etf_data_by_symbol, get_etf_data_by_symbols


DEFAULT_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_etf_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "date" not in df.columns:
        raise ValueError(f"ETF data from {source} must contain 'date' column")

    # Backtrader expects a monotonic datetime index.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    missing = [col for col in DEFAULT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"ETF data from {source} missing required columns: {missing}")

    numeric_columns = [col for col in ["open", "high", "low", "close", "volume", "value", "turnOver"] if col in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def load_etf_dataframe(symbol: str, data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    # Reuse data_manager for default loading path and metadata conventions.
    if data_dir is None:
        etf_data = get_etf_data_by_symbol(symbol)
        return _normalize_etf_dataframe(etf_data.data, source=f"data_manager:{symbol}")

    # Keep explicit path override for ad-hoc datasets/backtest batches.
    file_path = Path(data_dir) / f"{symbol}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"ETF data file not found: {file_path}")

    return _normalize_etf_dataframe(pd.read_csv(file_path), source=str(file_path))


def load_etf_dataframes(symbols: list[str]) -> dict[str, pd.DataFrame]:
    etf_list = get_etf_data_by_symbols(symbols)
    return {
        etf.symbol: _normalize_etf_dataframe(etf.data, source=f"data_manager:{etf.symbol}")
        for etf in etf_list
        if etf.symbol is not None
    }


def build_bt_feed_dataframe(
    symbol: str,
    data_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    df = load_etf_dataframe(symbol=symbol, data_dir=data_dir)

    # Extra `openinterest` is required by PandasData default schema.
    bt_df = df[["open", "high", "low", "close", "volume"]].copy()
    bt_df["openinterest"] = 0.0

    # Keep all additional numeric columns (e.g. value/turnOver/factor columns)
    # so strategy.next() can compose arbitrary entry/exit logic.
    reserved = set(DEFAULT_COLUMNS + ["openinterest"])
    extra_columns = [column for column in df.columns if column not in reserved]
    for column in extra_columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.notna().any():
            bt_df[column] = numeric_series.fillna(0.0).astype(float)

    return bt_df
