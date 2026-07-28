from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from core.models.daily_quote_data import DailyQuoteData
from data_manager.etf_data_manager import get_etf_data_by_symbol, get_etf_data_by_symbols


DEFAULT_COLUMNS = ["open", "high", "low", "close", "volume"]


def _normalize_daily_dataframe(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        raise ValueError(
            f"Data from {source} must contain 'date' column or DatetimeIndex"
        )

    df = df.sort_index()

    missing = [col for col in DEFAULT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Data from {source} missing required columns: {missing}")

    numeric_columns = [col for col in ["open", "high", "low", "close", "volume", "value", "turnOver"] if col in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def _to_bt_feed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    bt_df = df[["open", "high", "low", "close", "volume"]].copy()
    bt_df["openinterest"] = 0.0

    reserved = set(DEFAULT_COLUMNS + ["openinterest"])
    extra_columns = [column for column in df.columns if column not in reserved]
    for column in extra_columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.notna().any():
            bt_df[column] = numeric_series.fillna(0.0).astype(float)

    return bt_df


def load_daily_dataframe(symbol: str, data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    if data_dir is None:
        etf_data = get_etf_data_by_symbol(symbol)
        return _normalize_daily_dataframe(etf_data.data, source=f"data_manager:{symbol}")

    file_path = Path(data_dir) / f"{symbol}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return _normalize_daily_dataframe(pd.read_csv(file_path), source=str(file_path))


def build_bt_feed_dataframe_from_dataframe(df: pd.DataFrame, *, source: str = "in_memory") -> pd.DataFrame:
    normalized_df = _normalize_daily_dataframe(df, source=source)
    return _to_bt_feed_dataframe(normalized_df)


def build_bt_feed_dataframe_from_daily(daily_data: DailyQuoteData) -> pd.DataFrame:
    merged_df = daily_data.output_with_factors()
    source = f"daily_data:{daily_data.symbol or ''}"
    return build_bt_feed_dataframe_from_dataframe(merged_df, source=source)


# ---- legacy aliases for backward compatibility ----

def load_etf_dataframe(symbol: str, data_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Legacy alias for load_daily_dataframe."""
    return load_daily_dataframe(symbol, data_dir=data_dir)


def build_bt_feed_dataframe_from_etf_data(etf_data: DailyQuoteData) -> pd.DataFrame:
    """Legacy alias for build_bt_feed_dataframe_from_daily."""
    return build_bt_feed_dataframe_from_daily(etf_data)


def build_bt_feed_dataframe(
    symbol: str,
    data_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    df = load_daily_dataframe(symbol=symbol, data_dir=data_dir)
    return _to_bt_feed_dataframe(df)


def load_etf_dataframes(symbols: list[str]) -> dict[str, pd.DataFrame]:
    etf_list = get_etf_data_by_symbols(symbols)
    return {
        etf.symbol: _normalize_daily_dataframe(etf.data, source=f"data_manager:{etf.symbol}")
        for etf in etf_list
        if etf.symbol is not None
    }
