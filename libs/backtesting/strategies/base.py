from __future__ import annotations

from typing import Optional

import backtrader as bt


class BaseFactorTimingStrategy(bt.Strategy):
    """Base template for factor-driven timing strategies.

    Convention:
    - Strategy params that map to feed factor columns should use ``*_column``.
    - Column params can be required or optional.
    """

    params = (
        ("target_percent", 0.95),
    )

    def bind_required_line(self, data: bt.LineSeries, column: str):
        """Bind and validate a required feed line by column name."""
        self._validate_column_name(column, required=True)
        if not hasattr(data, column):
            raise ValueError(
                f"Missing required factor column '{column}' in feed lines. "
                "Ensure the factor is precomputed before backtest."
            )
        return getattr(data, column)

    def bind_optional_line(self, data: bt.LineSeries, column: Optional[str]):
        """Bind an optional feed line by column name."""
        if not column:
            return None
        self._validate_column_name(column, required=False)
        return getattr(data, column, None)

    @staticmethod
    def line_value(line, default: float) -> float:
        """Read current bar value from a line with fallback for missing lines."""
        if line is None:
            return default
        return float(line[0])

    @staticmethod
    def _validate_column_name(column: Optional[str], *, required: bool) -> None:
        if required and not column:
            raise ValueError("Column parameter must be a non-empty string.")
        if column is None:
            return
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"Invalid column parameter: {column!r}. Expected a non-empty string.")


__all__ = ["BaseFactorTimingStrategy"]
