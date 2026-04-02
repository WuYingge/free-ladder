from __future__ import annotations

from typing import Optional

import backtrader as bt
from factors.base_factor import BaseFactor


class BaseFactorTimingStrategy(bt.Strategy):
    """Base template for factor-driven timing strategies.

    Convention:
    - Strategy params that map to feed factor columns should use ``*_column``.
    - Column params can be required or optional.
    - ``factor_warmup_bars`` declares per-factor warm-up bars; batch metrics
      use the maximum declared value as the analysis warm-up period.
    """

    # Per-factor warm-up bars declared by subclass, for example:
    # {"RSRS": 600, "ATR_50": 50}
    factor_warmup_bars: dict[str, int] = {}
    # Preferred: declare concrete factor instances used by strategy so warm-up
    # can be derived from factor definitions and dependencies.
    involved_factors: tuple[BaseFactor, ...] = ()

    params = (
        ("target_percent", 0.95),
    )

    @classmethod
    def get_max_warmup_bars(cls) -> int:
        """Return strategy warm-up bars derived from factors or legacy map."""
        if cls.involved_factors:
            max_warmup = 0
            for factor in cls.involved_factors:
                factor_warmup = int(factor.get_max_warmup_period())
                if factor_warmup > max_warmup:
                    max_warmup = factor_warmup
            return max_warmup

        if not cls.factor_warmup_bars:
            return 0

        max_warmup = 0
        for factor_name, bars in cls.factor_warmup_bars.items():
            bars_int = int(bars)
            if bars_int < 0:
                raise ValueError(
                    f"Invalid warm-up bars for factor '{factor_name}': {bars_int}. "
                    "Expected a non-negative integer."
                )
            if bars_int > max_warmup:
                max_warmup = bars_int
        return max_warmup

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
