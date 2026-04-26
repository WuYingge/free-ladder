from __future__ import annotations

import math
from typing import Any, Callable, Optional

import backtrader as bt
import pandas as pd


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


WeightSignalFunction = Callable[[pd.DataFrame, dict[str, Any]], dict[str, float]]


class FunctionalPortfolioTimingStrategy(bt.Strategy):
    """Functional multi-symbol portfolio strategy."""

    params = (
        ("signal_func", None),
        ("signal_kwargs", None),
        ("rebalance_interval", 1),
        ("max_gross_exposure", 0.95),
        ("min_weight", 0.0),
        ("max_weight", 1.0),
        ("allow_short", False),
    )

    def __init__(self) -> None:
        if not callable(self.p.signal_func):
            raise ValueError("signal_func must be callable.")
        if int(self.p.rebalance_interval) <= 0:
            raise ValueError("rebalance_interval must be >= 1.")
        if float(self.p.max_gross_exposure) <= 0:
            raise ValueError("max_gross_exposure must be > 0.")

        self._signal_kwargs = dict(self.p.signal_kwargs or {})
        self._symbols = [d._name or f"data_{idx}" for idx, d in enumerate(self.datas)]
        self.last_target_weights: dict[str, float] = {sym: 0.0 for sym in self._symbols}

    def next(self) -> None:
        bar_index = len(self) - 1
        if bar_index < 0:
            return
        if bar_index % int(self.p.rebalance_interval) != 0:
            return

        snapshot = self._build_snapshot_frame()
        context = {
            "datetime": bt.num2date(self.datas[0].datetime[0]),
            "bar_index": bar_index,
            "current_weights": self._estimate_current_weights(),
        }
        raw_target = self.p.signal_func(snapshot, context, **self._signal_kwargs)
        target_weights = self._normalize_target_weights(raw_target)

        for data in self.datas:
            symbol = data._name or ""
            self.order_target_percent(data=data, target=target_weights.get(symbol, 0.0))
        self.last_target_weights = target_weights

    def _build_snapshot_frame(self) -> pd.DataFrame:
        rows: dict[str, dict[str, float]] = {}
        for idx, data in enumerate(self.datas):
            symbol = data._name or f"data_{idx}"
            row: dict[str, float] = {}
            for alias in data.lines.getlinealiases():
                try:
                    row[alias] = float(getattr(data, alias)[0])
                except Exception:
                    continue
            rows[symbol] = row
        return pd.DataFrame.from_dict(rows, orient="index")

    def _normalize_target_weights(self, raw: Optional[dict[str, float]]) -> dict[str, float]:
        target: dict[str, float] = {sym: 0.0 for sym in self._symbols}
        if not raw:
            return target

        allow_short = bool(self.p.allow_short)
        min_weight = float(self.p.min_weight)
        max_weight = float(self.p.max_weight)

        for symbol, value in raw.items():
            if symbol not in target:
                continue
            try:
                weight = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(weight):
                continue

            if allow_short:
                weight = max(-max_weight, min(max_weight, weight))
            else:
                weight = max(min_weight, min(max_weight, weight))
            target[symbol] = weight

        gross = float(sum(abs(weight) for weight in target.values()))
        max_gross = float(self.p.max_gross_exposure)
        if gross > max_gross and gross > 0.0:
            scale = max_gross / gross
            for symbol in target:
                target[symbol] *= scale

        return target

    def _estimate_current_weights(self) -> dict[str, float]:
        value = float(self.broker.getvalue())
        if value <= 0:
            return {sym: 0.0 for sym in self._symbols}

        weights: dict[str, float] = {}
        for data in self.datas:
            symbol = data._name or ""
            position = self.getposition(data)
            weights[symbol] = float(position.size * data.close[0] / value)
        return weights


__all__ = ["BaseFactorTimingStrategy", "WeightSignalFunction", "FunctionalPortfolioTimingStrategy"]
