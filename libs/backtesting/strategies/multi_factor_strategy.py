from __future__ import annotations

from .base import BaseFactorTimingStrategy


class MultiFactorTimingStrategy(BaseFactorTimingStrategy):
    """Template strategy that directly combines multiple factor columns in ``next()``.

    Factor values are expected to be precomputed outside the strategy and exposed
    as feed lines by the engine's auto data feed.
    """

    params = (
        ("entry_factor_column", "NewHigh"),
        ("exit_factor_column", "NewHigh"),
        ("trend_filter_column", "long_ma_filter"),
        ("volatility_column", "ATR_50"),
        ("entry_threshold", 1.0),
        ("exit_threshold", -1.0),
        ("min_volatility", 0.0),
        ("target_percent", 0.95),
    )

    def __init__(self) -> None:
        data0 = self.datas[0]
        self.entry_factor = self.bind_required_line(data0, self.p.entry_factor_column)
        self.exit_factor = self.bind_required_line(data0, self.p.exit_factor_column)
        self.trend_filter = self.bind_optional_line(data0, self.p.trend_filter_column)
        self.volatility = self.bind_optional_line(data0, self.p.volatility_column)

    def next(self) -> None:
        entry_signal = self.line_value(self.entry_factor, 0.0)
        exit_signal = self.line_value(self.exit_factor, 0.0)
        trend_ok = self.line_value(self.trend_filter, 1.0) >= 0.5
        volatility_ok = self.line_value(self.volatility, self.p.min_volatility) >= self.p.min_volatility

        if not self.position.size and entry_signal >= self.p.entry_threshold and trend_ok and volatility_ok:
            self.order_target_percent(target=self.p.target_percent)
            return

        if self.position.size and exit_signal <= self.p.exit_threshold:
            self.close()


__all__ = ["MultiFactorTimingStrategy"]
