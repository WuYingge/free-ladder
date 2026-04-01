from __future__ import annotations

from .base import BaseFactorTimingStrategy


class MinimalFactorTimingStrategyTemplate(BaseFactorTimingStrategy):
    """Minimal subclass template for building custom timing strategies.

    Copy this class, rename it, then only modify:
    1) params: declare your factor-column mapping and thresholds
    2) __init__: bind required/optional lines
    3) next: implement entry/exit rules
    """

    params = (
        ("entry_factor_column", "NewHigh"),
        ("exit_factor_column", "NewHigh"),
        ("entry_threshold", 1.0),
        ("exit_threshold", -1.0),
        ("target_percent", 0.95),
    )

    def __init__(self) -> None:
        data0 = self.datas[0]
        self.entry_factor = self.bind_required_line(data0, self.p.entry_factor_column)
        self.exit_factor = self.bind_required_line(data0, self.p.exit_factor_column)

    def next(self) -> None:
        entry_signal = self.line_value(self.entry_factor, 0.0)
        exit_signal = self.line_value(self.exit_factor, 0.0)

        if not self.position.size and entry_signal >= self.p.entry_threshold:
            self.order_target_percent(target=self.p.target_percent)
            return

        if self.position.size and exit_signal <= self.p.exit_threshold:
            self.close()


__all__ = ["MinimalFactorTimingStrategyTemplate"]
