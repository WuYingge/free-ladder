from __future__ import annotations

from .base import BaseFactorTimingStrategy


class NewHighBATRSTimingStrategy(BaseFactorTimingStrategy):
    """Minimal subclass template for building custom timing strategies.

    Copy this class, rename it, then only modify:
    1) params: declare your factor-column mapping and thresholds
    2) __init__: bind required/optional lines
    3) next: implement entry/exit rules
    """

    params = (
        ("entry_factor_column", "NewHigh"),
        ("exit_factor_column", "AverageTrueRange"),
        ("lookback_period", 25),
        ("atr_multiplier", 3.0),
        ("target_percent", 0.95),
    )

    def __init__(self) -> None:
        data0 = self.datas[0]
        self.new_high_value = self.bind_required_line(data0, self.p.entry_factor_column)
        self.atr_value = self.bind_required_line(data0, self.p.exit_factor_column)

    def next(self) -> None:
        entry_signal = self.line_value(self.new_high_value, 0.0)
        atr_now = self.line_value(self.atr_value, 0.0)

        window = min(self.p.lookback_period, len(self.datas[0]))
        highest_close = max(float(self.datas[0].close[-i]) for i in range(window))
        current_close = float(self.datas[0].close[0])
        drawdown_from_high = highest_close - current_close

        # Expose values on data for optional downstream diagnostics.
        self.datas[0].atr_value = atr_now
        self.datas[0].highest_close_lookback = highest_close
        self.datas[0].drawdown_from_high = drawdown_from_high

        if not self.position.size and entry_signal == 1.0:
            self.order_target_percent(target=self.p.target_percent)
            return

        if self.position.size and drawdown_from_high > self.p.atr_multiplier * atr_now:
            self.close()

__all__ = ["NewHighBATRSTimingStrategy"]