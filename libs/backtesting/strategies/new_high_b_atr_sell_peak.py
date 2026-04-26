from __future__ import annotations

from .base import BaseFactorTimingStrategy


class NewHighBATRSPeakTimingStrategy(BaseFactorTimingStrategy):
    """Similar to NewHighBATRSTimingStrategy, but exit is triggered when
    the price drops more than n * ATR from the **all-time high since entry**
    (i.e., trailing stop loss from peak), rather than from the highest close
    within a fixed lookback window.
    """

    params = (
        ("entry_factor_column", "NewHigh"),
        ("exit_factor_column", "AverageTrueRange"),
        ("atr_multiplier", 3.0),
        ("target_percent", 0.95),
    )

    def __init__(self) -> None:
        data0 = self.datas[0]
        self.new_high_value = self.bind_required_line(data0, self.p.entry_factor_column)
        self.atr_value = self.bind_required_line(data0, self.p.exit_factor_column)
        self._peak_close: float = 0.0

    def next(self) -> None:
        entry_signal = self.line_value(self.new_high_value, 0.0)
        atr_now = self.line_value(self.atr_value, 0.0)
        current_close = float(self.datas[0].close[0])

        if not self.position.size:
            if entry_signal == 1.0:
                self._peak_close = current_close
                self.order_target_percent(target=self.p.target_percent)
            return

        # Update peak while in position.
        if current_close > self._peak_close:
            self._peak_close = current_close

        drawdown_from_peak = self._peak_close - current_close

        # Expose values on data for optional downstream diagnostics.
        self.datas[0].atr_value = atr_now
        self.datas[0].peak_close = self._peak_close
        self.datas[0].drawdown_from_peak = drawdown_from_peak

        if drawdown_from_peak > self.p.atr_multiplier * atr_now:
            self._peak_close = 0.0
            self.close()


__all__ = ["NewHighBATRSPeakTimingStrategy"]
