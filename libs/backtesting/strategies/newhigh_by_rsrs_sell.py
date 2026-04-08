from __future__ import annotations

from .base import BaseFactorTimingStrategy
class NewHighByRSRSTimingStrategy(BaseFactorTimingStrategy):
    """Timing strategy that combines new high breakout with RSRS timing.

    Entry rule:  NewHigh == 1.0 and RSRS Z-Score > buy_threshold -> go long (full position)
    Exit rule:   NewHigh == 0.0 or RSRS Z-Score < sell_threshold -> close position
    """

    params = (
        ("new_high_column", "NewHigh"),
        ("rsrs_column", "RSRS"),
        ("buy_threshold", 0.7),
        ("sell_threshold", -0.7),
        ("target_percent", 0.95),
    )

    def __init__(self) -> None:
        data0 = self.datas[0]
        self.new_high_line = self.bind_required_line(data0, self.p.new_high_column)
        self.rsrs_line = self.bind_required_line(data0, self.p.rsrs_column)

    def next(self) -> None:
        new_high_signal = self.line_value(self.new_high_line, 0.0)
        zscore = self.line_value(self.rsrs_line, float("nan"))

        import math
        if math.isnan(zscore):
            return

        if not self.position.size and new_high_signal == 1.0:
            self.order_target_percent(target=self.p.target_percent)
            return

        if self.position.size and zscore <= self.p.sell_threshold:
            self.close()

__all__ = ["NewHighByRSRSTimingStrategy"]
