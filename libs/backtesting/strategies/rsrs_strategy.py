from __future__ import annotations

from .base import BaseFactorTimingStrategy
from factors.rsrs import RsrsFactor


class RsrsTimingStrategy(BaseFactorTimingStrategy):
    """Timing strategy driven by the RSRS Z-Score factor.

    Entry rule:  zscore > buy_threshold  -> go long (full position)
    Exit rule:   zscore < sell_threshold -> close position
    """

    involved_factors = (
        RsrsFactor(regression_window=18, zscore_window=600, output="zscore"),
    )

    params = (
        ("rsrs_column", "RSRS"),
        ("buy_threshold", 1.0),
        ("sell_threshold", -1.0),
        ("target_percent", 0.95),
    )

    def __init__(self) -> None:
        data0 = self.datas[0]
        self.rsrs_line = self.bind_required_line(data0, self.p.rsrs_column)

    def next(self) -> None:
        zscore = self.line_value(self.rsrs_line, float("nan"))

        import math
        if math.isnan(zscore):
            return

        if not self.position.size and zscore >= self.p.buy_threshold:
            self.order_target_percent(target=self.p.target_percent)
            return

        if self.position.size and zscore <= self.p.sell_threshold:
            self.close()


__all__ = ["RsrsTimingStrategy"]
