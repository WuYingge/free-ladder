from __future__ import annotations

import backtrader as bt


class SingleFactorSingleTargetDataFeed(bt.feeds.PandasData):
    # Expose custom factor signal line to strategy.
    lines = ("signal",)
    # Default mapping reads from a column literally named "signal".
    # Engine can override it dynamically, e.g. signal="NewHigh".
    params = (("signal", "signal"),)


class SingleFactorSingleTargetStrategy(bt.Strategy):
    params = (
        ("stake", 100),
        ("buy_signal", 1.0),
        ("sell_signal", -1.0),
    )

    def __init__(self) -> None:
        self.signal = self.datas[0].signal

    def next(self) -> None:
        signal_value = float(self.signal[0])
        # Current implementation is single-position (no pyramiding).
        has_position = bool(self.position.size)

        if not has_position and signal_value >= self.p.buy_signal:
            # Allocate all available portfolio value into the current instrument.
            self.order_target_percent(target=0.95)
            return

        if has_position and signal_value <= self.p.sell_signal:
            self.close()


# Backward-compatible aliases for existing imports.
SignalDataFeed = SingleFactorSingleTargetDataFeed
FactorSignalStrategy = SingleFactorSingleTargetStrategy
