from __future__ import annotations

import backtrader as bt


class ExampleCustomTimingStrategy(bt.Strategy):
    """Example timing strategy that trades directly inside ``next()``.

    This class is intentionally simple and lives in an importable module so it
    can be used by ``TimingBatchConfig`` with multi-process execution.
    """

    params = (
        ("entry_column", "signal"),
        ("exit_column", "signal"),
        ("entry_threshold", 1.0),
        ("exit_threshold", -1.0),
        ("target_percent", 0.95),
    )

    def __init__(self) -> None:
        self.entry_line = getattr(self.datas[0], self.p.entry_column)
        self.exit_line = getattr(self.datas[0], self.p.exit_column)

    def next(self) -> None:
        entry_value = float(self.entry_line[0])
        exit_value = float(self.exit_line[0])

        if not self.position.size and entry_value >= self.p.entry_threshold:
            self.order_target_percent(target=self.p.target_percent)
            return

        if self.position.size and exit_value <= self.p.exit_threshold:
            self.close()


__all__ = ["ExampleCustomTimingStrategy"]