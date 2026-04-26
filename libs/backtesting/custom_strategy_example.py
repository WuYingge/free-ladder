from __future__ import annotations

from typing import Any

import pandas as pd

from .strategies.custom_strategy_example import ExampleCustomTimingStrategy


def example_equal_weight_momentum_signal(
	snapshot: pd.DataFrame,
	context: dict[str, Any],
	*,
	momentum_column: str = "momentum",
	min_score: float = 0.0,
	max_positions: int = 5,
) -> dict[str, float]:
	"""Example functional portfolio signal.

	Select symbols with momentum >= min_score, keep top-N by momentum,
	and allocate equal weights.
	"""
	if snapshot.empty or momentum_column not in snapshot.columns:
		return {}

	candidates = snapshot[[momentum_column]].dropna().copy()
	candidates = candidates[candidates[momentum_column] >= float(min_score)]
	if candidates.empty:
		return {}

	top_n = max(1, int(max_positions))
	selected = candidates.sort_values(momentum_column, ascending=False).head(top_n)
	weight = 1.0 / float(len(selected))
	return {symbol: float(weight) for symbol in selected.index.tolist()}


__all__ = ["ExampleCustomTimingStrategy", "example_equal_weight_momentum_signal"]