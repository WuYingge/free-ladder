from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


class RollingOLS(BaseFactor):
    """Rolling OLS over a single price/value column against time index."""

    name = "RollingOLS"
    params = {
        "window": 20,
        "value_column": "close",
        "output": "slope",
    }

    def __init__(
        self,
        window: int = 20,
        value_column: str = "close",
        output: str = "slope",
    ) -> None:
        super().__init__()
        self.window = int(window)
        self.value_column = value_column
        self.output = output
        self.warmup_period = self.window
        self._set_params(
            window=window,
            value_column=value_column,
            output=output,
        )

    def get_output_name(self) -> str:
        output_name = "r2" if self.output in {"r2", "r_squared"} else self.output
        return f"{self.name}_{self.value_column}_{self.window}_{output_name}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)

        values = data[self.value_column].astype(float)
        slope = self._calculate_slope(values)
        r_squared = self._calculate_r_squared(values)

        if self.output == "slope":
            result = slope
        elif self.output in {"r2", "r_squared"}:
            result = r_squared
        else:
            raise ValueError("output must be one of: slope, r2, r_squared")

        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.value_column not in data.columns:
            raise ValueError(
                f"RollingOLS requires column '{self.value_column}', got columns {list(data.columns)}"
            )

    def _calculate_slope(self, values: pd.Series) -> pd.Series:
        x = np.arange(self.window, dtype=float)
        x_mean = x.mean()
        x_centered = x - x_mean
        x_denom = float(np.square(x_centered).sum())

        def compute(window_values: np.ndarray) -> float:
            if np.isnan(window_values).any():
                return np.nan
            y_centered = window_values - window_values.mean()
            return float(np.dot(x_centered, y_centered) / x_denom)

        result = values.rolling(window=self.window).apply(compute, raw=True)
        result.name = f"{self.name}_slope"
        return result

    def _calculate_r_squared(self, values: pd.Series) -> pd.Series:
        x = np.arange(self.window, dtype=float)

        def compute(window_values: np.ndarray) -> float:
            if np.isnan(window_values).any():
                return np.nan
            y = np.asarray(window_values, dtype=float)
            y_mean = float(y.mean())
            ss_tot = float(np.square(y - y_mean).sum())
            if ss_tot == 0.0:
                return np.nan

            slope, intercept = np.polyfit(x, y, deg=1)
            fitted = slope * x + intercept
            ss_res = float(np.square(y - fitted).sum())
            return float(1.0 - ss_res / ss_tot)

        result = values.rolling(window=self.window).apply(compute, raw=True)
        result.name = f"{self.name}_r2"
        return result