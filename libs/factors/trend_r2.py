from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class TrendR2Result:
    slope: pd.Series
    r2: pd.Series


def compute_trend_r2(close: pd.Series, window: int) -> TrendR2Result:
    """Compute rolling trend slope and R^2 from cumulative returns.

    For each rolling window, cumulative return is rebased to the first close in the
    window, then regressed on x = [0, 1, ..., window - 1].
    """

    if window < 2:
        raise ValueError("window must be at least 2")

    close_values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float, copy=False)
    result_index = close.index
    slope_values = np.full(close_values.shape[0], np.nan, dtype=float)
    r2_values = np.full(close_values.shape[0], np.nan, dtype=float)

    if close_values.shape[0] < window:
        return TrendR2Result(
            slope=pd.Series(slope_values, index=result_index, name=f"TrendR2Slope_{window}"),
            r2=pd.Series(r2_values, index=result_index, name=f"TrendR2R2_{window}"),
        )

    windows = sliding_window_view(close_values, window_shape=window)
    valid_window_mask = np.isfinite(windows).all(axis=1)
    valid_window_mask &= np.isfinite(windows[:, 0])
    valid_window_mask &= windows[:, 0] != 0.0

    if valid_window_mask.any():
        x = np.arange(window, dtype=float)
        x_mean = float(x.mean())
        x_centered = x - x_mean
        x_denom = float(np.square(x_centered).sum())

        valid_windows = windows[valid_window_mask]
        cum_ret = valid_windows / valid_windows[:, [0]] - 1.0
        y_mean = cum_ret.mean(axis=1)
        y_centered = cum_ret - y_mean[:, None]
        valid_slope = (y_centered @ x_centered) / x_denom

        intercept = y_mean - valid_slope * x_mean
        fitted = intercept[:, None] + valid_slope[:, None] * x
        ss_tot = np.square(y_centered).sum(axis=1)
        ss_res = np.square(cum_ret - fitted).sum(axis=1)

        valid_r2 = np.full(valid_windows.shape[0], np.nan, dtype=float)
        non_constant_mask = ss_tot > 0.0
        valid_r2[non_constant_mask] = 1.0 - ss_res[non_constant_mask] / ss_tot[non_constant_mask]
        valid_r2[non_constant_mask] = np.clip(valid_r2[non_constant_mask], 0.0, 1.0)

        result_positions = np.arange(window - 1, close_values.shape[0])[valid_window_mask]
        slope_values[result_positions] = valid_slope
        r2_values[result_positions] = valid_r2

    return TrendR2Result(
        slope=pd.Series(slope_values, index=result_index, name=f"TrendR2Slope_{window}"),
        r2=pd.Series(r2_values, index=result_index, name=f"TrendR2R2_{window}"),
    )


def compute_trend_r2_frame(close: pd.Series, window: int) -> pd.DataFrame:
    metrics = compute_trend_r2(close=close, window=window)
    return pd.DataFrame({"slope": metrics.slope, "r2": metrics.r2})