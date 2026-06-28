"""反转族因子。

本模块包含 4 个反转族因子，扩展 `libs/factors/` 的因子库：

- **ShortTermReversal**：短期反转，−N 日收益，捕捉短期超买超卖后的均值回归
- **ExtremeReversal**：极端反转，只在收益排名极值尾部取反向信号，中间为 0
- **VolumeReversal**：放量反转，反转信号乘以量比，放量反转更可靠
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor


class ShortTermReversal(BaseFactor):
    """短期反转因子：−N 日收益。

    与 PriceReturn（正动量追涨）相反，反转因子捕捉短期超买超卖后的均值回归。

    ``result = −(close / close.shift(N) − 1)``

    参数
    ----------
    window : int
        回看窗口天数。1 = 1 日反转，5 = 周反转，20 = 月反转。
    """

    name = "ShortTermReversal"
    params = {"window": 1}

    def __init__(self, window: int = 1) -> None:
        super().__init__()
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        self.warmup_period = self.window + 1
        self._set_params(window=window)

    def get_output_name(self) -> str:
        return f"ShortTermReversal_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        # PriceReturn = close / close.shift(N) - 1，取反即为反转信号
        result = -(close / close.shift(self.window) - 1.0)
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if "close" not in data.columns:
            raise ValueError(
                f"ShortTermReversal requires column 'close', got {list(data.columns)}"
            )


class ExtremeReversal(BaseFactor):
    """极端反转因子：在收益排名极值尾部取反向信号。

    计算逻辑:
        1. 计算 N 日收益率 ``ret = close / close.shift(N) - 1``
        2. 在滚动窗口内对收益率做百分位排名（0~1）
        3. 排名 ≥ 1−tail_pct（极端上涨尾部）→ −1（卖出）
        4. 排名 ≤ tail_pct（极端下跌尾部）→ +1（买入）
        5. 其他 → 0（中性）

    注意：当前实现为单标的滚动排名版本，适用于择时策略。
    横截面版本可继承 BaseCrossSectionFactor 另行实现。

    参数
    ----------
    window : int
        收益率计算窗口和滚动排名窗口。
    tail_pct : float
        尾部阈值，默认 0.1 表示前 10% 和后 10%。
    """

    name = "ExtremeReversal"
    params = {
        "window": 20,
        "tail_pct": 0.1,
    }

    def __init__(self, window: int = 20, tail_pct: float = 0.1) -> None:
        super().__init__()
        self.window = int(window)
        self.tail_pct = float(tail_pct)
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if not (0 < self.tail_pct < 0.5):
            raise ValueError("tail_pct must be in (0, 0.5)")
        self.warmup_period = self.window + 1
        self._set_params(window=window, tail_pct=tail_pct)

    def get_output_name(self) -> str:
        tail_pct_str = f"{self.tail_pct:.2f}".replace("0.", "").rstrip("0") or "0"
        return f"ExtremeReversal_{self.window}_p{tail_pct_str}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)

        ret = close / close.shift(self.window) - 1.0

        rank = ret.rolling(
            window=self.window, min_periods=self.window
        ).rank(pct=True)

        signal = pd.Series(0, index=data.index, name=self.get_output_name())
        signal = signal.where(
            rank.notna(), np.nan
        )
        signal[rank >= 1.0 - self.tail_pct] = -1
        signal[rank <= self.tail_pct] = 1
        signal.name = self.get_output_name()
        return signal

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if not (0 < self.tail_pct < 0.5):
            raise ValueError("tail_pct must be in (0, 0.5)")
        if "close" not in data.columns:
            raise ValueError(
                f"ExtremeReversal requires column 'close', got {list(data.columns)}"
            )


class VolumeReversal(BaseFactor):
    """放量反转因子：反转信号 × 量比，放量反转更可靠。

    计算逻辑::

        reversal = −(close / close.shift(ret_window) − 1)
        vol_ratio = volume / volume.rolling(vol_window).mean()
        result = reversal * vol_ratio

    直觉：放量下跌后的反弹更可靠，放量上涨后的回调也更可靠。

    参数
    ----------
    ret_window : int
        反转收益计算窗口。
    vol_window : int
        量比计算的均线窗口。
    """

    name = "VolumeReversal"
    params = {
        "ret_window": 5,
        "vol_window": 20,
    }

    def __init__(self, ret_window: int = 5, vol_window: int = 20) -> None:
        super().__init__()
        self.ret_window = int(ret_window)
        self.vol_window = int(vol_window)
        if self.ret_window < 1:
            raise ValueError("ret_window must be at least 1")
        if self.vol_window < 1:
            raise ValueError("vol_window must be at least 1")
        # ret_window: reversal 从 index=ret_window 起有效
        # vol_window:  rolling(vol_window, min_periods=vol_window) 从 index=vol_window-1 起有效
        self.warmup_period = max(self.ret_window, self.vol_window)
        self._set_params(ret_window=ret_window, vol_window=vol_window)

    def get_output_name(self) -> str:
        return f"VolumeReversal_{self.ret_window}_{self.vol_window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        close = data["close"].astype(float)
        volume = data["volume"].astype(float)

        reversal = -(close / close.shift(self.ret_window) - 1.0)
        vol_ma = volume.rolling(window=self.vol_window, min_periods=self.vol_window).mean()
        vol_ratio = volume / vol_ma.replace(0.0, np.nan)

        result = reversal * vol_ratio
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.ret_window < 1:
            raise ValueError("ret_window must be at least 1")
        if self.vol_window < 1:
            raise ValueError("vol_window must be at least 1")
        for col in ("close", "volume"):
            if col not in data.columns:
                raise ValueError(
                    f"VolumeReversal requires column {col!r}, got {list(data.columns)}"
                )
