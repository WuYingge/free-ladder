import numpy as np
import pandas as pd
from factors.base_factor import BaseFactor


def calculate_slope(series):
    if len(series) < 2:
        return np.nan
    x = np.arange(len(series))
    y = series.values
    # 线性回归：y = mx + b
    m, b = np.polyfit(x, y, 1)
    return m

def cal_n_day_linear_slope(series, n=20):
    return series.rolling(window=n).apply(calculate_slope, raw=False)

def cal_volatility_n_day(series: pd.Series, n=20) -> float:
    trading_days = 252
    volatility_series = (series/series.shift(1)).apply(np.log).rolling(window=n).std() * np.sqrt(trading_days)
    return volatility_series.tail(1).values[0]

def cal_predict_increase_percent(df, slope_name):
    df[f"predict_interest_{slope_name}"] = df[slope_name] / df["close"] * 100
    

def cal_ma_n(df, n=20, price_col="close"):
    df[f"ma{n}"] = df["close"].rolling(window=n).mean()
    
def cal_max_loss_of_ma_n(df, n=20):
    cur = (df["close"] - df[f"ma{n}"]) / df["close"] * 100
    df[f"max_loss_ma{n}"] = cur.apply(lambda x: x if x > 0 else np.inf)

def cal_ratio_of_interest_risk(df: pd.DataFrame, k=1, n=20) -> float:
    use_data = df.tail(1)
    c = cal_volatility_n_day(df["close"], n) * 0.5
    slope = use_data[f"predict_interest_{n}_slope"].values[0]
    max_loss = use_data[f"max_loss_ma{n}"].values[0]
    return k*slope / (abs(max_loss) + c)

def cal_one_etf(df: pd.DataFrame, k=1, n=20) -> float:
    slope_name = f"{n}_slope"
    df[slope_name] = cal_n_day_linear_slope(df["close"])
    cal_predict_increase_percent(df, slope_name)
    cal_ma_n(df)
    cal_max_loss_of_ma_n(df)
    return cal_ratio_of_interest_risk(df, k)
    

class MAPosition(BaseFactor):
    """close 相对 MA 的位置：(close - MA) / close。

    正值 = 在 MA 上方，负值 = 下方。
    用于 ThresholdFilter: field="MAPosition_close_200", operator=">", value=0。
    """

    name = "MAPosition"
    params = {
        "window": 200,
        "price_column": "close",
    }

    def __init__(self, window: int = 200, price_column: str = "close") -> None:
        super().__init__()
        self.window = int(window)
        self.price_column = price_column
        self.warmup_period = self.window
        self._set_params(window=window, price_column=price_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.price_column}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.price_column not in data.columns:
            raise ValueError(
                f"MAPosition requires column '{self.price_column}'"
            )
        price = data[self.price_column].astype(float)
        ma = price.rolling(window=self.window).mean()
        result = (price - ma) / price
        result.name = self.get_output_name()
        return result


class SlopeRiskRatio(BaseFactor):
    name = "SlopeRiskRatio"
    params = {}

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, data: pd.DataFrame) -> pd.Series:
        data["ratio"] = cal_one_etf(data)
        return data["ratio"]


class MAFactor(BaseFactor):
    name = "MA"
    params = {
        "window": 20,
        "price_column": "close",
    }

    def __init__(self, window: int = 20, price_column: str = "close") -> None:
        super().__init__()
        self.window = int(window)
        self.price_column = price_column
        self.warmup_period = self.window
        self._set_params(window=window, price_column=price_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.price_column}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        result = data[self.price_column].astype(float).rolling(window=self.window).mean()
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.price_column not in data.columns:
            raise ValueError(
                f"MAFactor requires column '{self.price_column}', got columns {list(data.columns)}"
            )


# ============================================================================
# 均线与偏离族扩展因子
# ============================================================================


class BIAS(BaseFactor):
    """乖离率因子：(close − MA) / MA，百分比偏离。

    BIAS 为正 = 收盘价高于均线（超买/强势），为负 = 低于均线（超卖/弱势）。
    极端乖离暗示均值回归可能。

    与 MAPosition 的区别：MAPosition 分母是 close，BIAS 分母是 MA。
    BIAS 更适合与传统乖离率概念（如 5 日乖离率）对齐。

    参数
    ----------
    window : int
        均线窗口，默认 20。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "BIAS"
    params = {
        "window": 20,
        "price_column": "close",
    }

    def __init__(self, window: int = 20, price_column: str = "close") -> None:
        super().__init__()
        self.window = int(window)
        self.price_column = price_column
        self.warmup_period = self.window
        self._set_params(window=window, price_column=price_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.price_column}_{self.window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if self.window < 1:
            raise ValueError("window must be at least 1")
        if self.price_column not in data.columns:
            raise ValueError(
                f"BIAS requires column '{self.price_column}'"
            )
        price = data[self.price_column].astype(float)
        ma = price.rolling(window=self.window).mean()
        result = (price - ma) / ma
        result.name = self.get_output_name()
        return result


class BollingerBandPosition(BaseFactor):
    """布林带位置因子：(close − MA) / (k × std)，波动率标准化后的偏离。

    值 ≈ 0 → close 在 MA 附近。
    值 ≈ 1 / −1 → 触及上/下轨（当 k 取默认值 2 时）。
    标准差用 std(ddof=1)（样本标准差）。

    参数
    ----------
    window : int
        均线/标准差窗口，默认 20。
    k : float
        标准差倍数，默认 2.0（经典布林带参数）。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "BollingerBandPosition"
    params = {
        "window": 20,
        "k": 2.0,
        "price_column": "close",
    }

    def __init__(
        self,
        window: int = 20,
        k: float = 2.0,
        price_column: str = "close",
    ) -> None:
        super().__init__()
        self.window = int(window)
        self.k = float(k)
        self.price_column = price_column
        self.warmup_period = self.window
        self._set_params(window=window, k=k, price_column=price_column)

    def get_output_name(self) -> str:
        return f"{self.name}_{self.price_column}_{self.window}_{self.k}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.price_column not in data.columns:
            raise ValueError(
                f"BollingerBandPosition requires column '{self.price_column}'"
            )
        price = data[self.price_column].astype(float)
        ma = price.rolling(window=self.window).mean()
        std = price.rolling(window=self.window).std(ddof=1)
        denom = self.k * std
        denom_safe = denom.replace(0.0, np.nan)
        result = (price - ma) / denom_safe
        result.name = self.get_output_name()
        return result


class MAAlignment(BaseFactor):
    """均线排列因子：短/中/长三条 MA 的相对排列关系，输出连续值 [-1, 1]。

    计算方式：对每日三条 MA 值从小到大排序（ascending rank），
    得分 = (rank_shortest − rank_longest) / 2。

    * +1：完全多头排列（短期 > 中期 > 长期）
    * −1：完全空头排列（短期 < 中期 < 长期）
    * 0：三条 MA 交叉或纠缠

    注意：mid 窗口不影响分数，但必须提供以固定三条 MA 的排序框架。

    参数
    ----------
    windows : list[int]
        三条 MA 的窗口列表，需恰好 3 个，默认 [5, 20, 60]。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "MAAlignment"
    params = {
        "windows": [5, 20, 60],
        "price_column": "close",
    }

    def __init__(
        self,
        windows: list[int] | None = None,
        price_column: str = "close",
    ) -> None:
        super().__init__()
        if windows is None:
            windows = [5, 20, 60]
        self.windows = sorted(int(w) for w in windows)
        if len(self.windows) != 3:
            raise ValueError(
                f"MAAlignment requires exactly 3 windows, got {len(self.windows)}"
            )
        self.price_column = price_column
        self.warmup_period = max(self.windows)
        self._set_params(windows=self.windows, price_column=price_column)

    def get_output_name(self) -> str:
        w_str = "_".join(str(w) for w in self.windows)
        return f"{self.name}_{self.price_column}_{w_str}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        price = data[self.price_column].astype(float)

        mas: dict[int, pd.Series] = {}
        for w in self.windows:
            mas[w] = price.rolling(window=w).mean()

        ma_df = pd.DataFrame({w: mas[w] for w in self.windows})
        # ascending: 最小值 rank=1，最大值 rank=3
        ranks = ma_df.rank(axis=1, ascending=True)
        w_min = min(self.windows)
        w_max = max(self.windows)
        result = (ranks[w_min] - ranks[w_max]) / 2.0
        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.price_column not in data.columns:
            raise ValueError(
                f"MAAlignment requires column '{self.price_column}'"
            )


class MASlope(BaseFactor):
    """均线斜率因子：MA 自身的 N 日变化率，衡量趋势在加速还是减速。

    计算方式：先算 MA(ma_window)，再计算其 pct_change(slope_window)。

    * 正值 → 均线向上，趋势延续
    * 负值 → 均线向下，趋势转弱
    * 绝对值增大 → 趋势在加速；减小 → 趋势在减速（钝化）

    参数
    ----------
    ma_window : int
        用于计算均线的窗口，默认 20。
    slope_window : int
        用于计算斜率（变化率）的窗口，默认 5。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "MASlope"
    params = {
        "ma_window": 20,
        "slope_window": 5,
        "price_column": "close",
    }

    def __init__(
        self,
        ma_window: int = 20,
        slope_window: int = 5,
        price_column: str = "close",
    ) -> None:
        super().__init__()
        self.ma_window = int(ma_window)
        self.slope_window = int(slope_window)
        self.price_column = price_column
        self.warmup_period = self.ma_window + self.slope_window
        self._set_params(
            ma_window=ma_window, slope_window=slope_window, price_column=price_column
        )

    def get_output_name(self) -> str:
        return f"{self.name}_{self.price_column}_{self.ma_window}_{self.slope_window}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if self.ma_window < 1:
            raise ValueError("ma_window must be at least 1")
        if self.slope_window < 1:
            raise ValueError("slope_window must be at least 1")
        if self.price_column not in data.columns:
            raise ValueError(
                f"MASlope requires column '{self.price_column}'"
            )
        price = data[self.price_column].astype(float)
        ma = price.rolling(window=self.ma_window).mean()
        result = ma.pct_change(periods=self.slope_window)
        result.name = self.get_output_name()
        return result


class MADistance(BaseFactor):
    """均线距离因子：(MA_short − MA_long) / close，金叉死叉的连续版本。

    * 正值 → 短期均线在长期均线上方（金叉状态）
    * 负值 → 短期均线在长期均线下方（死叉状态）
    * 绝对值大 → 均线发散；绝对值小 → 均线收敛

    参数
    ----------
    short_window : int
        短期均线窗口，默认 5。
    long_window : int
        长期均线窗口，默认 60。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "MADistance"
    params = {
        "short_window": 5,
        "long_window": 60,
        "price_column": "close",
    }

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 60,
        price_column: str = "close",
    ) -> None:
        super().__init__()
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.price_column = price_column
        self.warmup_period = max(self.short_window, self.long_window)
        self._set_params(
            short_window=short_window, long_window=long_window, price_column=price_column
        )

    def get_output_name(self) -> str:
        return (
            f"{self.name}_{self.price_column}_{self.short_window}_{self.long_window}"
        )

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        if self.short_window < 1 or self.long_window < 1:
            raise ValueError("windows must be at least 1")
        if self.price_column not in data.columns:
            raise ValueError(
                f"MADistance requires column '{self.price_column}'"
            )
        price = data[self.price_column].astype(float)
        ma_short = price.rolling(window=self.short_window).mean()
        ma_long = price.rolling(window=self.long_window).mean()
        result = (ma_short - ma_long) / price
        result.name = self.get_output_name()
        return result


class MADispersion(BaseFactor):
    """均线离散度因子：多条 MA 的标准差除以均值，度量均线粘合/发散程度。

    公式: std(MA_list, ddof=1) / mean(MA_list)

    * 值接近 0 → 均线高度粘合（收敛），通常为变盘前兆
    * 值增大 → 均线发散，趋势正在展开

    参数
    ----------
    windows : list[int]
        多条均线的窗口列表，默认 [5, 10, 20, 60]。
    price_column : str
        价格列名，默认 "close"。
    """

    name = "MADispersion"
    params = {
        "windows": [5, 10, 20, 60],
        "price_column": "close",
    }

    def __init__(
        self,
        windows: list[int] | None = None,
        price_column: str = "close",
    ) -> None:
        super().__init__()
        if windows is None:
            windows = [5, 10, 20, 60]
        self.windows = [int(w) for w in windows]
        if len(self.windows) < 2:
            raise ValueError("MADispersion requires at least 2 windows")
        self.price_column = price_column
        self.warmup_period = max(self.windows)
        self._set_params(windows=self.windows, price_column=price_column)

    def get_output_name(self) -> str:
        w_str = "_".join(str(w) for w in self.windows)
        return f"{self.name}_{self.price_column}_{w_str}"

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        self._validate_input(data)
        price = data[self.price_column].astype(float)

        mas: list[pd.Series] = []
        for w in self.windows:
            mas.append(price.rolling(window=w).mean())

        ma_stack = np.stack([s.values for s in mas], axis=1)
        ma_std = np.nanstd(ma_stack, axis=1, ddof=1)
        ma_mean = np.nanmean(ma_stack, axis=1)

        # 除零保护
        result_arr = np.full(len(price), np.nan, dtype=float)
        safe_mask = ma_mean > 0.0
        result_arr[safe_mask] = ma_std[safe_mask] / ma_mean[safe_mask]

        result = pd.Series(result_arr, index=data.index, name=self.get_output_name())
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if self.price_column not in data.columns:
            raise ValueError(
                f"MADispersion requires column '{self.price_column}'"
            )
