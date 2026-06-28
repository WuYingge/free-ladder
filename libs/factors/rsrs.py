from __future__ import annotations

import numpy as np
import pandas as pd

from factors.base_factor import BaseFactor
from factors.derived_factor import DerivedFactor


class RsrsFactor(BaseFactor):
    """
    RSRS 因子实现。

    计算流程分为四步：
    1. 在回归窗口内对 low 和 high 做线性回归，得到斜率 beta。
    2. 计算同一窗口内的相关系数平方，作为回归拟合优度 r_square。
    3. 可选地使用 beta * r_square 作为改进版 RSRS 分数，再做滚动 Z-Score 标准化。
    4. 按阈值将标准化结果映射为买入、观望、卖出信号。
    """
    name = "RSRS"
    params = {
        "regression_window": 18,
        "zscore_window": 200,
        "buy_threshold": 0.7,
        "sell_threshold": -0.7,
        "use_r2_adjustment": True,
        "output": "signal",
    }

    def __init__(
        self,
        regression_window: int = 18,
        zscore_window: int = 200,
        buy_threshold: float = 0.7,
        sell_threshold: float = -0.7,
        use_r2_adjustment: bool = True,
        output: str = "signal",
    ) -> None:
        # BaseFactor 内部会初始化依赖管理结构，因此这里必须先调用父类构造函数。
        super().__init__()

        # regression_window: 计算单日 RSRS 斜率时使用的 high/low 回归样本长度。
        self.regression_window = regression_window

        # zscore_window: 对 beta 或 beta*r_square 做历史标准化时使用的滚动窗口长度。
        self.zscore_window = zscore_window

        # buy_threshold / sell_threshold: 将标准化后的 RSRS 数值离散化为信号的上下阈值。
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        # use_r2_adjustment=True 时，使用 beta * r_square 作为改进版分数，
        # 这样可以削弱拟合质量较差的斜率带来的噪声。
        self.use_r2_adjustment = use_r2_adjustment

        # output 控制这个因子最终返回哪一层结果，既可返回最终 signal，
        # 也可返回中间结果用于调试或分析。
        self.output = output
        # 首个可用于交易决策的 zscore 需要先形成 regression_window 的 beta，
        # 再积累 zscore_window 个历史 score，因此 warm-up 需覆盖两段窗口。
        self.warmup_period = int(regression_window + zscore_window)
        self._set_params(
            regression_window=regression_window,
            zscore_window=zscore_window,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            use_r2_adjustment=use_r2_adjustment,
            output=output,
        )

    def get_output_name(self) -> str:
        return self._build_series_name()

    def __call__(self, data: pd.DataFrame) -> pd.Series:
        # 首先校验输入列和参数，避免在 rolling 计算中才暴露低质量错误。
        self._validate_input(data)

        # RSRS 定义只依赖 high 和 low，两列统一转成 float，
        # 以避免对象类型或整数类型在 rolling 运算中带来额外问题。
        low = data["low"].astype(float)
        high = data["high"].astype(float)

        # beta 是 high 对 low 回归后的斜率，代表支撑/阻力相对强度。
        beta = self._calculate_beta(low=low, high=high)

        # r_square 用于衡量该窗口内线性拟合的可信度。
        r_square = self._calculate_r_square(low=low, high=high)

        # 根据参数决定使用原始 beta，还是使用 beta * r_square 的修正版分数。
        score = beta * r_square if self.use_r2_adjustment else beta

        # 对分数做滚动标准化，使不同时间段的值更可比。
        zscore = self._calculate_zscore(score)

        # 最终把连续值映射为离散信号，便于回测或策略直接消费。
        signal = self._generate_signal(zscore)

        # output 提供多个层级的输出，便于既能做策略，也能做研究分析。
        if self.output == "beta":
            result = beta
        elif self.output == "r_square":
            result = r_square
        elif self.output == "score":
            result = score
        elif self.output == "zscore":
            result = zscore
        elif self.output == "signal":
            result = signal
        else:
            raise ValueError(
                "output must be one of: beta, r_square, score, zscore, signal"
            )

        result.name = self.get_output_name()
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        # RSRS 的数学定义要求 high 和 low 两列缺一不可。
        required_columns = {"high", "low"}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"RSRS requires columns: {sorted(required_columns)}, got missing {sorted(missing_columns)}"
            )

        # 回归窗口至少需要两个点，否则斜率没有意义。
        if self.regression_window < 2:
            raise ValueError("regression_window must be at least 2")

        # 标准化窗口至少需要两个点，否则标准差计算不成立。
        if self.zscore_window < 2:
            raise ValueError("zscore_window must be at least 2")

        # 下阈值必须严格小于上阈值，否则信号区间定义冲突。
        if self.sell_threshold >= self.buy_threshold:
            raise ValueError("sell_threshold must be smaller than buy_threshold")

    def _calculate_beta(self, low: pd.Series, high: pd.Series) -> pd.Series:
        # rolling_cov 对应公式中的协方差分子部分，
        # rolling_var 对应 low 的方差，也就是斜率公式的分母。
        rolling_cov = low.rolling(window=self.regression_window).cov(high)
        rolling_var = low.rolling(window=self.regression_window).var()

        # beta = cov(low, high) / var(low)
        # 当方差为 0 时，说明窗口内 low 没有波动，斜率不可定义，因此转成 NaN。
        beta = rolling_cov.divide(rolling_var.replace(0.0, np.nan))
        beta.name = f"{self.name}_beta"
        return beta

    def _calculate_r_square(self, low: pd.Series, high: pd.Series) -> pd.Series:
        # 对一元线性回归而言，决定系数 R^2 可以直接由相关系数的平方得到。
        # 这里用 rolling.corr 避免逐窗口手工回归，性能和表达都更直接。
        corr = low.rolling(window=self.regression_window).corr(high)
        r_square = corr.pow(2)
        r_square.name = f"{self.name}_r_square"
        return r_square

    def _calculate_zscore(self, score: pd.Series) -> pd.Series:
        # 严格历史标准化要求 t 时点的 Z-Score 只能使用 t 之前的历史分数，
        # 因此先整体向后移动一位，再对历史窗口做滚动均值和标准差。
        historical_score = score.shift(1)
        rolling_mean = historical_score.rolling(window=self.zscore_window).mean()
        rolling_std = historical_score.rolling(window=self.zscore_window).std()

        # 当标准差为 0 时，说明历史窗口内所有分数完全相同，
        # 此时无法标准化，结果应为 NaN 而不是无穷大。
        zscore = (score - rolling_mean).divide(rolling_std.replace(0.0, np.nan))
        zscore.name = f"{self.name}_zscore"
        return zscore

    def _generate_signal(self, zscore: pd.Series) -> pd.Series:
        # 默认信号为 0，表示中性或维持当前状态。
        signal = pd.Series(0, index=zscore.index, dtype="int64")

        # 当 Z-Score 高于上阈值时给出买入信号，
        # 低于下阈值时给出卖出信号，介于两者之间保持为 0。
        signal = signal.mask(zscore > self.buy_threshold, 1)
        signal = signal.mask(zscore < self.sell_threshold, -1)
        signal.name = self.name
        return signal

    def _build_series_name(self) -> str:
        # 输出名称中附带 output 类型、回归窗口和是否使用 R^2 修正，
        # 方便在同一 DataFrame 中同时比较多个 RSRS 变体。
        suffix = "adj" if self.use_r2_adjustment else "raw"
        return f"{self.name}_{self.output}_reg{self.regression_window}_z{self.zscore_window}_{suffix}"


class RsrsDerivedFactor(DerivedFactor):
    name = "RsrsDerived"
    params = {
        "regression_window": 18,
        "zscore_window": 200,
        "buy_threshold": 0.7,
        "sell_threshold": -0.7,
        "use_r2_adjustment": True,
        "short_ema_span": 5,
        "long_ema_span": 20,
        "output": "signal_duration",
    }

    def __init__(
        self,
        regression_window: int = 18,
        zscore_window: int = 200,
        buy_threshold: float = 0.7,
        sell_threshold: float = -0.7,
        use_r2_adjustment: bool = True,
        short_ema_span: int = 5,
        long_ema_span: int = 20,
        output: str = "signal_duration",
    ) -> None:
        super().__init__()
        self.regression_window = int(regression_window)
        self.zscore_window = int(zscore_window)
        self.buy_threshold = float(buy_threshold)
        self.sell_threshold = float(sell_threshold)
        self.use_r2_adjustment = use_r2_adjustment
        self.short_ema_span = int(short_ema_span)
        self.long_ema_span = int(long_ema_span)
        self.output = output
        self.signal_factor = RsrsFactor(
            regression_window=self.regression_window,
            zscore_window=self.zscore_window,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
            use_r2_adjustment=self.use_r2_adjustment,
            output="signal",
        )
        self.zscore_factor = RsrsFactor(
            regression_window=self.regression_window,
            zscore_window=self.zscore_window,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
            use_r2_adjustment=self.use_r2_adjustment,
            output="zscore",
        )
        self.add_dependency(self.signal_factor)
        self.add_dependency(self.zscore_factor)
        self.warmup_period = int(
            self.regression_window + self.zscore_window + max(self.long_ema_span - 1, 0)
        )
        self._set_params(
            regression_window=regression_window,
            zscore_window=zscore_window,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            use_r2_adjustment=use_r2_adjustment,
            short_ema_span=short_ema_span,
            long_ema_span=long_ema_span,
            output=output,
        )

    def get_output_name(self) -> str:
        suffix = "adj" if self.use_r2_adjustment else "raw"
        buy_threshold = self._format_param_token(self.buy_threshold)
        sell_threshold = self._format_param_token(self.sell_threshold)
        return (
            f"{self.name}_{self.output}_reg{self.regression_window}_z{self.zscore_window}"
            f"_buy{buy_threshold}_sell{sell_threshold}_ema{self.short_ema_span}_{self.long_ema_span}_{suffix}"
        )

    def get_source_columns(self) -> tuple[str, ...]:
        return ("close",)

    def get_dependency_column_map(self) -> dict[BaseFactor, str]:
        return {
            self.signal_factor: "signal",
            self.zscore_factor: "zscore",
        }

    def compute_from_frame(self, frame: pd.DataFrame) -> pd.Series:
        self._validate_params()

        signal = pd.to_numeric(frame["signal"], errors="coerce")
        zscore = pd.to_numeric(frame["zscore"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")

        first_buy_mask, entry_bar_position, entry_close = self._resolve_entry_state(
            signal=signal,
            close=close,
        )

        if self.output == "signal_duration":
            current_position = pd.Series(
                np.arange(len(frame), dtype=float),
                index=frame.index,
                dtype=float,
            )
            result = (current_position - entry_bar_position).where(entry_bar_position.notna())
        elif self.output == "zscore_trend":
            short_ema = zscore.ewm(span=self.short_ema_span, adjust=False).mean()
            long_ema = zscore.ewm(span=self.long_ema_span, adjust=False).mean()
            result = (short_ema - long_ema).where(zscore.notna())
        elif self.output == "return_since_buy":
            result = (close.divide(entry_close) - 1.0).where(entry_close.notna())
        else:
            raise ValueError(
                "output must be one of: signal_duration, zscore_trend, return_since_buy"
            )

        result.name = self.get_output_name()
        return result

    def _resolve_entry_state(
        self,
        signal: pd.Series,
        close: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        segment = signal.eq(-1).cumsum()
        buy_signal = signal.eq(1)
        first_buy_mask = buy_signal & (buy_signal.groupby(segment).cumsum() == 1)

        bar_position = pd.Series(
            np.arange(len(signal), dtype=float),
            index=signal.index,
            dtype=float,
        )
        entry_bar_position = bar_position.where(first_buy_mask).groupby(segment).ffill()
        entry_close = close.where(first_buy_mask).groupby(segment).ffill()
        return first_buy_mask, entry_bar_position, entry_close

    def _validate_params(self) -> None:
        if self.short_ema_span < 1:
            raise ValueError("short_ema_span must be at least 1")
        if self.long_ema_span < 1:
            raise ValueError("long_ema_span must be at least 1")
        if self.sell_threshold >= self.buy_threshold:
            raise ValueError("sell_threshold must be smaller than buy_threshold")

    def _format_param_token(self, value: float) -> str:
        token = f"{value:g}"
        return token.replace("-", "m").replace(".", "p")
    