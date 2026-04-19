# Invest Toolkit

## Backtrader 回测

已新增基于 `backtrader` 的回测能力，支持直接复用 `libs/factors` 中的因子信号（当前示例接入 `NewHigh`）。

### 核心模块

- `libs/backtesting/data.py`：读取 `data/etf_data/*.csv` 并构造 backtrader feed 数据。
- `libs/backtesting/single_factor_single_target_strategy.py`：基于信号线交易（`>=1` 买入，`<=-1` 卖出），用于单因子单标的。
- `libs/backtesting/engine.py`：回测运行器，返回收益、夏普、回撤、交易统计。
- `libs/backtesting/timing_batch.py`：批量编排与并行回测，供 notebook 统一调用。

### 快速开始

当前仅保留 notebook 入口：

- `notebooks/single_symbol_timing_framework.ipynb`：单标的择时主流程（筛选、因子计算、批量回测、结果落盘）。
- `notebooks/single_factor_backtest.ipynb`：单标的/单因子实验 notebook。

## 多标的组合回测模式（新增）

支持输入一批标的的 DataFrame（已包含 OHLCV + 预计算因子列），通过函数式策略输出目标权重，执行组合统一回测。

### 入口

- `libs/backtesting/timing_batch.py` 中的 `run_portfolio_backtest_batch`
- 配置对象：`PortfolioBatchConfig`

### 输入约定

- `symbol_feed_map: dict[str, pd.DataFrame]`
- 每个 DataFrame 至少包含列：`open`、`high`、`low`、`close`、`volume`
- 可包含任意因子列，策略函数按列名读取

### 策略函数签名

- `signal(snapshot: pd.DataFrame, context: dict[str, Any], **kwargs) -> dict[str, float]`
- 返回值为目标权重字典：`{symbol: target_weight}`
- 默认由引擎执行权重约束并根据总仓位上限缩放

### 最小示例

```python
from backtesting import (
	PortfolioBatchConfig,
	run_portfolio_backtest_batch,
	example_equal_weight_momentum_signal,
)

config = PortfolioBatchConfig(
	cash=100000,
	commission=0.0005,
	slippage_perc=0.0002,
	rebalance_interval=1,
	max_gross_exposure=0.95,
	strategy_kwargs={
		"momentum_column": "mom_20",
		"min_score": 0.0,
		"max_positions": 5,
	},
	output_dir="data/backtest_results/portfolio_mode_demo",
)

summary_df, errors, equity_curves = run_portfolio_backtest_batch(
	symbol_feed_map=symbol_feed_map,  # 每个 symbol 对应一个含因子列的 DataFrame
	strategy_callable=example_equal_weight_momentum_signal,
	config=config,
)
```

