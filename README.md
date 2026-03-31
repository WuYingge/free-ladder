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

