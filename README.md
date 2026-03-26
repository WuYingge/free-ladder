# Invest Toolkit

## Backtrader 回测

已新增基于 `backtrader` 的回测能力，支持直接复用 `libs/factors` 中的因子信号（当前示例接入 `NewHigh`）。

### 核心模块

- `libs/backtesting/data.py`：读取 `data/etf_data/*.csv` 并构造 backtrader feed 数据。
- `libs/backtesting/single_factor_single_target_strategy.py`：基于信号线交易（`>=1` 买入，`<=-1` 卖出），用于单因子单标的。
- `libs/backtesting/engine.py`：回测运行器，返回收益、夏普、回撤、交易统计。
- `libs/scripts/run_single_factor_single_target_backtest.py`：单因子单标的命令行示例入口。

### 快速开始

在项目根目录执行：

```bash
python libs/scripts/run_single_factor_single_target_backtest.py --symbol 159915
```

可调参数：

```bash
python libs/scripts/run_single_factor_single_target_backtest.py \
	--symbol 159915 \
	--cash 100000 \
	--commission 0.0005 \
	--stake 100 \
	--high-window 50 \
	--low-window 25
```

如需指定数据目录：

```bash
python libs/scripts/run_single_factor_single_target_backtest.py --symbol 159915 --data-dir data/etf_data
```

