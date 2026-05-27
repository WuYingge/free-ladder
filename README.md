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

## 代理配置

数据抓取默认通过 `libs/proxy/proxy.py` 统一取代理，现已支持按供应商切换：

- `PROXY_PROVIDER`：供应商名称，当前内置 `hailiang`、`luotuo` 和 `luotuo_static`
- `PROXY_API_URL`：通用代理提取地址，适合新供应商
- `PROXY_ENCRYPT_URL`：兼容旧版海量代理加密地址
- `PROXY_UNBIND_TIME`：代理有效时长，单位秒
- `PROXY_STATIC_TTL_SECONDS`：静态 IP 批次的默认有效期，单位秒；当接口不返回单个 IP 的过期时间时生效
- `PROXY_STATIC_FAILURE_COOLDOWN_SECONDS`：静态 IP 失败后的冷却时间，单位秒；冷却结束前不优先复用，但不会被移出池子
- `PROXY_SHARED_CACHE_DIR`：静态 IP 的跨进程共享缓存目录；同一批静态 IP 会在这里复用，直到过期才重新请求供应商 API

当前解析器支持两类常见返回：

- `{"code": 0 | "0", "data": [{"ip": "...", "port": "..."}]}`
- `{"code": 0 | "0", "data": "ip:port\nip:port"}`

`luotuo_static` 会把代理视为可复用静态 IP：单次使用后不会丢弃，只会在超过有效期后淘汰；如果接口返回 `expire` 字段则按实际过期时间处理，否则使用 `PROXY_STATIC_TTL_SECONDS` 作为默认有效期。

如果使用骆驼代理这类白名单模式接口（例如 `isAuth=false`），还需要先在供应商后台把当前出口 IP 加入白名单，否则会返回 `code=-3`。

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

## 低相关组合搜索脚本

- `libs/scripts/discover_three_asset_extensions.py`：从 `INDEX_KEYWORDS` 中为每个关键词挑选代表 ETF，并基于基础池搜索低相关扩展组合。
- 使用说明见 [docs/discover_three_asset_extensions.md](docs/discover_three_asset_extensions.md)

