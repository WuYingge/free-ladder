# Repository Structure Knowledge Base

## Repository Snapshot
- libs is the main code root.
- data stores ETF CSV inputs and backtest outputs.
- notebooks are the primary workflow entry points.
- playgrounds contains exploratory notebooks.
- config is environment-driven via libs/config.
- no dedicated automated test suite is currently present.

## Subsystems

### Configuration
- Responsibility: shared paths and env settings.
- Key file: libs/config.py

### Data Fetching
- Responsibility: fetch ETF market and history data from EastMoney and Akshare.
- Key files:
  - libs/fetcher/etf.py
  - libs/fetcher/utils.py
  - libs/proxy/proxy.py

### Data Management
- Responsibility: persist, update, and load ETF CSV datasets.
- Key files:
  - libs/data_manager/etf_data_manager.py
  - libs/data_manager/providers/etf_list_provider.py
  - libs/data_manager/providers/cluster_provider.py

### Core Models
- Responsibility: typed wrappers and reusable financial data behavior.
- Key files:
  - libs/core/models/data_base.py
  - libs/core/models/etf_daily_data.py
  - libs/core/models/calandar_df.py

### Factors
- Responsibility: signal generation for timing and portfolio analytics.
- Key files:
  - libs/factors/base_factor.py
  - libs/factors/new_high.py
  - libs/factors/average_true_range.py
  - libs/factors/portfolio/cluster_analysis.py
  - libs/factors/portfolio/correlation.py

### Backtesting
- Responsibility: transform data into Backtrader feeds, run strategies, batch execution, and performance metrics.
- Key files:
  - libs/backtesting/data.py
  - libs/backtesting/single_factor_single_target_strategy.py
  - libs/backtesting/engine.py
  - libs/backtesting/performance.py
  - libs/backtesting/timing_batch.py

### Portfolio Analytics
- Responsibility: position-level analysis, correlation checks, ATR sizing, cluster distribution.
- Key file:
  - libs/core/portfolios/portfolio.py

### Utility Scripts
- Responsibility: one-off data cleanup and conversion.
- Key files:
  - libs/scripts/rename_data.py
  - libs/scripts/tranfer_etf_columns.py

## Data Flow

### Timing Backtest Flow
1. Load ETF CSV data from data/etf_data.
2. Build normalized OHLCV dataframe in libs/backtesting/data.py.
3. Compute factor signal via BaseFactor implementation, commonly NewHigh.
4. Inject signal into Backtrader data feed.
5. Execute strategy in libs/backtesting/engine.py.
6. Aggregate batch metrics and persist results in libs/backtesting/timing_batch.py to data/backtest_results.

### Data Update Flow
1. Pull latest data using libs/fetcher/etf.py.
2. Update or create symbol CSVs via libs/data_manager/etf_data_manager.py.
3. Use notebooks for scheduled/manual orchestration.

## Important Entry Points
- notebooks/single_symbol_timing_framework.ipynb
- notebooks/single_factor_backtest.ipynb
- notebooks/dailyUpdate.ipynb
- notebooks/fetcher.ipynb
- libs/scripts/rename_data.py
- libs/scripts/tranfer_etf_columns.py

## High Value Symbols
- run_single_factor_single_target_backtest
- run_timing_backtest_batch
- build_bt_feed_dataframe
- update_etf_data
- batch_acquire_etf_data
- NewHigh
- Portfolio

## Dependency Highlights
- backtesting depends on factors through BaseFactor.
- backtesting data loaders depend on data_manager conventions.
- timing_batch requires picklable strategy and data feed classes for multiprocessing.
- portfolio analytics depend on provider singletons and factor outputs.
- provider singletons depend on env path configuration.

## Open Questions
- canonical process for generating provider backing files is not fully documented.
- proxy fallback behavior without proxy credentials is unclear.
- notebook-driven workflows are primary; CLI standardization is minimal.