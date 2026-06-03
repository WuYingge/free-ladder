# Project Instructions

This file provides context for AI assistants working on this project.

## Project Type: Python

### Commands
- Install: `pip install -e .`
- Test: `pytest libs/`
- Lint: `ruff check libs/`

### Documentation
See README.md for project overview.

### Version Control
This project uses Git. See .gitignore for excluded files.

## Agent Guidance
- **Language**: Prefer Simplified Chinese, never Japanese
- **CodeWhale reads this file as:** AGENTS.md
- **Priority reading order:** README → pyproject.toml → this file (AGENTS.md) → relevant subsystem files in `libs/` → notebook entry points (only when task is workflow-facing)
- **Approach:** Start each task by mapping the request to known subsystems before broad searching. Prefer editing within the smallest subsystem boundary that satisfies the request. When uncertain, verify with source files and update assumptions.
- **Read-only surface:** `data/` is storage for inputs and generated outputs. `data/backtest_results/` is output-only research artifacts — do not use them as source of truth for new business logic.
- **Never edit:** Files generated or owned by akshare/backtrader; `data/etf_data/*.csv` directly with `pd.read_csv` — always wrap via `EtfData` or `libs/data_manager/**`.
- **Always test with:** `pytest libs/` (run relevant test file for the changed subsystem)
- **Reuse mindset:** When you write code or logic that could be reused elsewhere, extract and settle it into `libs/` proactively.
- **Before creating a new script:** Check `libs/scripts/` first — any "run this analysis" task likely already has a CLI entry point there. Do not create one-off scripts in the project root.

## Architecture

A quantitative investment analysis toolkit. `libs/` contains reusable implementation, `notebooks/` contains daily interactive workflows, `data/` is storage for inputs and outputs.

### Entry Points
- `notebooks/single_symbol_timing_framework.ipynb` — 单标的择时回测主流程
- `notebooks/single_factor_backtest.ipynb` — 单标的/单因子实验
- `notebooks/dailyUpdate.ipynb` — 每日数据更新
- `notebooks/portfolio_backtest.ipynb` — 组合回测
- `libs/scripts/` — CLI 可执行入口："跑回测/导出数据/扫描因子"类任务优先查此目录，按文件名匹配
  - `run_wide_momentum_baseline.py` — 宽动量基线回测
  - `run_trend_r2_scan.py` — 趋势 R² 扫描
  - `discover_three_asset_extensions.py` — 三资产低相关组合搜索
  - `export_rsrs_local_etfs.py` — 导出 RSRS 因子
  - `tranfer_etf_columns.py` — ETF 列格式转换
  - `rename_data.py` — 数据文件重命名

### Key Modules

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| `libs/backtesting/` | Backtrader feeds, strategies, batch execution, performance | `engine.py`, `timing_batch.py`, `data.py`, `performance.py`, `strategies/` |
| `libs/factors/` | Signal generation (timing & portfolio) | `base_factor.py`, `rsrs.py`, `new_high.py`, `average_true_range.py`, `portfolio/` |
| `libs/core/models/` | Typed wrappers: `EtfData`, `IndexDailyData`, `FinancialData` | `data_base.py`, `etf_daily_data.py`, `daily_quote_data.py` |
| `libs/data_manager/` | Persist, update, load ETF/index CSV datasets | `etf_data_manager.py`, `index_data_manager.py`, `providers/` |
| `libs/fetcher/` | Fetch ETF/index market data from EastMoney / Akshare | `etf.py`, `index.py`, `utils.py` |
| `libs/proxy/` | Proxy pool management for data fetching | `proxy.py` |
| `libs/config.py` | Shared paths and env settings (`DataPath`) | `config.py` |

### Data Flow

**Timing Backtest Flow:**
1. Load ETF CSV via `EtfData` / `data_manager` (never `pd.read_csv` directly).
2. Build normalized OHLCV DataFrame in `libs/backtesting/data.py`.
3. Compute factor signal via `BaseFactor` subclass (e.g. `NewHigh`, `RsrsFactor`).
4. Inject signal into Backtrader data feed.
5. Execute strategy in `libs/backtesting/engine.py`.
6. Aggregate batch metrics and persist results via `libs/backtesting/timing_batch.py` → `data/backtest_results/`.

**Data Update Flow:**
1. Pull latest data using `libs/fetcher/etf.py` / `libs/fetcher/index.py`.
2. Update or create symbol CSVs via `libs/data_manager/etf_data_manager.py`.
3. Orchestrate via notebooks (`dailyUpdate.ipynb`, `fetcher.ipynb`).

### Architecture Rules (from `.github/instructions/etf-data-architecture.instructions.md`)

- **ETF data access:** Always use `core.models.etf_daily_data.EtfData` or `libs/data_manager/**` APIs (`get_etf_data_by_symbol`, `get_etf_data_by_symbols`, `etf_data_iter`). Never `pd.read_csv` on `data/etf_data/*.csv` anywhere — libs, notebooks, playgrounds, scripts.
- **Factor development:** Keep single-symbol factors under `libs/factors/`, inherit from `BaseFactor`.
- **Backtesting:** Keep Backtrader orchestration in `libs/backtesting/`. Reusable data abstractions in `libs/core/models/`. One-off execution entry points in `libs/scripts/`; move reusable logic back into `libs/`.
- **Constants:** For `data/const/` datasets, prefer provider/wrapper classes in `libs/data_manager/providers/` over hard-coded file paths when a provider already exists.
- **Notebooks:** Keep notebooks thin — orchestrate library code, load data through `EtfData`/`data_manager`, put reusable calculations/factors/helpers in `libs/`.
- **Backtest results:** Treat `data/backtest_results/` as output-only artifacts, not input for business logic.

## Dependency Highlights

- `libs/backtesting/` depends on `libs/factors/` through `BaseFactor`.
- `libs/backtesting/data.py` loaders depend on `libs/data_manager/` conventions.
- `timing_batch` requires picklable strategy and data feed classes for multiprocessing.
- Provider singletons depend on env path configuration from `libs/config.py`.

## High Value Symbols

Key functions, classes, and entry points to be aware of:

- `run_single_factor_single_target_backtest` — single-symbol single-factor backtest runner
- `run_timing_backtest_batch` — batch timing backtest orchestrator
- `build_bt_feed_dataframe` — construct Backtrader feed from OHLCV data
- `update_etf_data` / `batch_acquire_etf_data` — ETF data persistence / bulk fetch
- `NewHigh` — new-high breakout factor (inherits `BaseFactor`)
- `Portfolio` — position-level portfolio analytics
- **CLI Entry Points (按需执行的脚本):**
  - `libs/scripts/run_wide_momentum_baseline.py --min-momentum-value 0` — 宽动量基线回测
  - `libs/scripts/run_trend_r2_scan.py` — 趋势 R² 扫描
  - `libs/scripts/discover_three_asset_extensions.py` — 三资产低相关组合搜索
  - `libs/scripts/export_rsrs_local_etfs.py` — 导出 RSRS 因子到本地

## Open Questions

- Canonical process for generating provider backing files is not fully documented.
- Proxy fallback behavior without proxy credentials is unclear.
- Notebook-driven workflows are primary; CLI standardization is minimal.

## Cache Stability

<!-- DeepSeek V4 uses a byte-stable prefix cache (128-token granularity). -->
<!-- Keeping these things stable turn-over-turn saves ~90% on input tokens. -->

- **Frequently-rebuilt files:** `uv.lock`, `data/backtest_results/**`, `__pycache__/`
- **Stable scaffolding:** `AGENTS.md`, `pyproject.toml`, `README.md`, `.github/**`, `libs/config.py`
- **Append, don't reorder:** New context goes at the end of the request; reordering invalidates cache

## Guidelines

- Follow existing code style and patterns (type hints, `from __future__ import annotations`, dataclasses with `slots=True`)
- Write tests for new functionality in `libs/backtesting/tests/` or `libs/proxy/tests/`
- Keep changes focused and atomic; update this file (AGENTS.md) when architecture changes
- Document public APIs with docstrings
- Update this file when project conventions change
- Treat notebooks as runtime entry points and `libs/` as reusable implementation modules
- Do not infer architecture beyond what is documented in code and the repo knowledge base
- Write code with necessary comments in critical steps; extract reusable logic into `libs/`
