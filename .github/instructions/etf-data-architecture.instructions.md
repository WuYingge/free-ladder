---
description: "Use when writing Python modules or notebooks for ETF data access, factor development, backtesting, data_manager workflows, or reusable research code in this quantitative investment analysis repository. Enforces EtfData usage, BaseFactor inheritance, and library-vs-notebook boundaries."
name: "ETF Data Architecture"
applyTo: "libs/**/*.py, notebooks/**/*.ipynb, playgrounds/**/*.ipynb"
---

# ETF Data Architecture

- Treat this repository as a quantitative investment analysis toolkit: `libs/` contains reusable implementation, `notebooks/` contains daily interactive workflows, and `data/` is storage for inputs and generated outputs.
- When code needs ETF historical data, use `core.models.etf_daily_data.EtfData` or a `libs/data_manager/**` API that returns `EtfData` objects, such as `get_etf_data_by_symbol`, `get_etf_data_by_symbols`, or `etf_data_iter`.
- Do not read `data/etf_data/*.csv` directly with `pd.read_csv` anywhere in this repository, including `libs/`, notebooks, playgrounds, and one-off scripts. ETF file access must be wrapped by `EtfData` or routed through `libs/data_manager/**`.
- Convert `EtfData` to a plain DataFrame only at narrow integration boundaries, such as plotting, pandas-only analytics, serialization, or Backtrader feed preparation.
- Keep single-symbol factor implementations under `libs/factors/` and require them to inherit from `libs/factors/base_factor.py::BaseFactor`.
- Keep Backtrader-specific orchestration in `libs/backtesting/`. Keep reusable data abstractions in `libs/core/models/`. Keep one-off execution entry points in `libs/scripts/` and move reusable logic back into `libs/`.
- For constant datasets under `data/const/`, prefer provider or wrapper classes in `libs/data_manager/providers/` over hard-coded file paths when a provider already exists.
- Treat `data/backtest_results/` as output-only research artifacts, not a source of truth for new business logic.
- Keep notebooks thin: orchestrate library code from notebooks, load ETF data through `EtfData`/`data_manager`, and put reusable calculations, loaders, factor logic, and backtest helpers in `libs/`.