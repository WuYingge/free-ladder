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
- **Script placement (硬约束):**
  - 项目根目录 (`/`) **禁止**放置任何 `.py` 脚本。禁止创建、禁止保留。
  - 可复用的 CLI 入口 → `libs/scripts/`（含 docstring 用法说明）。
  - 一次性临时脚本 → 用完即删，不得提交。如果确需保留以供日后参考，移入 `libs/scripts/`。
  - 已有根目录脚本 `_analyze_factors.py` / `_cleanup_exec.py` / `_ic_filter.py` 为历史遗留，待后续迁移。

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
  - `generate_wide_momentum_configs.py` — 从 factors_to_analyze.csv 批量生成 config
  - `update_sw_industry_clf.py` — 刷新申万行业分类历史 CSV（data/const/stock_sw_industry_clf.csv）
  - `update_adj_factor.py` — 更新 data/adj_factor 复权因子（`--all-history` 首次回填 / 默认增量）
  - `check_adj_factor.py` — vwap 硬不变量体检 `low <= vwap <= high`（不联网）
  - `run_alpha101_scan.py` — Alpha101 扫描器（101 个公式；`--alpha 001 101` / `--only-plain` / `--universe stock|etf`）

### Key Modules

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| `libs/backtesting/` | Backtrader feeds, strategies, batch execution, performance | `engine.py`, `timing_batch.py`, `data.py`, `performance.py`, `strategies/` |
| `libs/factors/` | Signal generation (timing & portfolio) | `base_factor.py`, `rsrs.py`, `new_high.py`, `average_true_range.py`, `portfolio/`, `alpha101/`（101 公式全量） |
| `libs/core/models/` | Typed wrappers: `EtfData`, `IndexDailyData`, `FinancialData` | `data_base.py`, `etf_daily_data.py`, `daily_quote_data.py` |
| `libs/data_manager/` | Persist, update, load ETF/index CSV datasets | `etf_data_manager.py`, `index_data_manager.py`, `daily_basic_manager.py`, `adj_factor_manager.py`, `financial_manager.py`, `financial_schema.py`, `financial_health.py`, `financial_provenance.py`, `financial_audit.py`, `datasets.py`, `providers/` |
| `libs/fetcher/` | Fetch ETF/index/stock market data from EastMoney / Akshare | `etf.py`, `index.py`, `stock.py`, `industry.py`, `financial.py`, `utils.py` |
| `libs/proxy/` | Proxy pool management for data fetching | `proxy.py` |
| `libs/config.py` | Shared paths and env settings (`DataPath`) | `config.py` |

> `data/daily_basic/<code>.csv`（`date,circ_mv,total_mv,float_share`，与 stock_data 同代码集）：
> 2018-01-02 起为东财 `RPT_VALUEANALYSIS_DET` 真值；2016-2017 与退市股为成交额/换手率估算段。
> 统一 getter（`get_stock_data_by_symbol(..., with_basic=True)`）按需合并进 `StockDailyData`，
> 扩展数据集注册在 `libs/data_manager/datasets.py`（每个时序 CSV 对应一个 `withXXX` 开关）。

> `data/adj_factor/<code>.csv`（`date,close_raw,adj_factor`，股票 + ETF 同构）：
> **行情价格是后复权、成交额/成交量是不复权口径**，二者相差一个逐股、逐日复权因子，
> 跨口径量（alpha101 的 `vwap`）必须搬回同一尺度：
> `vwap = value/(volume*100) * adj_factor`（`volume` 单位是**手**；`close_raw`=东财不复权收盘，
> `adj_factor`=本地后复权收盘 ÷ `close_raw`）。抓取 `libs/data_manager/adj_factor_manager.py`
> （东财 K 线 `fqt=0` 走代理，与本地 hfq 同日相除，原子写），更新入口
> `libs/scripts/update_adj_factor.py`（`--all-history` 首次回填 / 默认增量 3 天窗口），
> 体检 `libs/scripts/check_adj_factor.py`（**硬不变量 `low <= vwap <= high`**，不联网 →
> `data/adj_factor/health_report.json`），provider `data_manager.providers.adj_factor_provider.ADJ_FACTOR`
> （`get_factor_series` 供 alpha101 panel，`get_events` 供按日合并）。
> 与行情合并：`get_stock_data_by_symbol(..., with_adj_factor=True)` → `close_raw` / `adj_factor`。
> **回填现状（2026-09-13）**：**全量已回填 —— 股票 5348/5348 + alpha101 ETF 池 489 只**
> （`--universe stock --all-history --max-workers 15`，15 进程约 5 分钟 / 557MB；失败 138 只补跑后清零）。
> ETF 池抽样体检 58.8 万行、越界 0.28%；股票抽样 400 只 128.6 万行、越界 0.11%，
> **越界集中在 1990 年代初上市的老股早期数据**（成交量单位/口径不一致，个别行偏差可达 100 倍）；
> 同一批抽样按 **2015 年起** 统计：84.7 万行、越界 337 行 = **0.0398%**（102/400 只有零星坏日），
> ETF 侧越界则集中在 61 只流动性极差的债券/货币 ETF。
> **面板侧对越界格统一置 NaN**（`panel._build_vwap_matrix` 的越界守门）→ alpha101 拿到的 vwap
> 恒满足 `low <= vwap <= high`；零成交日（`volume <= 0`）一律 NaN，绝不产生 ±inf。
> **东财后复权是仿射口径（不是乘性，也不是缺陷）**：其定义为
> `复权后 = 复权前×(1+流通股份变动比例) − 配股价×比例 + 现金红利` ⇒ **`H = a·P + b`**
> （a=1+累计股份变动、b=累计现金红利−累计配股成本；a、b 段内恒定，只在除权日更新）。
> 实测按真实除权日分段后段内线性拟合**最大残差 ≈0.005 元**（两位小数舍入级），
> 无分红标的退化为乘性（518880：a=1、b=0、日收益完全保真）。
> 数学后果：日收益被压缩 `a·P/(a·P+b)`，**逐股不同**（000001 `0.773`、601398 `0.606`、
> 600519 `0.845`、510300 `0.839`、510500 `0.974`），即该序列衡量的是
> 「股价 + 累计现金分红（**不再投资**）」而不是「分红再投资」。
> 实测影响：60 日波动低估约 **33%**、20 日动量幅度 β≈**0.73**、
> `|20日涨跌幅|>5%` 阈值信号漏报 **38%**、横截面排序 Spearman **0.970**（大体保留）。
> → **看/对账/复现软件、以及同日横截面量（vwap、alpha101 当日算子、无分红标的）可直接用 hfq；
> 吃日频收益绝对幅度的因子（波动/动量幅度/相关/IC 绝对值/阈值突破）需换乘性口径。**
> 本地文件与重新单次请求东财 hfq 逐日一致（比值恒 1.000000），故非抓取/管线问题；
> 对 vwap 无影响（用**同一天**的因子，与当日 OHLC 严格同尺度；体检 4.6 万行 0 越界）。
> 详见 `docs/eastmoney_hfq_convention.md`（含验证表、此前一次错误定性的更正、自建乘性因子方案）。
> **状态：仅记录，价格层未改动。**

> `libs/factors/alpha101/`（WorldQuant《101 Formulaic Alphas》**101 个公式已全部落地**）：
> `formulas.py` 公式本体（论文原文逐一实现，docstring 即公式）＋ `alphas.py` 注册表
> `ALPHA101_REGISTRY`（元数据 + 数据依赖标记）＋ `operators.py` 算子库 ＋ `panel.py` 面板 ＋
> `universe.py` 数据源（ETF/股票）＋ `testing.py` 合成面板（自检/单测用）。
> **数据依赖标记**（`AlphaSpec`，扫描器按可用性自动跳过；`data_requirements_summary()` 可打印）：
> 需 vwap **43** / 需 adv **45** / **需申万行业 18**（档位 1/2/3 = 论文 IndClass sector/industry/subindustry，
> 经 `panel.industry(level)` 取**逐日**标签做时点中性化，ETF 池无行业 → 跳过）/ 需 cap **1**（#56）/
> 无附加依赖 **52**（`--only-plain`）。
> **算子硬约定**：窗口算子用 `min_periods=_min_obs(d)`，即窗口内 ≥80% 有效观测即可算
> （`WINDOW_MISSING_TOLERANCE=0.2`）：A 股停牌缺口、零方差相关系数、ts_rank 长期 pin 极值都会
> 产生洞，严格满窗会让长链条公式（#96 需 ~40 天连续无洞）覆盖率塌到 ~3%；
> **数据无缺失时结果与满窗完全一致**。`correlation/covariance` 的 ±inf → NaN；
> `ts_argmax/ts_argmin` 返回窗口内日历位置（0-based）；比较型公式统一转 float（0/±1）。
> 用法：`python libs/scripts/run_alpha101_scan.py [--alpha 001 101] [--only-plain] [--universe stock|etf]`，
> **全池面板只加载一次**（`build_alpha101_panel(..., inputs=共享面板)`）供所有 alpha 复用。
> 口径提醒：面板价格是东财**仿射后复权**（日收益被压 `a·P/(a·P+b)`，逐股不同），含
> `returns/stddev/correlation` 的公式在**跨股幅度**上会带该效应；vwap 已搬回同尺度、
> 同日横截面量不受影响（见 `docs/eastmoney_hfq_convention.md`）。

> `data/const/stock_sw_industry_clf.csv`（申万官方全市场个股行业分类历史：
> `symbol,start_date,industry_code,level1_name,level2_name,level3_name,update_time`，含退市股；
> `start_date`=归属生效日，可做时点回溯防未来函数；6 位行业代码按前 2/4/6 位切片即申万一/二/三级档位；
> 名称按生效时代匹配对应版本标准表（2021-07-30 起 → `sw_industry_standard_2021.csv`；
> 2014-02-21 起 → `sw_industry_standard_2014.csv`（28 一级/104 二级/227 三级，自官网修订对照表旧侧提取）；
> 更早时代按代码跨版近似回退。行命名覆盖率 85.7%（2021/2014 时代均 100%））：
> 抓取 `libs/fetcher/industry.py::get_stock_sw_industry_clf_hist`（官网 xls 走代理，需 `verify=False`），
> 刷新入口 `libs/scripts/update_sw_industry_clf.py`，provider `data_manager.providers.sw_industry_provider.SW_INDUSTRY`
> （`get_mapping(asof)` / `get_industry` / `get_group_series` 供 alpha101 `indneutralize`；`get_events` 供按日合并）。
> 与行情合并：`get_stock_data_by_symbol(..., with_industry=True)` 按日 point-in-time 并入
> `industry_code/level1_name/level2_name/level3_name`（经 `data_manager/datasets.py` 事件式 asof 注册，见 with_basic 同款开关）。

> `data/financial/<表>.csv`（七张财报宽表：`income_q/balance_q/cashflow_q/forecast/share_capital/dividend/holder_num`；
> 前四列固定 `symbol,report_date,ann_date,report_type`；金额=元、股本=股、比率=小数、金额累计口径；
> 缺失留空；尾部方法论列 `currency,update_date,basis,update_time`）：
> **两条时间轴** `report_date`（会计截止日）与 `ann_date`（首次公告日）共同构成 PIT ——
> 合并按 `ann_date` 做 `merge_asof(direction="backward")`，公告日之前该期数据不可见（防前视）；
> `basis=first_reported|revised|latest_version` 标记该行数值是否在首次公告后被追溯调整，
> **加载默认取全部行**（`basis=None`）: 实测 2010-2024 年只有 3%~13% 的行是 `first_reported`
> （源端 `UPDATE_DATE` 普遍晚于 `NOTICE_DATE`），把它当默认会让 `with_financial=True` 取不到历史数据
> （实测 2020-06 仅 0/20 天有值）→ `first_reported` 改为**显式严格模式**（`get_events(..., basis="first_reported")`）。
> 取数 `libs/fetcher/financial.py`（东财 F10 / 东财数据中心 / 巨潮；**默认 `proxy_first`
> 代理优先、失败回退直连**，环境变量 `FINANCIAL_FETCH_MODE`、`FINANCIAL_FETCH_THREADS`；
> 代理链路只贵 +0.34s/请求，但财务全历史约 31~37 请求/股（F10 单次上限 5 期）→ 回填建议
> 临时切 `direct_first`；池被占满时代理档只重试 1 次即回退直连，不自旋）；
> 落盘与日更 `libs/data_manager/financial_manager.py`（`update_financials` 幂等、首版优先、
> 原子写；`batch_check_financials_updated` 不联网新鲜度体检；`load_financial_events` 行级门禁：
> ann_date 缺失 / 早于可见下界 / 越出行情区间的行一律剔除并计数）；
> provider `data_manager.providers.financial_provider.FINANCIAL`（`get_events` / `get_dataframe(asof)` / `get_latest` / `reload`）。
> 加载开关：`get_stock_data_by_symbol(..., with_financial=True)` 一键七表，或单表
> `with_income`/`with_balance`/`with_cashflow`/`with_forecast`/`with_share_capital`/`with_dividend`/`with_holder_num`。
> CLI：`libs/scripts/update_financials.py`（`--all-history` 首次回填 / 默认日更最近 2 期）、
> `libs/scripts/check_financials.py`（十三项体检 → `data/financial/health_report.json`）、
> `libs/scripts/crosscheck_financials.py`（一次性双源公告日对拍）。
> **日期文本规范是硬不变量**：落盘只允许 `YYYY-MM-DD`（`_canonical_for_write` 写前强制 datetime64 +
> `date_format`），读取一律走 `parse_date_column(format="mixed")` —— pandas 3 对混格式列会**静默**把
> 少数派格式判成 NaT（实测 12.7 万行公告日），写入端规范化与读取端容错必须同时存在（见 docs §4.6）。
> 源端哨兵 `1900-01-01`（上市前报告期无公告日）按"未知"置空，绝不当真实公告日（否则构成前视）。
> **`ann_date` 的源端"错位一年"缺陷：主体在 2000-2009 段**（`income_q` 上市后 3.97 万行；2010+ 修剪后仅 29 行）：
> 东财 F10 把部分报告期的公告日写成**下一期同日历期**的公告日（巨潮公告标题定案：000670 的 2016Q1 写成 2017 年一季报公告日），
> 且三张报表的缺陷互相独立、业绩报表 `RPT_LICO_FN_CPD` 共享同一缺陷（**它不是独立信源**）→ 只有巨潮公告原文能仲裁。
> 已于 2026-09-13 用巨潮逐条仲裁修复 **3,757 行**（同步重算 `basis` 1,716 行，全部留痕可回滚）；
> 剩余 2000-2009 段因巨潮档案缺失无法仲裁，**研究只用 2010+**。注意巨潮档案里有被错误命名的标题
> （"公告日早于报告期末"），抓取层与计划层各有护栏。错值方向偏晚（不构成前视，但 PIT 时点错）。
> **退市股是当前最大缺口**：153 只退市股中 152 只无三表数据（146 只有行情文件），已核实东财 F10 与数据中心
> 均不提供（新浪/巨潮可替代，尚未接入；不得用无公告日的源入库），见 docs §8。
> **溯源与跨源审计（"凭什么信这一格"）**：取数单锁死主表表头 → 用 sidecar：
> `data/financial/_provenance/<表>.meta.json`（逐列默认来源）+ `<表>.corrections.csv`（单元格级修正流水，
> **保留 old_value**，`financial_provenance.revert_corrections` 可回滚）；任意一格来源 = 默认来源 + 修正覆盖。
> **写盘前会重放修正流水**（`_reapply_corrections`）：否则一次 `--all-history --replace` 就会用东财原值
> 把 3,757 行公告日仲裁结果静默冲掉；撤销只能显式 `revert_corrections`。
> 跨源审计 `financial_audit.py` / `audit_financials.py`：`shares`（东财 balance_q.total_share vs 巨潮 share_capital，
> 离线全量，实测一致率 97.75%）、`quotes`（vs 行情 float_share，流通>总股本实测 2.23%）、
> `values`（vs 新浪三表，联网分层抽样，受判定字段实测 98.8~99.9%）。体检第 13 项 `cross_source_audit` 常驻读它。
> **已由跨源审计定位**：`balance_q.total_share` 约 2% 行偏旧/错值（巨潮与行情侧互相吻合），见 docs §8。
> 字段口径与实测坑位（利润表无股本字段、少数股东损益 vs 权益、预告过滤字段必须是 `REPORT_DATE`、
> companyType 需逐股解析否则银行/券商/保险三表为空、F10 单请求日期上限 5 期）见 `docs/financial_data_pipeline.md`。

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

**Financial Data Update Flow:**
1. 新鲜度体检 `batch_check_financials_updated()`（不联网，按 `update_time` + 披露窗口判定）。
2. 抓取 `libs/fetcher/financial.py`（东财 F10 逐股三表 / 东财数据中心逐期事件表 / 巨潮股本变动），
   **默认 `proxy_first`**（代理优先、失败回退直连）；逐股表默认只取最近 2 个报告期。
3. 落盘 `upsert_table`（主键 `(symbol, report_date[, forecast_indicator_cn])` —— **`ann_date` 不进键**，
   否则"缺公告日的旧行"会让带公告日的新行无法自愈；采纳顺序=字段填充多者优先→同分取新抓行，
   数值取首版；写前强制日期规范格式；原子写 + 幂等）。
4. 消费 `get_stock_data_by_symbol(..., with_financial=True)` 按 `ann_date` 时点合并；
   或 `FINANCIAL.get_dataframe(table, asof=...)` 做横截面研究。
5. 质量守门 `libs/scripts/check_financials.py` → `data/financial/health_report.json`。

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
  - `libs/scripts/update_financials.py` — 抓取/落盘财报七表（`--all-history` 首次回填，默认日更最近 2 期）
  - `libs/scripts/check_financials.py` — 财务数据十三项体检（不联网）→ `data/financial/health_report.json`
  - `libs/scripts/crosscheck_financials.py` — 一次性双源公告日对拍（验收用）
  - `libs/scripts/audit_financials.py` — 跨源审计（`shares`/`quotes` 离线全量；`values` 联网分层抽样；`--provenance` 看溯源）
  - `libs/scripts/update_adj_factor.py` — 抓取/落盘复权因子（`--universe stock|etf`，`--all-history` 首次回填，默认增量）
  - `libs/scripts/check_adj_factor.py` — vwap 边界体检（不联网）→ `data/adj_factor/health_report.json`
  - `libs/scripts/run_alpha101_scan.py` — Alpha101 扫描器：101 个公式跑 Layer1/2/3 分析并出报告
    （`--only-plain` 只跑无 vwap/行业/cap 依赖的 52 个；面板全池只加载一次供所有 alpha 复用）

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
