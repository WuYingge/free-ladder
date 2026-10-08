"""
复权因子数据管理 (Adjustment Factor Manager)

解决的问题：本地行情 CSV 的价格列是**后复权**（fetcher 默认 ``adjust="hfq"``），
而 ``value``(成交额) / ``volume``(成交量) 是**不复权**成交口径 —— 两者相差一个
逐股、逐日漂移的复权因子（实测 600519 2018→2026 因子 5.34→6.13；000001 130→161）。
因此 ``close - vwap`` 之类的差值会错量级，rank/correlation 也无法抵消这种漂移。
本模块把不复权成交均价搬到后复权尺度上：

    vwap_hfq = 成交额 / (成交量[手] x 100) x adj_factor

数据口径（``data/adj_factor/<code>.csv``，两个数据列 + date 索引）：

  * ``close_raw``  : 不复权收盘（东财 K 线 ``fqt=0`` 真值）
  * ``adj_factor`` : 后复权收盘 / 不复权收盘 = 本地行情 close / close_raw

后复权序列在两次除权之间因子恒定，故 ``adj_factor`` 为分段常数；这里仍**逐日落盘**，
不做分段压缩 —— 与主行情按日精确对齐最省心，也便于逐日核对。

关键不变量（体检用，见 ``check_vwap_bounds`` / ``libs/scripts/check_adj_factor.py``）：

    low <= vwap <= high

成交均价必然落在当日振幅内，且该不等式在乘上正的复权因子后依然成立 —— 这是
"因子配错/日期错位/单位错"的硬探针（例如忘了 x100 会立刻越界）。

成交量单位：本地 CSV 的 ``volume`` 是「手」（1 手 = 100 股），这是 A 股数据惯例；
``value`` 是元。实测 600519 2024-09-02 ``value/(volume*100)=1406.49`` 与东财
不复权 K 线的当日均价一致，而 ``value/volume=140650`` 明显偏离。
"""

from __future__ import annotations

import datetime
import os
import traceback
from multiprocessing import Pool

import pandas as pd

from config import DataPath

#: sidecar 的数据列（date 为索引）
ADJ_FACTOR_COLUMNS = ("close_raw", "adj_factor")

#: 成交量单位换算：本地行情 volume 单位是「手」
VOLUME_LOT_SIZE = 100.0

#: 增量更新回溯天数（覆盖末日期的复权因子变化，除权日重取当日即可）
UPDATE_LOOKBACK_DAYS = 3

#: vwap 越界判定的相对容差（覆盖价格落盘的 2~3 位小数舍入，约 2e-4）
DEFAULT_TOL_REL = 5e-4


# ---------------------------------------------------------------------------
# Path / load / write
# ---------------------------------------------------------------------------

def get_fp(symbol: str) -> str:
    os.makedirs(DataPath.ADJ_FACTOR_PATH, exist_ok=True)
    return os.path.join(DataPath.ADJ_FACTOR_PATH, f"{str(symbol).zfill(6)}.csv")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=list(ADJ_FACTOR_COLUMNS),
        index=pd.DatetimeIndex([], name="date"),
        dtype="float64",
    )


def load_adj_factor(symbol: str) -> pd.DataFrame:
    """读取单标的复权因子（date 索引，close_raw/adj_factor 两列）。

    文件缺失/为空返回同列空帧；缺列抛 ValueError。
    """
    fp = get_fp(symbol)
    if not os.path.exists(fp):
        return _empty_frame()
    df = pd.read_csv(fp, parse_dates=True, index_col=0)
    missing = [c for c in ADJ_FACTOR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"adj_factor {fp} 缺少列: {missing}")
    if df.empty:
        return _empty_frame()
    df = df[list(ADJ_FACTOR_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    return df.astype("float64")


def _write_frame(symbol: str, frame: pd.DataFrame) -> None:
    """原子写（临时文件 + os.replace），日期统一 YYYY-MM-DD。"""
    fp = get_fp(symbol)
    tmp_fp = fp + ".tmp"
    frame.to_csv(tmp_fp, encoding="utf-8-sig", index=True, date_format="%Y-%m-%d")
    os.replace(tmp_fp, fp)


def _merge_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """按日期去重合并（新行覆盖旧行），升序返回。"""
    merged = pd.concat([old, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.index.name = "date"
    return merged


def list_local_symbols(universe: str = "stock") -> list[str]:
    """本地已有行情文件的代码集（stock= data/stock_data，etf= data/etf_data）。"""
    root = DataPath.STOCK_PATH if universe == "stock" else DataPath.DEFAULT_PATH
    if not os.path.isdir(root):
        return []
    files = os.listdir(root)
    return sorted(
        os.path.splitext(f)[0] for f in files if f.endswith(".csv") and f[:6].isdigit()
    )


# ---------------------------------------------------------------------------
# 构造（纯函数）
# ---------------------------------------------------------------------------

def build_adj_factor_frame(
    raw_close: pd.Series,
    hfq_close: pd.Series,
) -> pd.DataFrame:
    """由不复权/后复权收盘序列构造 (close_raw, adj_factor) 逐日帧。

    :param raw_close: 不复权收盘（date 索引）
    :param hfq_close: 本地后复权收盘（date 索引）
    :return: date 索引、ADJ_FACTOR_COLUMNS 两列的 DataFrame；
             仅保留两边都有正值的日期（停牌/未上市的日子不产生因子）。
    """
    raw = pd.Series(raw_close).astype("float64").copy()
    hfq = pd.Series(hfq_close).astype("float64").copy()
    for series in (raw, hfq):
        series.index = pd.to_datetime(series.index, errors="coerce")
    raw = raw[raw.index.notna()].sort_index()
    hfq = hfq[hfq.index.notna()].sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    hfq = hfq[~hfq.index.duplicated(keep="last")]

    frame = pd.concat(
        [raw.rename("close_raw"), hfq.rename("close_hfq")], axis=1, join="inner"
    )
    frame = frame[(frame["close_raw"] > 0) & (frame["close_hfq"] > 0)].dropna()
    if frame.empty:
        return _empty_frame()
    frame["adj_factor"] = frame["close_hfq"] / frame["close_raw"]
    frame = frame[list(ADJ_FACTOR_COLUMNS)].sort_index()
    frame.index.name = "date"
    return frame.astype("float64")


# ---------------------------------------------------------------------------
# 本地/远端取数
# ---------------------------------------------------------------------------

def _local_quote_frame(symbol: str, universe: str = "stock") -> pd.DataFrame:
    """读取本地后复权行情（date 索引，open/high/low/close/volume/value）。

    数据访问遵守仓库约束：一律走 data_manager 的 get_*_data_by_symbol。
    """
    if universe == "stock":
        from data_manager.stock_data_manager import get_stock_data_by_symbol

        raw_df = get_stock_data_by_symbol(str(symbol).zfill(6)).data
    else:
        from data_manager.etf_data_manager import get_etf_data_by_symbol

        raw_df = get_etf_data_by_symbol(str(symbol).zfill(6)).data

    df = raw_df.copy()
    if "date" in df.columns:
        index = pd.to_datetime(df["date"], errors="coerce")
    else:
        index = pd.to_datetime(df.index, errors="coerce")
    cols = ["open", "high", "low", "close", "volume", "value"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"本地行情 {symbol} 缺少列: {missing}")
    out = pd.DataFrame(
        {c: pd.to_numeric(df[c], errors="coerce").to_numpy() for c in cols},
        index=pd.DatetimeIndex(index, name="date"),
    )
    out = out[out.index.notna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.astype("float64")
    # 与 Alpha101Universe 同口径：非正成交量/成交额视为无效交易日（无成交均价）
    out["volume"] = out["volume"].where(out["volume"] > 0)
    out["value"] = out["value"].where(out["value"] > 0)
    return out


def _local_hfq_close(symbol: str, universe: str = "stock") -> pd.Series:
    """本地后复权收盘序列（date 索引）。"""
    frame = _local_quote_frame(symbol, universe)
    if frame.empty:
        return pd.Series(dtype="float64", name="close")
    return frame["close"].rename("close")


def _fetch_raw_close(
    symbol: str,
    universe: str = "stock",
    start_date: str = "19700101",
    end_date: str = "20500101",
) -> pd.Series:
    """抓取东财**不复权**日线收盘（走代理，与本地后复权行情同源）。"""
    if universe == "stock":
        from fetcher.stock import get_stock_hist_em as fetch_hist
    else:
        from fetcher.etf import fund_etf_hist_em as fetch_hist

    df = fetch_hist(
        symbol=str(symbol).zfill(6),
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="",
    )
    if df is None or df.empty:
        return pd.Series(dtype="float64", name="close")
    series = pd.Series(
        pd.to_numeric(df["收盘"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(pd.to_datetime(df["日期"], errors="coerce"), name="date"),
    )
    return series.dropna().sort_index().rename("close")


# ---------------------------------------------------------------------------
# 回填 / 增量更新
# ---------------------------------------------------------------------------

def backfill_adj_factor_symbol(symbol: str, universe: str = "stock") -> bool:
    """单标的全历史回填：本地后复权收盘 ÷ 东财不复权收盘 → 逐日因子落盘。"""
    code = str(symbol).zfill(6)
    try:
        hfq = _local_hfq_close(code, universe)
        if hfq.empty:
            print(f"backfill adj_factor {code}: 无本地后复权行情, 跳过")
            return False
        raw = _fetch_raw_close(code, universe)
        if raw.empty:
            print(f"backfill adj_factor {code}: 接口无不复权数据, 跳过")
            return False
        frame = build_adj_factor_frame(raw, hfq)
        if frame.empty:
            print(f"backfill adj_factor {code}: 对齐后无有效行, 跳过")
            return False
        _write_frame(code, frame)
        return True
    except Exception as err:  # noqa: BLE001 — 单标的失败不影响批量
        traceback.print_exc()
        print(f"backfill adj_factor {code} 失败: {err}")
        return False


def update_adj_factor_symbol(symbol: str, universe: str = "stock") -> tuple[str, bool]:
    """单标的增量更新：无文件则回填；否则只重取 [末日期-3天, 今天] 的窗口。"""
    code = str(symbol).zfill(6)
    if not os.path.exists(get_fp(code)):
        return code, backfill_adj_factor_symbol(code, universe)
    try:
        existing = load_adj_factor(code)
        if existing.empty:
            return code, backfill_adj_factor_symbol(code, universe)
        last_date = pd.to_datetime(existing.index.max()).date()
        now = datetime.datetime.now()
        if now.date() - last_date < datetime.timedelta(days=1):
            return code, True

        start = last_date - datetime.timedelta(days=UPDATE_LOOKBACK_DAYS)
        raw = _fetch_raw_close(
            code, universe, start.strftime("%Y%m%d"), now.strftime("%Y%m%d")
        )
        if raw.empty:
            return code, False
        new = build_adj_factor_frame(raw, _local_hfq_close(code, universe))
        if new.empty:
            return code, False
        _write_frame(code, _merge_frames(existing, new))
        return code, True
    except Exception as err:  # noqa: BLE001 — 单标的失败不影响批量
        traceback.print_exc()
        print(f"update adj_factor {code} 失败: {err}")
        return code, False


def _update_worker(args: tuple[str, str]) -> tuple[str, bool]:
    """Pool.map 用的一元入口（唯一参数 = (symbol, universe)）。"""
    symbol, universe = args
    return update_adj_factor_symbol(symbol, universe)


def batch_update_adj_factor(
    symbols: list[str] | None = None,
    universe: str = "stock",
    *,
    all_history: bool = False,
    max_workers: int | None = None,
) -> list[tuple[str, bool]]:
    """批量更新（默认取本地已有行情的全部标的）。

    :param symbols: 指定标的；None 用 ``list_local_symbols(universe)``
    :param universe: "stock" / "etf"
    :param all_history: True 时逐标的全历史回填（首次建库用）
    :param max_workers: 并发进程数；默认复用股票更新管线的并发规模并预热代理池
    """
    targets = (
        [str(s).zfill(6) for s in symbols]
        if symbols is not None
        else list_local_symbols(universe)
    )
    if not targets:
        print("没有可更新的标的（本地行情目录为空？）")
        return []

    if max_workers is None or max_workers <= 1:
        return [
            (code, backfill_adj_factor_symbol(code, universe) if all_history
             else update_adj_factor_symbol(code, universe)[1])
            for code in targets
        ]

    from data_manager.stock_data_manager import (
        STOCK_UPDATE_POOL_SIZE,
        _initialize_stock_update_worker,
    )

    pool_size = max_workers or STOCK_UPDATE_POOL_SIZE
    worker = _backfill_worker if all_history else _update_worker
    with Pool(pool_size, initializer=_initialize_stock_update_worker) as pool:
        results = pool.map(worker, [(c, universe) for c in targets])

    failed = [code for code, ok in results if not ok]
    for code in failed:
        print(f"Failed to update adj_factor for {code}")
    return results


def _backfill_worker(args: tuple[str, str]) -> tuple[str, bool]:
    """Pool.map 用的一元回填入口。"""
    symbol, universe = args
    return symbol, backfill_adj_factor_symbol(symbol, universe)


# ---------------------------------------------------------------------------
# VWAP 计算与体检
# ---------------------------------------------------------------------------

def compute_vwap(
    value: pd.Series | pd.DataFrame,
    volume: pd.Series | pd.DataFrame,
    adj_factor: pd.Series | pd.DataFrame,
):
    """后复权尺度的 VWAP = 成交额 / (成交量[手] x 100) x 复权因子。

    :param value: 成交额（元）
    :param volume: 成交量（手）
    :param adj_factor: 复权因子（与 value 同日历；缺失日 → NaN，宁可缺口不给错值）
    :return: 与输入同形（同为 Series 或同为 date x symbol 的 DataFrame）的 VWAP

    注：``value``/``volume``/``adj_factor`` 三者的口径必须一致（同为 Series 或
    同为 DataFrame），且已按日期对齐；DataFrame 情形下按 index/columns 元素级对齐。
    **无成交日（volume<=0）或成交额缺失 → NaN**：债券/货币 ETF 常有零成交日，
    此时没有"成交均价"可言，绝不能变成 ±inf 混进因子。
    """
    valid = (volume > 0) & (value > 0)
    raw_vwap = value.where(valid) / (volume.where(valid) * VOLUME_LOT_SIZE)
    return raw_vwap * adj_factor


def check_vwap_bounds(
    symbol: str,
    universe: str = "stock",
    *,
    tol_rel: float = DEFAULT_TOL_REL,
) -> dict:
    """校验 ``low <= vwap <= high``（硬不变量），返回单标的体检结果。

    :return: dict(symbol/rows/violations/max_excess_rel/status)，
             status ∈ {ok, violation, missing_data}；行情或因子缺失/读取失败
             一律归为 missing_data（体检不应因单标的异常而中断）。
    """
    code = str(symbol).zfill(6)
    try:
        quote = _local_quote_frame(code, universe)
        factor = load_adj_factor(code)
    except Exception as exc:  # noqa: BLE001 — 单标的读取失败不影响整体体检
        return {
            "symbol": code, "rows": 0, "violations": 0,
            "max_excess_rel": float("nan"), "status": "missing_data",
            "first_date": None, "last_date": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if quote.empty or factor.empty:
        return {
            "symbol": code, "rows": 0, "violations": 0,
            "max_excess_rel": float("nan"), "status": "missing_data",
            "first_date": None, "last_date": None,
        }

    vwap = compute_vwap(quote["value"], quote["volume"], factor["adj_factor"])
    frame = pd.DataFrame(
        {"low": quote["low"], "high": quote["high"], "close": quote["close"], "vwap": vwap}
    ).dropna()
    if frame.empty:
        return {
            "symbol": code, "rows": 0, "violations": 0,
            "max_excess_rel": float("nan"), "status": "missing_data",
            "first_date": None, "last_date": None,
        }

    frame = frame.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if frame.empty:
        return {
            "symbol": code, "rows": 0, "violations": 0,
            "max_excess_rel": float("nan"), "status": "missing_data",
            "first_date": None, "last_date": None,
        }
    tolerance = tol_rel * frame["close"].abs()
    excess = pd.concat(
        [frame["vwap"] - frame["high"] - tolerance, frame["low"] - frame["vwap"] - tolerance],
        axis=1,
    ).max(axis=1)
    violations = int((excess > 0).sum())
    return {
        "symbol": code,
        "rows": int(len(frame)),
        "violations": violations,
        "max_excess_rel": float(
            (excess / frame["close"].abs()).max()
        ) if len(frame) else float("nan"),
        "status": "violation" if violations else "ok",
        "first_date": frame.index.min().strftime("%Y-%m-%d"),
        "last_date": frame.index.max().strftime("%Y-%m-%d"),
    }


def build_health_report(
    symbols: list[str] | None = None,
    universe: str = "stock",
    *,
    tol_rel: float = DEFAULT_TOL_REL,
) -> dict:
    """批量体检并汇总为 JSON 友好的报告 dict。"""
    targets = (
        [str(s).zfill(6) for s in symbols]
        if symbols is not None
        else list_local_symbols(universe)
    )
    rows = [check_vwap_bounds(s, universe, tol_rel=tol_rel) for s in targets]

    checked_rows = sum(r["rows"] for r in rows)
    violation_rows = sum(r["violations"] for r in rows)
    missing = [r["symbol"] for r in rows if r["status"] == "missing_data"]
    worst = sorted(
        (r for r in rows if r["violations"]),
        key=lambda r: r["max_excess_rel"],
        reverse=True,
    )[:10]
    return {
        "universe": universe,
        "tol_rel": tol_rel,
        "n_symbols": len(rows),
        "n_missing_factor": len(missing),
        "missing_factor_symbols": missing[:50],
        "n_checked_rows": checked_rows,
        "n_violation_rows": violation_rows,
        "violation_rate": (violation_rows / checked_rows) if checked_rows else 0.0,
        "worst_symbols": worst,
        "per_symbol": rows,
    }
