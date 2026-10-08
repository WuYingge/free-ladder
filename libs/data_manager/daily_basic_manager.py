"""
A股每日市值/股本数据管理 (Daily Basic Data Manager)

数据口径 (与 data/stock_data 同代码集, 东财):
  * 2018-01-02 起: 东财数据中心 RPT_VALUEANALYSIS_DET 真值 (唯一接口),
    circ_mv=流通市值(元) / total_mv=总市值(元) / float_share=流通A股(股)。
  * 2016-01-01 ~ 2017-12-31: 接口无此区间, 用现有 stock_data 的
    成交额 / 换手率 估算 (东财口径: circ_mv = value/(turnOver/100),
    float_share = volume*10000/turnOver, total_mv = circ_mv x ratio)。
  * 退市股: 接口查无, 全区间估算保底, total_mv 无 ratio 依据时为 NaN。

质量规则 (不额外加列):
  * 行日期 < 2018-01-02 即为估算段 (ESTIMATE_END_EXCLUSIVE 前);
  * 退市股 (STOCK_LIST.get_delist_date 非空) 全档为估算。
"""

from __future__ import annotations

import datetime
import os
import traceback
from multiprocessing import Pool

import pandas as pd

from config import DataPath
from data_manager.providers.stock_list_provider import STOCK_LIST
from fetcher.stock import get_stock_daily_basic_em

# 复用股票更新管线: 并发规模 + 代理池初始化 (错峰预热)
from data_manager.stock_data_manager import (  # noqa: E402
    STOCK_UPDATE_POOL_SIZE,
    _initialize_stock_update_worker,
)

DAILY_BASIC_COLUMNS = ["circ_mv", "total_mv", "float_share"]

# 估算段起点 (含); 真值自 2018-01-02 起 (接口硬边界)
ESTIMATE_START_DATE = datetime.date(2016, 1, 1)
TRUTH_START_DATE = pd.Timestamp("2018-01-02")


# ---------------------------------------------------------------------------
# Path / low-level helpers
# ---------------------------------------------------------------------------

def get_fp(symbol: str) -> str:
    os.makedirs(DataPath.DAILY_BASIC_PATH, exist_ok=True)
    return os.path.join(DataPath.DAILY_BASIC_PATH, f"{str(symbol).zfill(6)}.csv")


def list_stock_data_symbols() -> list[str]:
    """返回 data/stock_data 下的代码集 (与 daily_basic 规格"同代码集"对齐)。"""
    if not os.path.isdir(DataPath.STOCK_PATH):
        return []
    files = os.listdir(DataPath.STOCK_PATH)
    return sorted(
        os.path.splitext(f)[0] for f in files if f.endswith(".csv") and f[:6].isdigit()
    )


def _empty_basic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=DAILY_BASIC_COLUMNS,
        index=pd.DatetimeIndex([], name="date"),
    )


def load_daily_basic(symbol: str) -> pd.DataFrame:
    """
    读取单只股票的 daily_basic (date 索引, 3 列);
    文件缺失/为空返回空 DataFrame (标准列), 格式错误抛 ValueError。
    """
    fp = get_fp(symbol)
    if not os.path.exists(fp):
        return _empty_basic_frame()
    df = pd.read_csv(fp, parse_dates=True, index_col=0)
    missing = [c for c in DAILY_BASIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"daily_basic {fp} 缺少列: {missing}")
    if df.empty:
        return _empty_basic_frame()
    df = df[DAILY_BASIC_COLUMNS].sort_index()
    df.index.name = "date"
    return df


def get_daily_basic_last_local_date(symbol: str) -> datetime.date | None:
    try:
        df = load_daily_basic(symbol)
    except Exception:
        return None
    if df.empty:
        return None
    try:
        return pd.to_datetime(df.index.max()).date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Estimation (2016-2017 及退市股: 成交额 / 换手率)
# ---------------------------------------------------------------------------

def estimate_from_stock_history(
    df_stock: pd.DataFrame,
    ratio: float | None = None,
) -> pd.DataFrame:
    """
    由 stock_data 行情估算 daily_basic (东财口径)。

    :param df_stock: date 索引, 含 volume(手)/value(元)/turnOver(%) 列
    :param ratio: 总市值/流通市值 比例 (取接口首行真值, 视为常数);
                  None 时 total_mv 填 NaN (退市股无真值依据)
    :return: date 索引, DAILY_BASIC_COLUMNS 三列; 换手率<=0 或成交额缺失行剔除
    公式: circ_mv = value/(turnOver/100); float_share = volume*10000/turnOver
    """
    required = ["volume", "value", "turnOver"]
    missing = [c for c in required if c not in df_stock.columns]
    if missing:
        raise ValueError(f"estimate_from_stock_history 缺少列: {missing}")

    valid = df_stock[
        (pd.to_numeric(df_stock["turnOver"], errors="coerce") > 0)
        & (pd.to_numeric(df_stock["value"], errors="coerce").notna())
        & (pd.to_numeric(df_stock["volume"], errors="coerce").notna())
    ]
    if valid.empty:
        return _empty_basic_frame()

    # 换手率字段单位是百分数数值 (如 1.0 表示 1%)
    turn_over_pct = valid["turnOver"].astype(float)
    res = pd.DataFrame(index=valid.index, columns=DAILY_BASIC_COLUMNS, dtype="float64")
    res["circ_mv"] = valid["value"].astype(float) / (turn_over_pct / 100.0)
    res["float_share"] = valid["volume"].astype(float) * 10000.0 / turn_over_pct
    if ratio is not None and ratio > 0:
        res["total_mv"] = res["circ_mv"] * ratio
    else:
        res["total_mv"] = float("nan")
    res.index.name = "date"
    return res


# ---------------------------------------------------------------------------
# Backfill / incremental update
# ---------------------------------------------------------------------------

def backfill_daily_basic_symbol(code: str) -> bool:
    """
    单只股票首次回填:
      1) 接口拉真值 (2016 起, 接口实际自 2018-01-02);
      2) 估算段 = stock_data 区间 [2016-01-01, 首行真值日期) (无真值则全区间);
      3) ratio = 首行真值 total_mv/circ_mv; 拼接去重升序落盘。
    退市股 (接口空) -> 纯估算文件, total_mv 为 NaN。
    """
    code = str(code).zfill(6)
    try:
        truth = get_stock_daily_basic_em(code, start_date=ESTIMATE_START_DATE.strftime("%Y%m%d"))
        stock_fp = os.path.join(DataPath.STOCK_PATH, f"{code}.csv")
        if truth.empty and not os.path.exists(stock_fp):
            print(f"backfill {code}: 接口无数据且无本地行情, 跳过")
            return False
        if not os.path.exists(stock_fp):
            print(f"backfill {code}: 无 stock_data 行情文件, 仅真值落盘")
            df_stock = _empty_basic_frame()
        else:
            df_stock = pd.read_csv(stock_fp, parse_dates=True, index_col=0)

        first_truth_date = truth.index.min() if not truth.empty else None
        ratio: float | None = None
        if first_truth_date is not None:
            row0 = truth.loc[first_truth_date]
            if (
                pd.notna(row0["circ_mv"]) and row0["circ_mv"] > 0
                and pd.notna(row0["total_mv"])
            ):
                ratio = float(row0["total_mv"]) / float(row0["circ_mv"])

        if first_truth_date is None:
            # 退市股 (或接口查无): 全区间估算
            window = df_stock[df_stock.index >= pd.Timestamp(ESTIMATE_START_DATE)]
            est = estimate_from_stock_history(window, ratio=None)
        else:
            window = df_stock[
                (df_stock.index >= pd.Timestamp(ESTIMATE_START_DATE))
                & (df_stock.index < first_truth_date)
            ]
            est = estimate_from_stock_history(window, ratio=ratio)

        merged = pd.concat([est, truth])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "date"
        merged.to_csv(get_fp(code), encoding="utf-8-sig", index=True)
        return True
    except Exception as err:
        traceback.print_exc()
        print(f"backfill {code} 失败: {err}")
        return False


def update_daily_basic_symbol(code: str) -> tuple[str, bool]:
    """
    单只股票增量更新:
      * 无文件 -> backfill;
      * 已退市 -> 跳过 (行情停更, 估算段不再增长);
      * 否则按 [末日期-3天, 今天] 拉接口回补缺口 (窗口外旧缺失不追溯)。
    """
    normalized = str(code).zfill(6)
    fp = get_fp(normalized)
    if not os.path.exists(fp):
        return normalized, backfill_daily_basic_symbol(normalized)

    if STOCK_LIST.get_delist_date(normalized):
        return normalized, True

    try:
        df = load_daily_basic(normalized)
        last_date = pd.to_datetime(df.index.max()).date()
        now = datetime.datetime.now()
        if now.date() - last_date < datetime.timedelta(days=1):
            return normalized, True

        start = last_date - datetime.timedelta(days=3)
        new_rows = get_stock_daily_basic_em(
            normalized,
            start_date=start.strftime("%Y%m%d"),
            end_date=now.strftime("%Y%m%d"),
        )
        merged = pd.concat([df, new_rows])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "date"
        merged.to_csv(fp, encoding="utf-8-sig", index=True)
        return normalized, True
    except Exception as err:
        traceback.print_exc()
        print(f"Can't update daily_basic for {normalized} because: {err}")
        return normalized, False


def update_daily_basic(symbols: list[str] | None = None) -> list[tuple[str, bool]]:
    """批量增量更新 (默认在册全部股票), 并排复用股票更新管线的并发/代理池。"""
    all_symbols = (
        STOCK_LIST.get_all_symbol()
        if symbols is None
        else [str(s).zfill(6) for s in symbols]
    )
    if not all_symbols:
        print("No daily_basic symbols to update")
        return []

    with Pool(STOCK_UPDATE_POOL_SIZE, initializer=_initialize_stock_update_worker) as pool:
        results = pool.map(update_daily_basic_symbol, all_symbols)

    for code, result in results:
        if not result:
            print(f"Failed to update daily_basic for {code}")
    return results


# ---------------------------------------------------------------------------
# Status check (镜像 stock_data_manager.batch_check_stock_data_updated)
# ---------------------------------------------------------------------------

def batch_check_daily_basic_updated(
    symbols: list[str],
    target_date: str | datetime.date | datetime.datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    返回 status_df(symbol/exists/last_local_date/target_date/is_updated)。
    退市股不参与增量 -> 恒判 is_updated=True, 避免每日进入 failed 列表。
    """
    expected_date = (
        datetime.date.today()
        if target_date is None
        else pd.to_datetime(target_date).date()
    )
    rows = []
    for symbol in symbols:
        normalized = str(symbol).zfill(6)
        delisted = bool(STOCK_LIST.get_delist_date(normalized))
        last_local_date = get_daily_basic_last_local_date(normalized)
        is_updated = True if delisted else bool(last_local_date and last_local_date >= expected_date)
        rows.append(
            {
                "symbol": normalized,
                "exists": os.path.exists(get_fp(normalized)),
                "last_local_date": last_local_date,
                "target_date": expected_date,
                "is_updated": is_updated,
            }
        )
    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["is_updated", "symbol"], ascending=[True, True]).reset_index(drop=True)
    return res
