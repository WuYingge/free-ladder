"""申万行业分类数据的落盘/刷新/附名（data/const/stock_sw_industry_clf.csv）。

抓取逻辑见 libs/fetcher/industry.py（走代理访问申万宏源官网），
本模块负责：规范化排序、按官方行业标准表为每个 6 位行业代码附上一/二/三级中文名称
（level1_name/level2_name/level3_name），写入 utf-8-sig CSV。

名称按"代码生效时代"选择对应版本的标准表（同一代码在不同版本语义可能不同）：
  * data/const/sw_industry_standard_2021.csv — 2021 版标准树 (人工从官网下载转换)
  * data/const/sw_industry_standard_2014.csv — 2014 版标准树 (人工从官网"2021 版
    修订对照表"的旧侧提取转换; 28 一级/104 二级/227 三级)
2021-07-30 起生效的行用 2021 表; 2014-02-21 起生效的行用 2014 表;
更早时代 (2001 版) 行先查 2014 表再查 2021 表 (代码跨版稳定时近似可用), 都没有则留空。

供 libs/data_manager/providers/sw_industry_provider.py 的 SW_INDUSTRY 加载、
以及 libs/data_manager/datasets.py 的 with_industry 按日时点合并。
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config import DataPath

INDUSTRY_NAME_COLUMNS = ("level1_name", "level2_name", "level3_name")

# 申万行业分类标准版本生效日 (官方; 个股分类文件的计入日期与之对齐)
SW2014_START = pd.Timestamp("2014-02-21")
SW2021_START = pd.Timestamp("2021-07-30")


# ---------------------------------------------------------------------------
# 标准表 (代码 → 一/二/三级中文名称) 按版本加载
# ---------------------------------------------------------------------------

def load_sw_industry_standard_2021() -> pd.DataFrame:
    """读取 2021 版行业标准表 (industry_code/level1_name/level2_name/level3_name)。

    文件缺失或列不全时返回空 DataFrame (调用方按无名称处理)。
    """
    return _load_standard_csv(DataPath.STOCK_SW_INDUSTRY_STANDARD_CSV)


def load_sw_industry_standard_2014() -> pd.DataFrame:
    """读取 2014 版行业标准表 (列结构与 2021 版一致)。"""
    return _load_standard_csv(DataPath.STOCK_SW_INDUSTRY_STANDARD_2014_CSV)


def _load_standard_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except Exception:  # noqa: BLE001 — 名称表缺失不应阻断分类数据
        return pd.DataFrame()
    if df.empty:
        return df
    df.columns = df.columns.str.strip().str.lstrip("\ufeff")
    required = {"industry_code", *INDUSTRY_NAME_COLUMNS}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()
    df["industry_code"] = df["industry_code"].astype(str).str.strip()
    for col in INDUSTRY_NAME_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df[["industry_code", *INDUSTRY_NAME_COLUMNS]]


def _name_code_map(standard: pd.DataFrame) -> dict[str, tuple[str, str, str]]:
    """标准表 → {6 位代码: (一级名, 二级名, 三级名)}; 代码不在表中 → 缺省。"""
    if standard.empty or "industry_code" not in standard.columns:
        return {}
    return {
        row.industry_code: tuple(getattr(row, col) for col in INDUSTRY_NAME_COLUMNS)
        for row in standard.itertuples(index=False)
    }


def _resolve_names_series(
    codes: pd.Series,
    primary: dict[str, tuple[str, str, str]],
    fallback: dict[str, tuple[str, str, str]],
) -> pd.DataFrame:
    """代码 → 名称列: 优先 primary, 代码缺失时回退 fallback, 均无则空串。"""
    series_primary = codes.map(primary)
    series_fallback = codes.map(fallback)
    out = pd.DataFrame(index=codes.index)
    for i, col in enumerate(INDUSTRY_NAME_COLUMNS):
        p = series_primary.map(lambda t: t[i] if isinstance(t, tuple) else None)
        f = series_fallback.map(lambda t: t[i] if isinstance(t, tuple) else None)
        out[col] = p.fillna(f).fillna("")
    return out


def enrich_industry_names(
    df: pd.DataFrame,
    standard_2021: pd.DataFrame | None = None,
    standard_2014: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """按行生效时代选择标准表, 为分类历史 DataFrame 附一/二/三级中文名称。

    :param df: 含 industry_code 与 start_date(可解析为日期) 列的分类历史
    :param standard_2021: 2021 版标准表; None 时自动读取
    :param standard_2014: 2014 版标准表; None 时自动读取
    :return: 追加 level1_name/level2_name/level3_name 列 (无法命名行留空串)
    """
    if standard_2021 is None:
        standard_2021 = load_sw_industry_standard_2021()
    if standard_2014 is None:
        standard_2014 = load_sw_industry_standard_2014()
    m21 = _name_code_map(standard_2021)
    m14 = _name_code_map(standard_2014)
    out = df.copy()
    name_cols = list(INDUSTRY_NAME_COLUMNS)
    if not m21 and not m14:
        for col in name_cols:
            out[col] = ""
        return out

    if "start_date" in out.columns:
        starts = pd.to_datetime(out["start_date"], errors="coerce")
    else:
        starts = pd.Series([pd.NaT] * len(out), index=out.index)
    codes = out["industry_code"].astype(str).str.strip()

    era_2021 = starts >= SW2021_START
    era_2014 = (starts >= SW2014_START) & ~era_2021
    legacy = ~era_2021 & ~era_2014  # 更早时代(2001 版)行: 按代码跨版近似命名
    if m21 and m14:
        # 分时代选取 (回退次序见模块 docstring)
        resolved = pd.concat(
            [
                _resolve_names_series(codes[era_2021], m21, m14),
                _resolve_names_series(codes[era_2014], m14, m21),
                _resolve_names_series(codes[legacy], m14, m21),
            ]
        ).reindex(out.index)
    elif m21:
        resolved = _resolve_names_series(codes, m21, {})
    else:
        resolved = _resolve_names_series(codes, m14, {})
    for col in name_cols:
        out[col] = resolved[col]
    return out


# ---------------------------------------------------------------------------
# 落盘 / 刷新
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def refresh_sw_industry_clf(
    output_path: str | os.PathLike | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """抓取申万行业分类历史并刷新 data/const CSV（自动按时代附中文名称）。

    :param output_path: 输出 CSV 路径，默认 DataPath.STOCK_SW_INDUSTRY_CLF_CSV
    :param quiet: True 时不打印统计信息
    :return: 已写入磁盘的规范化 DataFrame
             (symbol/start_date/industry_code/level*_name/update_time)
    :raises RuntimeError: 抓取失败（fetcher 侧多次重试后仍失败）
    """
    from fetcher.industry import get_stock_sw_industry_clf_hist

    df = get_stock_sw_industry_clf_hist()
    if df.empty:
        raise RuntimeError("refresh_sw_industry_clf: 抓取结果为空")
    return write_sw_industry_clf(df, output_path=output_path, quiet=quiet)


def enrich_sw_industry_clf_file(
    output_path: str | os.PathLike | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """本地重生成: 读取现有分类历史 CSV, 重算名称列后覆盖写回 (无需联网)。

    用于首次升级 schema 或名称表更新后批量刷名。
    """
    path = Path(output_path or DataPath.STOCK_SW_INDUSTRY_CLF_CSV)
    if not path.exists():
        raise FileNotFoundError(f"enrich_sw_industry_clf_file: 文件不存在 {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    return write_sw_industry_clf(df, output_path=path, quiet=quiet)


def write_sw_industry_clf(
    df: pd.DataFrame,
    output_path: str | os.PathLike | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """规范化 + 附名 + 落盘分类历史 DataFrame。

    :return: 写入磁盘的完整 DataFrame; 列顺序:
             symbol/start_date/industry_code/level1_name/level2_name/level3_name/update_time
    """
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str).str.strip().str.zfill(6),
            "start_date": pd.to_datetime(df["start_date"], errors="coerce"),
            "industry_code": df["industry_code"].astype(str).str.strip(),
        }
    )
    if "update_time" in df.columns:
        out["update_time"] = pd.to_datetime(df["update_time"], errors="coerce")
    out = out.dropna(subset=["start_date"])
    out = out[out["symbol"].ne("") & out["industry_code"].ne("")]
    out = out.drop_duplicates(subset=["symbol", "start_date"], keep="last")
    out = out.sort_values(["symbol", "start_date"], ignore_index=True)
    out = enrich_industry_names(out)
    if "update_time" in out.columns:
        out["update_time"] = out["update_time"].dt.date
    out["start_date"] = out["start_date"].dt.date
    out = out[
        ["symbol", "start_date", "industry_code", *INDUSTRY_NAME_COLUMNS]
        + (["update_time"] if "update_time" in out.columns else [])
    ]
    _save(out, Path(output_path or DataPath.STOCK_SW_INDUSTRY_CLF_CSV))

    if not quiet:
        named = out[INDUSTRY_NAME_COLUMNS[0]].ne("").mean()
        print(f"刷新完成: {Path(output_path or DataPath.STOCK_SW_INDUSTRY_CLF_CSV)}")
        print(
            f"行数 {len(out)} | 股票数 {out['symbol'].nunique()} | "
            f"行业代码数 {out['industry_code'].nunique()} | 名称覆盖率 {named:.1%}"
        )
    return out
