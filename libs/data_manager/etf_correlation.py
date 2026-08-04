"""ETF 价格相关性分析与同指数合并的公共逻辑。

提供：
- load_close_matrix: 加载 symbol 池的 close 与日收益率对齐矩阵
- find_duplicate_pairs: 相关性超过阈值的 symbol 对（含 ratio_std 辅助列）
- build_merge_clusters: 把疑似同指数对聚类成簇（并查集，处理传递性）
- pick_representative: 簇内按上市时间（first_date）最早选代表
"""

from __future__ import annotations

import pandas as pd


def load_close_matrix(
    symbols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """加载 symbol 池数据，返回 (close 矩阵, 日收益率矩阵, 跳过列表)。

    两矩阵均为 date × symbol 对齐（外连接）。日收益率优先取 CSV 自带的
    gain（涨跌幅 %）列构造——每行是官方当日涨跌幅，本地序列若有数据
    缺口也不受影响；数据源无 gain 列时 fallback 到 close.pct_change()。
    """
    # 函数内懒加载：模块级 import 会拖入 fetcher → utils.interval_utils
    # 的依赖链（libs/utils 非可导入包），导致轻量场景（如纯函数单测）失败
    from data_manager.etf_data_manager import get_etf_data_by_symbol

    close_frames: dict[str, pd.Series] = {}
    ret_frames: dict[str, pd.Series] = {}
    skipped: list[tuple[str, str]] = []
    for symbol in symbols:
        try:
            etf = get_etf_data_by_symbol(symbol)
            df = etf.data
            df.columns = df.columns.str.lstrip("\ufeff")
            if "date" not in df.columns or "close" not in df.columns:
                skipped.append((symbol, "missing_columns"))
                continue
            df = df.assign(date=pd.to_datetime(df["date"], errors="coerce"))
            df = df.dropna(subset=["date"]).sort_values("date").set_index("date")
            close = df["close"].astype(float)
            if "gain" in df.columns:
                ret = pd.to_numeric(df["gain"], errors="coerce") / 100.0
            else:
                ret = close.pct_change()
            close_frames[symbol] = close
            ret_frames[symbol] = ret
        except Exception as err:
            skipped.append((symbol, str(err)))
    if not close_frames:
        return pd.DataFrame(), pd.DataFrame(), skipped
    close_matrix = pd.concat(close_frames, axis=1, sort=False)
    close_matrix.columns = close_matrix.columns.get_level_values(0)
    ret_matrix = pd.concat(ret_frames, axis=1, sort=False)
    ret_matrix.columns = ret_matrix.columns.get_level_values(0)
    return close_matrix, ret_matrix, skipped


def find_duplicate_pairs(
    close_matrix: pd.DataFrame,
    ret_matrix: pd.DataFrame,
    threshold: float,
    min_rows: int,
) -> list[dict[str, object]]:
    """返回相关性超过阈值的 symbol 对，含共同行数与价格比波动。

    相关性基于日收益率矩阵（gain 列或 pct_change 派生），价格比波动
    基于 close 矩阵。
    """
    if close_matrix.empty or len(close_matrix.columns) < 2:
        return []
    corr = ret_matrix.corr()  # pairwise 完整观测
    pairs: list[dict[str, object]] = []
    syms = list(corr.columns)
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = syms[i], syms[j]
            c = corr.iloc[i, j]
            if pd.isna(c) or c < threshold:
                continue
            pair = close_matrix[[a, b]].dropna()
            if len(pair) < min_rows:
                continue
            ratio_std = float((pair.iloc[:, 1] / pair.iloc[:, 0]).std())
            pairs.append(
                {
                    "symbol_a": a,
                    "symbol_b": b,
                    "corr": round(float(c), 4),
                    "rows": int(len(pair)),
                    "ratio_std": round(ratio_std, 4),
                }
            )
    return sorted(pairs, key=lambda r: -float(r["corr"]))


def build_merge_clusters(pairs: list[dict[str, object]]) -> list[list[str]]:
    """把疑似同指数对聚类成簇（并查集），处理 A-B、B-C 的传递性。"""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for p in pairs:
        a, b = p["symbol_a"], p["symbol_b"]
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    roots: dict[str, list[str]] = {}
    for sym in parent:
        roots.setdefault(find(sym), []).append(sym)
    return [sorted(members) for members in roots.values()]


def pick_representative(cluster: list[str], first_date_map: dict[str, str]) -> str:
    """簇内选上市时间最早（first_date 最小）的 symbol 作为代表。

    first_date 缺失或为空时视为最晚，不参与胜出。
    """
    return min(cluster, key=lambda s: (first_date_map.get(s) or "9999-12-31"))
