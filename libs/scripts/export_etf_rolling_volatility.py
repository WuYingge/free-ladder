"""计算 ETF_INDEX_MAP 中所有指数代表 ETF 的 N 日滚动波动率并导出。

用法（项目根目录执行）::

    python libs/scripts/export_etf_rolling_volatility.py
    python libs/scripts/export_etf_rolling_volatility.py --window 252 --output-dir output/etf_volatility
    python libs/scripts/export_etf_rolling_volatility.py --no-annualize

口径说明：
- 日收益率 = close.pct_change()（简单收益）
- 滚动波动率 = 日收益率 rolling(window, min_periods=window).std()
- 默认年化（乘以 sqrt(252)），可用 --no-annualize 关闭
- 复用 libs/factors/volatility.py 中的 Volatility 因子

输出（默认 output/etf_volatility/）：
- etf_volatility_{window}d_snapshot.csv    每个 ETF 最新一期滚动波动率 + 横截面分位
- etf_volatility_{window}d_timeseries.csv  date × symbol 宽表完整时间序列
- etf_volatility_{window}d_meta.json       运行参数与跳过明细

同时复制到 DEFAULT_WINDOWS_PATH（若已配置），用于 Windows 侧消费。
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from config import DataPath
from data_manager.etf_data_manager import get_etf_data_by_symbol
from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP
from factors.volatility import Volatility

DEFAULT_WINDOW = 252
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "etf_volatility"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export rolling volatility for all ETFs in ETF_INDEX_MAP."
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="滚动窗口（交易日）"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录"
    )
    parser.add_argument(
        "--no-annualize",
        action="store_true",
        help="输出原始日频波动率（不乘 sqrt(252)）",
    )
    return parser.parse_args()


def collect_symbol_meta() -> dict[str, dict[str, str]]:
    """从 ETF_INDEX_MAP 收集 symbol → {name, tracked_indices}。

    一个 symbol 可能被多个跟踪指数选中，tracked_indices 用逗号聚合。
    """
    meta: dict[str, dict[str, str]] = {}
    for tracked_index, symbol in ETF_INDEX_MAP.mapping().items():
        if not symbol:
            continue
        entry = meta.setdefault(
            symbol,
            {"name": ETF_INDEX_MAP.get_name(tracked_index), "tracked_indices": []},
        )
        entry["tracked_indices"].append(tracked_index)
    for entry in meta.values():
        entry["tracked_indices"] = ",".join(sorted(entry["tracked_indices"]))
    return meta


def compute_rolling_volatility(
    symbol: str, factor: Volatility
) -> tuple[pd.Series | None, dict[str, object]]:
    """计算单个 symbol 的滚动波动率序列，返回 (以 date 为索引的序列, 状态信息)。"""
    info: dict[str, object] = {"symbol": symbol}
    try:
        etf = get_etf_data_by_symbol(symbol)
    except FileNotFoundError:
        info["status"] = "missing_local_data"
        return None, info
    except Exception as err:
        info["status"] = f"load_error: {err}"
        return None, info

    df = etf.data
    # 本地 CSV 以 utf-8-sig 落盘，普通 read_csv 会把 BOM 留在首列名上
    df.columns = df.columns.str.lstrip("\ufeff")
    if "date" not in df.columns or "close" not in df.columns:
        info["status"] = "missing_columns"
        return None, info

    df = df.assign(date=pd.to_datetime(df["date"], errors="coerce"))
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    info["bar_count"] = int(len(df))
    if len(df) > 0:
        info["first_date"] = df.index[0].date().isoformat()
        info["last_date"] = df.index[-1].date().isoformat()

    if len(df) < factor.warmup_period:
        info["status"] = "insufficient_history"
        return None, info

    series = factor(df)
    series.name = symbol
    info["status"] = "ok"
    return series, info


def copy_outputs_to_windows(files: list[Path], subdir: str = "etf_volatility") -> None:
    """复制产出文件到 Windows 侧目录；未配置 DEFAULT_WINDOWS_PATH 则跳过。"""
    win_dir_str = (
        getattr(DataPath, "DEFAULT_WINDOWS_PATH", "")
        or getattr(DataPath, "DEFAULT_WINDOWS_PATH", "")
    )
    if not win_dir_str:
        print("DEFAULT_WINDOWS_PATH 未配置（检查 .env），跳过 Windows 备份")
        return
    dest_dir = Path(win_dir_str) / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fp in files:
        shutil.copy2(fp, dest_dir / fp.name)
        print(f"已复制到 Windows: {dest_dir / fp.name}")


def main() -> int:
    args = parse_args()
    window = int(args.window)
    annualize = not args.no_annualize
    factor = Volatility(window=window, annualize=annualize)

    symbol_meta = collect_symbol_meta()
    symbols = sorted(symbol_meta.keys())
    print(
        f"ETF_INDEX_MAP 共 {len(symbols)} 个去重 symbol，窗口 {window} 日，年化={annualize}"
    )

    vol_series: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        series, info = compute_rolling_volatility(symbol, factor)
        meta = symbol_meta[symbol]
        latest_vol = None
        if series is not None:
            valid = series.dropna()
            if not valid.empty:
                vol_series[symbol] = series
                latest_vol = float(valid.iloc[-1])
            else:
                info["status"] = "insufficient_history"
        rows.append(
            {
                "symbol": symbol,
                "name": meta["name"],
                "tracked_indices": meta["tracked_indices"],
                "first_date": info.get("first_date"),
                "last_date": info.get("last_date"),
                "bar_count": info.get("bar_count"),
                f"vol_{window}d{'_ann' if annualize else ''}": latest_vol,
                "status": info["status"],
            }
        )

    snapshot = pd.DataFrame(rows)
    vol_col = f"vol_{window}d{'_ann' if annualize else ''}"
    # 横截面百分位（0~1），供后续分组参考
    snapshot["vol_pct_rank"] = snapshot[vol_col].rank(pct=True)
    snapshot = snapshot.sort_values(
        vol_col, ascending=False, na_position="last"
    ).reset_index(drop=True)

    timeseries = (
        pd.DataFrame(vol_series).sort_index() if vol_series else pd.DataFrame()
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_fp = output_dir / f"etf_volatility_{window}d_snapshot.csv"
    timeseries_fp = output_dir / f"etf_volatility_{window}d_timeseries.csv"
    meta_fp = output_dir / f"etf_volatility_{window}d_meta.json"

    snapshot.to_csv(snapshot_fp, index=False, encoding="utf-8-sig")
    timeseries.to_csv(timeseries_fp, index=True, encoding="utf-8-sig")

    skipped = [row for row in rows if row["status"] != "ok"]
    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "window": window,
        "annualize": annualize,
        "return_type": "simple (close.pct_change)",
        "symbol_total": len(symbols),
        "ok_count": len(vol_series),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    meta_fp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # Windows 备份
    copy_outputs_to_windows([snapshot_fp, timeseries_fp, meta_fp])

    print(f"\n快照已写入: {snapshot_fp}")
    print(f"时间序列已写入: {timeseries_fp} (shape={timeseries.shape})")
    print(f"元信息已写入: {meta_fp}")
    print(f"成功 {len(vol_series)} / 跳过 {len(skipped)}")

    if len(vol_series) > 0:
        desc = snapshot[vol_col].describe(
            percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        )
        print(f"\n最新滚动波动率横截面分布（{len(vol_series)} 个标的）：")
        print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

        # 额外打印头尾各 20 供快速浏览
        print(f"\n波动率最高 20：")
        print(
            snapshot[["symbol", "name", vol_col, "status", "bar_count"]]
            .head(20)
            .to_string(index=False)
        )
        print(f"\n波动率最低 20：")
        print(
            snapshot[["symbol", "name", vol_col, "status", "bar_count"]]
            .tail(20)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
