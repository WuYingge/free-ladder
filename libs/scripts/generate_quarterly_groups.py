"""生成逐季分组矩阵（point-in-time），基于冻结阈值 <6% / 6-20% / 20-36% / ≥36%。

用法：python libs/scripts/generate_quarterly_groups.py

产出：
- output/etf_volatility/volatility_thresholds.json   冻结阈值配置（断点 + 组名）
- output/etf_volatility/quarterly_groups_wide.csv    quarter × symbol 宽表
- output/etf_volatility/quarterly_groups_long.csv    quarter, symbol, group 长表
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from config import DataPath

THRESHOLDS: list[tuple[float, float, str]] = [
    (0.00, 0.06, "bond"),
    (0.06, 0.20, "low"),
    (0.20, 0.36, "mid"),
    (0.36, 1.00, "high"),
]

OUTPUT_DIR = REPO_ROOT / "output" / "etf_volatility"


def assign_group(vol: float) -> str:
    for lo, hi, label in THRESHOLDS:
        if lo <= vol < hi:
            return label
    return ""


def main() -> None:
    ts_path = OUTPUT_DIR / "etf_volatility_252d_timeseries.csv"
    if not ts_path.exists():
        print(f"时间序列文件不存在: {ts_path}，请先运行 export_etf_rolling_volatility.py")
        return

    ts = pd.read_csv(ts_path, index_col=0, parse_dates=True)
    ts.index = pd.to_datetime(ts.index)

    # 每季度末最后一个实际交易日（日历季末可能无交易）
    ts["_period"] = ts.index.to_period("Q")
    quarter_last_idx = ts.groupby("_period").tail(1).index
    ts.drop(columns=["_period"], inplace=True)

    # 用每个季度末的波动率给各 symbol 打分组标签
    wide_rows: dict[str, dict[str, str]] = {}
    for qe in quarter_last_idx:
        row = ts.loc[qe].copy()
        row = row.dropna()
        groups = row.apply(assign_group)
        # 用季度末数据分组，分组结果用于**下一季度**回测（严格 point-in-time）
        next_quarter = (qe + pd.DateOffset(months=3)).to_period("Q")
        wide_rows[next_quarter.strftime("Q%q-%Y")] = groups.to_dict()

    wide = pd.DataFrame(wide_rows).T.sort_index()
    wide.index.name = "quarter"

    long_records = []
    for quarter, row in wide.iterrows():
        for symbol, group in row.items():
            if isinstance(group, str) and group:
                long_records.append(
                    {"quarter": quarter, "symbol": symbol, "group": group}
                )
    long = pd.DataFrame(long_records).sort_values(["quarter", "symbol"])

    # 保存阈值配置
    thresholds_config = {
        "description": "252d annualized rolling volatility thresholds, frozen once from full-history cross-section",
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "bins": [
            {"label": lab, "lower": lo, "upper": hi} for lo, hi, lab in THRESHOLDS
        ],
    }
    thresholds_fp = OUTPUT_DIR / "volatility_thresholds.json"
    thresholds_fp.write_text(
        json.dumps(thresholds_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    wide_fp = OUTPUT_DIR / "quarterly_groups_wide.csv"
    long_fp = OUTPUT_DIR / "quarterly_groups_long.csv"
    wide.to_csv(wide_fp, encoding="utf-8-sig")
    long.to_csv(long_fp, index=False, encoding="utf-8-sig")

    # 统计
    quarter_count = len(wide)
    symbol_count_per_quarter = wide.notna().sum(axis=1).describe()
    group_counts = long["group"].value_counts().to_dict()

    print(f"阈值配置: {thresholds_fp}")
    print(f"宽表: {wide_fp} ({wide.shape[0]} 季度 × {wide.shape[1]} symbol)")
    print(f"长表: {long_fp} ({long.shape[0]} 条记录)")
    print(f"季度范围: {wide.index[0]} → {wide.index[-1]}")
    print(f"每季度有效 symbol 数: mean={symbol_count_per_quarter['mean']:.0f}")
    print(f"分组总量（跨所有季度）: {group_counts}")

    # Windows 备份
    import shutil
    win_dir = getattr(DataPath, "DEFAULT_WINDOWS_PATH", "") or ""
    if win_dir:
        dest = Path(win_dir) / "etf_volatility"
        dest.mkdir(parents=True, exist_ok=True)
        for fp in [thresholds_fp, wide_fp, long_fp]:
            shutil.copy2(fp, dest / fp.name)
        print(f"已复制到 Windows: {dest}")
    else:
        print("DEFAULT_WINDOWS_PATH 未配置，跳过 Windows 备份")


if __name__ == "__main__":
    main()
