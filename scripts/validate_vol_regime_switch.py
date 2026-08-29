"""
波动率分位 × 因子切换 分桶验证脚本

验证假设:
- 高波动率状态 → 趋势自我强化 → 动量因子有效
- 低/正常波动率状态 → 均值回归主导 → C2(MAE)因子有效

两个维度:
1. 波动率水平: 20日波动率的600日ZScore
2. 波动率趋势: 20日波动率变化的ZScore
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==================== 路径配置 ====================
SRC = Path(__file__).resolve().parent.parent
INCOMING = Path("/root/.openclaw/workspace/opengouzi/incoming")
ETF_DATA_DIR = INCOMING / "etf_data"
CALENDAR_PATH = INCOMING / "const" / "calandar_df.csv"
OUTPUT_DIR = Path("/root/.openclaw/workspace/opengouzi/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 参数配置 ====================
MIN_DATA_DAYS = 900  # ETF最少交易日数

# 因子参数
VOL_WINDOW = 20
VOL_ZSCORE_WINDOW = 600
VOL_TREND_WINDOW = 5
VOL_TREND_ZSCORE_WINDOW = 120
MOM_WINDOW = 20
MOM_SKIP = 1
MAE_WINDOW = 40
MAE_ZSCORE_WINDOW = 120

# 前向收益
FWD_PERIODS = (10, 20)

# ZScore阈值扫描
ZSCORE_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# 分组数
N_GROUPS = 5  # 五分位


def load_etf_data() -> dict[str, pd.DataFrame]:
    """加载所有ETF数据, 返回 {symbol: DataFrame[close]}"""
    etfs = {}
    for f in sorted(ETF_DATA_DIR.glob("*.csv")):
        sym = f.stem.lstrip("-")  # 处理负数命名的文件
        try:
            df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        except Exception:
            continue
        if len(df) < MIN_DATA_DAYS:
            continue
        df = df.sort_index()
        close = pd.to_numeric(df.get("close"), errors="coerce")
        close = close.dropna()
        if len(close) >= MIN_DATA_DAYS:
            etfs[sym] = close.rename("close")
    print(f"加载 {len(etfs)} 只 ETF (≥{MIN_DATA_DAYS}天)")
    return etfs


def compute_factors(close: pd.Series) -> pd.DataFrame:
    """对单只ETF计算所有因子"""
    ret = close.pct_change()

    # ── 波动率 (20日) ──
    vol = ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()

    # ── 波动率 ZScore (600日) ──
    vol_mean = vol.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).mean()
    vol_std = vol.rolling(VOL_ZSCORE_WINDOW, min_periods=VOL_ZSCORE_WINDOW).std(ddof=1)
    vol_z = (vol - vol_mean) / vol_std.replace(0, np.nan)
    vol_z.name = "vol_zscore"

    # ── 波动率趋势: vol_20变化量的 ZScore ──
    vol_delta = vol - vol.shift(VOL_TREND_WINDOW)
    vol_delta_mean = vol_delta.rolling(VOL_TREND_ZSCORE_WINDOW, min_periods=VOL_TREND_ZSCORE_WINDOW).mean()
    vol_delta_std = vol_delta.rolling(VOL_TREND_ZSCORE_WINDOW, min_periods=VOL_TREND_ZSCORE_WINDOW).std(ddof=1)
    vol_trend_z = (vol_delta - vol_delta_mean) / vol_delta_std.replace(0, np.nan)
    vol_trend_z.name = "vol_trend_zscore"

    # ── 动量因子 PriceReturn_20 (skip_recent=1) ──
    past_close = close.shift(MOM_WINDOW + MOM_SKIP)
    ref_close = close.shift(MOM_SKIP)
    mom = (ref_close - past_close) / past_close
    mom.name = "momentum"

    # ── MAE_40 ──
    rolling_min = close.rolling(MAE_WINDOW, min_periods=MAE_WINDOW).min()
    ref = close.shift(MAE_WINDOW)
    mae = (rolling_min / ref) - 1.0
    mae.name = "mae"

    # ── C2: MAE_40__zscore_120__neg ──
    mae_mean = mae.rolling(MAE_ZSCORE_WINDOW, min_periods=MAE_ZSCORE_WINDOW).mean()
    mae_std = mae.rolling(MAE_ZSCORE_WINDOW, min_periods=MAE_ZSCORE_WINDOW).std(ddof=1)
    mae_z = (mae - mae_mean) / mae_std.replace(0, np.nan)
    c2 = -mae_z  # negate
    c2.name = "c2"

    # ── 前向收益 ──
    result = pd.DataFrame(
        {
            "close": close,
            "ret": ret,
            "vol": vol,
            "vol_zscore": vol_z,
            "vol_trend_zscore": vol_trend_z,
            "momentum": mom,
            "c2": c2,
            "mae": mae,
        },
        index=close.index,
    )

    for period in FWD_PERIODS:
        fwd = close.shift(-period) / close - 1.0
        result[f"fwd_{period}d"] = fwd

    return result


def compute_regime_ic_table(
    panel: pd.DataFrame,
    regime_col: str,
    factor_cols: tuple[str, ...],
    period: int,
    thresholds: list[float],
) -> pd.DataFrame:
    """
    按 regime_col 的阈值分两组(高/低), 分别计算各因子的IC。
    
    同时输出五分位(等分)的分组结果。
    """
    fwd_col = f"fwd_{period}d"
    
    # 过滤有效数据
    valid = panel.dropna(subset=[regime_col, "momentum", "c2", fwd_col]).copy()
    
    rows = []
    
    # ── 五分位分组 ──
    valid["regime_quintile"] = pd.qcut(
        valid[regime_col], N_GROUPS, labels=False, duplicates="drop"
    )
    
    for q in range(N_GROUPS):
        grp = valid[valid["regime_quintile"] == q]
        if len(grp) < 1000:
            continue
        
        row = {"regime": f"Q{q+1}", "n_obs": len(grp)}
        row[f"{regime_col}_mean"] = grp[regime_col].mean()
        
        for fc in factor_cols:
            ic = grp[[fc, fwd_col]].dropna().corr(method="spearman").iloc[0, 1]
            row[f"{fc}_IC"] = round(ic, 4)
            
            # Top-Bottom spread
            grp_sorted = grp.dropna(subset=[fc, fwd_col])
            if len(grp_sorted) >= 200:
                grp_sorted["q"] = pd.qcut(grp_sorted[fc], 5, labels=False, duplicates="drop")
                top = grp_sorted[grp_sorted["q"] == 4][fwd_col].mean()
                bot = grp_sorted[grp_sorted["q"] == 0][fwd_col].mean()
                row[f"{fc}_TB"] = round(top - bot, 6)
            else:
                row[f"{fc}_TB"] = np.nan
                
        rows.append(row)
    
    # ── 阈值分组 (ZScore >= threshold = 高波组) ──
    for thresh in thresholds:
        high = valid[valid[regime_col] >= thresh]
        low = valid[valid[regime_col] < thresh]
        
        for label, grp in [("low", low), ("high", high)]:
            if len(grp) < 500:
                continue
            row = {
                "regime": f"{label}(z>={thresh})" if label == "high" else f"{label}(z<{thresh})",
                "threshold": thresh,
                "n_obs": len(grp),
                f"{regime_col}_mean": grp[regime_col].mean(),
            }
            for fc in factor_cols:
                ic = grp[[fc, fwd_col]].dropna().corr(method="spearman").iloc[0, 1]
                row[f"{fc}_IC"] = round(ic, 4)
                
                grp_sorted = grp.dropna(subset=[fc, fwd_col])
                if len(grp_sorted) >= 200:
                    grp_sorted["q"] = pd.qcut(grp_sorted[fc], 5, labels=False, duplicates="drop")
                    top = grp_sorted[grp_sorted["q"] == 4][fwd_col].mean()
                    bot = grp_sorted[grp_sorted["q"] == 0][fwd_col].mean()
                    row[f"{fc}_TB"] = round(top - bot, 6)
                else:
                    row[f"{fc}_TB"] = np.nan
                    
            rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("波动率分位 × 因子切换 分桶验证")
    print(f"波动率窗口: {VOL_WINDOW}日, ZScore参考期: {VOL_ZSCORE_WINDOW}日")
    print(f"波动率趋势窗口: {VOL_TREND_WINDOW}日变化, ZScore: {VOL_TREND_ZSCORE_WINDOW}日")
    print(f"因子: PriceReturn_{MOM_WINDOW}_skip{MOM_SKIP}, MAE_{MAE_WINDOW}__zscore_{MAE_ZSCORE_WINDOW}__neg")
    print("=" * 70)

    # 1. 加载数据
    etfs = load_etf_data()
    if not etfs:
        print("无ETF数据, 退出")
        return

    # 2. 逐只计算因子, 构建面板
    panels = []
    print("计算因子...")
    for sym, close in etfs.items():
        try:
            df = compute_factors(close)
            df["symbol"] = sym
            panels.append(df)
        except Exception as e:
            pass

    panel = pd.concat(panels).reset_index()
    panel.columns = [c if c != "index" else "date" for c in panel.columns]
    if "date" not in panel.columns:
        # index was date from concat
        panel = panel.rename(columns={"index": "date"})
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"面板: {len(panel):,} 行, {panel['symbol'].nunique()} 只ETF")

    # 3. 过滤: 只保留 2020年后的数据
    panel = panel[panel["date"] >= "2020-01-01"]
    print(f"2020年后: {len(panel):,} 行")

    # 4. 打印总体IC基线
    factor_cols = ("momentum", "c2")
    for period in FWD_PERIODS:
        print(f"\n--- 全样本基线 IC (fwd={period}d) ---")
        for fc in factor_cols:
            ic = panel[[fc, f"fwd_{period}d"]].dropna().corr(method="spearman").iloc[0, 1]
            print(f"  {fc}: {ic:.4f}")

    # 5. 按波动率水平分桶
    print("\n" + "=" * 70)
    print("维度1: 波动率水平 (vol_zscore) 分桶")
    print("=" * 70)

    for period in FWD_PERIODS:
        print(f"\n### 前向收益: {period}日 ###")
        tbl = compute_regime_ic_table(
            panel, "vol_zscore", factor_cols, period, ZSCORE_THRESHOLDS
        )
        print(tbl.to_string(index=False))

        # 保存
        out_path = OUTPUT_DIR / f"vol_zscore_regime_fwd{period}d.csv"
        tbl.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  已保存: {out_path}")

    # 6. 按波动率趋势分桶
    print("\n" + "=" * 70)
    print("维度2: 波动率趋势 (vol_trend_zscore) 分桶")
    print("=" * 70)

    for period in FWD_PERIODS:
        print(f"\n### 前向收益: {period}日 ###")
        tbl = compute_regime_ic_table(
            panel, "vol_trend_zscore", factor_cols, period, ZSCORE_THRESHOLDS
        )
        print(tbl.to_string(index=False))

        out_path = OUTPUT_DIR / f"vol_trend_zscore_regime_fwd{period}d.csv"
        tbl.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  已保存: {out_path}")

    # 7. 关键交叉验证表
    print("\n" + "=" * 70)
    print("核心交叉验证: momentum(高波) vs C2(低波) IC对比")
    print("=" * 70)

    for period in FWD_PERIODS:
        fwd_col = f"fwd_{period}d"
        valid = panel.dropna(subset=["vol_zscore", *factor_cols, fwd_col]).copy()

        print(f"\n{fwd_col}:")
        print(f"{'阈值':>8}  {'高波-mom_IC':>12}  {'低波-C2_IC':>12}  {'高波-obs':>10}  {'低波-obs':>10}")
        print("-" * 62)

        for thresh in ZSCORE_THRESHOLDS:
            high = valid[valid["vol_zscore"] >= thresh]
            low = valid[valid["vol_zscore"] < thresh]
            if len(high) < 500 or len(low) < 500:
                continue
            mom_ic_high = high[["momentum", fwd_col]].dropna().corr(method="spearman").iloc[0, 1]
            c2_ic_low = low[["c2", fwd_col]].dropna().corr(method="spearman").iloc[0, 1]
            print(
                f"z>{thresh:<5.1f}  {mom_ic_high:>12.4f}  {c2_ic_low:>12.4f}  "
                f"{len(high):>10,}  {len(low):>10,}"
            )

    print("\n✅ 完成")


if __name__ == "__main__":
    main()
