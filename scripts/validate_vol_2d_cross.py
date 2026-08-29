"""
波动率水平 × 波动率趋势 二维交叉验证

用户假设:
- 波动率开始放大(低水平+上升趋势) → 趋势起点 → 动量因子
- 波动率绝对值极大 → 趋势末端 → C2(均值回归)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent
INCOMING = Path("/root/.openclaw/workspace/opengouzi/incoming")
ETF_DATA_DIR = INCOMING / "etf_data"
OUTPUT_DIR = Path("/root/.openclaw/workspace/opengouzi/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_DATA_DAYS = 900
VOL_WINDOW = 20
VOL_Z_WINDOW = 600
VOL_TREND_WINDOW = 5
VOL_TREND_Z_WINDOW = 120
MOM_WINDOW = 20
MOM_SKIP = 1
MAE_WINDOW = 40
MAE_Z_WINDOW = 120
FWD_PERIODS = (10, 20)


def load_all():
    etfs = {}
    for f in sorted(ETF_DATA_DIR.glob("*.csv")):
        sym = f.stem.lstrip("-")
        try:
            df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        except Exception:
            continue
        if len(df) < MIN_DATA_DAYS:
            continue
        close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
        if len(close) >= MIN_DATA_DAYS:
            etfs[sym] = close.rename("close")
    return etfs


def compute(close):
    ret = close.pct_change()

    vol = ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std()
    vm = vol.rolling(VOL_Z_WINDOW, min_periods=VOL_Z_WINDOW).mean()
    vs = vol.rolling(VOL_Z_WINDOW, min_periods=VOL_Z_WINDOW).std(ddof=1)
    vol_z = (vol - vm) / vs.replace(0, np.nan)

    vol_d = vol - vol.shift(VOL_TREND_WINDOW)
    vdm = vol_d.rolling(VOL_TREND_Z_WINDOW, min_periods=VOL_TREND_Z_WINDOW).mean()
    vds = vol_d.rolling(VOL_TREND_Z_WINDOW, min_periods=VOL_TREND_Z_WINDOW).std(ddof=1)
    vol_trend_z = (vol_d - vdm) / vds.replace(0, np.nan)

    mom = (close.shift(MOM_SKIP) - close.shift(MOM_WINDOW + MOM_SKIP)) / close.shift(MOM_WINDOW + MOM_SKIP)

    roll_min = close.rolling(MAE_WINDOW, min_periods=MAE_WINDOW).min()
    ref = close.shift(MAE_WINDOW)
    mae = (roll_min / ref) - 1.0
    mm = mae.rolling(MAE_Z_WINDOW, min_periods=MAE_Z_WINDOW).mean()
    ms = mae.rolling(MAE_Z_WINDOW, min_periods=MAE_Z_WINDOW).std(ddof=1)
    c2 = -((mae - mm) / ms.replace(0, np.nan))

    df = pd.DataFrame({
        "close": close, "ret": ret, "vol": vol,
        "vol_z": vol_z, "vol_trend_z": vol_trend_z,
        "momentum": mom, "c2": c2,
    }, index=close.index)

    for p in FWD_PERIODS:
        df[f"fwd_{p}d"] = close.shift(-p) / close - 1.0
    return df


def main():
    print("加载数据...")
    etfs = load_all()
    print(f"{len(etfs)} ETFs")

    panels = []
    for sym, close in etfs.items():
        try:
            df = compute(close)
            df["symbol"] = sym
            panels.append(df)
        except Exception:
            pass

    panel = pd.concat(panels).reset_index().rename(columns={"index": "date"})
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= "2020-01-01"]
    print(f"面板: {len(panel):,} 行")

    # ============================================================
    # 二维交叉: vol_z 水平分组 × vol_trend_z 方向分组
    # ============================================================
    valid = panel.dropna(subset=["vol_z", "vol_trend_z", "momentum", "c2"])
    
    # 分组定义
    vol_level_bins = [
        ("低波 (z<0)", lambda df: df["vol_z"] < 0),
        ("正常波 (0≤z<1)", lambda df: (df["vol_z"] >= 0) & (df["vol_z"] < 1)),
        ("高波 (1≤z<2.5)", lambda df: (df["vol_z"] >= 1) & (df["vol_z"] < 2.5)),
        ("极高波 (z≥2.5)", lambda df: df["vol_z"] >= 2.5),
    ]
    
    vol_dir_bins = [
        ("vol下降 (tz<-0.5)", lambda df: df["vol_trend_z"] < -0.5),
        ("vol平稳 (-0.5≤tz<0.5)", lambda df: (df["vol_trend_z"] >= -0.5) & (df["vol_trend_z"] < 0.5)),
        ("vol上升 (tz≥0.5)", lambda df: df["vol_trend_z"] >= 0.5),
    ]

    for period in FWD_PERIODS:
        fwd_col = f"fwd_{period}d"
        print(f"\n{'='*80}")
        print(f"二维交叉验证 — fwd={period}d")
        print(f"              momentum_IC  /  c2_IC   (观测数)")
        print(f"{'':>16}", end="")
        for vd_label, _ in vol_dir_bins:
            print(f"  {vd_label:>28}", end="")
        print()

        for vl_label, vl_fn in vol_level_bins:
            print(f"{vl_label:>16}", end="")
            for vd_label, vd_fn in vol_dir_bins:
                grp = valid[vl_fn(valid) & vd_fn(valid)]
                n = len(grp)
                if n < 200:
                    print(f"  {'—':>28}", end="")
                    continue
                g = grp.dropna(subset=["momentum", "c2", fwd_col])
                if len(g) < 200:
                    print(f"  {'—':>28}", end="")
                    continue
                mom_ic = g[["momentum", fwd_col]].corr(method="spearman").iloc[0, 1]
                c2_ic = g[["c2", fwd_col]].corr(method="spearman").iloc[0, 1]
                print(f"  {mom_ic:+7.4f}/{c2_ic:+7.4f} ({n//1000}k)", end="")
            print()

    # ============================================================
    # 关键场景: "vol刚开始放大" = 低波水平 + vol上升趋势
    #            "vol到极致"     = 极高波水平 (不管方向?)
    # ============================================================
    print(f"\n{'='*80}")
    print("关键场景验证")
    print(f"{'='*80}")

    scenarios = [
        ("① vol开始放大 (低波+上升)", lambda df: (df["vol_z"] < 0) & (df["vol_trend_z"] >= 0.5)),
        ("② vol持续高位 (高波+上升)", lambda df: (df["vol_z"] >= 1) & (df["vol_z"] < 2.5) & (df["vol_trend_z"] >= 0.5)),
        ("③ vol到达极致 (极高波)",     lambda df: df["vol_z"] >= 2.5),
        ("④ vol开始收敛 (极高波+下降)", lambda df: (df["vol_z"] >= 2.5) & (df["vol_trend_z"] < -0.5)),
        ("⑤ 正常低波 (低波+平稳)",     lambda df: (df["vol_z"] < 0) & (df["vol_trend_z"] >= -0.5) & (df["vol_trend_z"] < 0.5)),
    ]

    for period in FWD_PERIODS:
        fwd_col = f"fwd_{period}d"
        print(f"\n--- fwd={period}d ---")
        print(f"{'场景':<30}  {'momentum_IC':>12}  {'c2_IC':>12}  {'n_obs':>10}")
        print("-" * 70)

        for label, fn in scenarios:
            grp = valid[fn(valid)].dropna(subset=["momentum", "c2", fwd_col])
            if len(grp) < 200:
                print(f"{label:<30}  {'—':>12}  {'—':>12}  {len(grp):>10,}")
                continue
            mom_ic = grp[["momentum", fwd_col]].corr(method="spearman").iloc[0, 1]
            c2_ic = grp[["c2", fwd_col]].corr(method="spearman").iloc[0, 1]
            
            # Top-Bottom
            grp_s = grp.copy()
            grp_s["mom_q"] = pd.qcut(grp_s["momentum"], 5, labels=False, duplicates="drop")
            mom_tb = grp_s[grp_s["mom_q"]==4][fwd_col].mean() - grp_s[grp_s["mom_q"]==0][fwd_col].mean()
            grp_s["c2_q"] = pd.qcut(grp_s["c2"], 5, labels=False, duplicates="drop")
            c2_tb = grp_s[grp_s["c2_q"]==4][fwd_col].mean() - grp_s[grp_s["c2_q"]==0][fwd_col].mean()
            
            print(f"{label:<30}  {mom_ic:+12.4f}  {c2_ic:+12.4f}  {len(grp):>10,}")
            print(f"  → Top-Bottom spread:         mom={mom_tb:+.4%}  c2={c2_tb:+.4%}")

    print("\n✅ 完成")


if __name__ == "__main__":
    main()
