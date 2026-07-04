#!/usr/bin/env python3
"""从因子报告 JSON 中提取 Top 73 IC>0 因子的多维数据，输出 CSV"""

import json, os, csv

REPORT_DIR = "data/factors"
TARGETS = [
    "TrendR2_240_slope__rolling_mean_10", "TrendR2_240_slope__rolling_mean_5",
    "MAPosition_close_250__rolling_mean_5", "MAPosition_close_250__rolling_mean_10",
    "TimeSeriesMomentum_252__rolling_mean_10", "TimeSeriesMomentum_252__rolling_mean_5",
    "TimeSeriesMomentum_240__rolling_mean_5", "MADistance_close_10_40__delta_10",
    "AvgDrawdown_120__rolling_mean_10", "MASlope_close_20_10__delta_10",
    "MASlope_close_20_10__delta_5", "MAPosition_close_250",
    "TimeSeriesMomentum_252", "MASlope_close_20_5__delta_10",
    "MADispersion_close_5_10_20_60__delta_10", "TimeSeriesMomentum_240__rolling_mean_10",
    "MASlope_close_10_5__rolling_mean_5", "NewLowContinuous_50__delta_10",
    "MASlope_close_10_3__rolling_mean_5", "TimeSeriesMomentum_240",
    "CCI_10__binarize_winrate_10_0.0", "AvgDrawdown_120",
    "MFE_20__delta_10",
    "MADispersion_close_5_10_20_60__delta_5", "AvgDrawdown_120__rolling_mean_5",
    "BIAS_close_10__binarize_winrate_10_0.0",
    "BollingerBandPosition_close_10_1.5__binarize_winrate_10_0.0",
    "BollingerBandPosition_close_10_2.0__binarize_winrate_10_0.0",
    "BollingerBandPosition_close_10_2.5__binarize_winrate_10_0.0",
    "CCI_20__rolling_mean_5", "MASlope_close_10_3__rolling_mean_10",
    "MFI_14__rolling_mean_5", "DonchianPosition_20__rolling_mean_5",
    "MA_close_10__pct_change_10", "HighPointPosition_20",
    "MASlope_close_20_3__delta_10", "MAPosition_close_200__rolling_mean_5",
    "HighPointPosition_10__rolling_mean_10", "PriceReturn_10__rolling_mean_10",
    "ShortTermReversal_5__delta_5", "DonchianPosition_10__rolling_mean_10",
    "Stochastic_21_3_K__rolling_mean_5", "WilliamsR_21__rolling_mean_5",
    "MAPosition_close_200__pct_change_10", "MFI_14",
    "Stochastic_14_3_K__rolling_mean_10", "WilliamsR_14__rolling_mean_10",
    "TimeSeriesMomentum_252__binarize_winrate_10_0.0",
    "MADistance_close_20_40__delta_5", "MFI_7__rolling_mean_10",
    "TimeSeriesMomentum_252__binarize_winrate_20_0.0",
    "HighPointPosition_20__rolling_mean_5", "MAPosition_close_200__rolling_mean_10",
    "MAPosition_close_250__binarize_winrate_10_0.0",
    "Stochastic_14_3_K__rolling_mean_5", "WilliamsR_14__rolling_mean_5",
    "MADistance_close_10_60__delta_10", "TrendR2_120_r2__rolling_mean_10",
    "BIAS_close_10__rolling_mean_10", "MADistance_close_5_40__delta_10",
    "MADistance_close_10_40__delta_5", "MASlope_close_10_10__delta_10",
    "RSI_7__rolling_mean_5", "MASlope_close_10_5__rolling_mean_10",
    "TrendR2_240_r2__rolling_mean_10", "MASlope_close_10_10__rolling_mean_5",
    "TrendR2_120_r2__rolling_mean_5", "Stochastic_7_3_K__rolling_mean_10",
    "WilliamsR_7__rolling_mean_10",
    # 当前策略核心因子
    "PriceReturn_20", "RSRS_zscore_adj", "TrendR2_120_r2", "KaufmanER_20",
]

HEADERS = [
    "factor",
    "rank_ic_10d_mean", "rank_ic_10d_std", "rank_ic_10d_ir", "rank_ic_10d_t",
    "skewness", "kurtosis",
    "autocorr_lag10", "autocorr_lag20",
    "report_date",
]

rows = []
for factor_name in TARGETS:
    path = os.path.join(REPORT_DIR, factor_name)
    if not os.path.isdir(path):
        print(f"  MISSING: {factor_name}")
        continue

    reports = sorted([f for f in os.listdir(path) if f.endswith('.json')])
    if not reports:
        print(f"  NO JSON: {factor_name}")
        continue

    with open(os.path.join(path, reports[-1])) as f:
        d = json.load(f)

    l1 = d.get("layer1_quality", {})
    l2 = d.get("layer2_predictive", {})

    ic = l2.get("rank_ic", {}).get("10", {}).get("summary", {})
    dist = l1.get("distribution_stats", {}) or {}
    ac = l1.get("autocorr", []) or []

    ac_lag10 = next((a["mean_autocorr"] for a in ac if a["lag"] == 10), None)
    ac_lag20 = next((a["mean_autocorr"] for a in ac if a["lag"] == 20), None)

    rows.append([
        factor_name,
        ic.get("mean"),
        ic.get("std"),
        ic.get("ir"),
        ic.get("t_stat"),
        dist.get("skewness"),
        dist.get("kurtosis"),
        ac_lag10,
        ac_lag20,
        reports[-1],
    ])
    print(f"  OK: {factor_name}")

out_path = "output/factor_details_10d.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(HEADERS)
    w.writerows(rows)

print(f"\nDone: {len(rows)} factors -> {out_path}")
