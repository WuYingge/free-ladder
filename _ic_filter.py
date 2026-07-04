#!/usr/bin/env python3
"""筛选 data/factors/ 下所有因子中 5d rank IC 绝对值 > threshold 的因子，输出 CSV。"""
import json, os, glob, csv

FACTORS_DIR = "data/factors"
THRESHOLD = 0.03  # abs(rank_ic_5d_mean) > 0.03
OUTPUT_PATH = f"data/factors/ic5d_abs_gt_{THRESHOLD}.csv"

results = []
errors = []

factor_dirs = sorted(glob.glob(os.path.join(FACTORS_DIR, "*/")))
print(f"共找到 {len(factor_dirs)} 个因子目录")

for d in factor_dirs:
    name = os.path.basename(d.rstrip("/"))
    reports = sorted(glob.glob(os.path.join(d, "report_*.json")))
    if not reports:
        continue
    latest = reports[-1]
    try:
        with open(latest) as f:
            data = json.load(f)
    except Exception as e:
        errors.append((name, str(e)))
        continue

    try:
        rank_mean = data["layer2_predictive"]["rank_ic"]["5"]["summary"]["mean"]
        pearson_mean = data["layer2_predictive"]["pearson_ic"]["5"]["summary"]["mean"]
        if rank_mean is None:
            errors.append((name, "rank_ic mean is None"))
            continue
        rank_abs = abs(rank_mean)
        pearson_abs = abs(pearson_mean) if pearson_mean is not None else 0
    except (KeyError, TypeError) as e:
        errors.append((name, f"missing key: {e}"))
        continue

    if rank_abs > THRESHOLD:
        results.append({
            "factor": name,
            "rank_ic_5d_mean": round(rank_mean, 6),
            "rank_ic_5d_abs": round(rank_abs, 6),
            "pearson_ic_5d_mean": round(pearson_mean, 6) if pearson_mean is not None else "",
        })

results.sort(key=lambda x: x["rank_ic_5d_abs"], reverse=True)

print(f"\n筛选结果: {len(results)} 个因子满足 abs(rank_ic_5d) > {THRESHOLD}")
print(f"错误/缺失: {len(errors)} 个")

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["factor", "rank_ic_5d_mean", "rank_ic_5d_abs", "pearson_ic_5d_mean"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

print(f"结果已保存到: {OUTPUT_PATH}")

print(f"\n--- 符合条件的因子 (abs(rank_ic_5d) > {THRESHOLD}), 共 {len(results)} 个 ---")
print(f"{'因子名':<70} {'rank_ic':>10} {'abs':>8} {'pearson_ic':>10}")
print("-" * 102)
for r in results:
    p = r["pearson_ic_5d_mean"]
    print(f"{r['factor']:<70} {r['rank_ic_5d_mean']:>10.6f} {r['rank_ic_5d_abs']:>8.4f} {p if isinstance(p, str) else f'{p:>10.6f}'}")

if errors:
    print(f"\n--- 错误 ({len(errors)} 个) ---")
    for name, err in errors[:10]:
        print(f"  {name}: {err}")
    if len(errors) > 10:
        print(f"  ... 还有 {len(errors)-10} 个")
