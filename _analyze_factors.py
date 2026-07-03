#!/usr/bin/env python3
"""分析 data/factors/ 下多版本 report 分布"""
import os, glob
from collections import Counter

FACTORS_DIR = "data/factors"
all_dirs = sorted(glob.glob(os.path.join(FACTORS_DIR, "*/")))
report_counts = Counter()
total_old_reports = 0
total_old_html = 0
total_old_md = 0
multi_dirs = []
no_report = 0
single = 0
fp_count = 0

for d in all_dirs:
    name = os.path.basename(d.rstrip("/"))
    files = os.listdir(d)
    if ".task_fingerprint" in files:
        fp_count += 1

    rj = sorted(glob.glob(os.path.join(d, "report_*.json")))
    rh = sorted(glob.glob(os.path.join(d, "report_*.html")))
    rm = sorted(glob.glob(os.path.join(d, "report_*.md")))
    n = len(rj)
    report_counts[n] += 1
    if n == 0:
        no_report += 1
    elif n == 1:
        single += 1
    else:
        total_old_reports += n - 1
        total_old_html += max(0, len(rh) - 1)
        total_old_md += max(0, len(rm) - 1)
        multi_dirs.append((name, n, [os.path.basename(r) for r in rj]))

print(f"总目录: {len(all_dirs)}")
print(f"有 fingerprint: {fp_count}")
print(f"无 report: {no_report}")
print(f"仅1版本: {single}")
print(f"多版本: {len(multi_dirs)}")
print(f"\nreport_*.json 版本分布:")
for n in sorted(report_counts):
    print(f"  {n}版本: {report_counts[n]}")

total_del = total_old_reports + total_old_html + total_old_md
print(f"\n可清理: json={total_old_reports} html={total_old_html} md={total_old_md} 合计={total_del}")

print(f"\n多版本目录示例(前20):")
for name, n, rep in multi_dirs[:20]:
    print(f"  {name:<60} {n}版本 保留:{rep[-1]}")
