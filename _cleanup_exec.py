#!/usr/bin/env python3
"""清理 data/factors/ 下旧 report 文件，每个目录仅保留最新一组"""
import os, glob, re
from collections import defaultdict

FACTORS_DIR = "data/factors"
all_dirs = sorted(glob.glob(os.path.join(FACTORS_DIR, "*/")))
pattern = re.compile(r"report_(\d{4}-\d{2}-\d{2})\.(json|html|md)$")

deleted = 0
kept = 0

for d in all_dirs:
    name = os.path.basename(d.rstrip("/"))
    dated_files = []
    for f in os.listdir(d):
        m = pattern.match(f)
        if m:
            dated_files.append((m.group(1), f, os.path.join(d, f)))

    if not dated_files:
        continue

    by_date = defaultdict(dict)
    for date_str, fname, fullpath in dated_files:
        ext = fname.rsplit(".", 1)[-1]
        by_date[date_str][ext] = (fname, fullpath)

    latest_date = max(by_date.keys())

    for date_str, exts in by_date.items():
        for ext, (fname, fullpath) in exts.items():
            if date_str == latest_date:
                kept += 1
            else:
                os.remove(fullpath)
                deleted += 1
                print(f"  已删除: {fullpath}")

print(f"\n清理完成: 删除 {deleted} 个旧文件, 保留 {kept} 个最新文件")
