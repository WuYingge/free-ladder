"""基于价格相关性检测疑似同指数（重复）ETF 的审计工具。

用法（项目根目录执行）::

    python libs/scripts/detect_duplicate_etfs_by_price.py
    python libs/scripts/detect_duplicate_etfs_by_price.py --threshold 0.9 --min-rows 60
    python libs/scripts/detect_duplicate_etfs_by_price.py --symbols 159541 159289 563360 159338

逻辑说明：
- symbol 池默认取 ETF_INDEX_MAP 的全部选中 ETF（保持顺序去重）
- 通过 get_etf_data_by_symbol 加载每只 ETF 的 close 序列（不直接读 CSV），
  外连接对齐为 date × symbol 矩阵
- 日收益率优先取 CSV 自带的 gain（涨跌幅 %）列构造（每行官方当日涨跌幅，
  数据缺口不影响），无 gain 列时 fallback 到 close.pct_change()
- 两两计算日收益率 Pearson 相关性（pairwise 完整观测，可用共同交易日数
  过少会被 --min-rows 过滤）
- 相关性 > --threshold 的对即为疑似同指数 ETF；ratio_std 为两只 ETF
  价格比的标准差（同指数应接近 0），作为辅助人工复核的参考
- 局限：增强型 ETF（名称含"增强"）跟踪同一基准时相关性通常低于 0.99，
  价格验证无法覆盖，需结合官方跟踪基准判断

输出（默认 output/duplicate_etf_report.csv）：
- symbol_a, symbol_b, corr, rows, ratio_std, name_a, name_b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from data_manager.etf_correlation import find_duplicate_pairs, load_close_matrix
from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP

DEFAULT_OUTPUT = REPO_ROOT / "output" / "duplicate_etf_report.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于价格相关性检测疑似同指数（重复）ETF。"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="日收益率相关性阈值，默认 0.9",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=60,
        help="共同交易日数下限，默认 60",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="自定义 symbol 池；默认取 ETF_INDEX_MAP 全部选中 ETF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="报告 CSV 路径",
    )
    return parser.parse_args()


def symbol_name(symbol: str) -> str:
    info = ETF_INDEX_MAP.get_info(symbol)
    return info["name"] if info else ""


def main() -> int:
    args = parse_args()

    if args.symbols:
        symbols = list(dict.fromkeys(args.symbols))
    else:
        symbols = list(dict.fromkeys(ETF_INDEX_MAP.get_all_symbols()))
    print(f"symbol 池: {len(symbols)} 只")

    close_matrix, ret_matrix, skipped = load_close_matrix(symbols)
    if skipped:
        print(f"加载失败跳过 {len(skipped)} 只: {[s for s, _ in skipped]}")
    if close_matrix.empty:
        print("无可用数据")
        return 1
    print(f"对齐矩阵: {close_matrix.shape[0]} 个交易日 × {close_matrix.shape[1]} 只 ETF")

    pairs = find_duplicate_pairs(close_matrix, ret_matrix, args.threshold, args.min_rows)
    print(f"疑似同指数对（corr>{args.threshold}, 共同行数>={args.min_rows}）: {len(pairs)} 对")

    report = pd.DataFrame(
        [
            {
                **p,
                "name_a": symbol_name(p["symbol_a"]),
                "name_b": symbol_name(p["symbol_b"]),
            }
            for p in pairs
        ]
    )
    if not report.empty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"报告已写入: {args.output}")
        print(report.head(30).to_string(index=False))
    else:
        print("未发现疑似重复对")
    return 0


if __name__ == "__main__":
    sys.exit(main())
