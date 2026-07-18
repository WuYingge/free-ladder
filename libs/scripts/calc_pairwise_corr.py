"""计算两个因子的截面 Spearman 秩相关并持久化到 data/factor_cross_correlation/。
使用与 run_cross_correlation.py 相同的底层 API，结果格式完全一致。

用法:
    PYTHONPATH=libs uv run python libs/scripts/calc_pairwise_corr.py
"""
import sys, os, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "libs"))

from factors.distribution_family import MaxAdverseExcursion
from factors.trend_r2 import TrendR2Factor
from factors.meta_factor import TransformFactor, NegateFactor
from factor_analysis.panel import build_factor_panel
from factor_analysis.cross_correlation import compute_or_load_correlation
from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP
from config import DataPath


def build_pair():
    """构造两个目标因子。"""
    # 因子 A: TrendR2_60_slope__zscore_120__neg
    tr2 = TrendR2Factor(window=60, output="slope")
    tr2_z = TransformFactor(dependency=tr2, transform="zscore", window=120, threshold=0.0)
    fac_a = NegateFactor(tr2_z)

    # 因子 B: MAE_40__zscore_120__neg
    mae = MaxAdverseExcursion(window=40)
    mae_z = TransformFactor(dependency=mae, transform="zscore", window=120, threshold=0.0)
    fac_b = NegateFactor(mae_z)

    return fac_a, fac_b


def main():
    fac_a, fac_b = build_pair()
    name_a = fac_a.get_output_name()
    name_b = fac_b.get_output_name()
    print(f"因子 A: {name_a}")
    print(f"因子 B: {name_b}")

    symbols = ETF_INDEX_MAP.get_all_symbols()
    print(f"标的数: {len(symbols)}")

    # 构建面板
    t0 = time.time()
    panel_a = build_factor_panel(factor=fac_a, symbols=symbols, min_bars=252)
    print(f"面板 A ({name_a}): {panel_a.factor_values.shape} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    panel_b = build_factor_panel(factor=fac_b, symbols=symbols, min_bars=252)
    print(f"面板 B ({name_b}): {panel_b.factor_values.shape} ({time.time()-t0:.1f}s)")

    factor_values = {
        name_a: panel_a.factor_values,
        name_b: panel_b.factor_values,
    }

    # 计算并持久化到 data/factor_cross_correlation/
    output_dir = str(Path(DataPath.DATA_DIR) / "factor_cross_correlation")
    print(f"\n计算 & 落盘 → {output_dir}")

    result = compute_or_load_correlation(
        factor_values=factor_values,
        output_dir=output_dir,
        factor_fingerprints={
            name_a: name_a,
            name_b: name_b,
        },
        symbols_for_fingerprint=symbols,
        rebalance_interval=5,
        min_assets=30,
        end_date="2026-07-05",
        verbose=True,
    )

    print(f"\n有效调仓日数: {result.n_dates[0, 1]}")
    print(f"Spearman 秩相关系数均值: {result.mean_corr[0, 1]:.4f}")
    print(f"Spearman 秩相关系数标准差: {result.std_corr[0, 1]:.4f}")
    print("\n落盘文件:")
    for f in sorted(Path(output_dir).iterdir()):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
