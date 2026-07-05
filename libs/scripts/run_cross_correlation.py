#!/usr/bin/env python3
"""
因子截面相关性分析 CLI (Factor Cross-Correlation CLI)

一键式计算 N 个因子两两之间的截面 Spearman rank 相关性矩阵。
内置指纹缓存：日常运行（end_date 固定）零计算；参数/因子变化时自动增量或全量重算。

用法:
    # 默认：使用 FACTOR_REGISTRY 中所有基础因子
    python libs/scripts/run_cross_correlation.py

    # 展开衍生因子（包含 rolling_mean/delta 等变换）
    python libs/scripts/run_cross_correlation.py --meta-expand

    # 指定部分因子
    python libs/scripts/run_cross_correlation.py --factors PriceReturn TrendR2 RSRS

    # 自定义参数
    python libs/scripts/run_cross_correlation.py \\
        --end-date 2026-06-30 --rebalance-interval 10 --min-assets 50

    # 强制重算
    python libs/scripts/run_cross_correlation.py --force

    # 只检查指纹（不计算）
    python libs/scripts/run_cross_correlation.py --check-only

参数:
    --factors             因子名列表（空格分隔），默认全部 FACTOR_REGISTRY
    --meta-expand         展开衍生因子（含 Transform/Combine/Conditional）
    --rebalance-interval  调仓日期间隔（默认 5）
    --min-assets          最小共同标的数（默认 30）
    --start-date          起始日期 YYYY-MM-DD
    --end-date            结束日期 YYYY-MM-DD（默认昨天）
    --output-dir          输出目录（默认 output/factor_cross_correlation）
    --force               强制重算，跳过指纹检查
    --check-only          只检查指纹是否匹配
    --max-workers          构建面板时的并行 worker 数（默认 8）
    --no-heatmap          不生成热力图
    --no-cluster          不计算聚类
    --no-incremental      禁用增量更新（因子变化时全量重算）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

from factor_analysis.cross_correlation import (
    CrossCorrFingerprint,
    build_cross_corr_fingerprint,
    compute_or_load_correlation,
    compute_data_version,
    check_fingerprint_match,
    load_correlation_results,
)
from factor_analysis.panel import build_factor_panel
from factor_analysis.config import DEFAULT_MIN_BARS
from config import DataPath
from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP

# 复用 FACTOR_REGISTRY（来自 run_factor_analysis.py）
from scripts.run_factor_analysis import FACTOR_REGISTRY, _import_factor


def _compute_factor_fingerprint(
    factor_instance: "BaseFactor",
    end_date: str,
) -> str:
    """计算单个因子的独立指纹。

    包含: 类名 + 模块 + 输出名 + 可哈希参数 + end_date。
    """
    cls = factor_instance.__class__
    params_hashable = factor_instance._hashable_params(factor_instance.params)
    payload = json.dumps(
        {
            "class": cls.__name__,
            "module": cls.__module__,
            "output_name": factor_instance.get_output_name(),
            "params": params_hashable,
            "end_date": end_date,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _resolve_factor_instance(name: str, params: dict | None = None) -> "BaseFactor":
    """从 FACTOR_REGISTRY 中的因子名 + 参数构建因子实例。

    Parameters
    ----------
    name : str
        FACTOR_REGISTRY 键名（如 "PriceReturn"）。
    params : dict | None
        覆盖默认参数的构造参数。None = 使用默认参数。

    Returns
    -------
    BaseFactor
    """
    factor_cls = _import_factor(name)
    final_params = dict(factor_cls.params)  # 类级默认参数
    if params:
        final_params.update(params)
    return factor_cls(**final_params)


def _collect_base_factors(
    factor_names: list[str] | None,
) -> dict[str, "BaseFactor"]:
    """收集基础因子实例（使用 FACTOR_REGISTRY 默认参数）。

    Parameters
    ----------
    factor_names : list[str] | None
        因子名列表。None = 使用 FACTOR_REGISTRY 中所有因子。

    Returns
    -------
    dict[str, BaseFactor]
        因子输出名 → 因子实例。
    """
    if factor_names is None:
        factor_names = sorted(FACTOR_REGISTRY.keys())

    instances: dict[str, "BaseFactor"] = {}
    for fname in factor_names:
        try:
            inst = _resolve_factor_instance(fname)
            out_name = inst.get_output_name()
            instances[out_name] = inst
        except Exception as e:
            print(f"  警告: 跳过因子 {fname}: {e}")

    return instances


def _collect_base_factors_with_grid(
    factor_names: list[str] | None,
) -> dict[str, "BaseFactor"]:
    """收集基础因子 + 参数网格变体。

    对每个基础因子的 FULL_MODE_PARAM_GRIDS 做笛卡尔积展开。

    Returns
    -------
    dict[str, BaseFactor]
        因子输出名 → 因子实例。
    """
    from scripts.run_batch_factor_analysis import FULL_MODE_PARAM_GRIDS

    if factor_names is None:
        factor_names = sorted(FACTOR_REGISTRY.keys())

    instances: dict[str, "BaseFactor"] = {}

    for fname in factor_names:
        if fname not in FACTOR_REGISTRY:
            print(f"  警告: 跳过未知因子 {fname}")
            continue

        param_grid = FULL_MODE_PARAM_GRIDS.get(fname)
        if param_grid is None:
            # 无参数网格 → 只用默认参数
            try:
                inst = _resolve_factor_instance(fname)
                instances[inst.get_output_name()] = inst
            except Exception as e:
                print(f"  警告: 跳过 {fname}: {e}")
            continue

        # 先加入默认参数版本（避免漏掉不在网格中的默认值）
        try:
            inst_default = _resolve_factor_instance(fname)
            instances[inst_default.get_output_name()] = inst_default
        except Exception:
            pass

        # 笛卡尔积展开参数网格
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        import itertools

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            try:
                inst = _resolve_factor_instance(fname, params)
                instances[inst.get_output_name()] = inst
            except Exception as e:
                print(f"  警告: 跳过 {fname}({params}): {e}")

    return instances


def _collect_meta_factors() -> dict[str, "BaseFactor"]:
    """收集所有衍生因子（Transform + Combine + Conditional）。

    复用 run_batch_factor_analysis.py 中的配方生成逻辑。

    Returns
    -------
    dict[str, BaseFactor]
        因子输出名 → 因子实例。
    """
    from scripts.run_batch_factor_analysis import (
        generate_transform_specs,
        generate_combo_specs,
        generate_conditional_specs,
        TRANSFORM_CONFIGS,
        COMBO_WHITELIST,
        CONDITIONAL_WHITELIST,
        FULL_MODE_PARAM_GRIDS,
    )
    from factors.meta_factor import build_meta_factor

    all_factor_names = sorted(FACTOR_REGISTRY.keys())
    transform_names = list(TRANSFORM_CONFIGS.keys())

    specs = []
    # 传入 FULL_MODE_PARAM_GRIDS 以展开参数网格变体
    # （如 AvgDrawdown_120__rolling_mean_10 而非仅 AvgDrawdown_60__rolling_mean_10）
    specs.extend(
        generate_transform_specs(all_factor_names, transform_names,
                                 param_grids=FULL_MODE_PARAM_GRIDS)
    )
    specs.extend(generate_combo_specs(COMBO_WHITELIST))
    specs.extend(generate_conditional_specs(CONDITIONAL_WHITELIST))

    print(f"衍生因子配方数: {len(specs)}")

    instances: dict[str, "BaseFactor"] = {}
    for spec in specs:
        try:
            inst = build_meta_factor(spec)
            out_name = inst.get_output_name()
            instances[out_name] = inst
        except Exception as e:
            print(f"  警告: 跳过衍生因子配方: {e}")

    return instances


def _build_factor_values(
    factor_instances: dict[str, "BaseFactor"],
    symbols: list[str],
    *,
    start_date: str | None,
    end_date: str | None,
    max_workers: int,
    min_bars: int = DEFAULT_MIN_BARS,
) -> dict[str, "pd.DataFrame"]:
    """为每个因子构建 FactorPanel，提取 factor_values。

    Returns
    -------
    dict[str, pd.DataFrame]
        因子输出名 → date × symbol 因子值矩阵。
    """
    import pandas as pd

    factor_values: dict[str, pd.DataFrame] = {}
    total = len(factor_instances)

    for idx, (out_name, factor_inst) in enumerate(factor_instances.items()):
        print(f"  [{idx + 1}/{total}] 构建面板: {out_name}...")
        try:
            panel = build_factor_panel(
                factor=factor_inst,
                symbols=symbols,
                min_bars=min_bars,
                start_date=start_date,
                end_date=end_date,
                max_workers=max_workers,
            )
            factor_values[out_name] = panel.factor_values
            print(f"    → {panel.n_symbols} 标的, {panel.n_dates} 日")
        except Exception as e:
            print(f"    ✗ 失败: {e}")

    return factor_values


def _collect_factors_from_csv(csv_path: str) -> dict[str, "BaseFactor"]:
    """从 CSV 文件读取因子名列表，构建对应的因子实例。

    CSV 需有 'factor' 列（如 factor_details_10d.csv 或 ic10d_abs_gt_0.03.csv）。
    通过生成全量 MetaFactorSpec + 基础因子参数网格，
    然后筛选出 CSV 中的因子名来匹配实例。

    Parameters
    ----------
    csv_path : str
        CSV 文件路径。

    Returns
    -------
    dict[str, BaseFactor]
        因子输出名 → 因子实例（仅含 CSV 中出现的因子）。
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "factor" not in df.columns:
        raise ValueError(f"CSV 缺少 'factor' 列，实际列: {list(df.columns)}")
    target_names = set(df["factor"].dropna().tolist())
    print(f"CSV 因子数: {len(target_names)}")

    # ── 生成全量因子映射 ──
    all_instances: dict[str, "BaseFactor"] = {}

    # 1. 基础因子 + 参数网格
    print("  生成基础因子 + 参数网格...")
    base = _collect_base_factors_with_grid(None)
    all_instances.update(base)
    print(f"    基础因子: {len(base)} 个")

    # 2. 衍生因子
    print("  生成衍生因子配方...")
    meta = _collect_meta_factors()
    all_instances.update(meta)
    print(f"    衍生因子: {len(meta)} 个")

    # ── 筛选 ──
    matched: dict[str, "BaseFactor"] = {}
    missing: set[str] = set()
    for fname in target_names:
        if fname in all_instances:
            matched[fname] = all_instances[fname]
        else:
            missing.add(fname)

    if missing:
        print(f"  警告: {len(missing)} 个因子未匹配:")
        for m in sorted(missing)[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... 还有 {len(missing) - 10} 个")

    print(f"  匹配成功: {len(matched)}/{len(target_names)}")
    return matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="因子截面相关性分析 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="从 CSV 读取因子列表（需有 'factor' 列），自动匹配因子实例",
    )
    parser.add_argument(
        "--factors", type=str, nargs="*",
        help="因子名列表（空格分隔），默认全部 FACTOR_REGISTRY 基础因子",
    )
    parser.add_argument(
        "--meta-expand", action="store_true",
        help="展开衍生因子（含 Transform/Combine/Conditional）",
    )
    parser.add_argument(
        "--with-grid", action="store_true",
        help="对基础因子展开 FULL_MODE_PARAM_GRIDS 参数网格",
    )
    parser.add_argument(
        "--rebalance-interval", type=int, default=5,
        help="调仓日期间隔（交易日，默认 5）",
    )
    parser.add_argument(
        "--min-assets", type=int, default=30,
        help="最小共同标的数（默认 30）",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="起始日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="结束日期 YYYY-MM-DD（默认昨天）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output/factor_cross_correlation",
        help="输出目录（默认 output/factor_cross_correlation）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重算，跳过指纹检查",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="只检查指纹是否匹配，不计算",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8,
        help="构建面板时的并行 worker 数（默认 8）",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="不生成热力图",
    )
    parser.add_argument(
        "--no-cluster", action="store_true",
        help="不计算聚类",
    )
    parser.add_argument(
        "--no-incremental", action="store_true",
        help="禁用增量更新",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── 0. 确定 end_date ──
    end_date = args.end_date
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()
        print(f"end_date 未指定，取昨天: {end_date}")

    # ── 1. 收集因子实例 ──
    print("收集因子实例...")
    factor_instances: dict[str, "BaseFactor"] = {}

    if args.csv:
        print(f"  模式: 从 CSV 读取 ({args.csv})")
        factor_instances.update(_collect_factors_from_csv(args.csv))
    elif args.meta_expand:
        print("  模式: 展开衍生因子")
        factor_instances.update(_collect_meta_factors())
        factor_instances.update(_collect_base_factors(args.factors))
    elif args.with_grid:
        print("  模式: 基础因子 + 参数网格")
        factor_instances.update(_collect_base_factors_with_grid(args.factors))
    else:
        print("  模式: 基础因子（默认参数）")
        factor_instances.update(_collect_base_factors(args.factors))

    if not factor_instances:
        print("错误: 无有效因子")
        return 1

    print(f"共 {len(factor_instances)} 个因子")

    # ── 2. 获取标的列表 ──
    symbols = ETF_INDEX_MAP.get_all_symbols()
    print(f"标的数: {len(symbols)}")

    # ── 3. check-only 模式 ──
    if args.check_only:
        factor_fingerprints = {
            out_name: _compute_factor_fingerprint(inst, end_date)
            for out_name, inst in factor_instances.items()
        }
        fp = build_cross_corr_fingerprint(
            factor_names=list(factor_instances.keys()),
            factor_fingerprints=factor_fingerprints,
            rebalance_interval=args.rebalance_interval,
            min_assets=args.min_assets,
            start_date=args.start_date,
            end_date=end_date,
            symbols=symbols,
        )
        if check_fingerprint_match(fp, args.output_dir):
            print("指纹匹配——缓存有效，无需重算")
            return 0
        else:
            print("指纹不匹配——需要重算")
            cached = load_correlation_results(args.output_dir)
            if cached is not None:
                old_result, old_fp = cached
                old_set = set(old_fp.factor_fingerprints.keys())
                new_set = set(fp.factor_fingerprints.keys())
                added = new_set - old_set
                removed = old_set - new_set
                changed = {
                    f for f in old_set & new_set
                    if old_fp.factor_fingerprints.get(f) != fp.factor_fingerprints.get(f)
                }
                if added:
                    print(f"  新增因子: {sorted(added)}")
                if removed:
                    print(f"  删除因子: {sorted(removed)}")
                if changed:
                    print(f"  修改因子: {sorted(changed)}")
            return 1

    # ── 4. 构建面板 & 提取 factor_values ──
    print("\n构建因子面板...")
    factor_values = _build_factor_values(
        factor_instances=factor_instances,
        symbols=symbols,
        start_date=args.start_date,
        end_date=end_date,
        max_workers=args.max_workers,
    )

    if not factor_values:
        print("错误: 所有因子面板构建失败")
        return 1

    print(f"成功构建 {len(factor_values)} 个因子的面板")

    # ── 5. 计算因子指纹 ──
    factor_fingerprints = {
        out_name: _compute_factor_fingerprint(inst, end_date)
        for out_name, inst in factor_instances.items()
        if out_name in factor_values
    }

    # ── 6. 计算 / 加载结果 ──
    print("\n计算 / 加载相关性矩阵...")

    # 构建指纹
    fp = build_cross_corr_fingerprint(
        factor_names=list(factor_values.keys()),
        factor_fingerprints=factor_fingerprints,
        rebalance_interval=args.rebalance_interval,
        min_assets=args.min_assets,
        start_date=args.start_date,
        end_date=end_date,
        symbols=symbols,
    )

    # 检查是否需要增量
    if not args.force and not args.no_incremental:
        cached = load_correlation_results(args.output_dir)
        if cached is not None:
            _, fp_cached = cached
            if fp.compute_config_hash() == fp_cached.compute_config_hash():
                # 配置一致 → 使用 compute_or_load_correlation 走增量路径
                pass  # 下面的 compute_or_load_correlation 会处理

    result = compute_or_load_correlation(
        factor_values=factor_values,
        output_dir=args.output_dir,
        factor_fingerprints=factor_fingerprints,
        symbols_for_fingerprint=symbols,
        rebalance_interval=args.rebalance_interval,
        min_assets=args.min_assets,
        start_date=args.start_date,
        end_date=end_date,
        force=args.force,
        compute_clusters=not args.no_cluster,
        verbose=True,
    )

    # ── 7. 输出摘要 ──
    print("\n" + "=" * 60)
    print(result.summary_text())

    # 额外输出因子指纹（用于后续手动管理）
    print(f"\n因子指纹列表 ({len(factor_fingerprints)} 个):")
    for fname, fhash in sorted(factor_fingerprints.items()):
        print(f"  {fname}: {fhash}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
