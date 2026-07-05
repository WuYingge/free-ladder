#!/usr/bin/env python3
"""
批量因子分析编排器 (Batch Factor Analysis Runner)

一次性对 libs/factors/ 下全部已注册因子执行 factor_analysis，
每个因子支持自定义参数和参数网格扫描，并通过子进程并行加速。

用法:
    # 快速模式：所有因子默认参数，只跑 Layer 1+2（不含分组检验）
    python libs/scripts/run_batch_factor_analysis.py --mode quick

    # 标准模式：所有因子默认参数，跑 Layer 1+2+3
    python libs/scripts/run_batch_factor_analysis.py --mode standard

    # 完整模式：展开参数网格，每个参数组合独立跑 Layer 1+2+3
    python libs/scripts/run_batch_factor_analysis.py --mode full

    # 自定义：只分析特定因子族
    python libs/scripts/run_batch_factor_analysis.py \\
        --families 价格动量族 均线偏离族 \\
        --mode standard

    # 控制并行度
    python libs/scripts/run_batch_factor_analysis.py \\
        --mode quick --parallel 4 --max-workers 2

    # 强制重跑（默认会跳过 30 天内跑过且参数未变的因子）
    python libs/scripts/run_batch_factor_analysis.py --mode full --force

    # 延长数据有效期到 90 天
    python libs/scripts/run_batch_factor_analysis.py --mode full --max-age 90

    # 只分析指定因子
    python libs/scripts/run_batch_factor_analysis.py \\
        --factors PriceReturn TrendR2 RSRS --mode standard

参数:
    --mode        分析模式: quick / standard / full
    --factors     指定因子列表（空格分隔，默认全量）
    --families    指定因子族（如 价格动量族，默认全部）
    --parallel    并行度：同时运行的因子分析进程数（默认 16）
    --max-workers 每个因子分析内部的多进程 worker 数（默认 4）
    --resume      断点续跑（默认启用，跳过未过期且参数未变的因子）
    --force       强制重跑：忽略断点续跑，全量重跑
    --max-age     报告有效天数（默认 30，设为 0 表示永不过期）
    --dry-run     试运行：只打印会执行的分析任务，不实际运行
    --output-dir  汇总输出目录（默认 data/factors/_batch_summary/）
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import itertools
from datetime import date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))           # 让 libs 作为包可见（配置模块需要）
LIBS_DIR = REPO_ROOT / "libs"
if str(LIBS_DIR) not in sys.path:
    sys.path.insert(0, str(LIBS_DIR))

# 复用 CLI 中的 FACTOR_REGISTRY
from scripts.run_factor_analysis import FACTOR_REGISTRY, _import_factor

# 配置变量占位（由 main() 从配置文件加载后赋值）
FACTOR_FAMILIES: dict = {}
FULL_MODE_PARAM_GRIDS: dict = {}
CUSTOM_THRESHOLDS: dict = {}
TRANSFORM_CONFIGS: dict = {}
EXCLUSIONS: dict = {}
COMBO_WHITELIST: list = []
CONDITIONAL_WHITELIST: list = []


# ═══════════════════════════════════════════════════════════════════════════════
# 分析任务定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisTask:
    """单个因子分析任务的定义。"""
    factor_name: str          # FACTOR_REGISTRY 键名 或 衍生因子输出名
    factor_cls: type | None = None  # 基础因子类（meta_spec 存在时为 None）
    default_params: dict = field(default_factory=dict)
    layers: tuple[int, ...] = (1, 2)  # 分析层
    param_grid: dict | None = None   # 参数网格（None 表示不扫）
    extra_args: list[str] = field(default_factory=list)  # 额外 CLI 参数
    meta_spec: "MetaFactorSpec | None" = None  # 衍生因子配方

    @property
    def cli_args(self) -> list[str]:
        """生成 subprocess 调用 CLI 的参数列表。"""
        args = [
            sys.executable,
            str(LIBS_DIR / "scripts" / "run_factor_analysis.py"),
            "--layers", *[str(l) for l in self.layers],
        ]
        if self.meta_spec is not None:
            # 衍生因子路径：传 --meta-spec JSON
            from dataclasses import asdict
            spec_dict = asdict(self.meta_spec)
            args.extend(["--meta-spec", json.dumps(spec_dict, default=str)])
        else:
            # 基础因子路径：传 --factor + --param
            args.extend(["--factor", self.factor_name])
            for k, v in self.default_params.items():
                if isinstance(v, list):
                    vals = ",".join(str(x) for x in v)
                    args.extend(["--param", f"{k}=[{vals}]"])
                else:
                    args.extend(["--param", f"{k}={v}"])
        # 参数网格
        if self.param_grid:
            args.extend(["--param-grid", json.dumps(self.param_grid)])
        # 额外参数
        args.extend(self.extra_args)
        return args

    @property
    def output_dir(self) -> Path:
        """预期的输出目录路径。"""
        from config import DataPath
        if self.meta_spec is not None:
            # 衍生因子: 用 build_meta_factor 实例化以获取 output_name
            try:
                from factors.meta_factor import build_meta_factor
                factor_inst = build_meta_factor(self.meta_spec)
                sanitized = factor_inst.get_output_name().replace("/", "_").replace("\\", "_").replace(":", "_")
            except Exception:
                sanitized = self.factor_name
        else:
            # 基础因子: 用 factor_cls + default_params 实例化
            try:
                factor_inst = self.factor_cls(**self.default_params)
                sanitized = factor_inst.get_output_name().replace("/", "_").replace("\\", "_").replace(":", "_")
            except Exception:
                sanitized = self.factor_name
        return Path(DataPath.DATA_DIR) / "factors" / sanitized

    def _latest_report(self) -> Path | None:
        """返回 output_dir 中最新的 report_*.json 路径，不存在时返回 None。"""
        if not self.output_dir.exists():
            return None
        reports = sorted(self.output_dir.glob("report_*.json"))
        return reports[-1] if reports else None

    @property
    def report_json(self) -> Path | None:
        """最新的 report JSON 路径（向后兼容）。"""
        return self._latest_report()

    @property
    def report_end_date(self) -> str | None:
        """从最新报告中提取面板结束日期（YYYY-MM-DD），失败/不存在时返回 None。"""
        rp = self._latest_report()
        if rp is None:
            return None
        try:
            with open(rp) as f:
                report = json.load(f)
            return report.get("panel_summary", {}).get("end_date")
        except Exception:
            return None

    # ── 配置指纹 ──────────────────────────────────────────────────────────
    # 用于检测"这次跑的配置"和"上次跑的配置"是否一致，
    # 避免 layers / param_grid / forward_periods 等参数变了还没重跑。

    @property
    def config_fingerprint(self) -> str:
        """基于 (factor_name, layers, params, param_grid) 的 8 位短哈希。

        任何参数变化都会导致指纹不同，从而触发重跑。
        """
        payload = json.dumps({
            "factor": self.factor_name,
            "layers": list(self.layers),
            "params": self.default_params,
            "param_grid": self.param_grid,
            # 不包含 extra_args 中的 --forward-periods / --min-bars
            # 因为这些视为"数据窗口"参数而非"因子配置"参数，
            # 窗口不同的结果应共存而非互相覆盖。
            # 如需加入，在此添加。
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def fingerprint_matches(self) -> bool | None:
        """比对当前配置指纹和上次运行的指纹。

        Returns
        -------
        True:  指纹一致，配置没变
        False: 指纹存在但不一致，配置变了需要重跑
        None:  没有指纹文件（首次运行）
        """
        fp_path = self.output_dir / ".task_fingerprint"
        if not fp_path.exists():
            return None
        try:
            saved = fp_path.read_text().strip()
            return saved == self.config_fingerprint
        except Exception:
            return None

    def save_fingerprint(self) -> None:
        """保存当前配置指纹到 output_dir。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fp_path = self.output_dir / ".task_fingerprint"
        fp_path.write_text(self.config_fingerprint)

    # ── 数据新鲜度 ────────────────────────────────────────────────────────

    def is_fresh(self, max_age_days: int | None = None) -> bool:
        """报告是否存在且数据足够新鲜。

        Parameters
        ----------
        max_age_days: 报告 end_date 距今超过此天数视为过期。None = 永不视为过期。

        Returns
        -------
        bool: True = 报告存在且数据在 max_age_days 以内。
        """
        end_str = self.report_end_date
        if end_str is None:
            return False  # 报告不存在
        if max_age_days is None:
            return True   # 不检查数据新鲜度
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
            age = (date.today() - end_date).days
            return age <= max_age_days
        except ValueError:
            return False

    @property
    def is_done(self) -> bool:
        """检查是否已完成分析（output_dir 中存在任一 report_*.json）。

        注意：此方法不检查数据新鲜度。如需检查新鲜度请用 is_fresh()。
        """
        return self._latest_report() is not None

    @property
    def label(self) -> str:
        dir_name = self.output_dir.name
        end_str = self.report_end_date
        return f"{dir_name}" + (f" (已有报告, 数据至 {end_str})" if end_str else "")


# ═══════════════════════════════════════════════════════════════════════════════
# 批量编排器
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 衍生因子配方生成 (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════════


def _registry_entry(name: str) -> tuple[str, str, dict]:
    """获取 FACTOR_REGISTRY 中的条目: (module, class, params)。"""
    if name not in FACTOR_REGISTRY:
        raise ValueError(f"未知因子: {name!r}")
    module_path, class_name, default_params = FACTOR_REGISTRY[name]
    return module_path, class_name, dict(default_params)


def generate_transform_specs(
    factor_names: list[str],
    transforms: list[str],
    param_grids: dict[str, dict[str, list]] | None = None,
) -> list["MetaFactorSpec"]:
    """对每个基础因子 × 每种变换 × 每个 window/阈值，生成 MetaFactorSpec。

    根据 TRANSFORM_CONFIGS（多窗口）、EXCLUSIONS（排除规则）、
    CUSTOM_THRESHOLDS（多阈值）生成全量衍生因子配方。
    若提供 param_grids，对每个参数变体也生成独立的衍生因子 spec。

    Parameters
    ----------
    factor_names: 基础因子名列表（FACTOR_REGISTRY 键）。
    transforms: 变换类型列表，如 ["rolling_mean", "delta"]。值必须在 TRANSFORM_CONFIGS 中。
    param_grids: 可选，参数网格 {因子名: {参数名: [值, ...], ...}}。与 mode=full 共用 FULL_MODE_PARAM_GRIDS。

    Returns
    -------
    list[MetaFactorSpec]
    """
    from factors.meta_factor import MetaFactorSpec

    def _gen_for_params(name, module, cls, base_params):
        """对一组 base_params 生成 (因子名, 模块, 类, 参数) 的所有衍生 spec。"""
        for t_name in transforms:
            if t_name not in TRANSFORM_CONFIGS:
                continue
            if name in EXCLUSIONS.get(t_name, set()):
                continue

            cfg = TRANSFORM_CONFIGS[t_name]
            for window in cfg["windows"]:
                if t_name == "binarize_winrate":
                    thresholds = CUSTOM_THRESHOLDS.get(name, [0.0])
                else:
                    thresholds = [None]

                for threshold in thresholds:
                    meta_params: dict[str, Any] = {
                        "transform": t_name,
                        "window": window,
                    }
                    if threshold is not None:
                        meta_params["threshold"] = float(threshold)

                    specs.append(MetaFactorSpec(
                        base_factor_name=name,
                        base_factor_module=module,
                        base_factor_class=cls,
                        base_params=dict(base_params),
                        meta_type="transform",
                        meta_params=meta_params,
                    ))

    specs = []
    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            print(f"警告: 未知因子 '{name}'，跳过衍生")
            continue
        module, cls, default_params = _registry_entry(name)

        # 1. 默认参数的衍生因子
        _gen_for_params(name, module, cls, default_params)

        # 2. 参数网格展开（mode=full）
        if param_grids and name in param_grids:
            grid = param_grids[name]
            param_names = list(grid.keys())
            param_values = list(grid.values())
            for combo in itertools.product(*param_values):
                combo_params = dict(zip(param_names, combo))
                merged = dict(default_params)
                merged.update(combo_params)
                # 跳过与默认参数完全相同的组合
                if merged == default_params:
                    continue
                _gen_for_params(name, module, cls, merged)

    return specs


def generate_combo_specs(whitelist: list[dict]) -> list["MetaFactorSpec"]:
    """从组合白名单生成 MetaFactorSpec。

    白名单格式:
        {"a": "因子名", "a_params": {...}, "method": "...", "b": "因子名", "b_params": {...}}
    """
    from factors.meta_factor import MetaFactorSpec

    specs = []
    for entry in whitelist:
        a_module, a_cls, a_params = _registry_entry(entry["a"])
        b_module, b_cls, b_params = _registry_entry(entry["b"])
        # 合并 params: registry default + 白名单覆盖
        a_merged = dict(a_params)
        a_merged.update(entry.get("a_params", {}))
        b_merged = dict(b_params)
        b_merged.update(entry.get("b_params", {}))

        meta_params = {
            "method": entry["method"],
            "_b_module": b_module,
            "_b_class": b_cls,
            "_b_params": b_merged,
        }
        # 可选参数
        for opt in ("weight_a", "weight_b", "normalize", "normalize_window"):
            if opt in entry:
                meta_params[opt] = entry[opt]

        specs.append(MetaFactorSpec(
            base_factor_name=entry["a"],
            base_factor_module=a_module,
            base_factor_class=a_cls,
            base_params=a_merged,
            meta_type="combine",
            meta_params=meta_params,
        ))
    return specs


def generate_conditional_specs(whitelist: list[dict]) -> list["MetaFactorSpec"]:
    """从条件因子白名单生成 MetaFactorSpec。

    白名单格式:
        {"signal": "因子名", "signal_params": {...},
         "condition": "因子名", "condition_params": {...},
         "op": "gt", "threshold": 0.5, "false_value": "nan"}
    """
    from factors.meta_factor import MetaFactorSpec

    specs = []
    for entry in whitelist:
        s_module, s_cls, s_params = _registry_entry(entry["signal"])
        c_module, c_cls, c_params = _registry_entry(entry["condition"])
        s_merged = dict(s_params)
        s_merged.update(entry.get("signal_params", {}))
        c_merged = dict(c_params)
        c_merged.update(entry.get("condition_params", {}))

        meta_params = {
            "op": entry.get("op", "gt"),
            "threshold": entry.get("threshold", 0.0),
            "false_value": entry.get("false_value", "nan"),
            "_cond_module": c_module,
            "_cond_class": c_cls,
            "_cond_params": c_merged,
        }
        specs.append(MetaFactorSpec(
            base_factor_name=entry["signal"],
            base_factor_module=s_module,
            base_factor_class=s_cls,
            base_params=s_merged,
            meta_type="conditional",
            meta_params=meta_params,
        ))
    return specs


def _make_meta_task(spec: "MetaFactorSpec", mode: str, extra_args: list[str]) -> AnalysisTask:
    """从 MetaFactorSpec 和模式构造 AnalysisTask。"""
    # 用 spec 实例化一次获取 output_name 作为 factor_name
    from factors.meta_factor import build_meta_factor
    try:
        factor_inst = build_meta_factor(spec)
        factor_name = factor_inst.get_output_name()
    except Exception:
        factor_name = f"{spec.base_factor_name}__{spec.meta_type}"

    layers = (1, 2) if mode == "quick" else (1, 2, 3)
    return AnalysisTask(
        factor_name=factor_name,
        factor_cls=None,
        layers=layers,
        extra_args=list(extra_args),
        meta_spec=spec,
    )


def build_tasks(
    factor_names: list[str],
    mode: str,
    extra_args: list[str] | None = None,
) -> list[AnalysisTask]:
    """根据模式和因子列表构建分析任务。

    Parameters
    ----------
    factor_names: 要分析的因子名列表。
    mode: "quick" | "standard" | "full"
    extra_args: 传递给 CLI 的额外参数（如 --max-workers）。

    Returns
    -------
    list[AnalysisTask]
    """
    if extra_args is None:
        extra_args = []

    tasks = []

    for name in factor_names:
        if name not in FACTOR_REGISTRY:
            print(f"警告: 未知因子 '{name}'，跳过")
            continue

        module_path, class_name, default_params = FACTOR_REGISTRY[name]
        factor_cls = _import_factor(name)

        if mode == "quick":
            # 快速模式：默认参数，只跑 Layer 1+2（因子质量 + IC）
            tasks.append(AnalysisTask(
                factor_name=name,
                factor_cls=factor_cls,
                default_params=dict(default_params),
                layers=(1, 2),
                param_grid=None,
                extra_args=list(extra_args),
            ))
        elif mode == "standard":
            # 标准模式：默认参数，跑完整 Layer 1+2+3
            tasks.append(AnalysisTask(
                factor_name=name,
                factor_cls=factor_cls,
                default_params=dict(default_params),
                layers=(1, 2, 3),
                param_grid=None,
                extra_args=list(extra_args),
            ))
        elif mode == "full":
            # 完整模式：展开参数网格，每个参数组合独立跑完整 Layer 1+2+3
            pg = FULL_MODE_PARAM_GRIDS.get(name)
            if pg:
                param_names = list(pg.keys())
                param_values = list(pg.values())
                for combo in itertools.product(*param_values):
                    combo_params = dict(zip(param_names, combo))
                    merged = dict(default_params)
                    merged.update(combo_params)
                    tasks.append(AnalysisTask(
                        factor_name=name,
                        factor_cls=factor_cls,
                        default_params=merged,
                        layers=(1, 2, 3),
                        param_grid=None,
                        extra_args=list(extra_args),
                    ))
            else:
                tasks.append(AnalysisTask(
                    factor_name=name,
                    factor_cls=factor_cls,
                    default_params=dict(default_params),
                    layers=(1, 2, 3),
                    param_grid=None,
                    extra_args=list(extra_args),
                ))

    return tasks


def build_meta_tasks(
    factor_names: list[str],
    mode: str,
    generate_meta: list[str],
    combos_extra: list[str] | None,
    conditionals_extra: list[str] | None,
    extra_args: list[str] | None = None,
) -> list[AnalysisTask]:
    """构建元因子衍生任务。

    Parameters
    ----------
    factor_names: 基础因子名列表（用于 transform 衍生）。
    mode: "quick" | "standard" | "full"
    generate_meta: 要生成的衍生类型列表。
        "all" = 全部 6 种变换; 其他有效值: TRANSFORM_CONFIGS 的键 / "combos" / "conditionals".
    combos_extra: CLI --combos 传入的额外组合（字符串格式，V2 实现）。
    conditionals_extra: CLI --conditionals 传入的额外条件（字符串格式，V2 实现）。
    extra_args: 传递给 CLI 的额外参数。

    Returns
    -------
    list[AnalysisTask]
    """
    if extra_args is None:
        extra_args = []

    tasks = []
    ALL_TRANSFORMS = list(TRANSFORM_CONFIGS.keys())

    # 解析 generate_meta: "all" → 展开为全部 6 种变换
    meta_set = set(generate_meta)
    do_transforms: list[str] = []
    do_combos = False
    do_conditionals = False

    for item in meta_set:
        if item == "all":
            do_transforms = list(ALL_TRANSFORMS)
        elif item in TRANSFORM_CONFIGS:
            do_transforms.append(item)
        elif item == "combos":
            do_combos = True
        elif item == "conditionals":
            do_conditionals = True
        else:
            print(f"警告: 未知 --generate-meta 值 '{item}'，跳过")

    # ── 1. 单因子变换 ──────────────────────────────────────────────────────
    if do_transforms:
        pg = FULL_MODE_PARAM_GRIDS if mode == "full" else None
        specs = generate_transform_specs(factor_names, do_transforms, pg)
        for spec in specs:
            tasks.append(_make_meta_task(spec, mode, extra_args))

    # ── 2. 因子组合 ────────────────────────────────────────────────────────
    if do_combos:
        # 合并 Python 常量 + CLI 传入（CLI 传入暂为字符串，V2 解析）
        combo_entries = list(COMBO_WHITELIST)
        # TODO: 解析 combos_extra 字符串 → dict 追加到 combo_entries
        specs = generate_combo_specs(combo_entries)
        for spec in specs:
            tasks.append(_make_meta_task(spec, mode, extra_args))

    # ── 3. 条件因子 ────────────────────────────────────────────────────────
    if do_conditionals:
        cond_entries = list(CONDITIONAL_WHITELIST)
        # TODO: 解析 conditionals_extra 字符串 → dict 追加到 cond_entries
        specs = generate_conditional_specs(cond_entries)
        for spec in specs:
            tasks.append(_make_meta_task(spec, mode, extra_args))

    return tasks


def run_task(task: AnalysisTask) -> dict[str, Any]:
    """在子进程中运行单个分析任务。

    Returns
    -------
    dict: {"factor": str, "success": bool, "output_dir": str, "runtime_sec": float, "error": str|None}
    """
    start = time.monotonic()
    result = {
        "factor": task.factor_name,
        "success": False,
        "output_dir": str(task.output_dir),
        "runtime_sec": 0.0,
        "error": None,
    }

    try:
        proc = subprocess.run(
            task.cli_args,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 单因子最长 2 小时
        )
        result["runtime_sec"] = time.monotonic() - start

        if proc.returncode == 0:
            result["success"] = True
            # 尝试解析输出获取文件列表
            stdout = proc.stdout
            result["stdout_tail"] = stdout[-500:] if len(stdout) > 500 else stdout
        else:
            result["error"] = (
                f"exit_code={proc.returncode}\n"
                f"STDERR: {proc.stderr[-500:]}\n"
                f"STDOUT: {proc.stdout[-200:]}"
            )
    except subprocess.TimeoutExpired:
        result["runtime_sec"] = 7200
        result["error"] = "超时（2小时）"
    except Exception as e:
        result["runtime_sec"] = time.monotonic() - start
        result["error"] = f"{type(e).__name__}: {e}"

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 汇总报告生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_summary_csv(results: list[dict], output_dir: Path) -> Path:
    """生成汇总 CSV，包含每个因子分析的 key metrics。"""
    import pandas as pd

    rows = []
    for r in results:
        row = {
            "factor": r["factor"],
            "success": r["success"],
            "runtime_sec": round(r.get("runtime_sec", 0), 1),
            "output_dir": r.get("output_dir", ""),
            "error": (r.get("error", "") or "")[:200],
        }

        # 尝试从 report_*.json 提取关键指标（取最新一个）
        factor_dir = Path(r.get("output_dir", ""))
        report_files = sorted(factor_dir.glob("report_*.json")) if factor_dir.exists() else []
        report_path = report_files[-1] if report_files else None
        if report_path and report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)

                panel = report.get("panel_summary", {})
                row["n_symbols"] = panel.get("n_symbols")
                row["n_dates"] = panel.get("n_dates")
                row["start_date"] = panel.get("start_date")
                row["end_date"] = panel.get("end_date")
                row["coverage_mean"] = panel.get("coverage_mean")

                # Layer 2: IC（取 20 日持仓期）
                predictive = report.get("layer2_predictive", {})
                rank_ic = predictive.get("rank_ic", {})
                for period_key, ic_data in rank_ic.items():
                    summary = ic_data.get("summary", {})
                    period_int = int(period_key)
                    row[f"rank_ic_mean_{period_int}d"] = summary.get("mean")
                    row[f"rank_ic_ir_{period_int}d"] = summary.get("ir")

                # 参数网格最佳参数
                param_grid = predictive.get("param_grid", {})
                best = param_grid.get("best_params", {})
                if best:
                    row["best_params"] = json.dumps(best, default=str)
                    row["best_ic_mean"] = best.get("ic_mean")
            except Exception as e:
                row["report_parse_error"] = str(e)[:100]

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "batch_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n汇总 CSV 已保存: {csv_path}")
    print(f"  总任务: {len(rows)} | 成功: {df['success'].sum()} | 失败: {(~df['success']).sum()}")
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量因子分析 — 一键跑完全部因子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="quick",
        choices=["quick", "standard", "full"],
        help="分析模式: quick (Layer 1+2), standard (Layer 1+2+3), full (展开参数网格，每个组合独立跑 Layer 1+2+3)",
    )
    parser.add_argument(
        "--factors", nargs="*", default=None,
        help="指定因子列表（空格分隔），默认全量",
    )
    parser.add_argument(
        "--families", nargs="*", default=None,
        help=f"指定因子族，可选: {list(FACTOR_FAMILIES.keys())}",
    )
    parser.add_argument(
        "--parallel", type=int, default=16,
        help="并行度：同时运行的因子分析进程数（默认 16）",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="每个因子内部的多进程 worker 数（默认 4）",
    )
    parser.add_argument(
        "--resume", action="store_true", default=True,
        help="断点续跑：跳过已有报告的因子（默认启用，可与 --max-age 配合用）",
    )
    parser.add_argument(
        "--max-age", type=int, default=30,
        metavar="DAYS",
        help="报告有效天数。报告 end_date 距今超过此天数则视为过期、触发重跑。"
             " 默认 30 天；设为 0 表示永不过期（只要指纹匹配就跳过）。",
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="强制重跑：忽略已有报告和断点续跑，全量重跑",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="试运行：只打印任务列表，不实际运行",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="汇总输出目录（默认 data/factors/_batch_summary/）",
    )
    parser.add_argument(
        "--min-bars", type=int, default=252,
        help="最少交易日数，传递给 CLI",
    )
    parser.add_argument(
        "--forward-periods", nargs="+", type=int, default=[5, 10, 20, 60],
        help="前向持仓期，传递给 CLI",
    )
    parser.add_argument(
        "--generate-meta", nargs="*", default=None,
        help=(
            "生成衍生因子。可用值: all (全部6种变换), rolling_mean, rolling_std, "
            "delta, pct_change, binarize_winrate, zscore, combos, conditionals。"
            " 默认: 不生成衍生因子，只跑基础因子。"
        ),
    )
    parser.add_argument(
        "--combos", nargs="*", default=None,
        dest="combos_extra",
        help="额外因子组合（字符串格式，V2 支持），与 COMBO_WHITELIST 合并。",
    )
    parser.add_argument(
        "--conditionals", nargs="*", default=None,
        dest="conditionals_extra",
        help="额外条件因子（字符串格式，V2 支持），与 CONDITIONAL_WHITELIST 合并。",
    )
    return parser.parse_args(argv)


def main() -> int:
    global FACTOR_FAMILIES, FULL_MODE_PARAM_GRIDS, CUSTOM_THRESHOLDS, \
           TRANSFORM_CONFIGS, EXCLUSIONS, COMBO_WHITELIST, CONDITIONAL_WHITELIST

    # ── 预解析 --config ──
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="libs.scripts.factor_analysis_configs.default")
    config_args, remaining_argv = pre_parser.parse_known_args()

    cfg = importlib.import_module(config_args.config)
    FACTOR_FAMILIES        = cfg.FACTOR_FAMILIES
    FULL_MODE_PARAM_GRIDS  = cfg.FULL_MODE_PARAM_GRIDS
    CUSTOM_THRESHOLDS      = cfg.CUSTOM_THRESHOLDS
    TRANSFORM_CONFIGS      = cfg.TRANSFORM_CONFIGS
    EXCLUSIONS             = cfg.EXCLUSIONS
    COMBO_WHITELIST        = cfg.COMBO_WHITELIST
    CONDITIONAL_WHITELIST  = cfg.CONDITIONAL_WHITELIST

    args = parse_args(remaining_argv)

    # ── 1. 确定因子列表 ────────────────────────────────────────────────────
    if args.factors:
        factor_names = list(args.factors)
    elif args.families:
        factor_names = []
        for family in args.families:
            if family in FACTOR_FAMILIES:
                factor_names.extend(FACTOR_FAMILIES[family])
            else:
                print(f"警告: 未知因子族 '{family}'，可用: {list(FACTOR_FAMILIES.keys())}")
        factor_names = list(dict.fromkeys(factor_names))  # 去重保序
    else:
        factor_names = list(FACTOR_REGISTRY.keys())

    print(f"模式: {args.mode} | 因子数: {len(factor_names)} | 并行度: {args.parallel}")
    print()

    # ── 2. 构建任务 ────────────────────────────────────────────────────────
    extra_cli_args = [
        "--min-bars", str(args.min_bars),
        "--forward-periods", *[str(p) for p in args.forward_periods],
        "--max-workers", str(args.max_workers),
    ]

    tasks = build_tasks(factor_names, args.mode, extra_cli_args)

    # ── 2.5 衍生因子任务 ───────────────────────────────────────────────────
    if args.generate_meta is not None:
        meta_tasks = build_meta_tasks(
            factor_names=factor_names,
            mode=args.mode,
            generate_meta=args.generate_meta,
            combos_extra=args.combos_extra,
            conditionals_extra=args.conditionals_extra,
            extra_args=extra_cli_args,
        )
        if meta_tasks:
            print(f"衍生因子任务: {len(meta_tasks)} 个")
            tasks.extend(meta_tasks)
        print()

    # ── 3. 断点续跑 / 强制重跑 / 数据新鲜度逻辑 ───────────────────────────
    max_age = args.max_age if args.max_age > 0 else None

    if args.force:
        print("强制重跑: 将覆盖已有报告")
    else:
        # 默认断点续跑：配置指纹 → 数据新鲜度（默认 30 天过期）→ 文件是否存在
        config_changed: list[AnalysisTask] = []
        stale: list[AnalysisTask] = []
        skipped: list[AnalysisTask] = []

        for t in tasks:
            fp_match = t.fingerprint_matches()
            if fp_match is False:
                # 有指纹但和当前配置不一致 → 重跑
                config_changed.append(t)
            elif fp_match is True and t.is_fresh(max_age):
                # 指纹一致 + 数据新鲜 → 跳过
                skipped.append(t)
            elif fp_match is True and t.is_done:
                # 指纹一致但数据过期 → 重跑
                stale.append(t)
            # else: 无指纹或无报告 → 留在 tasks 中（首次运行）

        tasks = [t for t in tasks if t not in skipped]
        if skipped:
            print(f"断点续跑: 跳过 {len(skipped)} 个（配置&数据均一致）")
            for s in skipped:
                print(f"  ✓ {s.label}")
            print()
        if config_changed:
            print(f"配置变更: {len(config_changed)} 个任务将重跑（layers/params/grid 已变化）")
            for s in config_changed:
                print(f"  ↻ {s.label}")
            print()
        if stale:
            print(f"数据过期: {len(stale)} 个任务将重跑（报告 end_date 超过 {max_age} 天前）")
            for s in stale:
                print(f"  ↻ {s.label}")
            print()

    if not tasks:
        print("没有待执行的任务。")
        return 0

    print(f"待执行任务: {len(tasks)} 个")
    for t in tasks:
        print(f"  → {t.label}")
    print()

    # ── 4. 试运行 ──────────────────────────────────────────────────────────
    if args.dry_run:
        print("试运行模式：不执行实际分析。")
        print(f"\n示例命令 (第一个任务):")
        print(f"  {' '.join(tasks[0].cli_args)}")
        return 0

    # ── 5. 并行执行 ────────────────────────────────────────────────────────
    start_time = time.monotonic()
    results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_map = {
            executor.submit(run_task, task): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(future_map):
            task = future_map[future]
            completed += 1
            try:
                r = future.result()
                results.append(r)
                status = "✓" if r["success"] else "✗"
                elapsed = r.get("runtime_sec", 0)
                print(f"[{completed}/{len(tasks)}] {status} {task.label} ({elapsed:.0f}s)")
                if r["success"]:
                    # 成功完成 → 保存配置指纹，方便下次 resume 比对
                    try:
                        task.save_fingerprint()
                    except Exception:
                        pass
                else:
                    err = r.get("error", "未知错误")
                    print(f"      错误: {err[:200]}")
            except Exception as e:
                results.append({
                    "factor": task.factor_name,
                    "success": False,
                    "output_dir": str(task.output_dir),
                    "runtime_sec": 0,
                    "error": f"Future exception: {e}",
                })
                print(f"[{completed}/{len(tasks)}] ✗ {task.label} (异常: {e})")

    total_sec = time.monotonic() - start_time
    print(f"\n总耗时: {total_sec/60:.1f} 分钟 ({total_sec:.0f} 秒)")

    # ── 6. 汇总 ────────────────────────────────────────────────────────────
    from config import DataPath
    summary_dir = args.output_dir or (Path(DataPath.DATA_DIR) / "factors" / "_batch_summary")
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整结果 JSON
    results_path = summary_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"完整结果 JSON: {results_path}")

    # 生成汇总 CSV
    generate_summary_csv(results, summary_dir)

    # ── 同步到 Windows ────────────────────────────────────────────────────
    windows_base = DataPath.DEFAULT_WINDOWS_PATH
    if windows_base:
        from factor_analysis.reporter import copy_to_windows
        windows_summary = Path(windows_base) / "factors" / "_batch_summary"
        ok = copy_to_windows(summary_dir, windows_summary)
        print(f"  → Windows 同步: {'成功' if ok else '失败'} ({windows_summary})")

    # ── 7. 总结 ────────────────────────────────────────────────────────────
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    print(f"\n{'='*60}")
    print(f"批量分析完成: {success_count} 成功, {fail_count} 失败")
    if fail_count > 0:
        print("失败清单:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['factor']}: {r.get('error', '未知')[:150]}")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
