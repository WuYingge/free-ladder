#!/usr/bin/env python3
"""从 factors_to_analyze.csv 生成 wide_momentum_configs 配置文件。

用法:
    cd /home/gouzi/projects/invest
    uv run python libs/scripts/generate_wide_momentum_configs.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# 确保能导入项目模块
REPO_ROOT = Path(__file__).resolve().parents[2]
LIBS_DIR = REPO_ROOT / "libs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIBS_DIR))

# ═══════════════════════════════════════════════════════════════════════════
# 基础因子类映射表：输出名前缀 → (类名, import模块, 参数字典)
# ═══════════════════════════════════════════════════════════════════════════
BASE_FACTOR_MAP: dict[str, tuple[str, str, dict]] = {
    # 价格动量族
    "PriceReturn":               ("PriceReturn",               "factors.price_return",       {"window": 60}),
    "RiskAdjustedReturn":        ("RiskAdjustedReturn",        "factors.price_momentum",     {"window": 20}),
    "IntradayMomentum":          ("IntradayMomentum",          "factors.price_momentum",     {}),
    "OvernightReturn":           ("OvernightReturn",           "factors.price_momentum",     {}),
    "HighPointPosition":         ("HighPointPosition",         "factors.price_momentum",     {"window": 20}),
    "LowPointPosition":          ("LowPointPosition",          "factors.price_momentum",     {"window": 20}),
    "TimeSeriesMomentum":        ("TimeSeriesMomentum",        "factors.price_momentum",     {"window": 252}),
    # 反转族
    "ShortTermReversal":         ("ShortTermReversal",         "factors.reversal",           {"window": 1}),
    "VolumeReversal":            ("VolumeReversal",            "factors.reversal",           {"ret_window": 5, "vol_window": 20}),
    # 波动率族
    "DownsideVolatility":        ("DownsideVolatility",        "factors.volatility_family",  {"window": 20}),
    "AvgDrawdown":               ("AvgDrawdown",               "factors.volatility_family",  {"window": 60}),
    # 趋势质量族
    "KaufmanER":                 ("KaufmanEfficiencyRatio",    "factors.trend_quality",      {"window": 20}),
    # 超买超卖族
    "RSI":                       ("RSI",                       "factors.oscillator",         {"window": 14}),
    "Stochastic":                ("Stochastic",                "factors.oscillator",         {"n": 14, "m": 3, "output": "K"}),
    "CCI":                       ("CCI",                       "factors.oscillator",         {"window": 20}),
    "WilliamsR":                 ("WilliamsR",                 "factors.oscillator",         {"window": 14}),
    "MFI":                       ("MFI",                       "factors.oscillator",         {"window": 14}),
    # 均线偏离族
    "MAPosition":                ("MAPosition",                "factors.ma",                 {"window": 200, "price_column": "close"}),
    "MA":                        ("MAFactor",                  "factors.ma",                 {"window": 20, "price_column": "close"}),
    "BIAS":                      ("BIAS",                      "factors.ma",                 {"window": 20, "price_column": "close"}),
    "BollingerBandPosition":     ("BollingerBandPosition",     "factors.ma",                 {"window": 20, "k": 2.0, "price_column": "close"}),
    "MASlope":                   ("MASlope",                   "factors.ma",                 {"ma_window": 20, "slope_window": 5, "price_column": "close"}),
    "MADistance":                ("MADistance",                "factors.ma",                 {"short_window": 5, "long_window": 60, "price_column": "close"}),
    "MADispersion":              ("MADispersion",              "factors.ma",                 {"windows": [5, 10, 20, 60], "price_column": "close"}),
    # 分布形态族
    "MFE":                       ("MaxFavorableExcursion",     "factors.distribution_family",{"window": 20}),
    "MAE":                       ("MaxAdverseExcursion",       "factors.distribution_family",{"window": 20}),
    # 突破族
    "DailyRebound":              ("DailyRebound",              "factors.daily_rebound",      {}),
    "TrendR2":                   ("TrendR2Factor",             "factors.trend_r2",           {"window": 120, "output": "r2"}),
    "NewLowContinuous":          ("NewLowContinuous",          "factors.breakout_family",    {"window": 50}),
    "DonchianPosition":          ("DonchianChannelPosition",   "factors.breakout_family",    {"window": 20}),
}

# 变换方法名列表
TRANSFORM_METHODS = frozenset({"rolling_mean", "rolling_std", "delta",
                                "pct_change", "binarize_winrate", "zscore"})
# 组合方法名列表
COMBINE_METHODS = frozenset({"product", "ratio", "weighted_sum", "diff"})
# 条件运算符
COND_OPS = frozenset({"gt", "lt", "gte", "lte"})

# 用于变量名的计数器
_var_counter: dict[str, int] = {}

# 基础因子缓存：相同类+相同参数只创建一次，避免 hash 碰撞
# key = (cls_name, params_frozenset), value = varname
_base_factor_cache: dict[tuple, str] = {}


def make_varname(prefix: str) -> str:
    """生成唯一变量名，如 f1, f2, f3..."""
    _var_counter[prefix] = _var_counter.get(prefix, 0) + 1
    return f"{prefix}{_var_counter[prefix]}"


# ═══════════════════════════════════════════════════════════════════════════
# 基础因子解析
# ═══════════════════════════════════════════════════════════════════════════

def _parse_base_factor(name: str) -> tuple[str, dict, str, str] | None:
    """解析基础因子名称，返回 (类名, params_dict, 模块路径, 构造参数格式化字符串)。

    例如:
      "TimeSeriesMomentum_252" → ("TimeSeriesMomentum", {"window": 252}, "factors.price_momentum", "window=252")
    """
    # 尝试前缀匹配（从长到短）
    sorted_prefixes = sorted(BASE_FACTOR_MAP.keys(), key=len, reverse=True)
    for prefix in sorted_prefixes:
        if name == prefix:
            # 精确匹配（无参数基础因子，如 DailyRebound）
            cls_name, module, default_params = BASE_FACTOR_MAP[prefix]
            if not default_params:
                return (cls_name, {}, module, "")
            else:
                return (cls_name, dict(default_params), module,
                        _params_to_str(default_params))
        if name.startswith(prefix + "_"):
            rest = name[len(prefix) + 1:]
            cls_name, module, default_params = BASE_FACTOR_MAP[prefix]
            params = _parse_params_for_prefix(prefix, rest, default_params)
            if params is not None:
                return (cls_name, params, module, _params_to_str(params))
    return None


def _parse_params_for_prefix(prefix: str, rest: str, defaults: dict) -> dict | None:
    """根据前缀类型解析参数字符串部分。"""
    parts = rest.split("_")
    params = dict(defaults)

    if prefix in ("DailyRebound",):
        return params  # 无参数

    elif prefix in ("PriceReturn", "HighPointPosition", "LowPointPosition",
                     "TimeSeriesMomentum", "ShortTermReversal",
                     "CCI", "RSI", "WilliamsR", "MFI",
                     "AvgDrawdown", "MFE", "MAE", "NewLowContinuous",
                     "KaufmanER", "DownsideVolatility",
                     "DonchianPosition"):
        # 单窗口参数: {prefix}_{window}
        if len(parts) == 1:
            params["window"] = int(parts[0])
            return params

    elif prefix == "BIAS":
        # BIAS_close_{window}
        if len(parts) == 2 and parts[0] == "close":
            params["price_column"] = "close"
            params["window"] = int(parts[1])
            return params

    elif prefix == "BollingerBandPosition":
        # BollingerBandPosition_close_{window}_{k}
        if len(parts) >= 2 and parts[0] == "close":
            params["price_column"] = "close"
            params["window"] = int(parts[1])
            if len(parts) >= 3:
                params["k"] = float(parts[2])
            return params

    elif prefix == "MAPosition":
        # MAPosition_close_{window}
        if len(parts) >= 2 and parts[0] == "close":
            params["price_column"] = "close"
            params["window"] = int(parts[1])
            return params

    elif prefix == "MA":
        # MA_close_{window}
        if len(parts) >= 2 and parts[0] == "close":
            params["price_column"] = "close"
            params["window"] = int(parts[1])
            return params

    elif prefix == "MASlope":
        # MASlope_close_{ma_window}_{slope_window}
        if len(parts) >= 3 and parts[0] == "close":
            params["price_column"] = "close"
            params["ma_window"] = int(parts[1])
            params["slope_window"] = int(parts[2])
            return params

    elif prefix == "MADistance":
        # MADistance_close_{short}_{long}
        if len(parts) >= 3 and parts[0] == "close":
            params["price_column"] = "close"
            params["short_window"] = int(parts[1])
            params["long_window"] = int(parts[2])
            return params

    elif prefix == "MADispersion":
        # MADispersion_close_{w1}_{w2}_{w3}_{w4}
        if len(parts) >= 3 and parts[0] == "close":
            params["price_column"] = "close"
            params["windows"] = tuple(int(p) for p in parts[1:])
            return params

    elif prefix == "Stochastic":
        # Stochastic_{n}_{m}_{output}
        if len(parts) == 3:
            params["n"] = int(parts[0])
            params["m"] = int(parts[1])
            params["output"] = parts[2]
            return params

    elif prefix in ("TrendR2",):
        # TrendR2_{window}_{output}
        if len(parts) >= 1:
            params["window"] = int(parts[0])
            if len(parts) >= 2:
                params["output"] = parts[1]
            return params

    elif prefix == "VolumeReversal":
        # VolumeReversal_{ret_window}_{vol_window}
        if len(parts) == 2:
            params["ret_window"] = int(parts[0])
            params["vol_window"] = int(parts[1])
            return params

    return None


def _params_tuple(params: dict) -> tuple:
    """将参数字典转为可哈希的排序元组，用于缓存键。"""
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, tuple):
            v = v  # tuple already hashable
        elif isinstance(v, list):
            v = tuple(v)
        items.append((k, v))
    return tuple(items)


def _params_to_str(params: dict) -> str:
    """将参数字典格式化为 Python 构造参数字符串。"""
    parts = []
    for k, v in params.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, tuple):
            parts.append(f"{k}=({', '.join(str(x) for x in v)})")
        elif isinstance(v, list):
            parts.append(f"{k}=[{', '.join(str(x) for x in v)}]")
        elif isinstance(v, float):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 完整因子名称解析 → 生成 Python 代码
# ═══════════════════════════════════════════════════════════════════════════

def parse_factor_name(name: str) -> dict | None:
    """解析因子名称，返回描述性字典，包含 code 字段表示生成的 Python 代码。

    返回格式:
      {
        "varname": "f1",           # 变量名
        "code": "f1 = DailyRebound()",  # Python 代码行
        "imports": {"factors.daily_rebound"},  # 需要的 import 集合
        "output_name": "DailyRebound",
      }
    """
    # 0. 处理 _neg 后缀
    is_negate = name.endswith("_neg") and not name.endswith("__neg")
    if is_negate:
        inner_name = name[:-4]  # 去掉 _neg 后缀
        inner = parse_factor_name(inner_name)
        if inner is None:
            return None
        varname = make_varname("f")
        inner_varname = inner["varname"]
        code = f"{varname} = _Negate({inner_varname})\n"
        imports = inner["imports"] | {"factors.derived_factor"}
        return {
            "varname": varname,
            "code": inner["code"] + code,
            "imports": imports,
            "output_name": f"{inner['output_name']}__neg",
        }

    # 1. SwitchFactor: __if_...__else_...  (支持单条件和多条件)
    if "__if_" in name and "__else_" in name:
        # 格式: {true信号}__if_{条件}__else_{false信号}
        # 先按 __else_ 分割（取最后一个 __else_）
        else_idx = name.rfind("__else_")
        true_if_part = name[:else_idx]
        false_part = name[else_idx + 7:]  # 去掉 "__else_"

        # true_if_part 格式: {true信号}__if_{条件}
        if_idx = true_if_part.rfind("__if_")
        true_signal_name = true_if_part[:if_idx]
        cond_part = true_if_part[if_idx + 5:]  # 去掉 "__if_"

        true_signal = parse_factor_name(true_signal_name)
        false_signal = parse_factor_name(false_part)
        if true_signal is None or false_signal is None:
            return None

        varname = make_varname("f")
        code = true_signal["code"] + false_signal["code"]
        all_imports = true_signal["imports"] | false_signal["imports"] | {"factors.meta_factor"}

        # 检查条件部分是否含 _and_ 或 _or_（多条件）
        if "_and_" in cond_part or "_or_" in cond_part:
            logic = "and" if "_and_" in cond_part else "or"
            sep = f"_{logic}_"
            cond_strs = cond_part.split(sep)
            cond_infos = []
            for cs in cond_strs:
                ci = _parse_condition(cs)
                if ci is None:
                    return None
                cond_infos.append(ci)
                # 包含条件因子代码
                code += ci["parsed"]["code"]
                all_imports |= ci["parsed"]["imports"]
            code += _gen_switch_multi_cond_code(varname, true_signal, false_signal, cond_infos, logic)
        else:
            # 单条件
            cond_info = _parse_condition(cond_part)
            if cond_info is None:
                return None
            # 包含条件因子代码
            code += cond_info["parsed"]["code"]
            all_imports |= cond_info["parsed"]["imports"]
            code += _gen_switch_code(varname, true_signal, false_signal, cond_info)

        return {
            "varname": varname,
            "code": code,
            "imports": all_imports,
            "output_name": name,
        }

    # 1b. 检查是否有条件结构但被误判（多条件 AND 可能含 _and_）
    if "__if_" in name:
        # 分离信号和条件部分
        if_idx = name.index("__if_")
        signal_name = name[:if_idx]
        cond_part = name[if_idx + 5:]

        # 检查是否含 _and_ 或 _or_
        if "_and_" in cond_part or "_or_" in cond_part:
            # MultiConditionalFactor
            logic = "and" if "_and_" in cond_part else "or"
            sep = f"_{logic}_"
            cond_strs = cond_part.split(sep)

            cond_infos = []
            for cs in cond_strs:
                ci = _parse_condition(cs)
                if ci is None:
                    return None
                cond_infos.append(ci)

            signal = parse_factor_name(signal_name)
            if signal is None:
                return None

            # 确保所有条件因子的代码被包含
            varname = make_varname("f")
            code = signal["code"]
            for ci in cond_infos:
                code += ci["parsed"]["code"]
            code += _gen_multi_cond_code(varname, signal, cond_infos, logic)

            imports = signal["imports"] | {"factors.meta_factor"}
            for ci in cond_infos:
                imports |= ci["parsed"]["imports"]
            return {
                "varname": varname,
                "code": code,
                "imports": imports,
                "output_name": name,
            }
        else:
            # ConditionalFactor: 单条件
            cond_info = _parse_condition(cond_part)
            if cond_info is None:
                return None

            signal = parse_factor_name(signal_name)
            if signal is None:
                return None

            # 确保条件因子的代码也被包含
            cond_parsed = cond_info["parsed"]

            varname = make_varname("f")
            code = signal["code"] + cond_parsed["code"]
            code += _gen_conditional_code(varname, signal, cond_info, cond_parsed)

            imports = signal["imports"] | cond_parsed["imports"] | {"factors.meta_factor"}
            return {
                "varname": varname,
                "code": code,
                "imports": imports,
                "output_name": name,
            }

    # 2. CombineFactor: 检查四种组合方法
    for method in COMBINE_METHODS:
        marker = f"__{method}_"
        if marker in name:
            idx = name.index(marker)
            factor_a_name = name[:idx]
            factor_b_name = name[idx + len(marker):]

            factor_a = parse_factor_name(factor_a_name)
            factor_b = parse_factor_name(factor_b_name)
            if factor_a is None or factor_b is None:
                return None

            varname = make_varname("f")
            code = factor_a["code"] + factor_b["code"]
            code += f"{varname} = CombineFactor(factor_a={factor_a['varname']}, factor_b={factor_b['varname']}, method=\"{method}\")\n"

            imports = factor_a["imports"] | factor_b["imports"] | {"factors.meta_factor"}
            return {
                "varname": varname,
                "code": code,
                "imports": imports,
                "output_name": name,
            }

    # 3. TransformFactor: 检查六种变换
    for tmethod in TRANSFORM_METHODS:
        marker = f"__{tmethod}_"
        if marker in name:
            idx = name.index(marker)
            base_name = name[:idx]
            rest = name[idx + len(marker):]

            base = parse_factor_name(base_name)
            if base is None:
                return None

            varname = make_varname("f")
            code = base["code"]

            if tmethod == "binarize_winrate":
                # 格式: binarize_winrate_{window}_{threshold}
                parts = rest.split("_")
                if len(parts) < 1:
                    return None
                win = int(parts[0])
                threshold = float(parts[1]) if len(parts) >= 2 else 0.0
                code += f"{varname} = TransformFactor(dependency={base['varname']}, transform=\"binarize_winrate\", window={win}, threshold={threshold})\n"
            else:
                win = int(rest.split("_")[0])
                code += f"{varname} = TransformFactor(dependency={base['varname']}, transform=\"{tmethod}\", window={win})\n"

            imports = base["imports"] | {"factors.meta_factor"}
            return {
                "varname": varname,
                "code": code,
                "imports": imports,
                "output_name": name,
            }

    # 4. 基础因子（带缓存，避免同参数多实例导致 hash 碰撞）
    base_info = _parse_base_factor(name)
    if base_info is not None:
        cls_name, params, module, param_str = base_info
        # 构造缓存键
        cache_key = (cls_name, _params_tuple(params))
        if cache_key in _base_factor_cache:
            # 复用已有变量
            varname = _base_factor_cache[cache_key]
            return {
                "varname": varname,
                "code": "",  # 不重复创建
                "imports": {module},
                "output_name": name,
            }
        # 首次创建
        varname = make_varname("f")
        _base_factor_cache[cache_key] = varname
        if param_str:
            code = f"{varname} = {cls_name}({param_str})\n"
        else:
            code = f"{varname} = {cls_name}()\n"
        return {
            "varname": varname,
            "code": code,
            "imports": {module},
            "output_name": name,
        }

    return None


def _parse_condition(cond_str: str) -> dict | None:
    """解析条件字符串 "{因子}_{op}_{阈值}"。

    返回 {"factor_name": ..., "op": ..., "threshold": ..., "parsed": ..., "imports": ...}
    """
    for op in COND_OPS:
        op_marker = f"_{op}_"
        if op_marker in cond_str:
            idx = cond_str.index(op_marker)
            factor_name = cond_str[:idx]
            threshold_str = cond_str[idx + len(op_marker):]
            try:
                threshold = float(threshold_str)
            except ValueError:
                return None

            parsed = parse_factor_name(factor_name)
            if parsed is None:
                return None
            return {
                "factor_name": factor_name,
                "op": op,
                "threshold": threshold,
                "parsed": parsed,
                "imports": parsed["imports"],
            }
    return None


def _gen_switch_code(varname: str, true_signal: dict, false_signal: dict,
                     cond_info: dict) -> str:
    """生成 SwitchFactor 代码（单条件）。"""
    cond = cond_info["parsed"]
    lines = [
        f"# SwitchFactor: {true_signal['output_name']} / {false_signal['output_name']}",
        f"{varname} = SwitchFactor(",
        f"    signal_true={true_signal['varname']},",
        f"    signal_false={false_signal['varname']},",
        f"    condition={cond['varname']},",
        f"    op=\"{cond_info['op']}\",",
        f"    threshold={cond_info['threshold']},",
        ")",
        "",
    ]
    return "\n".join(lines)


def _gen_switch_multi_cond_code(varname: str, true_signal: dict, false_signal: dict,
                                  cond_infos: list[dict], logic: str) -> str:
    """生成 SwitchFactor 代码（多条件）。"""
    lines = [
        f"# SwitchFactor (multi-cond, {logic}): {true_signal['output_name']} / {false_signal['output_name']}",
        f"{varname} = SwitchFactor(",
        f"    signal_true={true_signal['varname']},",
        f"    signal_false={false_signal['varname']},",
        "    conditions=[",
    ]
    for ci in cond_infos:
        cond = ci["parsed"]
        lines.append(f"        ConditionSpec(condition={cond['varname']}, op=\"{ci['op']}\", threshold={ci['threshold']}),")
    lines.append("    ],")
    lines.append(f'    logic="{logic}",')
    lines.append(")")
    lines.append("")
    return "\n".join(lines)


def _gen_conditional_code(varname: str, signal: dict, cond_info: dict,
                          cond_parsed: dict) -> str:
    """生成 ConditionalFactor 代码。"""
    lines = [
        f"# ConditionalFactor: {signal['output_name']} if {cond_info['factor_name']} {cond_info['op']} {cond_info['threshold']}",
        f"{varname} = ConditionalFactor(",
        f"    signal={signal['varname']},",
        f"    condition={cond_parsed['varname']},",
        f"    op=\"{cond_info['op']}\",",
        f"    threshold={cond_info['threshold']},",
        ")",
        "",
    ]
    return "\n".join(lines)


def _gen_multi_cond_code(varname: str, signal: dict, cond_infos: list[dict],
                          logic: str) -> str:
    """生成 MultiConditionalFactor 代码。"""
    lines = [
        f"# MultiConditionalFactor ({logic}): {signal['output_name']}",
        f"{varname} = MultiConditionalFactor(",
        f"    signal={signal['varname']},",
        "    conditions=[",
    ]
    for ci in cond_infos:
        cond = ci["parsed"]
        lines.append(f"        ConditionSpec(condition={cond['varname']}, op=\"{ci['op']}\", threshold={ci['threshold']}),")
    lines.append("    ],")
    lines.append(f'    logic="{logic}",')
    lines.append(")\n")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 导入语句收集
# ═══════════════════════════════════════════════════════════════════════════

def _import_map(module: str) -> str:
    """将模块路径映射为 import 语句。"""
    # 需要导入的类按模块组织
    IMPORT_CLASSES: dict[str, set[str]] = {
        "factors.price_momentum": {"HighPointPosition", "LowPointPosition", "TimeSeriesMomentum"},
        "factors.price_return": {"PriceReturn"},
        "factors.volatility_family": {"AvgDrawdown", "DownsideVolatility"},
        "factors.oscillator": {"RSI", "Stochastic", "CCI", "WilliamsR", "MFI"},
        "factors.reversal": {"ShortTermReversal", "VolumeReversal"},
        "factors.trend_r2": {"TrendR2Factor"},
        "factors.trend_quality": {"KaufmanEfficiencyRatio"},
        "factors.ma": {"MAPosition", "MAFactor", "BIAS", "BollingerBandPosition", "MASlope", "MADistance", "MADispersion"},
        "factors.distribution_family": {"MaxAdverseExcursion", "MaxFavorableExcursion"},
        "factors.daily_rebound": {"DailyRebound"},
        "factors.breakout_family": {"DonchianChannelPosition", "NewLowContinuous"},
        "factors.meta_factor": {"TransformFactor", "CombineFactor", "ConditionalFactor", "MultiConditionalFactor", "ConditionSpec", "SwitchFactor"},
        "factors.derived_factor": {"DerivedFactor"},
        "factors.rsrs": {"RsrsFactor"},
        "factors.volatility": {"Volatility"},
    }
    return IMPORT_CLASSES.get(module, set())


# ═══════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH = REPO_ROOT / "output" / "factor_cross_correlation" / "factors_to_analyze.csv"
OUTPUT_PATH = REPO_ROOT / "libs" / "scripts" / "wide_momentum_configs" / "factors0706_all.py"


def main():
    # 读取因子列表
    factor_names: list[str] = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("factor", "").strip()
            if name:
                factor_names.append(name)

    print(f"读取到 {len(factor_names)} 个因子名称")

    # 解析所有因子
    parsed: list[dict] = []
    failed: list[str] = []
    for name in factor_names:
        result = parse_factor_name(name)
        if result is None:
            failed.append(name)
            print(f"  ✗ 无法解析: {name}")
        else:
            parsed.append(result)

    print(f"\n成功解析: {len(parsed)} 个")
    print(f"解析失败: {len(failed)} 个")
    if failed:
        print("失败列表:")
        for fname in failed:
            print(f"  - {fname}")

    # ── 收集所有需要的导入 ──
    all_imports: dict[str, set[str]] = {}  # module → {class names}
    for p in parsed:
        for module in p["imports"]:
            classes = _import_map(module)
            if module not in all_imports:
                all_imports[module] = set()
            all_imports[module] |= classes

    # ── 生成代码 ──
    lines: list[str] = []
    lines.append('"""宽动量基线回测 — 全量因子扫描（来自 factors_to_analyze.csv）。')
    lines.append('')
    lines.append(f'包含 {len(parsed)} 个排名因子（纯排名，无过滤）。')
    lines.append('')
    lines.append('用法:')
    lines.append('    uv run python libs/scripts/run_wide_momentum_custom.py \\')
    lines.append('        --config libs.scripts.wide_momentum_configs.factors0706_all')
    lines.append('"""')
    lines.append('from __future__ import annotations')
    lines.append('')

    # 导入
    lines.append('from backtesting.wide_momentum_baseline import (')
    lines.append('    equal_weight_allocator,')
    lines.append('    make_factor_weighted_allocator,')
    lines.append('    score_proportional_allocator,')
    lines.append(')')
    lines.append('')

    # 按模块组织导入
    for module in sorted(all_imports.keys()):
        classes = sorted(all_imports[module])
        if classes:
            cls_list = ", ".join(classes)
            lines.append(f"from {module} import {cls_list}")

    lines.append('from factors.base_factor import BaseFactor')
    lines.append('from factors.volatility import Volatility')
    lines.append('')

    # _Negate 工具类
    lines.append('')
    lines.append('# ====================================================================')
    lines.append('# 小工具：取反包装')
    lines.append('# ====================================================================')
    lines.append('')
    lines.append('class _Negate(DerivedFactor):')
    lines.append('    """对依赖因子取反（乘 -1），用于反转因子方向。"""')
    lines.append('    name = "_Negate"')
    lines.append('')
    lines.append('    def __init__(self, dependency: BaseFactor) -> None:')
    lines.append('        super().__init__()')
    lines.append('        self.add_dependency(dependency)')
    lines.append('        self.warmup_period = dependency.get_max_warmup_period()')
    lines.append('')
    lines.append('    def get_output_name(self) -> str:')
    lines.append('        dep_name = self._dependencies[0].get_output_name()')
    lines.append('        return f"{dep_name}__neg"')
    lines.append('')
    lines.append('    def compute_from_frame(self, frame):')
    lines.append('        import pandas as pd')
    lines.append('        dep_name = self._dependencies[0].get_output_name()')
    lines.append('        result = -frame[dep_name]')
    lines.append('        result.name = self.get_output_name()')
    lines.append('        return result')
    lines.append('')
    lines.append('')
    lines.append('# ====================================================================')
    lines.append('# 因子定义')
    lines.append('# ====================================================================')
    lines.append('')
    lines.append(f'# 共 {len(parsed)} 个因子')
    lines.append('')

    # 因子代码（按依赖顺序排列）
    # 将所有因子代码拼接在一起
    all_code: list[str] = []
    for i, p in enumerate(parsed):
        all_code.append(f"# ── [{i + 1}] {p['output_name']} ──")
        all_code.append(p["code"].rstrip())
        all_code.append("")

    lines.extend(all_code)

    # 共享管道（仅保留权重分配器需要的 vol20）
    lines.append('')
    lines.append('# ====================================================================')
    lines.append('# 共享管道因子')
    lines.append('# ====================================================================')
    lines.append('vol20 = Volatility(window=20)')
    lines.append('')
    lines.append('SHARED_PIPELINE: tuple = (vol20,)')
    lines.append('')

    # GROUPS
    lines.append('')
    lines.append(f'# ====================================================================')
    lines.append(f'# 组定义: {len(parsed)} 组（{len(parsed)} 因子，无过滤器）')
    lines.append(f'# ====================================================================')
    lines.append('# (label, ranking_factor, builtin_filters)')
    lines.append('GROUPS: list[tuple[str, object, tuple]] = [')

    for i, p in enumerate(parsed):
        # 生成简短 label
        short_name = p["output_name"]
        # 截断过长的名称
        if len(short_name) > 80:
            short_name = short_name[:77] + "..."
        # 简化 label（替换特殊字符）
        label_base = short_name.replace("__", "_").replace(".", "_").replace("-", "_")
        lines.append(f"    # [{i + 1}] {p['output_name']}")
        lines.append(f'    ("{label_base}",          {p["varname"]}, ()),')

    lines.append(']')
    lines.append('')

    # Grid Search
    lines.append('')
    lines.append('# ====================================================================')
    lines.append('# Grid Search 参数')
    lines.append('# ====================================================================')
    lines.append('GRID_TOP_N: tuple[int, ...] = (1, 5, 10, 20)')
    lines.append('GRID_MIN_MOMENTUM: tuple = (None,)')
    lines.append('GRID_CLUSTER_MAX_PER_GROUP: tuple[int, ...] = (0,)')
    lines.append('GRID_REBALANCE_INTERVAL: tuple[int, ...] = (5, 10, 20)')
    lines.append('GRID_EXCLUDE_BONDS: tuple[bool, ...] = (True, False)')
    lines.append('GRID_HOLD_OVERLAP: tuple[bool, ...] = (False, True)')
    lines.append('')

    # Weight Allocators
    lines.append('')
    lines.append('# ====================================================================')
    lines.append('# 权重分配器')
    lines.append('# ====================================================================')
    lines.append('alloc_equal = equal_weight_allocator')
    lines.append('')
    lines.append('alloc_momentum = score_proportional_allocator')
    lines.append('alloc_momentum.__name__ = "momentum"')
    lines.append('')
    lines.append('alloc_inv_vol = make_factor_weighted_allocator(vol20.get_output_name(), inverse=True)')
    lines.append('alloc_inv_vol.__name__ = "invvol"')
    lines.append('')
    lines.append('def _adaptive_tiered(candidates):')
    lines.append('    """自适应分档：前 40% 权重 1.5，后 60% 权重 1.0。"""')
    lines.append('    if not candidates:')
    lines.append('        return {}')
    lines.append('    n = len(candidates)')
    lines.append('    top_count = max(1, round(n * 0.4))')
    lines.append('    weights = {}')
    lines.append('    for i, c in enumerate(candidates):')
    lines.append('        weights[c.symbol] = 1.5 if i < top_count else 1.0')
    lines.append('    return weights')
    lines.append('_adaptive_tiered.__name__ = "tiered"')
    lines.append('alloc_tiered = _adaptive_tiered')
    lines.append('')
    lines.append('')
    lines.append('WEIGHT_ALLOCATORS: tuple = (')
    lines.append('    alloc_inv_vol,')
    lines.append(')')
    lines.append('')

    # 执行参数
    lines.append('')
    lines.append('# ====================================================================')
    lines.append('# 执行参数')
    lines.append('# ====================================================================')
    lines.append('OUTPUT_BASE_DIR: str = "/mnt/c/Users/wyg/Documents/invest/backtest"')
    lines.append('TITLE: str              = "宽动量基线回测 — 全量因子扫描 (factors0706_all)"')
    lines.append('START_DATE: str         = "2020-01-01"')
    lines.append('END_DATE: str           = "2026-05-29"')
    lines.append('MAX_WORKERS: int | None = None')
    lines.append('PERIOD_FREQ: str | None = None')
    lines.append('CUSTOM_PERIODS: tuple[tuple[str, str], ...] | None = None')

    # 写入文件
    content = "\n".join(lines) + "\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(content)

    print(f"\n配置文件已生成: {OUTPUT_PATH}")
    print(f"文件大小: {OUTPUT_PATH.stat().st_size} 字节")
    print(f"总行数: {len(lines)}")


if __name__ == "__main__":
    main()
