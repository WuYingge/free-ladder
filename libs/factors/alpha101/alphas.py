"""
Alpha101 注册表 (Alpha101 Registry)

汇总 WorldQuant《101 Formulaic Alphas》全部 101 个公式的元数据。公式本体在
``factors.alpha101.formulas``，本模块只负责：

  * ``ALPHA101_REGISTRY``：alpha_id → :class:`AlphaSpec`（公式 + warmup + 数据依赖）
  * ``get_alpha_spec`` / ``get_computable_alpha_ids`` 等查询辅助

数据依赖标记（供扫描器按可用性启用/跳过，别再用"是否可算"一句话概括）：

  * ``uses_cap``        —— 依赖流通市值（仅 #56）；数据源 ``data/daily_basic``
  * ``needs_vwap``      —— 依赖 vwap（45 个公式）；面板 vwap 由
    ``value/(volume*100) × adj_factor`` 构造，需要 ``data/adj_factor``（东财不复权
    收盘作锚点），详见 ``factors.alpha101.panel``
  * ``needs_adv``       —— 依赖 adv{d}（平均成交额），用面板的 ``value`` 现算
  * ``needs_industry``  —— 需要申万行业分类做时点中性化：1/2/3 = 一/二/三级
    （对应论文 IndClass.sector/industry/subindustry），数据源
    ``data/const/stock_sw_industry_clf.csv``；ETF 池无行业分类 → 这 18 个公式跳过

``warmup`` 为经验值：在合成面板（20 标的 × 1000 交易日）上从首日起到出现首个
有效值的交易日数；用于记录/展示，不作为硬性切片依据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from factors.alpha101.formulas import FORMULAS
from factors.alpha101.panel import Alpha101Inputs


@dataclass(frozen=True)
class AlphaSpec:
    """单个 Alpha101 因子的元数据。"""

    alpha_id: str
    func: Callable[[Alpha101Inputs], pd.DataFrame]
    warmup: int = 0
    uses_cap: bool = False          # 依赖流通市值（#56）
    needs_vwap: bool = False        # 依赖 vwap（需 data/adj_factor 提供复权因子）
    needs_adv: bool = False         # 依赖 adv{d}（成交额滑动均值）
    needs_industry: int = 0         # 0=不需要；1/2/3=申万一/二/三级行业中性化
    description: str = ""


#: alpha_id → (warmup, 依赖标记串)
#: 标记串字符：c=cap, v=vwap, a=adv, i{1,2,3}=行业档位（如 "vai2"）
_SPEC_TABLE: dict[str, tuple[int, str]] = {
    "001": (3, ""), "002": (6, ""), "003": (7, ""), "004": (6, ""),
    "005": (7, "v"), "006": (7, ""), "007": (0, "a"), "008": (14, ""),
    "009": (1, ""), "010": (1, ""), "011": (3, "v"), "012": (1, ""),
    "013": (3, ""), "014": (7, ""), "015": (2, ""), "016": (3, ""),
    "017": (18, "a"), "018": (7, ""), "019": (200, ""), "020": (1, ""),
    "021": (0, "a"), "022": (15, ""), "023": (0, ""), "024": (3, ""),
    "025": (15, "va"), "026": (7, ""), "027": (0, "v"), "028": (18, "a"),
    "029": (10, ""), "030": (15, ""), "031": (24, "a"), "032": (188, "v"),
    "033": (0, ""), "034": (4, ""), "035": (26, ""), "036": (159, "va"),
    "037": (160, ""), "038": (7, ""), "039": (200, "a"), "040": (7, ""),
    "041": (0, "v"), "042": (0, "v"), "043": (30, "a"), "044": (3, ""),
    "045": (20, ""), "046": (0, ""), "047": (15, "va"), "048": (201, "i3"),
    "049": (1, ""), "050": (6, "v"), "051": (1, ""), "052": (192, ""),
    "053": (9, ""), "054": (0, ""), "055": (13, ""), "056": (8, "c"),
    "057": (23, "v"), "058": (11, "vi1"), "059": (19, "vi2"), "060": (7, ""),
    "061": (0, "va"), "062": (0, "va"), "063": (191, "vai2"), "064": (0, "va"),
    "065": (0, "va"), "066": (13, "v"), "067": (1, "vai3"), "068": (0, "a"),
    "069": (6, "vai2"), "070": (1, "vai2"), "071": (179, "va"), "072": (44, "va"),
    "073": (16, "v"), "074": (0, "va"), "075": (0, "va"), "076": (111, "vai1"),
    "077": (36, "va"), "078": (51, "va"), "079": (0, "vai1"), "080": (14, "ai2"),
    "081": (0, "va"), "082": (27, "i1"), "083": (5, "v"), "084": (27, "v"),
    "085": (30, "a"), "086": (0, "va"), "087": (86, "vai2"), "088": (74, "a"),
    "089": (21, "vai2"), "090": (35, "ai3"), "091": (26, "vai2"), "092": (38, "a"),
    "093": (97, "vai2"), "094": (9, "va"), "095": (0, "a"), "096": (79, "va"),
    "097": (94, "vai2"), "098": (43, "va"), "099": (0, "a"), "100": (23, "ai3"),
    "101": (0, ""),
}


def _parse_flags(flags: str) -> dict[str, object]:
    """解析依赖标记串（见 _SPEC_TABLE 注释）。"""
    level = 0
    idx = flags.find("i")
    if idx >= 0 and idx + 1 < len(flags) and flags[idx + 1].isdigit():
        level = int(flags[idx + 1])
    return {
        "uses_cap": "c" in flags,
        "needs_vwap": "v" in flags,
        "needs_adv": "a" in flags,
        "needs_industry": level,
    }


def _formula_description(alpha_id: str) -> str:
    """取公式函数 docstring 的首行作为描述（即论文公式原文）。"""
    doc = (FORMULAS[alpha_id].__doc__ or "").strip().splitlines()
    text = doc[0].strip() if doc else ""
    return text.split(":", 1)[-1].strip()


def _build_registry() -> dict[str, AlphaSpec]:
    """按 _SPEC_TABLE + formulas.FORMULAS 组装注册表（并在导入期做完整性校验）。"""
    registry: dict[str, AlphaSpec] = {}
    for alpha_id, (warmup, flags) in _SPEC_TABLE.items():
        func = FORMULAS.get(alpha_id)
        if func is None:  # pragma: no cover — 表与公式不同步时立刻暴露
            raise RuntimeError(f"Alpha101 注册表缺公式实现: #{alpha_id}")
        registry[alpha_id] = AlphaSpec(
            alpha_id=alpha_id,
            func=func,
            warmup=warmup,
            description=_formula_description(alpha_id),
            **_parse_flags(flags),  # type: ignore[arg-type]
        )

    expected = {f"{i:03d}" for i in range(1, 102)}
    if set(registry) != expected:  # pragma: no cover
        missing = sorted(expected - set(registry))
        extra = sorted(set(registry) - expected)
        raise RuntimeError(f"Alpha101 注册表不完整: 缺 {missing}, 多 {extra}")
    if set(FORMULAS) != expected:  # pragma: no cover
        raise RuntimeError("formulas.FORMULAS 与 101 个 id 不一致")
    return registry


ALPHA101_REGISTRY: dict[str, AlphaSpec] = _build_registry()


def get_computable_alpha_ids(exclude_cap: bool = True) -> list[str]:
    """返回可计算的 alpha id 列表（exclude_cap=True 时排除依赖 cap 的，保持旧默认）。"""
    return [
        alpha_id
        for alpha_id, spec in ALPHA101_REGISTRY.items()
        if not (exclude_cap and spec.uses_cap)
    ]


def get_alpha_spec(alpha_id: str) -> AlphaSpec:
    """按 id 取 AlphaSpec，未知 id 抛 ValueError。"""
    if alpha_id not in ALPHA101_REGISTRY:
        raise ValueError(f"未知 Alpha101 因子: {alpha_id!r}，可用: {sorted(ALPHA101_REGISTRY)}")
    return ALPHA101_REGISTRY[alpha_id]


def data_requirements_summary() -> dict[str, int]:
    """统计各数据依赖的公式数量（供 CLI/文档展示）。"""
    specs = ALPHA101_REGISTRY.values()
    return {
        "total": len(ALPHA101_REGISTRY),
        "needs_vwap": sum(1 for s in specs if s.needs_vwap),
        "needs_adv": sum(1 for s in specs if s.needs_adv),
        "needs_industry": sum(1 for s in specs if s.needs_industry),
        "uses_cap": sum(1 for s in specs if s.uses_cap),
        "plain": sum(
            1
            for s in specs
            if not (s.needs_vwap or s.needs_industry or s.uses_cap)
        ),
    }


__all__ = [
    "ALPHA101_REGISTRY",
    "AlphaSpec",
    "FORMULAS",
    "data_requirements_summary",
    "get_alpha_spec",
    "get_computable_alpha_ids",
]
