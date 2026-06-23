"""
因子分析框架 (Factor Analysis Framework)

提供单因子从"体检"到"验证"的完整分析流水线：

- quality: 因子自身质量（覆盖率、分布、缺失模式、自相关）
- predictive: 因子预测力（Rank/Peason IC、IC 衰减、滚动 IC、参数网格）
- grouping: 分组检验（分位数组收益、多空组合、单调性）

典型用法:
    >>> from factor_analysis.config import FactorAnalysisConfig
    >>> from factor_analysis.runner import run_factor_analysis
    >>> from factors.price_return import PriceReturn
    >>>
    >>> factor = PriceReturn(window=60)  # 3 个月动量因子
    >>> config = FactorAnalysisConfig(factor=factor, layers=(1, 2, 3))
    >>> results = run_factor_analysis(config)
"""

__all__ = [
    "FactorAnalysisConfig",
    "FactorPanel",
    "build_factor_panel",
    "run_factor_analysis",
]


def __getattr__(name: str):
    """懒加载子模块，避免缺少依赖模块时整个包无法导入。"""
    if name in ("FactorAnalysisConfig",):
        from factor_analysis.config import FactorAnalysisConfig as _obj
        return _obj
    if name in ("FactorPanel", "build_factor_panel"):
        from factor_analysis.panel import FactorPanel as _Panel, build_factor_panel as _build
        _mapping = {"FactorPanel": _Panel, "build_factor_panel": _build}
        return _mapping[name]
    if name == "run_factor_analysis":
        from factor_analysis.runner import run_factor_analysis as _run
        return _run
    raise AttributeError(f"module 'factor_analysis' has no attribute {name!r}")
