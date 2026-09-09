"""
Alpha101Factor 适配类 (Alpha101 Factor Adapter)

把注册表里的单个 alpha 公式包装成符合 `factor_analysis` 配置/报告命名惯例的因子对象。

factor_analysis 的 `FeatureAnalysisConfig`/`generate_and_save_reports` 需要：
  * config.factor.params          -> dict，用于报告记录因子参数
  * config.factor.get_output_name() -> str，用于输出目录名
  * type(config.factor).__name__   -> str，用于报告记录因子类型

本类只做元数据适配，真正计算走 `build_alpha101_panel`（面板化），
这里的 __call__ 不实现逐标的运算。
"""

from __future__ import annotations


class Alpha101Factor:
    """单个 Alpha101 因子的元数据适配器。"""

    name = "Alpha101"

    def __init__(self, alpha_id: str, universe_kind: str = "etf") -> None:
        self.alpha_id = str(alpha_id)
        self.universe_kind = universe_kind
        self.params = {"alpha": self.alpha_id, "universe": universe_kind}
        self.warmup_period = self._resolve_warmup()

    def _spec(self):
        from factors.alpha101.alphas import ALPHA101_REGISTRY

        if self.alpha_id not in ALPHA101_REGISTRY:
            raise ValueError(f"未知 Alpha101 因子: {self.alpha_id!r}")
        return ALPHA101_REGISTRY[self.alpha_id]

    def _resolve_warmup(self) -> int:
        try:
            return int(self._spec().warmup)
        except Exception:
            return 0

    def get_output_name(self) -> str:
        return f"Alpha101_{self.alpha_id}"

    @property
    def is_computable(self) -> bool:
        """该 alpha 在当前数据源下是否可计算（依赖 cap 的标记为不可计算）。"""
        return not self._spec().uses_cap

    def __repr__(self) -> str:
        return f"Alpha101Factor(alpha={self.alpha_id}, universe={self.universe_kind})"
