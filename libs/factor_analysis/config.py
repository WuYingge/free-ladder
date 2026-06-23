"""
因子分析配置 (Factor Analysis Configuration)

定义单因子验证框架的所有输入参数、默认值和输出路径规则。
所有参数位置均带有详细的金融语义注释，方便人类调整。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factors.base_factor import BaseFactor

from config import DataPath
from data_manager.providers.etf_index_map_provider import ETF_INDEX_MAP


# ── 默认前向持仓期 ──────────────────────────────────────────────────────────
# 这些是 IC 计算和分组检验中使用的未来收益回看窗口（单位：交易日）。
# A 股每年约 242 个交易日，可按此比例折算自然月/季：
#   5  ≈ 1 周（短期 alpha 衰减速度——信号能维持多久）
#   10 ≈ 2 周（中短期信号有效性的中间观察点）
#   20 ≈ 1 个月（月频调仓场景的标准回看窗口）
#   60 ≈ 1 个季度（季频趋势跟踪，较长持仓期检验因子持续性）
# 不同持仓期的 IC 差异可以揭示因子是"短期反转型"还是"长期趋势型"。
DEFAULT_FORWARD_PERIODS: tuple[int, ...] = (5, 10, 20, 60)


# ── 最少交易日数 ────────────────────────────────────────────────────────────
# 单个标的至少需要的历史交易日数。低于此值的标的不参与分析。
#   252 ≈ 1 个自然年（保证有完整的年度周期覆盖）
#   过低：小截面噪声大，IC 不稳定，统计显著性差
#   过高：会过滤掉上市不满 1 年的新 ETF，减少截面广度
# 通常保持默认值即可；若关注次新 ETF 表现可适当下调至 120（≈半年）。
DEFAULT_MIN_BARS: int = 252


# ── 滚动 IC 窗口 ────────────────────────────────────────────────────────────
# 计算滚动 IC 均值时使用的历史窗口（单位：交易日）。
#   120 ≈ 半年（适合观察 IC 在牛熊切换前后的变化）
#   60  ≈ 1 个季度（更敏感，但噪声也更大）
#   240 ≈ 1 年（更平滑，但对近期变化反应迟钝）
# 这个参数不影响任何指标的计算，只影响可视化中滚动均线的平滑程度。
DEFAULT_ROLLING_IC_WINDOW: int = 120


# ── 分组数 ──────────────────────────────────────────────────────────────────
# 分位数分组的数量。
#   5  = 五分位（Q1 Top 20% ... Q5 Bottom 20%）——最常用的分组方式
#   10 = 十分位——更细粒度，适合观察顶部和底部极端组的 alpha 集中度
# 注意：分组过多会导致每组标的数太少（特别是截面较窄时），
#       从而单组收益波动剧烈，影响单调性检验的可靠性。
DEFAULT_N_QUANTILES: int = 5


# ── 自相关滞后天数 ──────────────────────────────────────────────────────────
# 因子值 rank autocorrelation 的滞后天数。
#   1  ≈ 日频自相关（衡量因子值一天内的变化幅度——换手率代理）
#   5  ≈ 周频（周度调仓场景下的因子稳定性）
#   10 ≈ 双周（中期稳定性）
#   20 ≈ 月频（月频调仓场景——自相关高 → 换手低 → 交易成本低）
# 注意：此指标对截面因子（如 PriceReturn）意义有限，对时序因子
#       （如 RSRS zscore、TrendR2 r2）更有参考价值。
DEFAULT_AUTOCORR_LAGS: tuple[int, ...] = (1, 5, 10, 20)


@dataclass
class FactorAnalysisConfig:
    """单因子分析完整配置。

    所有字段均为可选（除 factor 外），默认值已覆盖常见分析场景。
    对默认值不满意的地方直接赋值覆盖即可。

    金融语义速查:
        forward_periods  → 你想检验因子在"多长持仓期"上的预测力
        min_bars         → 标的至少有多少天数据才值得分析
        rolling_ic_window → IC 滚动的平滑窗口，只看图的话通常不需要调
        n_quantiles      → 分几组来比较"高因子值"和"低因子值"的收益差异
        layers           → 只跑某几层可以节省时间（如只看质量时选 (1,)）
    """

    # ── 因子实例（必填） ─────────────────────────────────────────────────
    # 传入已构造好参数的具体因子对象，如 PriceReturn(window=60)。
    # 框架会调用 factor(data) 对每个标的逐一计算因子值。
    factor: "BaseFactor"

    # ── 标的列表 ─────────────────────────────────────────────────────────
    # None → 自动从 ETF_INDEX_MAP 获取全部去重后的代表 ETF（约 484 个）。
    # 如需缩小范围，传入手动挑选的 symbol 列表。
    symbols: list[str] | None = None

    # ── 前向持仓期（交易日） ─────────────────────────────────────────────
    # 未来 N 日收益率 = close_{t+N} / close_t - 1。
    # 用于计算 IC（信息系数）和分组收益。
    # 默认覆盖短期 (5d/10d)、中期 (20d)、长期 (60d) 四个观察点。
    forward_periods: tuple[int, ...] = DEFAULT_FORWARD_PERIODS

    # ── 标的质量过滤 ─────────────────────────────────────────────────────
    # 单个标的历史数据低于此值时直接剔除（不参与任何分析）。
    # 该过滤在面板构建阶段执行，被剔除标的会记录到报告中。
    min_bars: int = DEFAULT_MIN_BARS

    # ── 日期范围（可选） ─────────────────────────────────────────────────
    # 分析的时间窗口。None 表示不限制，使用全部可用数据。
    # 格式: "YYYY-MM-DD"，如 "2020-01-01"。
    # 用于排除早期数据噪声或聚焦特定行情阶段。
    start_date: str | None = None
    end_date: str | None = None

    # ── 分析层选择 ───────────────────────────────────────────────────────
    # (1,)  = 只跑因子质量（不涉及未来收益，最快）
    # (2,)  = 只跑预测力（IC 相关指标）
    # (3,)  = 只跑分组检验
    # (1,2,3) = 全跑（默认）
    layers: tuple[int, ...] = (1, 2, 3)

    # ── Layer 2：滚动 IC 窗口（交易日） ───────────────────────────────────
    # 用于 compute_rolling_ic 的滚动窗口长度。
    # 只影响"滚动 IC 时序图"的可视化效果，不改变 IC 本身的数值。
    rolling_ic_window: int = DEFAULT_ROLLING_IC_WINDOW

    # ── Layer 2：参数敏感度网格 ──────────────────────────────────────────
    # 若不为 None，则对因子进行参数网格扫描。
    # 格式: {"参数名": [值1, 值2, ...], ...}
    # 示例: {"window": [20, 60, 120], "skip_recent": [0, 20]}
    # 笛卡尔积 × 每个 forward_period → IC 热力图矩阵。
    # None → 跳过参数网格分析（因子无可调参数或不需要此分析时）。
    param_grid: dict[str, list] | None = None

    # ── Layer 3：分位数分组数 ─────────────────────────────────────────────
    # 将每日截面的标的按因子值等分为 n_quantiles 组。
    # 5 组时每组约 20% 标的；10 组时每组约 10% 标的。
    n_quantiles: int = DEFAULT_N_QUANTILES

    # ── 自相关滞后天数 ───────────────────────────────────────────────────
    # 用于 compute_autocorr 的滞后天数。
    autocorr_lags: tuple[int, ...] = DEFAULT_AUTOCORR_LAGS

    # ── 输出根目录（Linux 端） ───────────────────────────────────────────
    # None → 自动使用 data/factors/{factor_name}/
    # 手动指定时请确保目录存在。
    output_root: Path | None = None

    # ── 报告日期标签 ─────────────────────────────────────────────────────
    # None → 使用当天日期 (YYYY-MM-DD)。
    # 手动指定用于区分同一因子的多次分析结果。
    output_date: str | None = None

    # ── 多进程 worker 数 ─────────────────────────────────────────────────
    # None → 使用 os.cpu_count()。
    # 在因子计算、参数网格等计算密集型环节使用。
    max_workers: int | None = None

    # ── 调试模式 ─────────────────────────────────────────────────────────
    # True → 即使单个标的计算出错也继续，错误信息聚合到报告中。
    # False → 任一标的出错则整体报错。
    debug: bool = False

    def resolve_symbols(self) -> list[str]:
        """获取去重后的标的列表。None 时自动回退到 ETF_INDEX_MAP。"""
        if self.symbols is not None:
            return list(self.symbols)
        # ETF_INDEX_MAP 包含 484 个跟踪指数的最优代表 ETF，已按跟踪指数字母序排序
        return ETF_INDEX_MAP.get_all_symbols()

    def resolve_output_root(self) -> Path:
        """获取 Linux 端输出根目录: data/factors/{factor_name}/"""
        if self.output_root is not None:
            return Path(self.output_root)
        # 因子名称用作子目录名（如 PriceReturn_60）
        factor_name = self._sanitized_factor_name()
        return Path(DataPath.DATA_DIR) / "factors" / factor_name

    def resolve_windows_output_root(self) -> Path | None:
        """获取 Windows 端输出根目录，仅在 DEFAULT_WINDOWS_PATH 非空时返回。"""
        windows_base = DataPath.DEFAULT_WINDOWS_PATH
        if not windows_base:
            return None
        factor_name = self._sanitized_factor_name()
        return Path(windows_base) / "factors" / factor_name

    def resolve_output_date(self) -> str:
        """获取报告日期标签 (YYYY-MM-DD)。"""
        return self.output_date or date.today().isoformat()

    def _sanitized_factor_name(self) -> str:
        """将因子实例转为安全的目录名（去除特殊字符）。"""
        # 使用因子的输出列名作为标识，如 "PriceReturn_60"
        raw = self.factor.get_output_name()
        # 替换文件系统不安全字符
        return raw.replace("/", "_").replace("\\", "_").replace(":", "_")
