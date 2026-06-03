"""验证 ThresholdFilter 重构功能。"""
import sys
sys.path.insert(0, '.')

from backtesting.wide_momentum_baseline import (
    BaselineCandidate,
    ThresholdFilter,
    WideMomentumBaselineConfig,
    _build_candidate_filters,
)

# 1. 向后兼容：min_momentum_value=0 自动转为 builtin_filters
config = WideMomentumBaselineConfig(min_momentum_value=0)
assert len(config.builtin_filters) == 1, f"Expected 1, got {len(config.builtin_filters)}"
bf = config.builtin_filters[0]
assert bf.field == "score", f"Expected 'score', got {bf.field!r}"
assert bf.operator == ">=", f"Expected '>=', got {bf.operator!r}"
assert bf.value == 0.0, f"Expected 0.0, got {bf.value!r}"
print(f"[PASS] 向后兼容: {bf.field}{bf.operator}{bf.value}")

# 2. 内置过滤器：score<0 的 candidate 被过滤
cand_negative = BaselineCandidate(symbol="X", score=-1.0, factor_values={"score": -1.0})
cand_positive = BaselineCandidate(symbol="Y", score=1.0, factor_values={"score": 1.0})
cand_zero = BaselineCandidate(symbol="Z", score=0.0, factor_values={"score": 0.0})

filters = _build_candidate_filters(config)
assert len(filters) == 1
name, fn = filters[0]
assert not fn(cand_negative), f"Negative should be filtered out by {name}"
assert fn(cand_positive), f"Positive should pass {name}"
assert fn(cand_zero), f"Zero should pass {name}"
print(f"[PASS] 过滤逻辑: {name}")

# 3. 新增 builtin_filters 直接传入
config2 = WideMomentumBaselineConfig(
    builtin_filters=(
        ThresholdFilter(field="score", operator=">=", value=0),
        ThresholdFilter(field="volume_ratio", operator=">=", value=1.5),
    ),
)
assert len(config2.builtin_filters) == 2
filters2 = _build_candidate_filters(config2)
assert len(filters2) == 2
print(f"[PASS] 新 builtin_filters: {[n for n, _ in filters2]}")

# 4. CandidateFilterSpec.params 支持
from backtesting.wide_momentum_baseline import CandidateFilterSpec
spec = CandidateFilterSpec(filter_fn=lambda x: True, name="my_filter", params={"threshold": 0.5})
assert spec.params == {"threshold": 0.5}
print(f"[PASS] CandidateFilterSpec.params: {spec.params}")

# 5. _serialize_candidate_filters 新格式
from backtesting.wide_momentum_baseline import _serialize_candidate_filters
serialized = _serialize_candidate_filters(config2)
assert len(serialized) == 2
assert serialized[0]["field"] == "score"
assert serialized[0]["operator"] == ">="
assert serialized[0]["value"] == 0.0
print(f"[PASS] 序列化: {serialized}")

# 6. 实验名自动生成
from backtesting.wide_momentum_baseline import _resolve_experiment_name
name = _resolve_experiment_name(config2)
assert "score_ge_0" in name, f"Expected 'score_ge_0' in {name!r}"
assert "volume_ratio_ge_1.5" in name, f"Expected 'volume_ratio_ge_1.5' in {name!r}"
print(f"[PASS] 实验名: {name}")

print("\n✅ 所有验证通过")
