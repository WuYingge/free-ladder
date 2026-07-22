"""回测结果 HTML 报告生成器。

生成自包含、人类可读的 HTML 报告，包含参数策略概览和绩效汇总表。
支持宽动量基线回测的输出目录结构。
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ====================================================================
# 公开接口
# ====================================================================


def generate_wide_momentum_html_report(
    groups: list[tuple[str, object, tuple]],
    all_summaries: list[dict[str, Any]],
    output_base: str | Path,
    grid_params: dict[str, Any] | None = None,
    title: str = "宽动量基线回测报告",
) -> str:
    """生成自包含的 HTML 报告。

    Parameters
    ----------
    groups:
        GROUPS 配置，每项为 (group_label, ranking_factor, builtin_filters)。
    all_summaries:
        所有 grid 组合的 variant_result.summary 列表。
        每条应包含 group_label、grid_label、top_n 及所有绩效字段。
    output_base:
        输出根目录，用于定位各组各 grid 的净值曲线图。
    grid_params:
        可选的 Grid 搜索参数字典，展示在概览区。
    title:
        报告标题。

    Returns
    -------
    HTML 字符串。
    """
    output_base = Path(output_base)

    # 将 summaries 转为 DataFrame 便于处理
    df = pd.DataFrame(all_summaries)

    # 确保关键列存在
    if "group_label" not in df.columns:
        df["group_label"] = "?"

    # 构建报告各部分
    sections: list[str] = []

    # CSS 样式
    sections.append(_css())

    # 头部
    sections.append(_build_header(title=title, output_base=output_base))

    # §1 总体参数
    if grid_params:
        sections.append(_build_overview(grid_params))

    # §2 策略定义
    sections.append(_build_strategy_table(groups))

    # §3 绩效汇总表
    sections.append(_build_performance_table(df, groups))

    # §4 各组详情（净值曲线图，来自已有输出）
    sections.append(_build_group_details(df, output_base, groups))

    # 排序 JS
    sections.append(_sortable_table_js())

    return "\n".join(sections)


# ====================================================================
# 内部：CSS
# ====================================================================


def _css() -> str:
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    --bg: #f5f6fa;
    --card-bg: #ffffff;
    --text: #2d3436;
    --muted: #636e72;
    --accent: #0984e3;
    --positive: #00b894;
    --negative: #d63031;
    --warn: #fdcb6e;
    --border: #dfe6e9;
    --radius: 10px;
    --shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 20px;
  }
  .container { max-width: 1400px; margin: 0 auto; }

  /* 头部 */
  .header {
    background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
    color: #fff;
    padding: 32px 40px;
    border-radius: var(--radius);
    margin-bottom: 24px;
  }
  .header h1 { font-size: 26px; font-weight: 700; }
  .header .meta { font-size: 13px; opacity: 0.75; margin-top: 6px; }

  /* 卡片 */
  .card {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 24px 28px;
    margin-bottom: 20px;
  }
  .card h2 {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 2px solid var(--accent);
  }
  .card h3 {
    font-size: 14px;
    font-weight: 600;
    color: var(--muted);
    margin: 20px 0 10px 0;
  }

  /* 参数网格 */
  .param-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px 24px;
  }
  .param-item { display: flex; gap: 8px; font-size: 13px; }
  .param-item .key { color: var(--muted); white-space: nowrap; }
  .param-item .val { font-weight: 600; }

  /* 表格 */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  thead th {
    background: #2d3436;
    color: #fff;
    padding: 10px 8px;
    text-align: left;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    position: sticky;
    top: 0;
    z-index: 1;
  }
  thead th:hover { background: #3d4547; }
  thead th .sort-arrow { font-size: 10px; margin-left: 4px; }
  tbody td {
    padding: 7px 8px;
    border-bottom: 1px solid var(--border);
  }
  tbody tr:hover { background: #f8f9ff; }

  /* 颜色编码 */
  .num-pos { color: var(--positive); font-weight: 600; }
  .num-neg { color: var(--negative); font-weight: 600; }
  .num-neutral { color: var(--muted); }
  .highlight { background: #ffeaa7 !important; }

  /* 净值曲线 */
  .equity-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 12px;
  }
  .equity-item {
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: #fafafa;
  }
  .equity-item .caption {
    font-size: 11px;
    font-weight: 600;
    padding: 6px 10px;
    color: var(--muted);
    background: #fff;
  }
  .equity-item img { width: 100%; display: block; }

  /* 标签 */
  .tag {
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: #dfe6e9;
    color: var(--muted);
    margin: 1px 2px;
    white-space: nowrap;
  }
  .tag.filter { background: #74b9ff; color: #fff; }
  .tag.ranking { background: #a29bfe; color: #fff; }
  .tag.weight { background: #55efc4; color: #2d3436; }

  /* 折叠 */
  details { margin-bottom: 8px; }
  details summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 15px;
    padding: 10px 14px;
    background: var(--card-bg);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    list-style: none;
  }
  details summary::-webkit-details-marker { display: none; }
  details summary::before { content: "▸ "; font-size: 12px; }
  details[open] summary::before { content: "▾ "; }

  /* 响应式 */
  @media (max-width: 768px) {
    body { padding: 10px; }
    .header { padding: 20px; }
    .card { padding: 16px; }
    .equity-gallery { grid-template-columns: 1fr; }
  }

  @media print {
    body { background: #fff; }
    .card { box-shadow: none; border: 1px solid #ccc; break-inside: avoid; }
    .header { background: #2d3436 !important; -webkit-print-color-adjust: exact; }
  }
</style>
</head>
<body>
<div class="container">
"""


# ====================================================================
# 内部：各节
# ====================================================================


def _build_header(title: str, output_base: Path) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<div class="header">
  <h1>{title}</h1>
  <div class="meta">生成时间: {now} &nbsp;|&nbsp; 输出目录: {output_base}</div>
</div>"""


def _build_overview(grid_params: dict[str, Any]) -> str:
    """构建总体参数概览卡片。"""
    # 将 tuple 转为可读字符串
    display: dict[str, str] = {}
    for k, v in grid_params.items():
        if isinstance(v, tuple):
            if len(v) <= 5:
                display[k] = ", ".join(str(x) for x in v)
            else:
                display[k] = f"{v[0]} .. {v[-1]}  ({len(v)} 种)"
        elif isinstance(v, list):
            display[k] = str(v)
        elif v is None:
            display[k] = "auto"
        else:
            display[k] = str(v)

    items = "\n".join(
        f'<div class="param-item"><span class="key">{k}:</span><span class="val">{v}</span></div>'
        for k, v in display.items()
    )
    return f"""<div class="card">
  <h2>§1 总体参数概览</h2>
  <div class="param-grid">{items}</div>
</div>"""


def _build_strategy_table(groups: list[tuple]) -> str:
    """构建各组策略定义表。支持 3 或 4 元素 GROUP entry。"""
    rows: list[str] = []
    for group_entry in groups:
        group_label = group_entry[0]
        ranking_factor = group_entry[1]
        builtin_filters = group_entry[2]
        cross_sectional_filters = group_entry[3] if len(group_entry) >= 4 else ()

        ranking_str = _factor_readable_label(ranking_factor)
        filter_parts: list[str] = []
        if builtin_filters:
            for f in builtin_filters:
                filter_parts.append(
                    f'<span class="tag filter">{_filter_readable(f)}</span>'
                )
        if cross_sectional_filters:
            for rf in cross_sectional_filters:
                filter_parts.append(
                    f'<span class="tag rank-filter">{rf.name or rf.factor.get_output_name()}</span>'
                )
        filter_str = ", ".join(filter_parts) if filter_parts else '<span class="num-neutral">(无)</span>'
        rows.append(
            f"<tr>"
            f"<td><strong>{group_label}</strong></td>"
            f"<td><span class=\"tag ranking\">{ranking_str}</span></td>"
            f"<td>{filter_str}</td>"
            f"</tr>"
        )
    return f"""<div class="card">
  <h2>§2 各组策略定义</h2>
  <table>
    <thead><tr><th>组</th><th>排名因子</th><th>过滤器</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""


def _build_performance_table(
    df: pd.DataFrame, groups: list[tuple]
) -> str:
    """构建主绩效汇总表（可排序）。"""
    # 选择要展示的列，排好顺序
    desired_cols = [
        "start_date",
        "end_date",
        "group_label",
        "grid_label",
        "top_n",
        "rebalance_interval_days",
        "cumulative_return_pct",
        "annualised_return_pct",
        "annualised_volatility_pct",
        "sharpe",
        "max_drawdown_pct",
        "calmar",
        "rebalance_win_rate_pct",
        "monthly_turnover_pct",
        "rebalance_count",
        "completed_period_count",
    ]
    # 只保留存在的列
    cols = [c for c in desired_cols if c in df.columns]
    work_df = df[cols].copy()

    # 表头中英文映射
    col_labels: dict[str, str] = {
        "start_date": "起始日期",
        "end_date": "结束日期",
        "group_label": "组",
        "grid_label": "Grid 标签",
        "top_n": "持仓数",
        "rebalance_interval_days": "调仓间隔(日)",
        "cumulative_return_pct": "累计收益(%)",
        "annualised_return_pct": "年化收益(%)",
        "annualised_volatility_pct": "年化波动(%)",
        "sharpe": "Sharpe",
        "max_drawdown_pct": "最大回撤(%)",
        "calmar": "Calmar",
        "rebalance_win_rate_pct": "调仓胜率(%)",
        "monthly_turnover_pct": "月换手(%)",
        "rebalance_count": "调仓次数",
        "completed_period_count": "完成期数",
    }

    thead = "<tr>" + "".join(
        f'<th onclick="sortTable({i})">{col_labels.get(c, c)}<span class="sort-arrow"></span></th>'
        for i, c in enumerate(cols)
    ) + "</tr>"

    # 数值格式化 + 颜色编码
    def _fmt_cell(col: str, val: Any) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return '<td class="num-neutral">—</td>'
        if col in ("group_label", "grid_label"):
            return f"<td>{val}</td>"
        if col == "top_n":
            return f"<td>{int(val)}</td>"
        # 带颜色的数值
        try:
            num = float(val)
        except (TypeError, ValueError):
            return f"<td>{val}</td>"
        # 收益类：正绿负红
        if col in (
            "cumulative_return_pct",
            "annualised_return_pct",
            "sharpe",
            "calmar",
            "rebalance_win_rate_pct",
        ):
            css_class = "num-pos" if num > 0 else "num-neg" if num < 0 else "num-neutral"
            return f'<td class="{css_class}">{num:.2f}</td>'
        # 风险类：回撤 / 波动用负值逻辑即绝对值
        if col in ("max_drawdown_pct", "annualised_volatility_pct"):
            css_class = "num-neg"  # 回撤和波动不是好事，标记为红色
            return f'<td class="{css_class}">{num:.2f}</td>'
        if col == "monthly_turnover_pct":
            return f"<td>{num:.1f}</td>"
        if col == "rebalance_interval_days":
            return f"<td>{int(num)}</td>"
        return f"<td>{num:.2f}</td>"

    tbody_rows: list[str] = []
    for _, row in work_df.iterrows():
        cells = "".join(_fmt_cell(c, row[c]) for c in cols)
        tbody_rows.append(f"<tr>{cells}</tr>")

    return f"""<div class="card">
  <h2>§3 绩效汇总（点击表头排序）</h2>
  <div style="overflow-x:auto;">
  <table id="perfTable">
    <thead>{thead}</thead>
    <tbody>{"".join(tbody_rows)}</tbody>
  </table>
  </div>
  <p style="font-size:11px;color:var(--muted);margin-top:8px;">
    💡 点击表头可按该列排序；再次点击切换升降序。绿色 = 正面指标，红色 = 负面指标。
  </p>
</div>"""


def _build_group_details(
    df: pd.DataFrame,
    output_base: Path,
    groups: list[tuple],
    max_images_per_group: int = 4,
) -> str:
    """构建各组详情折叠区，嵌入净值曲线图。"""
    parts: list[str] = ['<div class="card"><h2>§4 各组详情</h2>']

    for group_entry in groups:
        group_label = group_entry[0]
        ranking_factor = group_entry[1]
        builtin_filters = group_entry[2]

        group_df = df[df["group_label"] == group_label]
        if group_df.empty:
            parts.append(f"<p style=\"color:var(--muted);\">⚠️ {group_label}: 无数据</p>")
            continue

        # 组基本参数
        ranking_str = _factor_readable_label(ranking_factor)
        filter_str = ", ".join(_filter_readable(f) for f in builtin_filters) if builtin_filters else "无"
        top_ns = sorted(group_df["top_n"].dropna().unique())

        summary_text = (
            f"排名因子: {ranking_str} &nbsp;|&nbsp; "
            f"过滤器: {filter_str} &nbsp;|&nbsp; "
            f"Top-N: {', '.join(str(int(t)) for t in top_ns)} &nbsp;|&nbsp; "
            f"共 {len(group_df)} 条结果"
        )

        parts.append(
            f"<details><summary>{group_label}</summary>"
            f"<p style=\"font-size:12px;color:var(--muted);margin:8px 0;\">{summary_text}</p>"
        )

        # 嵌入该组各 grid_label 的净值曲线图
        images_found = 0
        gallery_parts: list[str] = ['<div class="equity-gallery">']
        seen_grids: set[str] = set()

        for _, row in group_df.iterrows():
            grid_label = row.get("grid_label", "")
            top_n = int(row.get("top_n", 0))
            if not grid_label:
                continue
            if grid_label in seen_grids:
                continue
            if images_found >= max_images_per_group:
                break

            # 净值曲线位置: output_base/wide_momentum_{group_label}/{grid_label}/top_{top_n}/equity_curve.png
            group_dir = output_base / f"wide_momentum_{group_label}"
            png_path = group_dir / str(grid_label) / f"top_{top_n}" / "equity_curve.png"

            b64 = _read_image_base64(png_path)
            if b64:
                gallery_parts.append(
                    f'<div class="equity-item">'
                    f'<div class="caption">Top {top_n} · {grid_label}</div>'
                    f'<img src="data:image/png;base64,{b64}" alt="净值曲线 {grid_label}" loading="lazy">'
                    f'</div>'
                )
                seen_grids.add(grid_label)
                images_found += 1

        gallery_parts.append("</div>")
        parts.extend(gallery_parts)

        if images_found == 0:
            parts.append(
                '<p style="color:var(--muted);font-size:12px;">(未找到净值曲线图)</p>'
            )
        elif images_found < len(group_df["grid_label"].unique()):
            remaining = len(group_df["grid_label"].unique()) - images_found
            parts.append(
                f'<p style="font-size:11px;color:var(--muted);">'
                f'（仅展示前 {images_found} 张，另有 {remaining} 张未嵌入）</p>'
            )

        parts.append("</details>")

    parts.append("</div>")
    return "\n".join(parts)


# ====================================================================
# 内部：辅助
# ====================================================================


def _factor_readable_label(factor: object) -> str:
    """从因子对象生成人类可读标签。"""
    # 尝试获取 __str__ / get_output_name
    if hasattr(factor, "__name__"):
        return str(getattr(factor, "__name__"))
    if hasattr(factor, "get_output_name"):
        return str(factor.get_output_name())  # type: ignore[union-attr]
    if hasattr(factor, "__class__"):
        cls_name = factor.__class__.__name__
        if hasattr(factor, "__str__"):
            s = str(factor)
            if s and s != cls_name:
                return s
        return cls_name
    return str(factor)


def _filter_readable(filt: object) -> str:
    """从 ThresholdFilter 生成可读字符串。"""
    if hasattr(filt, "field") and hasattr(filt, "operator") and hasattr(filt, "value"):
        field = str(getattr(filt, "field"))
        op = str(getattr(filt, "operator"))
        val = getattr(filt, "value")
        # 简化字段名
        short_field = field.replace("MAPosition_close_", "MA").replace("RSRS_", "RSRS:").replace("TrendR2_", "TR²:")
        short_field = short_field.replace("_r2", "").replace("_zscore_", ":").replace("_reg", "(reg")
        short_field = short_field.replace("_adj", ")")
        return f"{short_field} {op} {val}"
    return str(filt)


def _read_image_base64(path: Path, max_size_mb: float = 0.5) -> str | None:
    """将 PNG 图片读取为 base64 字符串。跳过不存在的或太大的文件。"""
    try:
        if not path.is_file():
            return None
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except (OSError, PermissionError):
        return None


# ====================================================================
# 内部：可排序表格 JS
# ====================================================================


def _sortable_table_js() -> str:
    return """<script>
function sortTable(colIdx) {
  const table = document.getElementById('perfTable');
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const th = table.querySelectorAll('thead th')[colIdx];

  // 判断当前排序方向
  const currentDir = th.getAttribute('data-sort-dir') || 'none';
  const nextDir = currentDir === 'asc' ? 'desc' : 'asc';

  // 清除所有箭头
  table.querySelectorAll('thead th .sort-arrow').forEach(el => el.textContent = '');
  // 设置当前箭头
  th.querySelector('.sort-arrow').textContent = nextDir === 'asc' ? ' ▲' : ' ▼';
  th.setAttribute('data-sort-dir', nextDir);

  rows.sort((a, b) => {
    let aVal = a.cells[colIdx]?.textContent?.trim() || '';
    let bVal = b.cells[colIdx]?.textContent?.trim() || '';

    // 尝试数字比较
    let aNum = parseFloat(aVal.replace(/[,%¥￥]/g, ''));
    let bNum = parseFloat(bVal.replace(/[,%¥￥]/g, ''));
    if (!isNaN(aNum) && !isNaN(bNum)) {
      aVal = aNum;
      bVal = bNum;
    }

    if (aVal < bVal) return nextDir === 'asc' ? -1 : 1;
    if (aVal > bVal) return nextDir === 'asc' ? 1 : -1;
    return 0;
  });

  rows.forEach(row => tbody.appendChild(row));
}
</script>
</div>
</body>
</html>"""
