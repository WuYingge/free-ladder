import { useState, useEffect, useCallback } from "react";
import ReactECharts from "echarts-for-react";
import { fetchFactors, fetchFactorDetail, fetchCorrelation, rebuildIndex } from "../api";
import type { FactorSummary } from "../api";

export default function FactorOverview() {
  const [factors, setFactors] = useState<FactorSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);

  // 筛选状态
  const [icPeriod, setIcPeriod] = useState(10);
  const [icType, setIcType] = useState("rank_ic");
  const [sortBy, setSortBy] = useState("icir");
  const [sortOrder, setSortOrder] = useState("desc");
  const [icMin, setIcMin] = useState("");
  const [icMax, setIcMax] = useState("");
  const [icirMin, setIcirMin] = useState("");
  const [icirMax, setIcirMax] = useState("");
  const [search, setSearch] = useState("");

  // 选中因子详情
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [corrData, setCorrData] = useState<{ factor: string; correlation: number }[]>([]);
  const [corrThreshold, setCorrThreshold] = useState(0.7);
  const [corrDirection, setCorrDirection] = useState("above");

  const loadFactors = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchFactors({
        ic_period: icPeriod, ic_type: icType, sort_by: sortBy, sort_order: sortOrder,
        page, page_size: 50,
        ...(icMin && { ic_min: Number(icMin) }),
        ...(icMax && { ic_max: Number(icMax) }),
        ...(icirMin && { icir_min: Number(icirMin) }),
        ...(icirMax && { icir_max: Number(icirMax) }),
        ...(search && { search }),
      });
      setFactors(data.factors);
      setTotal(data.total);
    } catch (e) { console.error(e); }
    setLoading(false);
  }, [icPeriod, icType, sortBy, sortOrder, page, icMin, icMax, icirMin, icirMax, search]);

  useEffect(() => { loadFactors(); }, [loadFactors]);

  const handleRebuild = async () => {
    setRebuilding(true);
    try {
      const r = await rebuildIndex();
      alert(`索引已刷新: ${r.n_factors} 个因子, 生成时间 ${r.generated_at}`);
      loadFactors();
    } catch (e) {
      alert(`刷新失败: ${e}`);
    }
    setRebuilding(false);
  };

  const loadDetail = async (name: string) => {
    setSelected(name);
    try {
      const [d, c] = await Promise.all([
        fetchFactorDetail(name),
        fetchCorrelation(name, corrThreshold, corrDirection, 50),
      ]);
      setDetail(d);
      setCorrData(c.related || []);
    } catch (e) { console.error(e); }
  };

  const icKey = icType === "rank_ic" ? "ic_rank" : "ic_pearson";
  const periodKey = String(icPeriod);

  const getVal = (f: FactorSummary, field: string) => {
    const stats = f[icKey]?.[periodKey] as Record<string, number | null> | undefined;
    if (!stats) return null;
    return stats[field] ?? null;
  };

  const getTopVal = (f: FactorSummary, field: string) => {
    const stats = f.top_ic?.[periodKey] as Record<string, number | null> | undefined;
    if (!stats) return null;
    return stats[field] ?? null;
  };

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>因子概览</h2>
        <button onClick={handleRebuild} disabled={rebuilding} style={btnStyle}>
          {rebuilding ? "刷新中..." : "🔄 刷新索引"}
        </button>
      </div>

      {/* 筛选栏 */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12, alignItems: "center" }}>
        <select value={icPeriod} onChange={e => { setIcPeriod(Number(e.target.value)); setPage(1); }}
          style={selectStyle}>
          <option value={5}>IC 5日</option><option value={10}>IC 10日</option>
          <option value={20}>IC 20日</option><option value={60}>IC 60日</option>
        </select>
        <select value={icType} onChange={e => { setIcType(e.target.value); setPage(1); }} style={selectStyle}>
          <option value="rank_ic">Rank IC</option><option value="pearson_ic">Pearson IC</option>
        </select>
        <input placeholder="IC min" value={icMin} onChange={e => setIcMin(e.target.value)} style={inputStyle} />
        <input placeholder="IC max" value={icMax} onChange={e => setIcMax(e.target.value)} style={inputStyle} />
        <input placeholder="ICIR min" value={icirMin} onChange={e => setIcirMin(e.target.value)} style={inputStyle} />
        <input placeholder="ICIR max" value={icirMax} onChange={e => setIcirMax(e.target.value)} style={inputStyle} />
        <input placeholder="搜索因子名..." value={search} onChange={e => { setSearch(e.target.value); setPage(1); }} style={{ ...inputStyle, width: 180 }} />
        <select value={sortBy} onChange={e => setSortBy(e.target.value)} style={selectStyle}>
          <option value="icir">按 ICIR</option><option value="ic_mean">按 IC</option>
          <option value="sharpe">按 Sharpe</option>
          <option value="top_icir">按 Top10% ICIR</option>
        </select>
        <button onClick={() => setSortOrder(o => o === "desc" ? "asc" : "desc")}
          style={btnStyle}>{sortOrder === "desc" ? "↓降序" : "↑升序"}</button>
      </div>

      {/* 表格 */}
      <div style={{ maxHeight: "60vh", overflow: "auto", marginBottom: 16 }}>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th>因子名</th><th>类型</th><th>IC Mean</th><th>IC Std</th><th>ICIR</th><th>T-stat</th>
              <th>Top10% IC</th><th>Top10% ICIR</th><th>Sharpe(5d)</th>
            </tr>
          </thead>
          <tbody>
            {factors.map(f => (
              <tr key={f.name} onClick={() => loadDetail(f.name)}
                style={{ cursor: "pointer", background: selected === f.name ? "#1f2937" : undefined }}>
                <td style={{ maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                  title={f.name}>{f.name}</td>
                <td>{f.factor_type}</td>
                <td>{getVal(f, "mean")?.toFixed(4)}</td>
                <td>{getVal(f, "std")?.toFixed(4)}</td>
                <td style={{ color: (getVal(f, "ir") ?? 0) > 0 ? "#3fb950" : "#f85149" }}>
                  {getVal(f, "ir")?.toFixed(4)}</td>
                <td>{getVal(f, "t_stat")?.toFixed(2)}</td>
                <td>{getTopVal(f, "mean")?.toFixed(4)}</td>
                <td style={{ color: (getTopVal(f, "ir") ?? 0) > 0 ? "#3fb950" : "#f85149" }}>
                  {getTopVal(f, "ir")?.toFixed(4)}</td>
                <td>{f.longshort_sharpe_5d?.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* 分页 */}
      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
        <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1} style={btnStyle}>上一页</button>
        <span>第 {page} 页 / 共 {Math.ceil(total / 50)} 页 ({total} 个因子)</span>
        <button onClick={() => setPage(p => p + 1)} disabled={page * 50 >= total} style={btnStyle}>下一页</button>
        {loading && <span>加载中...</span>}
      </div>

      {/* 详情面板 */}
      {selected && detail && (
        <div style={{ border: "1px solid #30363d", borderRadius: 8, padding: 16, background: "#161b22" }}>
          <h3 style={{ marginTop: 0 }}>{selected}</h3>
          <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
            <DetailSection title="预测力 (Layer 2)" detail={detail} />
            <DetailSection title="分组收益 (Layer 3)" detail={detail} />
            <DetailSection title="统计分布" detail={detail} />
          </div>

          {/* IC 衰减图 */}
          <IcDecayChart detail={detail} />

          {/* Top 10% IC 衰减图 */}
          <TopIcDecayChart detail={detail} />

          {/* 分组收益柱状图 */}
          <QuantileBarChart detail={detail} />

          {/* 相关性筛选 */}
          <div style={{ marginTop: 16 }}>
            <h4>相关性筛选</h4>
            <label>阈值 |ρ|: </label>
            <select value={corrThreshold} onChange={e => setCorrThreshold(Number(e.target.value))} style={selectStyle}>
              <option value={0.9}>≥ 0.9</option><option value={0.8}>≥ 0.8</option>
              <option value={0.7}>≥ 0.7</option><option value={0.5}>≥ 0.5</option><option value={0.3}>≥ 0.3</option>
            </select>
            <select value={corrDirection} onChange={e => setCorrDirection(e.target.value)} style={{ ...selectStyle, marginLeft: 8 }}>
              <option value="above">高于</option><option value="below">低于</option>
            </select>
            <button onClick={() => loadDetail(selected)} style={{ ...btnStyle, marginLeft: 8 }}>查询</button>
            {corrData.length > 0 && (
              <table style={{ ...tableStyle, marginTop: 8, maxWidth: 400 }}>
                <thead><tr><th>因子</th><th>Spearman ρ</th></tr></thead>
                <tbody>
                  {corrData.map(c => (
                    <tr key={c.factor} onClick={() => loadDetail(c.factor)} style={{ cursor: "pointer" }}>
                      <td style={{ maxWidth: 250, overflow: "hidden", textOverflow: "ellipsis" }}>{c.factor}</td>
                      <td>{c.correlation.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {corrData.length === 0 && <p style={{ color: "#8b949e" }}>无匹配因子 (共 145 个因子在相关性矩阵中)</p>}
          </div>
        </div>
      )}
    </div>
  );
}

// ── 子组件 ────────────────────────────────────────────────────────────────────

function DetailSection({ title, detail }: { title: string; detail: Record<string, unknown> }) {
  let content: React.ReactNode = null;
  if (title.includes("Layer 2")) {
    const pred = detail.layer2_predictive as Record<string, unknown> | undefined;
    if (pred) {
      const rankIc = pred.rank_ic as Record<string, { summary: Record<string, number> }> | undefined;
      const topIc = pred.top_ic as Record<string, { summary: Record<string, number> }> | undefined;
      content = (
        <div>
          {rankIc && Object.entries(rankIc).map(([p, v]) => (
            <div key={p}>IC {p}d: mean={v.summary?.mean?.toFixed(4)}, IR={v.summary?.ir?.toFixed(4)}, t={v.summary?.t_stat?.toFixed(2)}</div>
          ))}
          {topIc && (
            <div style={{ marginTop: 8, borderTop: "1px solid #30363d", paddingTop: 8 }}>
              <strong>Top 10% IC:</strong>
              {Object.entries(topIc).map(([p, v]) => (
                <div key={p} style={{ color: "#8b949e", fontSize: 13 }}>
                  {p}d: mean={v.summary?.mean?.toFixed(4)}, IR={v.summary?.ir?.toFixed(4)}
                </div>
              ))}
            </div>
          )}
        </div>
      );
    }
  } else if (title.includes("Layer 3")) {
    const grp = detail.layer3_grouping as Record<string, unknown> | undefined;
    if (grp) {
      const ls5 = (grp["5"] as Record<string, unknown>)?.longshort as Record<string, number> | undefined;
      content = (
        <div>
          {ls5 && <div>LongShort 5d: Sharpe={ls5.sharpe?.toFixed(2)}, 年化收益={(ls5.annualised_return * 100)?.toFixed(1)}%</div>}
        </div>
      );
    }
  } else {
    const dist = detail.distribution as Record<string, unknown> | undefined;
    if (dist) {
      const stats = dist.stats as Record<string, number> | undefined;
      const pct = dist.daily_percentiles as Record<string, number> | undefined;
      content = (
        <div>
          {stats && <div>mean={stats.mean?.toFixed(4)}, std={stats.std?.toFixed(4)}</div>}
          {pct && <div>p5={pct.p5?.toFixed(4)}, p50={pct.p50?.toFixed(4)}, p95={pct.p95?.toFixed(4)}</div>}
        </div>
      );
    }
  }
  return (
    <div style={{ flex: 1, minWidth: 250 }}>
      <h4 style={{ margin: "8px 0" }}>{title}</h4>
      {content || <span style={{ color: "#8b949e" }}>无数据</span>}
    </div>
  );
}

function IcDecayChart({ detail }: { detail: Record<string, unknown> }) {
  const pred = detail.layer2_predictive as Record<string, unknown> | undefined;
  if (!pred?.ic_decay) return null;
  const decay = pred.ic_decay as Array<{ period: number; ic_mean: number; ic_ir: number }>;
  const option = {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: decay.map(d => `${d.period}d`) },
    yAxis: [{ type: "value", name: "IC Mean" }, { type: "value", name: "ICIR" }],
    series: [
      { name: "IC Mean", type: "bar", data: decay.map(d => d.ic_mean), itemStyle: { color: "#58a6ff" } },
      { name: "ICIR", type: "line", yAxisIndex: 1, data: decay.map(d => d.ic_ir), itemStyle: { color: "#3fb950" } },
    ],
  };
  return <ReactECharts option={option} style={{ height: 250, marginTop: 12 }} theme="dark" />;
}

function TopIcDecayChart({ detail }: { detail: Record<string, unknown> }) {
  const pred = detail.layer2_predictive as Record<string, unknown> | undefined;
  if (!pred?.top_ic_decay) return null;
  const decay = pred.top_ic_decay as Array<{ period: number; ic_mean: number; ic_ir: number }>;
  const option = {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: decay.map(d => `${d.period}d`) },
    yAxis: [{ type: "value", name: "Top10% IC" }, { type: "value", name: "Top10% ICIR" }],
    series: [
      { name: "Top10% IC Mean", type: "bar", data: decay.map(d => d.ic_mean), itemStyle: { color: "#d29922" } },
      { name: "Top10% ICIR", type: "line", yAxisIndex: 1, data: decay.map(d => d.ic_ir), itemStyle: { color: "#3fb950" } },
    ],
  };
  return <ReactECharts option={option} style={{ height: 250, marginTop: 12 }} theme="dark" />;
}

function QuantileBarChart({ detail }: { detail: Record<string, unknown> }) {
  const grp = detail.layer3_grouping as Record<string, unknown> | undefined;
  const q5 = (grp?.["5"] as Record<string, unknown>)?.quantile_summary as Array<{ mean_return: number; win_rate: number }> | undefined;
  if (!q5) return null;
  const option = {
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: ["Q1 (Top)", "Q2", "Q3", "Q4", "Q5 (Bottom)"] },
    yAxis: [{ type: "value", name: "Mean Return" }, { type: "value", name: "Win Rate" }],
    series: [
      { name: "Mean Return", type: "bar", data: q5.map(q => q.mean_return), itemStyle: { color: "#58a6ff" } },
      { name: "Win Rate", type: "line", yAxisIndex: 1, data: q5.map(q => q.win_rate), itemStyle: { color: "#d29922" } },
    ],
  };
  return <ReactECharts option={option} style={{ height: 250, marginTop: 12 }} theme="dark" />;
}

// ── 样式 ──────────────────────────────────────────────────────────────────────

const selectStyle: React.CSSProperties = {
  background: "#21262d", color: "#e1e4e8", border: "1px solid #30363d", borderRadius: 4, padding: "4px 8px",
};
const inputStyle: React.CSSProperties = {
  background: "#21262d", color: "#e1e4e8", border: "1px solid #30363d", borderRadius: 4, padding: "4px 8px", width: 100,
};
const btnStyle: React.CSSProperties = {
  background: "#21262d", color: "#e1e4e8", border: "1px solid #30363d", borderRadius: 4, padding: "4px 12px", cursor: "pointer",
};
const tableStyle: React.CSSProperties = {
  width: "100%", borderCollapse: "collapse", fontSize: 13,
};
