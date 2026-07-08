import { useState, useEffect } from "react";
import ReactECharts from "echarts-for-react";
import { fetchTrend, fetchTrendableFactors, fetchSymbols } from "../api";
import type { TrendPoint } from "../api";

export default function FactorTrend() {
  const [factors, setFactors] = useState<string[]>([]);
  const [symbols, setSymbols] = useState<{ symbol: string; name: string }[]>([]);
  const [factorName, setFactorName] = useState("");
  const [symbol, setSymbol] = useState("");
  const [factorInput, setFactorInput] = useState("");
  const [symbolInput, setSymbolInput] = useState("");
  const [data, setData] = useState<TrendPoint[]>([]);
  const [stats, setStats] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [symbolName, setSymbolName] = useState("");

  useEffect(() => {
    fetchTrendableFactors().then(r => setFactors(r.factors)).catch(() => {});
    fetchSymbols().then(r => setSymbols(r.symbols)).catch(() => {});
  }, []);

  const loadTrend = async () => {
    const fName = factorName || factorInput;
    const sCode = symbol || symbolInput;
    if (!fName || !sCode) { setError("请选择因子和ETF"); return; }
    setLoading(true); setError("");
    try {
      const res = await fetchTrend(fName, sCode);
      setData(res.data);
      setStats(res.stats);
      setSymbolName(res.symbol_name);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "加载失败");
    }
    setLoading(false);
  };

  const filteredFactors = factorInput
    ? factors.filter(f => f.toLowerCase().includes(factorInput.toLowerCase())).slice(0, 20)
    : factors.slice(0, 50);
  const filteredSymbols = symbolInput
    ? symbols.filter(s => s.symbol.includes(symbolInput) || s.name.includes(symbolInput)).slice(0, 20)
    : symbols.slice(0, 50);

  // 图表配置
  const volumeChartOption = buildChartOption(data, {
    series1Name: "成交额", series1Data: data.map(d => d.volume ? d.volume / 1e8 : null),
    series1Type: "bar", series1Color: "#30363d",
    series2Name: "因子值", series2Data: data.map(d => d.factor_value),
    y1Name: "成交额(亿)", y2Name: "因子值",
  });

  const priceChartOption = buildChartOption(data, {
    series1Name: "收盘价", series1Data: data.map(d => d.close),
    series1Type: "line", series1Color: "#58a6ff",
    series2Name: "因子值", series2Data: data.map(d => d.factor_value),
    y1Name: "收盘价(元)", y2Name: "因子值",
  });

  return (
    <div>
      <h2 style={{ marginBottom: 16 }}>因子走势</h2>

      {/* 选择区域 */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16, alignItems: "flex-start" }}>
        <div>
          <label style={{ display: "block", marginBottom: 4 }}>因子 (可输入搜索):</label>
          <input placeholder="输入因子名筛选..." value={factorInput} onChange={e => { setFactorInput(e.target.value); setFactorName(""); }}
            style={inputStyle} list="factor-list" />
          <datalist id="factor-list">
            {filteredFactors.map(f => <option key={f} value={f} />)}
          </datalist>
          <select size={6} value={factorName} onChange={e => setFactorName(e.target.value)}
            style={{ ...selectStyle, width: 320, marginTop: 4 }}>
            {filteredFactors.map(f => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>

        <div>
          <label style={{ display: "block", marginBottom: 4 }}>ETF 标的:</label>
          <input placeholder="输入代码或名称..." value={symbolInput} onChange={e => { setSymbolInput(e.target.value); setSymbol(""); }}
            style={inputStyle} list="symbol-list" />
          <datalist id="symbol-list">
            {filteredSymbols.map(s => <option key={s.symbol} value={s.symbol}>{s.symbol} {s.name}</option>)}
          </datalist>
          <select size={6} value={symbol} onChange={e => setSymbol(e.target.value)}
            style={{ ...selectStyle, width: 280, marginTop: 4 }}>
            {filteredSymbols.map(s => <option key={s.symbol} value={s.symbol}>{s.symbol} {s.name}</option>)}
          </select>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 22 }}>
          <button onClick={loadTrend} disabled={loading} style={goBtnStyle}>
            {loading ? "加载中..." : "📈 加载走势"}
          </button>
          {error && <div style={{ color: "#f85149", fontSize: 13 }}>{error}</div>}
        </div>
      </div>

      {/* 统计摘要 */}
      {Object.keys(stats).length > 0 && (
        <div style={{ marginBottom: 12, padding: "8px 12px", background: "#161b22", borderRadius: 6, display: "flex", gap: 24, flexWrap: "wrap", fontSize: 13 }}>
          <span>标的: <b>{symbolName || symbol}</b></span>
          <span>数据点: <b>{String(stats.data_points)}</b></span>
          <span>因子均值: <b>{Number(stats.factor_mean)?.toFixed(4)}</b></span>
          <span>因子标准差: <b>{Number(stats.factor_std)?.toFixed(4)}</b></span>
          <span>最新因子值: <b>{Number(stats.factor_latest)?.toFixed(4)}</b></span>
          <span>最新收盘价: <b>{Number(stats.close_latest)?.toFixed(2)}</b></span>
        </div>
      )}

      {/* 图表 */}
      {data.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <h4 style={{ margin: "4px 0" }}>成交额 + 因子值</h4>
            <ReactECharts option={volumeChartOption} style={{ height: 400 }} theme="dark" />
          </div>
          <div>
            <h4 style={{ margin: "4px 0" }}>收盘价 + 因子值</h4>
            <ReactECharts option={priceChartOption} style={{ height: 400 }} theme="dark" />
          </div>
        </div>
      )}

      {!loading && data.length === 0 && !error && (
        <p style={{ color: "#8b949e" }}>选择一个因子和 ETF，然后点击"加载走势"</p>
      )}
    </div>
  );
}

// ── 图表构建 ──────────────────────────────────────────────────────────────────

interface ChartSeriesConfig {
  series1Name: string;
  series1Data: (number | null)[];
  series1Type: string;
  series1Color: string;
  series2Name: string;
  series2Data: (number | null)[];
  y1Name: string;
  y2Name: string;
}

function buildChartOption(data: TrendPoint[], cfg: ChartSeriesConfig) {
  return {
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "cross" },
      formatter: (params: Array<{ seriesName: string; value: number; axisValue: string }>) => {
        if (!params || params.length === 0) return "";
        let tip = `<b>${params[0].axisValue}</b><br/>`;
        params.forEach(p => {
          tip += `${p.seriesName}: ${p.value?.toFixed?.(4) ?? p.value}<br/>`;
        });
        return tip;
      },
    },
    legend: { data: [cfg.series1Name, cfg.series2Name], textStyle: { color: "#e1e4e8" } },
    grid: { left: 60, right: 60, top: 40, bottom: 40 },
    dataZoom: [
      { type: "inside", start: 0, end: 100 },
      { type: "slider", start: 0, end: 100, height: 24, bottom: 8 },
    ],
    xAxis: {
      type: "category", data: data.map(d => d.date),
      axisLabel: { color: "#8b949e", rotate: 45 },
    },
    yAxis: [
      { type: "value", name: cfg.y1Name, nameTextStyle: { color: "#8b949e" }, axisLabel: { color: "#8b949e" } },
      { type: "value", name: cfg.y2Name, nameTextStyle: { color: "#8b949e" }, axisLabel: { color: "#8b949e" } },
    ],
    series: [
      {
        name: cfg.series1Name, type: cfg.series1Type, data: cfg.series1Data,
        yAxisIndex: 0, itemStyle: { color: cfg.series1Color },
      },
      {
        name: cfg.series2Name, type: "line", data: cfg.series2Data,
        yAxisIndex: 1, itemStyle: { color: "#f0883e" }, smooth: true,
      },
    ],
  };
}

// ── 样式 ──────────────────────────────────────────────────────────────────────

const selectStyle: React.CSSProperties = {
  background: "#21262d", color: "#e1e4e8", border: "1px solid #30363d", borderRadius: 4, padding: "4px",
};
const inputStyle: React.CSSProperties = {
  background: "#21262d", color: "#e1e4e8", border: "1px solid #30363d", borderRadius: 4, padding: "6px 8px", width: 260,
};
const goBtnStyle: React.CSSProperties = {
  background: "#238636", color: "#fff", border: "none", borderRadius: 6, padding: "8px 20px", cursor: "pointer",
  fontSize: 14, fontWeight: 600,
};
