import * as echarts from "echarts";

// ECharts 6.x 不再内置 dark 主题，手动注册
echarts.registerTheme("dark", {
  backgroundColor: "#0f1117",
  color: ["#58a6ff", "#3fb950", "#d29922", "#f0883e", "#f85149", "#a371f7"],
  title: { textStyle: { color: "#e1e4e8" } },
  legend: { textStyle: { color: "#8b949e" } },
  tooltip: { backgroundColor: "#161b22", borderColor: "#30363d", textStyle: { color: "#e1e4e8" } },
  categoryAxis: { axisLine: { lineStyle: { color: "#30363d" } }, axisLabel: { color: "#8b949e" }, splitLine: { lineStyle: { color: "#21262d" } } },
  valueAxis: { axisLine: { lineStyle: { color: "#30363d" } }, axisLabel: { color: "#8b949e" }, splitLine: { lineStyle: { color: "#21262d" } } },
  dataZoom: { textStyle: { color: "#8b949e" } },
});
