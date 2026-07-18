// API 客户端
const BASE = "/api";

export interface FactorSummary {
  name: string;
  factor_type: string;
  params: Record<string, unknown>;
  n_symbols: number;
  start_date: string;
  end_date: string;
  coverage_mean: number;
  ic_rank: Record<string, ICStats>;
  ic_pearson: Record<string, ICStats>;
  longshort_sharpe_5d: number | null;
  top_ic: Record<string, ICStats>;
  analysis_date: string;
}

export interface ICStats {
  mean: number | null;
  std: number | null;
  ir: number | null;
  t_stat: number | null;
  pos_ratio: number | null;
}

export interface FactorListResponse {
  total: number;
  page: number;
  page_size: number;
  factors: FactorSummary[];
}

export interface TrendPoint {
  date: string;
  factor_value: number | null;
  close: number | null;
  volume: number | null;
}

export interface TrendResponse {
  factor_name: string;
  symbol: string;
  symbol_name: string;
  start_date: string;
  end_date: string;
  data: TrendPoint[];
  stats: Record<string, unknown>;
}

export async function fetchFactors(params: Record<string, string | number>): Promise<FactorListResponse> {
  const qs = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") qs.set(k, String(v)); });
  const res = await fetch(`${BASE}/factors?${qs}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchFactorDetail(name: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${BASE}/factors/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchCorrelation(name: string, threshold: number, direction: string, limit: number) {
  const qs = new URLSearchParams({ threshold: String(threshold), direction, limit: String(limit) });
  const res = await fetch(`${BASE}/factors/${encodeURIComponent(name)}/correlation?${qs}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchTrend(factorName: string, symbol: string, startDate?: string, endDate?: string): Promise<TrendResponse> {
  const qs = new URLSearchParams({ symbol });
  if (startDate) qs.set("start_date", startDate);
  if (endDate) qs.set("end_date", endDate);
  const res = await fetch(`${BASE}/trend/${encodeURIComponent(factorName)}?${qs}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchSymbols(): Promise<{ symbols: { symbol: string; name: string }[] }> {
  const res = await fetch(`${BASE}/trend/_/symbols`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function fetchTrendableFactors(): Promise<{ factors: string[] }> {
  const res = await fetch(`${BASE}/trend/_/trendable-factors`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}
