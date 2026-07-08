import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import FactorOverview from "./pages/FactorOverview";
import FactorTrend from "./pages/FactorTrend";

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ minHeight: "100vh", background: "#0f1117", color: "#e1e4e8" }}>
        <nav style={{
          display: "flex", gap: 24, padding: "12px 24px",
          background: "#161b22", borderBottom: "1px solid #30363d"
        }}>
          <span style={{ fontWeight: 700, fontSize: 18, marginRight: 24 }}>📊 投资框架</span>
          <NavLink to="/" style={linkStyle}>因子概览</NavLink>
          <NavLink to="/trend" style={linkStyle}>因子走势</NavLink>
        </nav>
        <main style={{ padding: 24 }}>
          <Routes>
            <Route path="/" element={<FactorOverview />} />
            <Route path="/trend" element={<FactorTrend />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

const linkStyle = ({ isActive }: { isActive: boolean }) => ({
  color: isActive ? "#58a6ff" : "#8b949e",
  textDecoration: "none",
  fontSize: 15,
  padding: "4px 8px",
  borderRadius: 4,
  background: isActive ? "#1f2937" : "transparent",
});
