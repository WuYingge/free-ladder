"""
投资框架网站 — FastAPI 后端入口

启动方式:
    cd /home/gouzi/projects/invest
    python -m uvicorn libs.web.main:app --reload --port 8000

或:
    cd /home/gouzi/projects/invest
    PYTHONPATH=. uvicorn libs.web.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根和 libs 在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LIBS_DIR = PROJECT_ROOT / "libs"
WEB_DIR = Path(__file__).resolve().parent
for p in [str(PROJECT_ROOT), str(LIBS_DIR), str(WEB_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="投资框架",
    description="因子分析 & 因子走势可视化",
    version="0.1.0",
)

# CORS — 允许前端 localhost 访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from routers.factors import router as factors_router    # noqa: E402
from routers.trend import router as trend_router        # noqa: E402

app.include_router(factors_router)
app.include_router(trend_router)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


# 启动时自动构建因子索引（如果不存在）
@app.on_event("startup")
async def startup_event():
    from services.factor_index import build_index
    print("[startup] 检查因子索引...")
    idx = build_index(force=False)
    print(f"[startup] 因子索引就绪: {idx['n_factors']} 个因子")
