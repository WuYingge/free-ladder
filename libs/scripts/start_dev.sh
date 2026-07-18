#!/bin/bash
# ============================================================
# 同时启动前后端开发服务器
#
# 用法:
#     bash libs/scripts/start_dev.sh
# 或:
#     chmod +x libs/scripts/start_dev.sh
#     ./libs/scripts/start_dev.sh
#
# 后端: uvicorn → http://localhost:8000
# 前端: vite    → http://localhost:5173
# 按 Ctrl+C 停止所有服务
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ---------- 激活虚拟环境 ----------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif ! command -v uvicorn &>/dev/null; then
    echo "❌ 未找到 .venv 且 uvicorn 不可用，请先执行: uv sync"
    exit 1
fi

# ---------- 清理函数 ----------
cleanup() {
    echo ""
    echo "正在停止所有服务..."
    kill "$BACKEND_PID" 2>/dev/null
    kill "$FRONTEND_PID" 2>/dev/null
    wait "$BACKEND_PID" 2>/dev/null
    wait "$FRONTEND_PID" 2>/dev/null
    echo "已停止。"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ---------- 前置检查 ----------

if ! command -v npm &>/dev/null; then
    echo "❌ npm 未安装"
    exit 1
fi

if [ ! -d "web/node_modules" ]; then
    echo "📦 前端依赖未安装，正在安装..."
    cd web && npm install && cd "$PROJECT_ROOT"
fi

# ---------- 启动后端 ----------
echo "=== 启动后端 (uvicorn) ==="
uvicorn libs.web.main:app --reload --port 8000 &
BACKEND_PID=$!

# 等待后端先就绪
sleep 1

# ---------- 启动前端 ----------
echo "=== 启动前端 (vite) ==="
cd web
npm run dev &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  后端 : http://localhost:8000"
echo "  前端 : http://localhost:5173"
echo "  健康检查: http://localhost:8000/api/health"
echo "  按 Ctrl+C 停止所有服务"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

wait