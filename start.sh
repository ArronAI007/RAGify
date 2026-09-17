#!/usr/bin/env bash
# 一键启动/停止/重启 RAGify（API + 前端两个后台进程）。
#
# 用法:
#   ./start.sh            # 启动（若已在运行，先自动停止旧进程再启动，幂等）
#   ./start.sh start      # 同上
#   ./start.sh restart    # 显式重启
#   ./start.sh stop       # 停止
#   ./start.sh status     # 查看运行状态
#
# 日志: tail -f .run/api.log 或 .run/frontend.log
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
RUN_DIR="$PROJECT_ROOT/.run"

PID_FILE="$RUN_DIR/frontend.pid"
LOG_FILE="$RUN_DIR/frontend.log"
PORT="${PORT:-3000}"

API_PID_FILE="$RUN_DIR/api.pid"
API_LOG_FILE="$RUN_DIR/api.log"
API_PORT="${API_PORT:-8000}"

info() { printf '\033[1;34m[start]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[start]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[start]\033[0m %s\n' "$1" >&2; }

mkdir -p "$RUN_DIR"

is_service_running() {
  [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

stop_service() {
  local label="$1" pid_file="$2" port="$3"
  if is_service_running "$pid_file"; then
    local pid
    pid="$(cat "$pid_file")"
    info "停止${label} (PID $pid) ..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"

  if command -v lsof >/dev/null 2>&1; then
    local port_pids
    port_pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    if [ -n "$port_pids" ]; then
      warn "端口 ${port} 仍被占用 (PID: $port_pids)，一并终止"
      kill $port_pids 2>/dev/null || true
    fi
  fi
}

stop_all() {
  stop_service "前端服务" "$PID_FILE" "$PORT"
  stop_service "API 服务" "$API_PID_FILE" "$API_PORT"
}

status_all() {
  if is_service_running "$PID_FILE"; then
    info "前端服务运行中 (PID $(cat "$PID_FILE"))，http://localhost:${PORT}"
  else
    info "前端服务未运行"
  fi
  if is_service_running "$API_PID_FILE"; then
    info "API 服务运行中 (PID $(cat "$API_PID_FILE"))，http://localhost:${API_PORT}"
  else
    info "API 服务未运行"
  fi
}

ensure_backend_ready() {
  # 1. Python 环境
  if ! command -v uv >/dev/null 2>&1; then
    err "未找到 uv，请先安装：pip install uv"
    exit 1
  fi

  if [ ! -x "$VENV_PYTHON" ]; then
    info "未发现虚拟环境，正在创建 .venv ..."
    (cd "$PROJECT_ROOT" && uv venv)
  fi

  if ! "$VENV_PYTHON" -c "import ragify" >/dev/null 2>&1; then
    info "安装/同步 Python 依赖 ..."
    (cd "$PROJECT_ROOT" && uv pip install -e .)
  fi

  # 2. 配置文件
  if [ ! -f "$PROJECT_ROOT/config/config.yaml" ]; then
    if [ -f "$PROJECT_ROOT/config/config.yaml.example" ]; then
      warn "config/config.yaml 不存在，从示例文件复制一份"
      cp "$PROJECT_ROOT/config/config.yaml.example" "$PROJECT_ROOT/config/config.yaml"
    else
      err "缺少 config/config.yaml，请先手动创建配置文件"
      exit 1
    fi
  fi

  if [ ! -f "$PROJECT_ROOT/.env" ]; then
    warn ".env 不存在，请确认已配置所需的 API Key（如 DASHSCOPE_API_KEY）"
  fi

  # 3. 前端依赖
  if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    info "安装前端依赖 ..."
    (cd "$FRONTEND_DIR" && npm install)
  fi

  # 4. 数据库迁移
  info "执行数据库迁移 ..."
  (cd "$PROJECT_ROOT" && "$VENV_PYTHON" -m alembic upgrade head)
}

start_api() {
  info "启动 API 服务（后台运行，http://localhost:${API_PORT}）..."
  (
    cd "$PROJECT_ROOT"
    nohup "$VENV_PYTHON" -m uvicorn ragify.api.main:app --host 0.0.0.0 --port "$API_PORT" >"$API_LOG_FILE" 2>&1 &
    echo $! > "$API_PID_FILE"
  )
  sleep 1
  if is_service_running "$API_PID_FILE"; then
    info "API 已启动 (PID $(cat "$API_PID_FILE"))，日志: tail -f $API_LOG_FILE"
  else
    err "API 启动失败，请查看日志: $API_LOG_FILE"
    exit 1
  fi
}

start_frontend() {
  info "启动前端开发服务器（后台运行，http://localhost:${PORT}）..."
  (
    cd "$FRONTEND_DIR"
    nohup npm run dev -- -p "$PORT" >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
  )
  sleep 1
  if is_service_running "$PID_FILE"; then
    info "前端已启动 (PID $(cat "$PID_FILE"))，日志: tail -f $LOG_FILE"
  else
    err "前端启动失败，请查看日志: $LOG_FILE"
    exit 1
  fi
}

start_all() {
  stop_all
  ensure_backend_ready
  start_api
  start_frontend
}

ACTION="${1:-start}"
case "$ACTION" in
  start)
    start_all
    ;;
  stop)
    stop_all
    info "已停止"
    ;;
  restart)
    start_all
    ;;
  status)
    status_all
    ;;
  *)
    err "未知参数: ${ACTION}（支持 start|stop|restart|status）"
    exit 1
    ;;
esac
