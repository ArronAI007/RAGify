#!/usr/bin/env bash
# 一键启动/停止/重启 RAGify 前端（后台守护进程模式）。
#
# 用法:
#   ./start.sh            # 启动（若已在运行，先自动停止旧进程再启动，幂等）
#   ./start.sh start      # 同上
#   ./start.sh restart    # 显式重启
#   ./start.sh stop       # 停止
#   ./start.sh status     # 查看运行状态
#
# 日志: tail -f .run/frontend.log
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
RUN_DIR="$PROJECT_ROOT/.run"
PID_FILE="$RUN_DIR/frontend.pid"
LOG_FILE="$RUN_DIR/frontend.log"
PORT="${PORT:-3000}"

info() { printf '\033[1;34m[start]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[start]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[start]\033[0m %s\n' "$1" >&2; }

mkdir -p "$RUN_DIR"

is_running() {
  [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

stop_frontend() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    info "停止正在运行的前端服务 (PID $pid) ..."
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"

  # 兜底：杀掉任何仍占用该端口的进程（例如脚本外手动起的实例）
  if command -v lsof >/dev/null 2>&1; then
    local port_pids
    port_pids="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
    if [ -n "$port_pids" ]; then
      warn "端口 $PORT 仍被占用 (PID: $port_pids)，一并终止"
      kill $port_pids 2>/dev/null || true
    fi
  fi
}

status_frontend() {
  if is_running; then
    info "前端服务运行中 (PID $(cat "$PID_FILE"))，http://localhost:$PORT"
  else
    info "前端服务未运行"
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
}

start_frontend() {
  stop_frontend
  ensure_backend_ready

  info "启动前端开发服务器（后台运行，http://localhost:${PORT}）..."
  (
    cd "$FRONTEND_DIR"
    nohup npm run dev -- -p "$PORT" >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
  )
  sleep 1
  if is_running; then
    info "已启动 (PID $(cat "$PID_FILE"))，日志: tail -f $LOG_FILE"
  else
    err "启动失败，请查看日志: $LOG_FILE"
    exit 1
  fi
}

ACTION="${1:-start}"
case "$ACTION" in
  start)
    start_frontend
    ;;
  stop)
    stop_frontend
    info "已停止"
    ;;
  restart)
    start_frontend
    ;;
  status)
    status_frontend
    ;;
  *)
    err "未知参数: ${ACTION}（支持 start|stop|restart|status）"
    exit 1
    ;;
esac
