#!/usr/bin/env bash
# ============================================================
# AI 模拟面试官 - Linux 云服务器一键部署脚本
#
# 用法（项目根目录或任意目录）：
#   bash deploy/deploy.sh                 # 构建 → 启动 → 验活 → 摄入知识库 → 打印访问地址
#   bash deploy/deploy.sh --no-build      # 跳过镜像构建（仅重启）
#   bash deploy/deploy.sh --skip-ingest   # 跳过知识库摄入（例如已有数据）
#   bash deploy/deploy.sh --help          # 查看说明（含 Windows 本地使用方式）
#
# 脚本假设：已安装 Docker Engine + docker compose v2 插件，
# 当前用户可直接执行 docker（否则请用 sudo 或将用户加入 docker 组）。
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WAIT_TIMEOUT="${WAIT_TIMEOUT:-300}" # 等待容器 healthy 的最长秒数
DO_BUILD=1
DO_INGEST=1

# ---------------- 输出辅助 ----------------
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[警告] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[错误] %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Windows 本地（PowerShell）等价操作：
  Copy-Item .env.example .env      # 复制后用编辑器填入真实 Key
  docker compose build
  docker compose up -d
  docker compose exec api python scripts/check_models.py
  docker compose exec api python scripts/ingest_kb.py --rebuild
  # 浏览器打开 http://localhost:8501 ；日志： docker compose logs -f api
EOF
}

for arg in "$@"; do
  case "$arg" in
    --help|-h) usage; exit 0 ;;
    --no-build) DO_BUILD=0 ;;
    --skip-ingest) DO_INGEST=0 ;;
    *) die "未知参数：$arg（可用 --help 查看说明）" ;;
  esac
done

# ---------------- 1. 环境检查 ----------------
check_docker() {
  log "检查 Docker 环境"
  command -v docker >/dev/null 2>&1 \
    || die "未检测到 docker。请先安装 Docker Engine（参见 README 部署章节）。"
  docker compose version >/dev/null 2>&1 \
    || die "未检测到 docker compose v2 插件。请安装 docker-compose-plugin 后重试。"
  docker info >/dev/null 2>&1 \
    || die "无法连接 Docker 守护进程（未启动，或当前用户无权限）。
        Linux 可执行：sudo systemctl start docker && sudo usermod -aG docker \$USER（重新登录后生效）"
  printf '    Docker 可用，compose 版本：%s\n' "$(docker compose version --short 2>/dev/null || echo 'unknown')"
}

# ---------------- 2. .env 引导 ----------------
ensure_env() {
  log "检查 .env 配置"
  if [ ! -f .env ]; then
    cp .env.example .env
    warn "未找到 .env，已从 .env.example 生成模板：${ROOT_DIR}/.env"
    warn "请填入真实的 DEEPSEEK_API_KEY 与 SILICONFLOW_API_KEY 后重新运行本脚本。"
    exit 1
  fi
  # 模板占位符未被替换时提前拦截，避免启动后才报鉴权失败
  if grep -q 'sk-your-deepseek-key-here' .env || grep -q 'sk-your-siliconflow-key-here' .env; then
    die ".env 中仍是 .env.example 的占位符，请填入真实 API Key 后重试。"
  fi
  printf '    .env 已存在（密钥仅注入容器环境变量，不会写入镜像）\n'
}

# ---------------- 3. 构建与启动 ----------------
build_and_up() {
  if [ "$DO_BUILD" -eq 1 ]; then
    log "构建镜像（首次构建需下载依赖，耗时较长）"
    docker compose build
  else
    log "跳过镜像构建（--no-build）"
  fi
  log "启动三个容器（qdrant / api / streamlit）"
  docker compose up -d
}

# ---------------- 4. 等待 healthy ----------------
_container_health() { # $1=服务名
  local cid
  cid="$(docker compose ps -q "$1" 2>/dev/null || true)"
  [ -n "$cid" ] || { echo "missing"; return; }
  docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || echo "missing"
}

wait_healthy() {
  log "等待容器进入 healthy 状态（最长 ${WAIT_TIMEOUT}s）"
  local services=(qdrant api streamlit)
  local deadline=$((SECONDS + WAIT_TIMEOUT))
  while :; do
    local all_ok=1 line=""
    for svc in "${services[@]}"; do
      local status
      status="$(_container_health "$svc")"
      line+="    ${svc}=${status}"
      [ "$status" = "healthy" ] || all_ok=0
    done
    if [ "$all_ok" -eq 1 ]; then
      printf '%s\n' "$line"
      printf '    三个容器均已 healthy\n'
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      printf '%s\n' "$line"
      warn "等待超时。排查：docker compose ps；docker compose logs --tail=80 api"
      die "容器未在 ${WAIT_TIMEOUT}s 内全部 healthy。"
    fi
    printf '%s\n' "$line"
    sleep 5
  done
}

# ---------------- 5. 模型验活 ----------------
verify_models() {
  log "运行三接口验活（DeepSeek 对话 / SiliconFlow embedding / rerank）"
  if docker compose exec -T api python scripts/check_models.py; then
    printf '    三接口均可用\n'
  else
    warn "验活存在失败项（详见上表，常见原因：Key 无效、余额不足、网络不通）"
    return 1
  fi
}

# ---------------- 6. 摄入知识库 ----------------
ingest_kb() {
  log "摄入知识库到 Qdrant（--rebuild 清空后全量重建）"
  docker compose exec -T api python scripts/ingest_kb.py --rebuild
}

# ---------------- 7. 打印访问地址 ----------------
print_url() {
  local ip=""
  ip="$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || true)"
  if [ -z "$ip" ]; then
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  log "部署完成"
  if [ -n "$ip" ]; then
    printf '    前端地址：http://%s:8501   （在浏览器完成一次完整模拟面试）\n' "$ip"
  else
    printf '    前端地址：http://<服务器公网IP>:8501\n'
  fi
  cat <<'EOF'
    后端地址：http://<服务器IP>:8000/health（建议安全组仅放行 8501 与 22）
    常用命令：
      docker compose ps                      # 容器与健康状态
      docker compose logs -f api             # 后端日志（含 SSE 首字节耗时）
      docker compose logs -f streamlit       # 前端日志
      docker compose exec api python scripts/ingest_kb.py --rebuild   # 重新摄入知识库
      bash deploy/deploy.sh --no-build       # 拉取更新后重建并重启
    评测（RAGAS）在本地开发环境运行：uv sync --extra eval；uv run python scripts/run_eval.py
EOF
}

# ---------------- 主流程 ----------------
main() {
  printf '\033[1mAI 模拟面试官 - 部署\033[0m  项目目录：%s\n' "$ROOT_DIR"
  check_docker
  ensure_env
  build_and_up
  wait_healthy
  verify_models || warn "已启动服务，但接口验活未全部通过，请修复后重试。"
  if [ "$DO_INGEST" -eq 1 ]; then
    ingest_kb
  else
    log "跳过知识库摄入（--skip-ingest）"
  fi
  print_url
}

main
