# AI 模拟面试官 - 生产镜像（多阶段构建）
#
# 一个镜像同时承载后端 API（uvicorn）与前端（Streamlit），
# 由 docker-compose 用不同 command 分别启动，避免重复构建与版本漂移。
#
# 构建：
#   docker build -t ai-interviewer:latest .
# 本地直接运行后端：
#   docker run --rm -p 8000:8000 --env-file .env ai-interviewer:latest

# ============================================================
# Stage 1：用 uv 安装依赖（独立层，充分利用构建缓存）
# ============================================================
FROM python:3.11-slim AS builder

# PyPI 源：默认走清华 TUNA 镜像。实测国内服务器直连 files.pythonhosted.org
# 仅约 70 KB/s（17.9MB 的 uv 包耗时 265s），依赖层要下 ~250MB，会拖到近一小时；
# 换国内镜像可提升 5-10 倍。
# 注意：uv.lock 里逐个包写死了官方 CDN 绝对直链，--frozen 会按该 URL 直接下载，
# 只设 UV_INDEX_URL 不生效，还须在构建期改写直链主机（见下方 sed）。
# 海外环境或需官方源时覆盖：
#   docker build --build-arg PIP_INDEX_URL=https://pypi.org/simple \
#                --build-arg UV_INDEX_URL=https://pypi.org/simple \
#                --build-arg UV_MIRROR_HOST=files.pythonhosted.org .
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 与 files.pythonhosted.org 路径结构一致的包文件直链主机
ARG UV_MIRROR_HOST=pypi.tuna.tsinghua.edu.cn

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    UV_INDEX_URL=${UV_INDEX_URL}
# uv 版本与开发环境保持一致（uv --version = 0.13.0），保证锁定解析结果一致
RUN pip install "uv==0.13.0"

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 仅复制依赖声明文件，先同步第三方依赖：
# 源码变动不会使依赖层缓存失效；同时锁定 uv.lock 保证可复现。
# --no-install-project：本项目以 /app 下的源码直接运行（见下），
#   若把项目装进 site-packages，settings.py 的 BASE_DIR 会指向 site-packages，
#   导致 .env 与 data/ 相对路径解析错位，故不安装项目本身。
# README.md 被 pyproject 的 readme 字段引用，构建元数据需要它存在。
COPY pyproject.toml uv.lock README.md ./
# uv.lock 中每个包都内嵌了官方 CDN 绝对直链（https://files.pythonhosted.org/...），
# --frozen 下 uv 直接按该 URL 下载、忽略 UV_INDEX_URL，国内直连会长时间卡住。
# 国内镜像与官方文件字节一致、路径结构相同，故只改写主机名，不改 sha256 校验。
RUN sed -i "s#https://files.pythonhosted.org/#https://${UV_MIRROR_HOST}/#g" uv.lock
RUN uv sync --frozen --no-dev --no-install-project --extra obs

# ============================================================
# Stage 2：运行时（只带依赖与源码，非 root 运行）
# ============================================================
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 虚拟环境与源码路径：scripts/*.py 依赖 "当前目录即项目根"
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    HOME=/home/appuser \
    LANG=C.UTF-8

# 非 root 运行：UID/GID 1000 与常见云主机部署用户一致
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
# 知识库随镜像分发（不是卷）：保证 ingest_kb.py 读到的始终是本次构建的文档
COPY data/knowledge/ ./data/knowledge/
COPY data/eval/ ./data/eval/
COPY pyproject.toml README.md ./

# 可写目录：命名卷挂载点。命名卷首次创建时会继承镜像内目录属主，
# 因此非 root 的 appuser 对卷有写权限（无需启动时 chown）。
RUN mkdir -p /app/data/db /app/data/reports /app/logs \
    && chown -R appuser:appuser /app/data /app/logs

USER appuser
EXPOSE 8000 8501

# 默认启动后端；Streamlit 前端由 compose 覆写 command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
