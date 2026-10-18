"""健康检查路由。

``GET /health`` 为全系统唯一的存活探测地址：
- 始终返回各组件的配置/连通状态（不因未配置 key 而拒绝请求），
  便于部署验活与前端"开始页"展示 API 连通情况；
- 仅当关键组件不可达时整体 status 标记为 degraded。
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.api.quota import get_quota
from app.config import get_settings

router = APIRouter(tags=["health"])


async def _check_qdrant() -> dict[str, object]:
    """探测 Qdrant：server 模式请求 /healthz；local 模式确认数据目录可写。"""
    settings = get_settings()
    if settings.qdrant_mode == "local":
        path = settings.qdrant_path_abs
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".healthprobe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return {"status": "ok", "mode": "local", "detail": f"数据目录可写: {path}"}
        except OSError as exc:
            return {"status": "error", "mode": "local", "detail": f"数据目录不可用: {exc}"}

    # server 模式：Qdrant 提供 GET /healthz（就绪返回 200）
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.qdrant_url.rstrip('/')}/healthz")
        if resp.status_code == 200:
            return {"status": "ok", "mode": "server", "detail": settings.qdrant_url}
        return {
            "status": "error",
            "mode": "server",
            "detail": f"HTTP {resp.status_code}: {resp.text[:200]}",
        }
    except httpx.HTTPError as exc:
        return {"status": "error", "mode": "server", "detail": f"连接失败: {exc}"}


@router.get("/health")
async def health() -> dict[str, object]:
    """组件健康状态。HTTP 200 表示进程存活；整体 status 区分 healthy/degraded。"""
    settings = get_settings()
    qdrant = await _check_qdrant()
    config = settings.config_status()
    keys_ready = bool(config["deepseek_configured"] and config["siliconflow_configured"])
    overall = "healthy" if qdrant["status"] == "ok" and keys_ready else "degraded"
    return {
        "status": overall,
        "version": "0.1.0",
        "config": config,
        "components": {"qdrant": qdrant},
    }


@router.get("/quota")
async def quota() -> dict[str, object]:
    """配额快照（P2-③）：限速配置 + 当日 LLM 用量/成本与熔断阈值，供运维观测。"""
    return get_quota().snapshot()
