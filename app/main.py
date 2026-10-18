"""FastAPI 应用入口（唯一事实源）。

本地启动：  uv run uvicorn app.main:app --reload --port 8000
健康检查：  GET http://localhost:8000/health
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.health import router as health_router
from app.api.sessions import router as sessions_router
from app.api.train import router as train_router
from app.config import get_settings

access_logger = logging.getLogger("app.access")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI 模拟面试官 API",
        description="Agentic RAG + LangGraph 面试 Agent 后端服务",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    @app.middleware("http")
    async def access_log(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """访问日志：method/path/status/total_ms（SSE 端点另自记 first_event_ms）。"""
        start = time.perf_counter()
        response = await call_next(request)
        access_logger.info(
            "%s %s -> %d total_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(sessions_router)
    app.include_router(train_router)
    return app


app = create_app()
