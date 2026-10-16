"""统一错误响应与全局异常处理。

所有非 SSE 接口的错误响应结构统一为 ``ErrorResponse``（error.code/message/details）；
SSE 接口的错误以 ``SseEvent.error`` 事件下发（见 sessions 路由 ``_to_error_event``）。
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agent.report import ReportGenerationError
from app.api.schemas import ErrorBody, ErrorResponse
from app.core.exceptions import ModelProviderError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """业务错误：路由层显式抛出，由全局处理器转换为统一错误响应。"""

    def __init__(
        self, status_code: int, code: str, message: str, details: dict | None = None
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


def _payload(code: str, message: str, details: dict | None = None) -> dict:
    return ErrorResponse(
        error=ErrorBody(code=code, message=message, details=details)
    ).model_dump()


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器（顺序无关，FastAPI 按异常类型匹配）。"""

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "请求错误"
        return JSONResponse(
            status_code=exc.status_code,
            content=_payload(f"HTTP_{exc.status_code}", message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []))
        message = f"请求参数校验失败：{loc} {first.get('msg', '')}".strip()
        return JSONResponse(status_code=422, content=_payload("VALIDATION_ERROR", message))

    @app.exception_handler(ModelProviderError)
    async def _model_error(request: Request, exc: ModelProviderError) -> JSONResponse:
        # 上游模型错误（消息已含中文排查建议）：502 Bad Gateway
        return JSONResponse(status_code=502, content=_payload("UPSTREAM_ERROR", str(exc)))

    @app.exception_handler(ReportGenerationError)
    async def _report_error(request: Request, exc: ReportGenerationError) -> JSONResponse:
        return JSONResponse(
            status_code=502, content=_payload("REPORT_GENERATION_FAILED", str(exc))
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常 %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500, content=_payload("INTERNAL_ERROR", f"服务内部错误：{exc}")
        )
