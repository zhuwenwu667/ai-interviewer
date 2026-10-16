"""模型供应商 HTTP 调用的共享能力：异常翻译与异步指数退避重试。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import openai

from app.core.exceptions import (
    AuthenticationError,
    BadRequestError,
    ModelNotFoundError,
    RateLimitError,
    RetryExhaustedError,
    UpstreamServerError,
    UpstreamTimeoutError,
)


def translate_openai_exception(
    exc: Exception, *, provider: str, model: str = ""
) -> Exception:
    """把 openai SDK / httpx 异常映射为中文领域异常（不泄露 Key）。"""
    suffix = f"（供应商：{provider}" + (f"，模型：{model}）" if model else "）")
    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(
            "API Key 无效或已过期（HTTP 401）。请检查 .env 中的密钥是否正确、"
            "是否已在对应平台激活" + suffix,
            provider=provider,
            model=model,
        )
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(
            "触发限流或账户额度不足（HTTP 429）。请稍后重试，并检查账户余额与"
            "速率限制" + suffix,
            provider=provider,
            model=model,
        )
    if isinstance(exc, openai.NotFoundError):
        return ModelNotFoundError(
            "模型不存在或当前账号无该模型权限（HTTP 404）。请检查 .env 中配置的"
            "模型名" + suffix,
            provider=provider,
            model=model,
        )
    if isinstance(exc, openai.BadRequestError):
        return BadRequestError(
            f"请求参数被供应商拒绝（HTTP 400）：{str(exc)[:300]}" + suffix,
            provider=provider,
            model=model,
        )
    if isinstance(exc, (openai.APITimeoutError, httpx.TimeoutException)):
        return UpstreamTimeoutError(
            "调用模型超时，请检查网络或稍后重试" + suffix,
            provider=provider,
            model=model,
        )
    if isinstance(exc, openai.InternalServerError):
        return UpstreamServerError(
            "模型供应商服务暂时不可用（HTTP 5xx），重试后仍失败，请稍后再试" + suffix,
            provider=provider,
            model=model,
        )
    if isinstance(exc, openai.APIConnectionError):
        return UpstreamServerError(
            "无法连接模型供应商，请检查 base_url、网络与代理设置" + suffix,
            provider=provider,
            model=model,
        )
    return exc


async def retry_async(
    operation: Callable[[], Awaitable],
    *,
    max_retries: int = 2,
    provider: str = "",
    model: str = "",
) -> object:
    """对可重试异常指数退避（1s, 2s...）；401/400/404 立即失败。"""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001 - 统一翻译后再判定
            translated = translate_openai_exception(exc, provider=provider, model=model)
            last_exc = translated
            if not getattr(translated, "retryable", False) or attempt == max_retries:
                raise translated from exc
            await asyncio.sleep(2**attempt)
    raise RetryExhaustedError(str(last_exc), provider=provider, model=model)
