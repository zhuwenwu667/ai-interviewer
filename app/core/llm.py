"""DeepSeek 聊天客户端（OpenAI 兼容协议，全异步）。

职责：
- 统一 chat / stream_chat 接口与按"用途"绑定模型、温度的工厂参数；
- 指数退避重试（429/超时/5xx 最多 2 次，401/400/404 立即失败）；
- 异常翻译为中文领域异常（不泄露 Key）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

from app.config.settings import Settings
from app.core.http_utils import retry_async, translate_openai_exception
from app.observability.tracer import get_tracer

# 用途 -> (模型配置属性名, 默认温度)
_PURPOSE_PARAMS: dict[str, tuple[str, float]] = {
    "question": ("question_model", 0.7),
    "followup": ("followup_model", 0.8),
    "eval": ("eval_model", 0.2),
    "rewrite": ("rewrite_model", 0.3),
}

ChatMessage = dict[str, str]


@dataclass
class ChatResult:
    """一次非流式对话的结果。"""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class DeepSeekChatClient:
    """DeepSeek 对话客户端。``transport`` 仅供测试注入（httpx.MockTransport）。"""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        http_client = httpx.AsyncClient(transport=transport) if transport else None
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key or "missing-key",
            base_url=settings.deepseek_base_url,
            timeout=timeout,
            max_retries=0,  # 重试策略由本类统一实现
            http_client=http_client,
        )
        self._settings = settings
        self._max_retries = max_retries
        self._provider = "DeepSeek"

    def _model_and_temperature(
        self, purpose: str, model: str | None, temperature: float | None
    ) -> tuple[str, float]:
        attr, default_temp = _PURPOSE_PARAMS.get(purpose, ("deepseek_model", 0.7))
        resolved_model = model or getattr(self._settings, attr)
        return resolved_model, (default_temp if temperature is None else temperature)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "question",
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        model_name, temp = self._model_and_temperature(purpose, model, temperature)
        kwargs: dict = {"model": model_name, "messages": messages, "temperature": temp}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        async def _call() -> ChatResult:
            resp = await self._client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            usage = resp.usage
            return ChatResult(
                content=choice.message.content or "",
                model=resp.model or model_name,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                finish_reason=choice.finish_reason,
            )

        with get_tracer().span("llm.chat", purpose=purpose, model=model_name, json_mode=json_mode) as span:
            result = await retry_async(
                _call, max_retries=self._max_retries, provider=self._provider, model=model_name
            )
            span.set(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                finish_reason=result.finish_reason,
            )
        return result  # type: ignore[return-value]

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "question",
        model: str | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """流式产出文本增量。建链阶段纳入重试，开始产出后失败直接抛领域异常。"""
        model_name, temp = self._model_and_temperature(purpose, model, temperature)

        with get_tracer().span("llm.stream_chat", purpose=purpose, model=model_name) as span:

            async def _open_stream():
                return await self._client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                    stream=True,
                )

            stream = await retry_async(
                _open_stream, max_retries=self._max_retries, provider=self._provider, model=model_name
            )
            chunks = 0
            chars = 0
            try:
                async for event in stream:
                    if event.choices and event.choices[0].delta.content:
                        text = event.choices[0].delta.content
                        chunks += 1
                        chars += len(text)
                        yield text
            except Exception as exc:  # noqa: BLE001
                raise translate_openai_exception(
                    exc, provider=self._provider, model=model_name
                ) from exc
            finally:
                await stream.close()
            span.set(chunks=chunks, chars=chars)

    async def close(self) -> None:
        await self._client.close()
