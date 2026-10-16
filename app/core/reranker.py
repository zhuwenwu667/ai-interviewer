"""硅基流动 Reranker 客户端（bge-reranker-v2-m3，/rerank 端点）。

注意：rerank 不属于 OpenAI 标准接口，使用 httpx 直接调用；
按 HTTP 状态码映射到与其他客户端一致的领域异常。
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.config.settings import Settings
from app.core.exceptions import (
    AuthenticationError,
    BadRequestError,
    ModelNotFoundError,
    RateLimitError,
    UpstreamServerError,
    UpstreamTimeoutError,
)
from app.core.http_utils import retry_async

PROVIDER = "SiliconFlow"


@dataclass
class RerankItem:
    """单条重排结果。"""

    index: int  # 入参 documents 中的下标
    score: float


class SiliconFlowReranker:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 2,
        timeout: float = 30.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.siliconflow_base_url,
            headers={"Authorization": f"Bearer {settings.siliconflow_api_key or 'missing-key'}"},
            timeout=timeout,
            transport=transport,
        )
        self._model = settings.reranker_model
        self._max_retries = max_retries

    def _raise_by_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 200:
            return
        suffix = f"（供应商：{PROVIDER}，模型：{self._model}）"
        if resp.status_code == 401:
            raise AuthenticationError("Rerank API Key 无效（HTTP 401），请检查 SILICONFLOW_API_KEY" + suffix)
        if resp.status_code == 404:
            raise ModelNotFoundError("Rerank 模型不存在或无权限（HTTP 404），请检查 RERANKER_MODEL" + suffix)
        if resp.status_code == 429:
            raise RateLimitError("Rerank 触发限流或额度不足（HTTP 429）" + suffix)
        if resp.status_code == 400:
            raise BadRequestError(f"Rerank 请求参数非法（HTTP 400）：{resp.text[:300]}" + suffix)
        if resp.status_code >= 500:
            raise UpstreamServerError(f"Rerank 服务异常（HTTP {resp.status_code}）" + suffix)
        raise BadRequestError(f"Rerank 未预期的状态码 HTTP {resp.status_code}: {resp.text[:200]}")

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankItem]:
        """返回按相关性降序的重排结果（index 指向 documents 原始下标）。"""
        if not documents:
            return []
        top_n = top_n or len(documents)

        async def _call() -> list[RerankItem]:
            try:
                resp = await self._client.post(
                    "/rerank",
                    json={
                        "model": self._model,
                        "query": query,
                        "documents": documents,
                        "top_n": top_n,
                        "return_documents": False,
                    },
                )
            except httpx.TimeoutException as exc:
                raise UpstreamTimeoutError("Rerank 请求超时") from exc
            self._raise_by_status(resp)
            data = resp.json()
            return [
                RerankItem(index=int(item["index"]), score=float(item["relevance_score"]))
                for item in data.get("results", [])
            ]

        return await retry_async(  # type: ignore[return-value]
            _call, max_retries=self._max_retries, provider=PROVIDER, model=self._model
        )

    async def close(self) -> None:
        await self._client.aclose()
