"""硅基流动 Embedding 客户端（bge-m3，OpenAI 兼容协议）。"""

from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from app.config.settings import Settings
from app.core.http_utils import retry_async


class SiliconFlowEmbedder:
    """文本向量化。文档侧按 batch 切分调用，维度由 bge-m3 固定为 1024。"""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        batch_size: int = 32,
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        http_client = httpx.AsyncClient(transport=transport) if transport else None
        self._client = AsyncOpenAI(
            api_key=settings.siliconflow_api_key or "missing-key",
            base_url=settings.siliconflow_base_url,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
        self._model = settings.embedding_model
        self._dim = settings.embedding_dim
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._provider = "SiliconFlow"

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化，保证返回顺序与输入一致（SDK 已按 index 排序，这里再防御一次）。"""
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]

            async def _call(st: int = start, b: list[str] = batch) -> list[list[float]]:
                resp = await self._client.embeddings.create(model=self._model, input=b)
                ordered = sorted(resp.data, key=lambda item: item.index)
                for offset, item in enumerate(ordered):
                    results[st + offset] = item.embedding
                return [item.embedding for item in ordered]

            await retry_async(
                _call, max_retries=self._max_retries, provider=self._provider, model=self._model
            )
        return [v for v in results if v is not None]  # type: ignore[list-item]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]

    async def close(self) -> None:
        await self._client.close()
