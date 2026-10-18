"""P2-① 语义缓存单测（真实 Qdrant local 模式 + 假 embedding/检索）。

- 序列化往返：``dataclasses.asdict`` → ``AdaptiveResult.from_dict`` 等价；
- 命中/未命中：同一向量（cos=1.0）命中，正交向量（cos=0）低于阈值不命中；
- 关闭开关：``enabled=False`` 时 get 恒 None、put 空操作；
- 清空：``clear()`` 后同查询不再命中（知识库变更失效路径）；
- 装配：``AdaptiveRetriever`` 二次查询命中缓存，底层 Retriever 只被调用一次。
"""

from __future__ import annotations

import dataclasses
import json

import httpx

from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient
from app.rag.adaptive import AdaptiveResult, AdaptiveRetriever
from app.rag.cache import SemanticCache
from app.rag.models import Chunk
from app.rag.retriever import RetrievalResult, RetrievalTrace, RetrievedChunk
from app.storage.qdrant_store import QdrantStore

_DIM = 4


class FakeEmbedder:
    """按文本返回预设向量；未命中的文本返回 default（用于控制相似度）。"""

    def __init__(self, vectors: dict[str, list[float]], default: list[float]) -> None:
        self._vectors = vectors
        self._default = default
        self.calls: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vectors.get(text, self._default))

    async def close(self) -> None:  # pragma: no cover - 测试替身
        pass


class CountingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(
        self, query: str, kb_types: list[str] | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        self.queries.append(query)
        chunk = Chunk(
            chunk_id=f"c-{len(self.queries)}", source="qa.md", kb_type="qa",
            heading="题库", text=f"召回片段 {len(self.queries)}", chunk_index=0,
        )
        trace = RetrievalTrace(
            original_query=query, vector_queries=[query], hyde_answer=None,
            vector_hit_count=1, bm25_hit_count=0, fused_count=1, rerank_candidate_count=1,
        )
        return RetrievalResult(
            chunks=[RetrievedChunk(chunk=chunk, vector_rank=1, rerank_score=0.8)], trace=trace
        )


def _settings(tmp_path, **overrides) -> Settings:
    base: dict = {
        "_env_file": None,
        "deepseek_api_key": "sk-test-ds",
        "deepseek_base_url": "https://ds.test/v1",
        "siliconflow_api_key": "sk-test-sf",
        "siliconflow_base_url": "https://sf.test/v1",
        "qdrant_mode": "local",
        "qdrant_path": tmp_path / "q",
        "qdrant_collection": "cache_test",
        "embedding_dim": _DIM,
    }
    base.update(overrides)
    return Settings(**base)


def _result(n: int = 1) -> AdaptiveResult:
    chunk = Chunk(
        chunk_id=f"c-{n}", source="qa.md", kb_type="qa",
        heading="题库", text=f"片段{n}", chunk_index=0,
    )
    trace = RetrievalTrace(
        original_query="Q", vector_queries=["Q"], hyde_answer=None,
        vector_hit_count=1, bm25_hit_count=0, fused_count=1, rerank_candidate_count=1,
    )
    return AdaptiveResult(
        retrieval=RetrievalResult(
            chunks=[RetrievedChunk(chunk=chunk, vector_rank=1, rerank_score=0.9)], trace=trace
        ),
        rounds=2,
        stopped_reason="max_rounds_reached",
    )


def test_from_dict_roundtrip_is_lossless():
    original = _result()
    restored = AdaptiveResult.from_dict(json.loads(json.dumps(dataclasses.asdict(original))))
    assert restored == original


async def test_hit_and_miss_by_similarity(tmp_path):
    settings = _settings(tmp_path, semantic_cache_threshold=0.95)
    embedder = FakeEmbedder(
        {"精确查询": [1.0, 0.0, 0.0, 0.0], "另一主题": [0.0, 1.0, 0.0, 0.0]},
        default=[1.0, 0.0, 0.0, 0.0],
    )
    store = QdrantStore(settings)
    try:
        cache = SemanticCache(settings, store, embedder)
        await cache.put("精确查询", "*|6", _result())

        hit = await cache.get("精确查询", "*|6")
        assert hit is not None and hit.rounds == 2
        # 正交向量相似度 0 < 0.95 → 未命中
        assert await cache.get("另一主题", "*|6") is None
        # 不同 scope 不互相污染
        assert await cache.get("精确查询", "qa|10") is None
        assert cache.stats()["hit_rate"] == 0.3333  # 3 次查询中 1 次命中
    finally:
        store.close()


async def test_disabled_cache_is_noop(tmp_path):
    settings = _settings(tmp_path, semantic_cache_enabled=False)
    embedder = FakeEmbedder({}, default=[1.0, 0.0, 0.0, 0.0])
    store = QdrantStore(settings)
    try:
        cache = SemanticCache(settings, store, embedder)
        assert cache.enabled is False
        await cache.put("Q", "*|6", _result())
        assert await cache.get("Q", "*|6") is None
        assert cache.stats()["stores"] == 0 and cache.stats()["lookups"] == 0
        assert embedder.calls == []  # 未启用时不产生 embedding 往返
    finally:
        store.close()


async def test_clear_invalidates_entries(tmp_path):
    settings = _settings(tmp_path)
    embedder = FakeEmbedder({}, default=[1.0, 0.0, 0.0, 0.0])
    store = QdrantStore(settings)
    try:
        cache = SemanticCache(settings, store, embedder)
        await cache.put("Q", "*|6", _result())
        assert await cache.get("Q", "*|6") is not None
        assert cache.clear() == 1
        assert await cache.get("Q", "*|6") is None
    finally:
        store.close()


async def test_adaptive_retriever_serves_second_call_from_cache(tmp_path):
    settings = _settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"sufficient": True, "reason": "充足"}),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    llm = DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))
    retriever = CountingRetriever()
    embedder = FakeEmbedder({}, default=[1.0, 0.0, 0.0, 0.0])
    store = QdrantStore(settings)
    try:
        cache = SemanticCache(settings, store, embedder)
        adaptive = AdaptiveRetriever(settings, retriever, llm, cache=cache)

        first = await adaptive.retrieve("AI 应用开发 方向面试题", top_k=6)
        second = await adaptive.retrieve("AI 应用开发 方向面试题", top_k=6)

        assert len(retriever.queries) == 1  # 第二次命中缓存，未再触发底层检索
        assert first.rounds == second.rounds == 1
        assert cache.stats()["hits"] == 1
    finally:
        store.close()
        await llm.close()
