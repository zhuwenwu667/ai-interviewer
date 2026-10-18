"""P2-① 语义缓存：复用现有 Qdrant，缓存「查询 → 自适应检索结果」。

背景（P0 实测）：面试出题每轮都用同一句 ``{track} 方向面试题`` 触发
``AdaptiveRetriever``，该查询在同一会话内被反复执行，且几乎每次都跑到
``max_rewrite_rounds`` 上限（实测 54 次 rewrite / 18 次出题 = 3.0 次/题），
导致每道题都要重复付出「查询改写 LLM + 向量化 + 双路召回 + 精排 + 充足性自评」
共约 6 次 LLM 往返。语义缓存把这一整套结果按查询缓存，命中即整体跳过。

复用已有 Qdrant 而非引入 Redis：新增独立集合 ``{collection}__semantic_cache``，
与知识库集合隔离但**共用同一 Qdrant 客户端**（local 模式下同路径多客户端会争抢
文件锁）。命中判定为查询向量的余弦相似度 ≥ ``semantic_cache_threshold``；
``scope``（kb_types + top_k）作为 payload 过滤条件，避免不同检索范围互相污染；
结果带 TTL，超期视为未命中并删除该点；知识库变更时由调用方 ``clear()`` 失效。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import uuid

from app.config.settings import Settings
from app.core.embedding import SiliconFlowEmbedder
from app.rag.adaptive import AdaptiveResult
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

_CACHE_NAMESPACE = uuid.UUID("b1d2c3e4-5f60-4a7b-8c9d-0e1f2a3b4c5d")


def _point_id(scope: str, query: str) -> str:
    """稳定点位 id：同 (scope, query) 必得同 id，天然去重。"""
    return str(uuid.uuid5(_CACHE_NAMESPACE, f"semantic-cache:{scope}:{query}"))


class SemanticCache:
    """查询级语义缓存；``enabled=False`` 时 get/put 均空操作（零行为变化）。"""

    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        embedder: SiliconFlowEmbedder,
        *,
        enabled: bool | None = None,
        threshold: float | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._enabled = settings.semantic_cache_enabled if enabled is None else enabled
        self._threshold = (
            settings.semantic_cache_threshold if threshold is None else threshold
        )
        self._ttl = settings.semantic_cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        self._lock = asyncio.Lock()
        self._stats = {"lookups": 0, "hits": 0, "misses": 0, "stores": 0, "expired": 0}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """运行时开关（供 A/B 对照实测：同一实例可即时关闭，模拟无缓存基线）。"""
        self._enabled = enabled

    async def get(self, query: str, scope: str) -> AdaptiveResult | None:
        """查缓存；未命中（含低于阈值 / 已过期 / 未启用）返回 None。"""
        if not self._enabled:
            return None
        self._stats["lookups"] += 1
        vector = await self._embedder.embed_query(query)
        hits = self._store.search_semantic_cache(
            vector, scope=scope, score_threshold=self._threshold, limit=1
        )
        if not hits:
            self._stats["misses"] += 1
            return None
        payload, score = hits[0]
        created = float(payload.get("created_at") or 0.0)
        if self._ttl > 0 and (time.time() - created) > self._ttl:
            self._stats["expired"] += 1
            self._stats["misses"] += 1
            self._store.delete_semantic_cache(str(payload.get("point_id", "")))
            return None
        result = AdaptiveResult.from_dict(json.loads(payload["result_json"]))
        self._stats["hits"] += 1
        logger.info(
            "语义缓存命中 query=%r scope=%s similarity=%.4f rounds=%d",
            query, scope, score, result.rounds,
        )
        return result

    async def put(self, query: str, scope: str, result: AdaptiveResult) -> None:
        """写缓存（未启用时空操作）。"""
        if not self._enabled:
            return
        vector = await self._embedder.embed_query(query)
        point_id = _point_id(scope, query)
        payload = {
            "point_id": point_id,
            "query": query,
            "scope": scope,
            "created_at": time.time(),
            "result_json": json.dumps(dataclasses.asdict(result), ensure_ascii=False),
        }
        async with self._lock:  # Qdrant local 客户端非并发安全，写入串行化
            self._store.upsert_semantic_cache(point_id, vector, payload)
        self._stats["stores"] += 1

    def clear(self) -> int:
        """清空缓存（知识库摄入 / 重建后调用），返回删除点数。"""
        if not self._enabled:
            return 0
        removed = self._store.clear_semantic_cache()
        logger.info("语义缓存已清空，删除 %d 条", removed)
        return removed

    def stats(self) -> dict:
        lookups = self._stats["lookups"]
        hits = self._stats["hits"]
        return {
            **self._stats,
            "enabled": self._enabled,
            "threshold": self._threshold,
            "ttl_seconds": self._ttl,
            "hit_rate": round(hits / lookups, 4) if lookups else 0.0,
        }
