"""混合检索器：查询改写 → 向量/BM25 双路召回 → RRF 融合 → Rerank 精排。

流程：
1. Multi-Query / HyDE 改写（开关可控，失败自动降级）；
2. 向量路：每个（改写）查询各做一次 Qdrant 检索，各算一路排名；
   BM25 路：用原查询检索内存索引，按 kb_type 过滤；
3. RRF 融合全部排名（k 取 settings.retrieval_rrf_k）；
4. 融合前列送 SiliconFlow reranker 精排，取 Top-K。

返回结果携带完整溯源：各阶段分数、命中查询、改写策略与命中统计。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config.settings import Settings
from app.core.embedding import SiliconFlowEmbedder
from app.core.llm import DeepSeekChatClient
from app.core.reranker import SiliconFlowReranker
from app.observability.tracer import get_tracer
from app.rag.bm25 import BM25Index
from app.rag.fusion import rrf_fuse
from app.rag.models import Chunk
from app.rag.rewrite import QueryRewriter
from app.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """一条召回结果：chunk 元数据 + 各阶段分数与命中溯源。"""

    chunk: Chunk
    vector_score: float | None = None  # 向量路最佳 cosine 相似度（多查询取最大）
    vector_rank: int | None = None  # 向量路最佳名次（1-based）
    bm25_rank: int | None = None  # BM25 路名次（1-based），None 表示未被该路命中
    rrf_score: float = 0.0
    rerank_score: float | None = None
    matched_queries: list[str] = field(default_factory=list)  # 向量路命中本 chunk 的查询（含改写）

    @property
    def hit_by_vector(self) -> bool:
        return self.vector_rank is not None

    @property
    def hit_by_bm25(self) -> bool:
        return self.bm25_rank is not None


@dataclass
class RetrievalTrace:
    """一次检索的完整溯源，供日志 / trace / 前端展示。"""

    original_query: str
    vector_queries: list[str]  # 实际参与向量检索的查询（原查询 + 改写）
    hyde_answer: str | None  # HyDE 假设答案（未启用为 None）
    vector_hit_count: int
    bm25_hit_count: int
    fused_count: int
    rerank_candidate_count: int

    @property
    def hyde_used(self) -> bool:
        return self.hyde_answer is not None

    @property
    def multi_query_used(self) -> bool:
        return len(self.vector_queries) > 1


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]  # 按 rerank_score 降序
    trace: RetrievalTrace


class Retriever:
    """统一检索入口。``llm`` 不传时跳过全部改写策略（纯双路 + 融合 + 精排）。"""

    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        embedder: SiliconFlowEmbedder,
        reranker: SiliconFlowReranker,
        llm: DeepSeekChatClient | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._rewriter = QueryRewriter(llm, settings) if llm is not None else None
        self._bm25 = bm25_index or BM25Index()

    def refresh_bm25(self) -> int:
        """从向量库全量重建 BM25 索引（摄入 / 重建后调用），返回索引 chunk 数。"""
        chunks = self._store.scroll_all()
        self._bm25.build(chunks)
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        kb_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        settings = self._settings
        final_k = top_k or settings.retrieval_final_top_k
        if self._bm25.size == 0:
            refreshed = self.refresh_bm25()
            logger.info("BM25 索引为空，已自动重建（%d chunks）", refreshed)

        # 1) 查询改写
        vector_queries = [query]
        hyde_answer: str | None = None
        if self._rewriter is not None:
            with get_tracer().span(
                "rewrite.multi_query", n=settings.retrieval_multi_query_n
            ) as rw_span:
                vector_queries = await self._rewriter.multi_query(query)
                rw_span.set(queries=list(vector_queries), count=len(vector_queries))
            with get_tracer().span("rewrite.hyde") as hyde_span:
                hyde_answer = await self._rewriter.hyde(query)
                hyde_span.set(used=hyde_answer is not None, chars=len(hyde_answer or ""))
        embed_texts = list(vector_queries) + ([hyde_answer] if hyde_answer else [])

        # 2) 向量路：每个改写查询一路召回；HyDE 以假设答案向量追加一路
        embeddings = await self._embedder.embed(embed_texts)
        rankings: list[list[str]] = []
        entries: dict[str, RetrievedChunk] = {}
        for text, vector in zip(embed_texts, embeddings, strict=True):
            hits = self._store.search(vector, kb_types, settings.retrieval_vector_top_k)
            rankings.append([chunk.chunk_id for chunk, _ in hits])
            # HyDE 路的命中溯源记到原查询上（假设答案不是"查询"）
            tag = text if text in vector_queries else query
            for rank, (chunk, score) in enumerate(hits, start=1):
                entry = entries.get(chunk.chunk_id)
                if entry is None:
                    entry = entries[chunk.chunk_id] = RetrievedChunk(chunk=chunk)
                if entry.vector_score is None or score > entry.vector_score:
                    entry.vector_score = score
                    entry.vector_rank = rank
                if tag not in entry.matched_queries:
                    entry.matched_queries.append(tag)
        vector_hit_count = sum(1 for e in entries.values() if e.hit_by_vector)

        # 3) BM25 路：原查询精确匹配（改写查询对关键词检索收益低且引入噪声）
        kb_filter = set(kb_types) if kb_types else None
        bm25_ranking: list[str] = []
        for chunk_id, rank in self._bm25.search(query, settings.retrieval_bm25_top_k):
            chunk = self._bm25.get(chunk_id)
            if chunk is None or (kb_filter is not None and chunk.kb_type not in kb_filter):
                continue
            bm25_ranking.append(chunk_id)
            entry = entries.get(chunk_id)
            if entry is None:
                entry = entries[chunk_id] = RetrievedChunk(chunk=chunk)
            entry.bm25_rank = rank
        rankings.append(bm25_ranking)
        bm25_hit_count = len(bm25_ranking)

        # 4) RRF 融合
        fused = rrf_fuse(rankings, k=settings.retrieval_rrf_k)
        for chunk_id, score in fused:
            entries[chunk_id].rrf_score = score
        fused_ids = [chunk_id for chunk_id, _ in fused]
        candidate_n = min(len(fused_ids), max(final_k * 4, 20))
        candidates = fused_ids[:candidate_n]

        # 5) Rerank 精排取 Top-K
        ordered: list[RetrievedChunk] = []
        if candidates:
            with get_tracer().span("retrieval.rerank", candidates=len(candidates)) as rr_span:
                documents = [entries[cid].chunk.text for cid in candidates]
                reranked = await self._reranker.rerank(query, documents, top_n=len(candidates))
                for item in reranked:
                    entry = entries[candidates[item.index]]
                    entry.rerank_score = item.score
                    ordered.append(entry)
                ordered.sort(key=lambda e: e.rerank_score or 0.0, reverse=True)
                rr_span.set(
                    returned=len(ordered),
                    top_score=ordered[0].rerank_score if ordered else None,
                )

        trace = RetrievalTrace(
            original_query=query,
            vector_queries=vector_queries,
            hyde_answer=hyde_answer,
            vector_hit_count=vector_hit_count,
            bm25_hit_count=bm25_hit_count,
            fused_count=len(fused_ids),
            rerank_candidate_count=len(candidates),
        )
        get_tracer().event(
            "retrieval.summary",
            query=query,
            kb_types=list(kb_types) if kb_types else None,
            vector_hit_count=vector_hit_count,
            bm25_hit_count=bm25_hit_count,
            fused_count=len(fused_ids),
            rerank_candidate_count=len(candidates),
            returned=len(ordered[:final_k]),
            vector_queries=len(vector_queries),
            hyde_used=hyde_answer is not None,
            top_score=ordered[0].rerank_score if ordered else None,
        )
        logger.info(
            "检索完成 query=%r vector_hits=%d bm25_hits=%d fused=%d rerank_candidates=%d",
            query, vector_hit_count, bm25_hit_count, len(fused_ids), len(candidates),
        )
        return RetrievalResult(chunks=ordered[:final_k], trace=trace)
