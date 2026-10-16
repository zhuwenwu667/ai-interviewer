"""BM25 关键词检索（rank_bm25 + jieba 中文分词）。

方案说明（简单可靠优先）：
- 中文经 jieba.lcut 分词，英文/数字小写化，二者统一作为 term；
- 索引全量内存构建（题库规模 ≤2000 chunk，构建为毫秒级），不持久化；
- heading 与 text 拼接参与打分，提升标题关键词（如题目标题）的权重；
- 知识库变更（摄入/重建）后由调用方触发 ``build`` 重建索引。
"""

from __future__ import annotations

import jieba
from rank_bm25 import BM25Okapi

from app.rag.models import Chunk

# jieba 首次分词时才加载词典，这里提前初始化避免首次检索抖动，并关闭启动日志
jieba.setLogLevel(60)


def tokenize(text: str) -> list[str]:
    """jieba 分词 + 小写化 + 去空白。"""
    return [tok.strip().lower() for tok in jieba.lcut(text) if tok.strip()]


class BM25Index:
    """chunk 级 BM25 内存索引，同时持有 chunk_id -> Chunk 映射供元数据查询。"""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._chunks = {c.chunk_id: c for c in chunks}
        corpus = [tokenize(f"{c.heading}\n{c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    @property
    def size(self) -> int:
        return len(self._chunk_ids)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def search(self, query: str, top_k: int) -> list[tuple[str, int]]:
        """返回 (chunk_id, 名次) 列表，按 BM25 分数降序；名次从 1 开始，剔除 0 分结果。"""
        if self._bm25 is None or not self._chunk_ids or top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
        hits: list[tuple[str, int]] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            if score <= 0:
                break
            hits.append((self._chunk_ids[idx], rank))
            if len(hits) >= top_k:
                break
        return hits
