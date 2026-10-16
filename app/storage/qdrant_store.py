"""Qdrant 向量存储访问层。

双模式（由 settings.qdrant_mode 决定）：
- local：嵌入式文件存储（QdrantClient(path=...)），Windows 开发零依赖；
- server：连接独立 Qdrant 服务（Docker/云端）。

幂等摄入保证：point id 由 chunk_id 稳定映射（uuid5），重建索引前先按 source
删除旧点，因此同一文件重复摄入不会产生重复数据。
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config.settings import Settings
from app.rag.models import Chunk

_POINT_NAMESPACE = uuid.UUID("6f3a1c2e-9b4d-4e7f-8a2c-1d5e6f708190")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"ai-interviewer:{chunk_id}"))


def _kb_filter(kb_types: Sequence[str] | None) -> qm.Filter | None:
    if not kb_types:
        return None
    return qm.Filter(
        must=[qm.FieldCondition(key="kb_type", match=qm.MatchAny(any=list(kb_types)))]
    )


def _source_filter(source: str) -> qm.Filter:
    return qm.Filter(
        must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=source))]
    )


class QdrantStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.collection = settings.qdrant_collection
        if settings.qdrant_mode == "local":
            settings.qdrant_path_abs.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(settings.qdrant_path_abs))
        else:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=self._settings.embedding_dim,
                    distance=qm.Distance.COSINE,
                ),
            )
        # payload 索引：local 模式不支持（会告警且无效），仅 server 模式创建
        if self._settings.qdrant_mode == "server":
            for field in ("kb_type", "source"):
                try:
                    self._client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field,
                        field_schema="keyword",
                    )
                except Exception:  # noqa: BLE001 - 索引已存在属正常
                    pass

    # ---------- 写入侧 ----------
    def reset_collection(self) -> None:
        """清空集合内全部点（--rebuild 使用）。

        显式按 point id 删除而非 delete_collection 后同名重建：
        Qdrant local 模式下后者存在同进程重建后旧数据复活的问题。
        """
        point_ids: list[str | int] = []
        offset: int | None = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.collection,
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(record.id for record in records)
            offset = next_offset
            if not offset:
                break
        if point_ids:
            self._client.delete(
                collection_name=self.collection,
                points_selector=point_ids,
                wait=True,
            )

    def delete_source(self, source: str) -> int:
        """删除某来源的全部点，返回删除前数量（重建索引前置步骤）。"""
        deleted = self._client.count(
            collection_name=self.collection,
            count_filter=_source_filter(source),
            exact=True,
        ).count
        if deleted:
            self._client.delete(
                collection_name=self.collection,
                points_selector=_source_filter(source),
                wait=True,
            )
        return deleted

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 与 vectors 数量必须一致")
        if not chunks:
            return
        points = [
            qm.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self.collection, points=points, wait=True)

    # ---------- 读取侧 ----------
    def count(self, kb_types: Sequence[str] | None = None) -> int:
        return self._client.count(
            collection_name=self.collection,
            count_filter=_kb_filter(kb_types),
            exact=True,
        ).count

    def scroll_all(
        self, kb_types: Sequence[str] | None = None, page_size: int = 256
    ) -> list[Chunk]:
        """取出（某类型下的）全部 chunk，供 BM25 索引构建与数据校验使用。"""
        chunks: list[Chunk] = []
        offset: int | None = None
        while True:
            records, next_offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=_kb_filter(kb_types),
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(Chunk.from_payload(dict(record.payload)) for record in records)
            offset = next_offset
            if not offset:
                break
        return chunks

    def search(
        self,
        query_vector: Sequence[float],
        kb_types: Sequence[str] | None,
        top_k: int,
    ) -> list[tuple[Chunk, float]]:
        """向量检索，返回 (Chunk, cosine 相似度) 按分数降序。"""
        response = self._client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            query_filter=_kb_filter(kb_types),
            limit=top_k,
        )
        return [
            (Chunk.from_payload(dict(point.payload)), float(point.score))
            for point in response.points
        ]

    def close(self) -> None:
        self._client.close()
