"""知识摄入服务：加载 → 切块 → 向量化 → 幂等写入。

离线脚本与在线上传（Task 9）共用同一套逻辑，保证行为一致。
"""

from __future__ import annotations

from pathlib import Path

from app.core.embedding import SiliconFlowEmbedder
from app.rag.chunker import chunk_document
from app.rag.loader import load_file
from app.rag.models import KbType, LoadedDocument
from app.storage.qdrant_store import QdrantStore


async def ingest_loaded_document(
    store: QdrantStore,
    embedder: SiliconFlowEmbedder,
    doc: LoadedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """摄入已加载文档：先删同 source 旧点再 upsert，天然幂等。"""
    chunks = chunk_document(
        doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if not chunks:
        return {"source": doc.source, "kb_type": doc.kb_type.value, "chunks": 0, "deleted": 0}
    vectors = await embedder.embed([chunk.text for chunk in chunks])
    deleted = store.delete_source(doc.source)
    store.upsert_chunks(chunks, vectors)
    return {
        "source": doc.source,
        "kb_type": doc.kb_type.value,
        "chunks": len(chunks),
        "deleted": deleted,
    }


async def ingest_path(
    store: QdrantStore,
    embedder: SiliconFlowEmbedder,
    path: Path,
    kb_type: KbType,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    doc = load_file(path, kb_type)
    return await ingest_loaded_document(
        store,
        embedder,
        doc,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
