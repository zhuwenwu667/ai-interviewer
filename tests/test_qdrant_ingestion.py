"""Qdrant 本地存储 + 摄入幂等性测试。

使用 Qdrant local 嵌入式模式（临时目录），不依赖外部 Qdrant 服务；
Embedding 用确定性 FakeEmbedder，不发起网络请求。
"""

from __future__ import annotations

import pytest

from app.config.settings import Settings
from app.rag.ingestion import ingest_loaded_document
from app.rag.models import KbType, LoadedDocument, Section
from app.storage.qdrant_store import QdrantStore

DIM = 8


class FakeEmbedder:
    """按关键词把文本映射到不同坐标轴，确定性向量，便于检索断言。"""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * DIM
            if "GIL" in text:
                vec[0] = 1.0
            elif "Redis" in text:
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            vectors.append(vec)
        return vectors

    async def close(self) -> None:
        pass


@pytest.fixture
def store(tmp_path):
    settings = Settings(
        _env_file=None,
        qdrant_mode="local",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="test_kb",
        embedding_dim=DIM,
    )
    s = QdrantStore(settings)
    yield s
    s.close()


def _doc(source: str, kb_type: KbType, sections: list[tuple[str, str]]) -> LoadedDocument:
    return LoadedDocument(
        source=source,
        kb_type=kb_type,
        sections=[Section(heading=h, text=t) for h, t in sections],
    )


async def _ingest(store, doc):
    return await ingest_loaded_document(
        store, FakeEmbedder(), doc, chunk_size=500, chunk_overlap=80
    )


async def test_ingest_and_count_by_kb_type(store):
    await _ingest(store, _doc("resume.md", KbType.RESUME, [("技能", "熟悉 Python 与 FastAPI")]))
    await _ingest(store, _doc("qa.md", KbType.QA, [
        ("GIL", "讲讲 Python 的 GIL 与多线程"),
        ("Redis", "Redis 缓存穿透如何解决"),
    ]))
    assert store.count() == 3
    assert store.count(kb_types=["qa"]) == 2
    assert store.count(kb_types=["resume"]) == 1


async def test_reingest_same_document_is_idempotent(store):
    doc = _doc("qa.md", KbType.QA, [
        ("GIL", "讲讲 Python 的 GIL 与多线程"),
        ("Redis", "Redis 缓存穿透如何解决"),
    ])
    first = await _ingest(store, doc)
    assert first["chunks"] == 2
    second = await _ingest(store, doc)
    # 第二次：先删同 source 的 2 个旧点，再写入 2 个，总数不变
    assert second["deleted"] == 2
    assert second["chunks"] == 2
    assert store.count() == 2


async def test_scroll_payload_has_all_required_fields(store):
    await _ingest(store, _doc("resume.md", KbType.RESUME, [("技能", "Python 与 Redis 技能")]))
    chunks = store.scroll_all()
    assert len(chunks) == 1
    payload = chunks[0].payload()
    for field in ("chunk_id", "source", "kb_type", "heading", "text"):
        assert field in payload and payload[field] != ""


async def test_vector_search_returns_most_similar_first(store):
    await _ingest(store, _doc("qa.md", KbType.QA, [
        ("GIL", "讲讲 Python 的 GIL 与多线程"),
        ("Redis", "Redis 缓存穿透如何解决"),
        ("其他", "谈谈职业规划"),
    ]))
    query = [1.0] + [0.0] * (DIM - 1)  # 指向 GIL 坐标轴
    results = store.search(query, kb_types=["qa"], top_k=3)
    assert len(results) == 3
    top_chunk, top_score = results[0]
    assert "GIL" in top_chunk.text
    assert all(score <= top_score for _, score in results)


async def test_kb_type_filter_excludes_other_types(store):
    await _ingest(store, _doc("resume.md", KbType.RESUME, [("技能", "GIL 相关经验")]))
    await _ingest(store, _doc("qa.md", KbType.QA, [("GIL", "GIL 题目")]))
    qa_results = store.search([1.0] + [0.0] * (DIM - 1), kb_types=["qa"], top_k=5)
    assert all(chunk.kb_type == "qa" for chunk, _ in qa_results)
    assert len(qa_results) == 1


async def test_reset_collection_wipes_everything(store):
    await _ingest(store, _doc("qa.md", KbType.QA, [("GIL", "GIL 题目")]))
    assert store.count() == 1
    store.reset_collection()
    assert store.count() == 0
