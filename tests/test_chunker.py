"""结构化切块器测试（TR-2.2）。"""

from __future__ import annotations

import pytest

from app.rag.chunker import chunk_document, parse_markdown, window_split
from app.rag.models import KbType, LoadedDocument, Section


def test_parse_markdown_preserves_heading_hierarchy():
    content = (
        "# 一级标题\n\n序言段落。\n\n## 二级标题 A\n\nA 的内容。\n\n"
        "### 三级标题\n\n深层内容。\n\n## 二级标题 B\n\nB 的内容。\n"
    )
    sections = parse_markdown(content)
    assert [s.heading for s in sections] == [
        "一级标题",
        "一级标题 > 二级标题 A",
        "一级标题 > 二级标题 A > 三级标题",
        "一级标题 > 二级标题 B",
    ]
    # 标题文本计入正文，序言归入第一个标题
    assert "序言段落" in sections[0].text
    assert "二级标题 A" in sections[1].text


def test_window_split_no_boundary_has_exact_overlap():
    text = "啊" * 1000
    pieces = window_split(text, chunk_size=300, overlap=100)
    assert len(pieces) > 1
    # 无标点边界时不回退，相邻块严格重叠 overlap 个字符
    assert pieces[1][:100] == pieces[0][-100:]
    assert all(len(p) <= 300 for p in pieces)


def test_window_split_prefers_sentence_boundary():
    text = "这是第一句话。" * 40 + "这是第二句话。" * 40
    pieces = window_split(text, chunk_size=120, overlap=20)
    assert len(pieces) >= 2
    # 第一块的切点应回退到搜索窗口内最后一个句号边界，而非硬切在第 120 字符
    assert pieces[0].endswith("。")
    assert len(pieces[0]) <= 120
    # overlap 让下一块前缀与上一块尾部衔接
    assert pieces[1][:10] in pieces[0][-30:]


def test_window_split_rejects_overlap_ge_size_indirectly():
    with pytest.raises(ValueError):
        chunk_document(
            LoadedDocument(source="x", kb_type=KbType.QA, sections=[Section("h", "正文")]),
            chunk_size=100,
            chunk_overlap=100,
        )


def test_chunk_ids_are_deterministic():
    doc = LoadedDocument(
        source="resume.md",
        kb_type=KbType.RESUME,
        sections=[Section("技能", "Python、FastAPI、RAG。" * 50)],
    )
    first = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    second = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    assert len(first) > 1
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_document_carries_full_metadata():
    doc = LoadedDocument(
        source="qa_bank.md",
        kb_type=KbType.QA,
        sections=[Section("Python > GIL", "GIL 是全局解释器锁。")],
    )
    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=80)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.kb_type == "qa"
    assert chunk.source == "qa_bank.md"
    assert chunk.heading == "Python > GIL"
    assert chunk.chunk_index == 0
    # 直接构造的 Section 正文即最终块文本（标题拼接发生在 parse_markdown 阶段）
    assert chunk.text == "GIL 是全局解释器锁。"
    assert chunk.chunk_id == chunk.payload()["chunk_id"]
