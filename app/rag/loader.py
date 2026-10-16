"""文档加载器：PDF / Markdown / TXT -> LoadedDocument。"""

from __future__ import annotations

import io
from pathlib import Path

from app.rag.chunker import parse_markdown
from app.rag.models import KbType, LoadedDocument, Section

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt"}


def load_text_content(content: str, *, source: str, kb_type: KbType) -> LoadedDocument:
    """从 Markdown/TXT 文本内容直接构建文档（便于上传场景与测试）。"""
    sections = parse_markdown(content)
    if not sections:
        sections = [Section(heading=Path(source).stem, text=content.strip())]
    return LoadedDocument(source=source, kb_type=kb_type, sections=sections)


def load_pdf_content(content: bytes, *, source: str, kb_type: KbType) -> LoadedDocument:
    """从 PDF 字节构建文档：按页成 section，再交给 chunker 窗口切分。"""
    from pypdf import PdfReader  # 延迟导入：仅 PDF 场景需要

    reader = PdfReader(io.BytesIO(content))
    sections: list[Section] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(heading=f"PDF 第 {page_num} 页", text=text))
    if not sections:
        raise ValueError(f"PDF {source} 未能提取到任何文本（可能是扫描件，需 OCR）")
    return LoadedDocument(source=source, kb_type=kb_type, sections=sections)


def load_file(path: Path, kb_type: KbType) -> LoadedDocument:
    """从磁盘加载文件，按扩展名分发。source 取文件名（同类型内应唯一）。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"不支持的文件类型 {suffix}，仅支持 {sorted(SUPPORTED_SUFFIXES)}")
    source = path.name
    if suffix == ".pdf":
        return load_pdf_content(path.read_bytes(), source=source, kb_type=kb_type)
    return load_text_content(
        path.read_text(encoding="utf-8"), source=source, kb_type=kb_type
    )
