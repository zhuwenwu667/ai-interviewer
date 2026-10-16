"""结构化文档切块（自研，便于面试讲清原理）。

两层策略：
1. Markdown 按标题层级切逻辑 section，保留 "一级 > 二级" 标题路径，标题文本计入正文
   （题库场景下问题标题本身就是关键检索信号）；
2. section 内按字符数滑窗二次切分，带 overlap，切点优先对齐段落/句子边界，
   避免把句子拦腰截断。中文按字符数近似 token 数（bge 系模型对字符长度稳健）。
"""

from __future__ import annotations

import re

from app.rag.models import Chunk, LoadedDocument, Section

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# 友好切分边界：中文句读/换行，或英文句号/问号/叹号后接空白
_BOUNDARY_RE = re.compile(r"[。！？；\n]|(?:[.!?])\s")


def parse_markdown(content: str) -> list[Section]:
    """把 Markdown 解析为带标题层级路径的 section 列表。"""
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(Section(heading=heading, text=text))

    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            heading = " > ".join(t for _, t in stack)
            buffer = [title]  # 标题纳入正文，保证题库"问题标题"可被检索
        else:
            buffer.append(line)
    flush()
    return sections


def _find_boundary(window_tail: str) -> int:
    """返回窗口尾部区域内最后一个友好边界的结束位置；无则 -1。"""
    last = -1
    for match in _BOUNDARY_RE.finditer(window_tail):
        last = match.end()
    return last


def window_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """带 overlap 的字符滑窗，切点向最近的句子/段落边界回退。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + chunk_size, total)
        if end < total:
            search_from = start + int(chunk_size * 0.6)
            boundary_rel = _find_boundary(text[search_from:end])
            if boundary_rel != -1:
                candidate = search_from + boundary_rel
                if candidate > start:
                    end = candidate
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= total:
            break
        start = max(end - overlap, start + 1)  # 保证至少前进 1 字符
    return pieces


def chunk_document(
    doc: LoadedDocument,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """把加载后的文档切成带确定性 ID 的 Chunk 列表（index 跨 section 连续）。"""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    chunks: list[Chunk] = []
    index = 0
    for section in doc.sections:
        for piece in window_split(section.text, chunk_size, chunk_overlap):
            chunk_id = Chunk.make_id(doc.kb_type.value, doc.source, index, piece)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=doc.source,
                    kb_type=doc.kb_type.value,
                    heading=section.heading,
                    text=piece,
                    chunk_index=index,
                )
            )
            index += 1
    return chunks
