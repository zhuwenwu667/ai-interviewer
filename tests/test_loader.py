"""文档加载器测试（Markdown / TXT / PDF，TR-2.2）。"""

from __future__ import annotations

import pytest

from app.rag.loader import load_file, load_pdf_content, load_text_content
from app.rag.models import KbType


def test_load_markdown_file(tmp_path):
    md = tmp_path / "resume.md"
    md.write_text("# 标题\n\n正文内容。\n\n## 小节\n\n小节内容。", encoding="utf-8")
    doc = load_file(md, KbType.RESUME)
    assert doc.source == "resume.md"
    assert doc.kb_type == KbType.RESUME
    assert len(doc.sections) == 2
    assert doc.sections[1].heading == "标题 > 小节"


def test_load_txt_file(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("纯文本第一行\n纯文本第二行", encoding="utf-8")
    doc = load_file(txt, KbType.JD)
    assert doc.kb_type == KbType.JD
    assert "纯文本第一行" in doc.sections[0].text


def test_unsupported_suffix_raises(tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的文件类型"):
        load_file(bad, KbType.JD)


def test_empty_markdown_falls_back_to_single_section():
    doc = load_text_content("", source="empty.md", kb_type=KbType.QA)
    assert doc.sections == [] or len(doc.sections) == 1


def _build_minimal_pdf(text: str) -> bytes:
    """动态拼装一个合法的单页 PDF（含 xref 偏移表），避免引入 reportlab 依赖。"""
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    stream = b"BT /F1 12 Tf 72 720 Td (" + text.encode("latin-1") + b") Tj ET"
    objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_pos = len(pdf)
    pdf.extend(b"xref\n0 " + str(len(objects) + 1).encode() + b"\n")
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n" + str(xref_pos).encode() + b"\n%%EOF"
    )
    return bytes(pdf)


def test_load_pdf_extracts_text():
    pdf_bytes = _build_minimal_pdf("Python GIL interview knowledge")
    doc = load_pdf_content(pdf_bytes, source="sample.pdf", kb_type=KbType.QA)
    assert len(doc.sections) == 1
    assert "PDF 第 1 页" == doc.sections[0].heading
    assert "Python GIL interview knowledge" in doc.sections[0].text
