"""RAG 领域数据模型：知识库类型、文档 section、检索块。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum


class KbType(StrEnum):
    """知识库三类分区。"""

    RESUME = "resume"  # 简历
    JD = "jd"  # 岗位描述
    QA = "qa"  # 八股/知识点题库

    @classmethod
    def from_dir(cls, dirname: str) -> KbType:
        mapping = {"resume": cls.RESUME, "jd": cls.JD, "qa": cls.QA}
        if dirname not in mapping:
            raise ValueError(f"无法从目录名 {dirname!r} 推断知识库类型，应为 resume/jd/qa 之一")
        return mapping[dirname]


@dataclass
class Section:
    """文档按标题切出的逻辑段。heading 为标题层级路径，如 "项目经历 > 智能客服"。"""

    heading: str
    text: str


@dataclass
class LoadedDocument:
    """加载器产出：一个来源文件对应一个文档。"""

    source: str  # 唯一来源标识（相对路径或上传文件名）
    kb_type: KbType
    sections: list[Section]


@dataclass
class Chunk:
    """写入向量库的最小检索单元。"""

    chunk_id: str
    source: str
    kb_type: str
    heading: str
    text: str
    chunk_index: int

    def payload(self) -> dict:
        """Qdrant payload（不含向量；chunk_id 同时作为稳定外部 ID）。"""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "kb_type": self.kb_type,
            "heading": self.heading,
            "text": self.text,
            "chunk_index": self.chunk_index,
        }

    @staticmethod
    def make_id(kb_type: str, source: str, index: int, text: str) -> str:
        """内容寻址的确定性 ID：同输入必得同 ID（幂等去重基础）。"""
        raw = f"{kb_type}|{source}|{index}|{text}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]

    @classmethod
    def from_payload(cls, payload: dict) -> Chunk:
        return cls(
            chunk_id=payload["chunk_id"],
            source=payload["source"],
            kb_type=payload["kb_type"],
            heading=payload.get("heading", ""),
            text=payload["text"],
            chunk_index=payload.get("chunk_index", 0),
        )
