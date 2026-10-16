"""离线评测集加载与结构校验（TR-11.1）。

评测集为 JSON 文件，字段见 ``EvalQuestion``。四类问题：

- ``resume``：简历题，考察简历信息召回；
- ``jd``：JD 匹配题（中/英），考察岗位信息与多语言检索；
- ``qa``：八股题，考察技术题库召回与答案质量；
- ``oos``：题库外诱导题，期望系统显式拒答而非编造（``expected_sources`` 为空）。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

EvalCategory = Literal["resume", "jd", "qa", "oos"]

CATEGORIES: tuple[str, ...] = ("resume", "jd", "qa", "oos")
VALID_KB_TYPES: frozenset[str] = frozenset({"resume", "jd", "qa"})
# 各类最少条数；总数另需 >= MIN_TOTAL，保证 TR-11.1「四类齐全且不少于 40 条」
MIN_PER_CATEGORY: dict[str, int] = {"resume": 8, "jd": 8, "qa": 16, "oos": 6}
MIN_TOTAL = 40


class EvalQuestion(BaseModel):
    """单条评测样例。"""

    id: str
    category: EvalCategory
    question: str
    # 原子化答案要点，供 RAGAS context_recall / answer_relevancy 评判
    ground_truth: list[str] = Field(min_length=1)
    # 期望命中的来源文件名；oos 题为空列表（不期望命中文档）
    expected_sources: list[str] = Field(default_factory=list)
    # 检索时限定的 kb_type；空列表表示不限分区（全库检索）
    expected_kb_types: list[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("question", "id")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("字段不能为空字符串")
        return v

    @field_validator("ground_truth")
    @classmethod
    def _points_non_empty(cls, v: list[str]) -> list[str]:
        if any(not p.strip() for p in v):
            raise ValueError("ground_truth 要点不能为空字符串")
        return v

    @property
    def ground_truth_text(self) -> str:
        """RAGAS ground_truth 列：要点拼成一段。"""
        return "；".join(self.ground_truth)

    @property
    def kb_type_filter(self) -> list[str] | None:
        """检索分区过滤；空列表 => 全库。"""
        return self.expected_kb_types or None

    @property
    def expects_refusal(self) -> bool:
        """题库外诱导题：期望显式拒答。"""
        return self.category == "oos"


class EvalDataset(BaseModel):
    """整个评测集。"""

    version: str = "1.0"
    description: str = ""
    questions: list[EvalQuestion]

    def by_category(self, category: str) -> list[EvalQuestion]:
        return [q for q in self.questions if q.category == category]

    def filter(
        self, categories: list[str] | None = None, limit: int | None = None
    ) -> list[EvalQuestion]:
        rows = self.questions
        if categories:
            unknown = set(categories) - set(CATEGORIES)
            if unknown:
                raise ValueError(f"未知评测类别：{sorted(unknown)}")
            rows = [q for q in rows if q.category in categories]
        return rows[:limit] if limit else rows


def load_dataset(path: str | Path) -> EvalDataset:
    """从 JSON 文件加载评测集。"""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(data)


class DatasetValidationError(ValueError):
    """评测集结构不满足 TR-11.1 约定。"""


def validate_dataset(
    dataset: EvalDataset, known_sources: set[str] | None = None
) -> None:
    """校验评测集结构；不通过抛 ``DatasetValidationError``（错误信息聚合全部问题）。

    - 总数 >= 40；
    - 四类齐全且各自达到最小条数；
    - id 唯一、字段非空；
    - expected_sources 非空时必须能在知识库文件（``known_sources``）中找到；
    - oos 题 expected_sources 必须为空（诱导题不应期望命中文档）。
    """
    errors: list[str] = []
    questions = dataset.questions

    if len(questions) < MIN_TOTAL:
        errors.append(f"评测集仅 {len(questions)} 条，要求不少于 {MIN_TOTAL} 条")

    counts = Counter(q.category for q in questions)
    for category in CATEGORIES:
        need = MIN_PER_CATEGORY[category]
        actual = counts.get(category, 0)
        if actual < need:
            errors.append(f"类别 {category} 仅 {actual} 条，要求至少 {need} 条")

    ids = [q.id for q in questions]
    dup = [qid for qid, n in Counter(ids).items() if n > 1]
    if dup:
        errors.append(f"id 重复：{dup}")

    for q in questions:
        bad_types = [t for t in q.expected_kb_types if t not in VALID_KB_TYPES]
        if bad_types:
            errors.append(f"{q.id}: expected_kb_types 含非法值 {bad_types}")
        if q.category == "oos" and q.expected_sources:
            errors.append(f"{q.id}: oos 诱导题 expected_sources 必须为空")
        if q.category != "oos" and not q.expected_sources:
            errors.append(f"{q.id}: 非诱导题必须给出 expected_sources")
        if known_sources is not None:
            unknown_sources = [s for s in q.expected_sources if s not in known_sources]
            if unknown_sources:
                errors.append(
                    f"{q.id}: expected_sources 中存在知识库不存在的文件 {unknown_sources}"
                )

    if errors:
        raise DatasetValidationError("评测集校验失败：\n- " + "\n- ".join(errors))
