"""训练题库：直接解析 ``data/knowledge/qa/*.md`` 的八股题（不依赖向量库）。

题库 Markdown 约定（与切块器一致：每题一个二级标题）::

    ## Q01: 题干……

    - tags: python, 并发
    - 答案要点：
      - 要点一
      - 要点二

- ``question_id`` 为 ``<文件名>#<Q 编号>``（内容寻址之外的稳定外部 ID，落库可溯源）；
- 抽题优先排除该用户已作答过的题（``train_attempts``），全部答完后回到全量随机；
- 参考答案要点只在「提交作答后」由 answer 接口返回，next 响应绝不下发（防提前泄题）。
"""

from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^##\s+(Q\d+)\s*[:：]\s*(?P<question>.+?)\s*$", re.MULTILINE)
_TAGS_RE = re.compile(r"^-\s*tags\s*[:：]\s*(?P<tags>.+?)\s*$", re.MULTILINE)
_POINT_RE = re.compile(r"^\s{2,}-\s+(?P<point>.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class TrainQuestion:
    """一道训练题（含仅供服务端使用的参考答案要点）。"""

    question_id: str  # "<source>#Q01"
    source: str  # 题库文件名
    ref: str  # 题号（如 Q01）
    question: str
    tags: tuple[str, ...]
    points: tuple[str, ...]  # 答案要点（不下发到 next 响应）


def _parse_file(path: Path) -> list[TrainQuestion]:
    text = path.read_text(encoding="utf-8")
    matches = list(_HEADING_RE.finditer(text))
    questions: list[TrainQuestion] = []
    for i, match in enumerate(matches):
        ref = match.group(1)
        body = text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        tags_match = _TAGS_RE.search(body)
        tags = tuple(
            t.strip() for t in (tags_match.group("tags") if tags_match else "").split(",")
            if t.strip()
        )
        # 答案要点：取「- 答案要点：」行之后的缩进列表项
        points: tuple[str, ...] = ()
        points_marker = re.search(r"^-\s*答案要点\s*[:：]\s*$", body, re.MULTILINE)
        if points_marker:
            points_block = body[points_marker.end():]
            points = tuple(m.group("point") for m in _POINT_RE.finditer(points_block))
        questions.append(
            TrainQuestion(
                question_id=f"{path.name}#{ref}",
                source=path.name,
                ref=ref,
                question=match.group("question"),
                tags=tags,
                points=points,
            )
        )
    return questions


class QuestionBank:
    """内存题库：加载 qa 目录全部题目，支持按排除集随机抽题。"""

    def __init__(self, questions: list[TrainQuestion]) -> None:
        self._questions = tuple(questions)
        self._by_id = {q.question_id: q for q in questions}
        self._rng = random.SystemRandom()

    @classmethod
    def load(cls, qa_dir: Path) -> QuestionBank:
        questions: list[TrainQuestion] = []
        for path in sorted(qa_dir.glob("*.md")):
            questions.extend(_parse_file(path))
        return cls(questions)

    @property
    def total(self) -> int:
        return len(self._questions)

    def get(self, question_id: str) -> TrainQuestion | None:
        return self._by_id.get(question_id)

    def draw(self, exclude_ids: set[str] | None = None) -> TrainQuestion | None:
        """随机抽一题；优先从未答过的题中抽，全部答完则回退到全量随机。"""
        if not self._questions:
            return None
        excluded = exclude_ids or set()
        fresh = [q for q in self._questions if q.question_id not in excluded]
        pool = fresh or list(self._questions)
        return self._rng.choice(pool)


_BANKS: dict[Path, QuestionBank] = {}
_BANKS_LOCK = threading.Lock()


def get_question_bank(qa_dir: Path) -> QuestionBank:
    """按目录缓存的题库单例（题库文件变更需重启进程生效）；测试用 reset 清缓存。"""
    key = qa_dir.resolve()
    with _BANKS_LOCK:
        bank = _BANKS.get(key)
        if bank is None:
            bank = QuestionBank.load(key)
            _BANKS[key] = bank
        return bank


def reset_question_bank() -> None:
    with _BANKS_LOCK:
        _BANKS.clear()
