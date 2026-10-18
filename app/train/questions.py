"""训练题库：直接解析 ``data/knowledge/qa/*.md`` 的八股题（不依赖向量库）。

题库 Markdown 约定（与切块器一致：每题一个二级标题）::

    ## Q01: 题干……

    - tags: python, 并发
    - 答案要点：
      - 要点一
      - 要点二

- ``question_id`` 为 ``<文件名>#<Q 编号>``（内容寻址之外的稳定外部 ID，落库可溯源）；
- 抽题按「错题优先」定序：poor > partial > 未作答 > good，同级取最久未练的题；
  未作答的题（含只 reveal 看过答案的）优先于已掌握（good）的题；``avoid_id`` 用于
  避免「下一题」连出同一道题；
- 参考答案要点只在「提交作答后」由 answer 接口返回，next 响应绝不下发（防提前泄题）。
"""

from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_HEADING_RE = re.compile(r"^##\s+(Q\d+)\s*[:：]\s*(?P<question>.+?)\s*$", re.MULTILINE)
_TAGS_RE = re.compile(r"^-\s*tags\s*[:：]\s*(?P<tags>.+?)\s*$", re.MULTILINE)
_POINT_RE = re.compile(r"^\s{2,}-\s+(?P<point>.+?)\s*$", re.MULTILINE)

# 抽题优先级：数值越小越先被抽到。未作答的题（states 中无记录）单列一档，
# 排在 partial 之后、good 之前——已掌握（good）的题最后复练。
_QUALITY_PRIORITY = {"poor": 0, "partial": 1, "good": 3}
_UNSEEN_PRIORITY = 2

# 每题的「最近一次作答」状态：quality 与作答时间（用于同级定序）
TrainState = tuple[str, datetime]


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

    def draw(
        self,
        states: dict[str, TrainState] | None = None,
        avoid_id: str | None = None,
    ) -> TrainQuestion | None:
        """错题优先抽一题；``avoid_id`` 用于避免连出同一题（如「下一题」跳过）。

        ``states`` 为每题最近一次作答的 ``{question_id: (quality, 作答时间)}``：
        - 优先级 poor > partial > 未作答 > good（未作答含只 reveal 看过答案的题）；
        - 同级取最久未练（作答时间最早）的题，未作答者之间随机；
        - 若最高优先的题恰为 ``avoid_id`` 则顺延到下一题；若只有这一道题仍返回它。
        """
        if not self._questions:
            return None
        state_map = states or {}
        ordered = list(self._questions)
        # 先打乱：同级且「未练时长」相同的题在稳定排序下保持随机顺序
        self._rng.shuffle(ordered)

        def _key(q: TrainQuestion) -> tuple[int, datetime]:
            state = state_map.get(q.question_id)
            priority = _QUALITY_PRIORITY.get(state[0]) if state else None
            if state is None or priority is None:  # 未作答 / 未知 quality → 未作答档
                return (_UNSEEN_PRIORITY, datetime.min)
            return (priority, state[1])

        ordered.sort(key=_key)
        for question in ordered:
            if question.question_id != avoid_id:
                return question
        return ordered[0]


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
