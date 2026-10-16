"""评测批跑管线：逐条 检索 → 组装上下文 → 受控生成。

检索默认走生产路径 ``AdaptiveRetriever``（自评 + 改写重检索，Task 5），
可用 ``use_adaptive=False`` 退回单轮 ``Retriever`` 做消融对比。
组件以鸭子类型注入，离线测试可传 Fake 检索器与 Fake LLM。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from app.eval.dataset import EvalQuestion
from app.eval.generator import generate_answer
from app.eval.refusal import is_refusal

logger = logging.getLogger(__name__)


@dataclass
class RowOutcome:
    """单条评测的完整执行结果（检索侧 + 生成侧）。"""

    id: str
    category: str
    question: str
    ground_truth: str
    expected_sources: list[str]
    contexts: list[str]
    retrieved_sources: list[str]
    answer: str = ""
    refusal_detected: bool = False
    source_hit: bool | None = None
    rounds: int = 1
    stopped_reason: str | None = None
    top_rerank_score: float | None = None
    error: str | None = None

    def to_ragas_record(self) -> dict[str, Any]:
        """RAGAS evaluate 所需列（旧列名，经 dataset 转换）。"""
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


class SupportsRetrieve(Protocol):
    async def retrieve(
        self, query: str, kb_types: list[str] | None, top_k: int | None
    ) -> Any: ...


def _normalize_retrieval(result: Any) -> tuple[list[Any], int, str | None]:
    """兼容 AdaptiveResult 与 RetrievalResult，返回 (chunks, rounds, stopped_reason)。"""
    # AdaptiveRetriever 返回 AdaptiveResult（含 .retrieval）
    retrieval = getattr(result, "retrieval", result)
    chunks = list(retrieval.chunks)
    rounds = int(getattr(result, "rounds", 1))
    stopped_reason = getattr(result, "stopped_reason", None)
    return chunks, rounds, stopped_reason


async def _run_one(
    question: EvalQuestion,
    retriever: SupportsRetrieve,
    llm: Any,
    *,
    top_k: int | None,
) -> RowOutcome:
    outcome = RowOutcome(
        id=question.id,
        category=question.category,
        question=question.question,
        ground_truth=question.ground_truth_text,
        expected_sources=list(question.expected_sources),
        contexts=[],
        retrieved_sources=[],
    )
    try:
        result = await retriever.retrieve(
            question.question, question.kb_type_filter, top_k
        )
        chunks, rounds, stopped_reason = _normalize_retrieval(result)
        outcome.rounds = rounds
        outcome.stopped_reason = stopped_reason

        seen: set[str] = set()
        sources: list[str] = []
        contexts: list[str] = []
        for item in chunks:
            chunk = item.chunk
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            contexts.append(chunk.text)
            if chunk.source not in sources:
                sources.append(chunk.source)
            rerank_score = getattr(item, "rerank_score", None)
            if outcome.top_rerank_score is None or (
                rerank_score is not None and rerank_score > outcome.top_rerank_score
            ):
                outcome.top_rerank_score = rerank_score
        outcome.contexts = contexts
        outcome.retrieved_sources = sources

        outcome.answer = await generate_answer(llm, question.question, contexts)
        outcome.refusal_detected = is_refusal(outcome.answer)
        if question.expected_sources:
            outcome.source_hit = any(
                s in question.expected_sources for s in sources
            )
    except Exception as exc:  # noqa: BLE001 - 单条失败不应中断整批评测
        logger.exception("评测样例 %s 执行失败", question.id)
        outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


async def run_pipeline(
    questions: list[EvalQuestion],
    retriever: SupportsRetrieve,
    llm: Any,
    *,
    top_k: int | None = None,
    concurrency: int = 4,
    on_row_done: Any = None,
) -> list[RowOutcome]:
    """批量执行 检索+生成；信号量限流，单条异常被捕获并记录到 ``error``。"""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RowOutcome | None] = [None] * len(questions)

    async def _worker(index: int, question: EvalQuestion) -> None:
        async with semaphore:
            outcome = await _run_one(question, retriever, llm, top_k=top_k)
        results[index] = outcome
        if on_row_done is not None:
            on_row_done(outcome)

    await asyncio.gather(
        *[_worker(i, q) for i, q in enumerate(questions)]
    )
    return [r for r in results if r is not None]
