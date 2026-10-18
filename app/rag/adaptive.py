"""Agentic RAG：检索质量自评与自适应重检索。

流程：
1. 首轮混合检索（复用 Task 4 的 ``Retriever``）；
2. 评估节点：LLM 依据"问题 + 召回上下文 Top-3 摘要"判定上下文是否足以回答问题
   （JSON 约束输出 + 解析兜底；连续非法输出重试一次后保守放行）；
3. 不足时用评估反馈的改进查询（fallback：原查询拼接不足理由）重检索，
   最多 ``settings.adaptive_max_rewrite_rounds`` 次；
4. 达到上限时返回各轮中 Top-1 rerank 分数最优的一轮结果（"当前最优结果"）。

每轮判定、改进查询与命中变化写结构化日志，供 trace / 前端展示。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient
from app.observability.tracer import get_tracer
from app.rag.retriever import RetrievalResult, Retriever

if TYPE_CHECKING:  # 仅类型标注，避免与 cache 模块形成运行时循环导入
    from app.rag.cache import SemanticCache

logger = logging.getLogger(__name__)

_EVAL_SYSTEM = (
    "你是检索质量评估员。给定用户问题与检索召回的上下文片段，"
    "判断这些上下文是否足以准确、完整地回答该问题。"
    '严格输出 JSON，格式为 {"sufficient": true/false, "reason": "简短理由", '
    '"improved_query": "不足时给出的改进检索查询，充分时为空字符串"}。'
    "不要输出 JSON 以外的任何内容。"
)
_EVAL_USER = "问题：{query}\n\n召回上下文：\n{context}\n\n请评估。"

# 关键词兜底：英文用词边界（避免 insufficient/cannot 误命中），中文限定开头窗口
_NO_PATTERN = re.compile(r"\b(?:no|false|insufficient)\b|不足|不够|不充足|否")
_YES_PATTERN = re.compile(r"\b(?:yes|true|sufficient)\b|充足|足够|是")


@dataclass
class SufficiencyVerdict:
    """一次检索充分性判定。"""

    sufficient: bool
    reason: str = ""
    improved_query: str | None = None


@dataclass
class AdaptiveResult:
    """自适应检索的最终结果。"""

    retrieval: RetrievalResult  # 各轮中 Top-1 rerank 分数最优的一轮
    rounds: int  # 实际检索轮次（含首轮）
    stopped_reason: str  # "sufficient" | "max_rounds_reached"
    verdicts: list[SufficiencyVerdict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> AdaptiveResult:
        """从 ``dataclasses.asdict`` 产物还原（语义缓存用）。"""
        return cls(
            retrieval=RetrievalResult.from_dict(data["retrieval"]),
            rounds=int(data["rounds"]),
            stopped_reason=str(data["stopped_reason"]),
            verdicts=[SufficiencyVerdict(**v) for v in data.get("verdicts", [])],
        )


def _parse_verdict(content: str) -> SufficiencyVerdict | None:
    """解析评估输出。优先 JSON；失败时按 yes/no/是/否 关键词兜底；均失败返回 None。"""
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            raw = data.get("sufficient")
            if isinstance(raw, bool):
                improved = str(data.get("improved_query") or "").strip() or None
                return SufficiencyVerdict(
                    sufficient=raw,
                    reason=str(data.get("reason") or "").strip(),
                    improved_query=improved,
                )
    except json.JSONDecodeError:
        pass
    head = content.strip().lower()[:20]
    if _NO_PATTERN.search(head):
        return SufficiencyVerdict(sufficient=False, reason=content.strip()[:200])
    if _YES_PATTERN.search(head):
        return SufficiencyVerdict(sufficient=True, reason=content.strip()[:200])
    return None


def _top_rerank_score(result: RetrievalResult) -> float:
    if not result.chunks:
        return float("-inf")
    return result.chunks[0].rerank_score or 0.0


class AdaptiveRetriever:
    """自评 + 重检索封装的检索入口，供出题 / 追问节点调用。"""

    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        llm: DeepSeekChatClient,
        *,
        cache: SemanticCache | None = None,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm
        self._cache = cache

    def _scope(self, kb_types: list[str] | None, top_k: int | None) -> str:
        """缓存作用域：区分不同检索范围（kb_types + top_k），避免互相污染。"""
        kb = ",".join(sorted(kb_types)) if kb_types else "*"
        final_k = top_k if top_k is not None else self._settings.retrieval_final_top_k
        return f"{kb}|{final_k}"

    async def _evaluate(self, query: str, result: RetrievalResult) -> SufficiencyVerdict:
        """LLM 判定召回充分性；非法输出重试一次，仍非法则保守放行（视为充足）。"""
        top_chunks = result.chunks[:3]
        context = (
            "\n".join(f"[{i}] {c.chunk.text[:200]}" for i, c in enumerate(top_chunks, start=1))
            or "（无召回内容）"
        )
        messages = [
            {"role": "system", "content": _EVAL_SYSTEM},
            {"role": "user", "content": _EVAL_USER.format(query=query, context=context)},
        ]
        for attempt in range(2):
            try:
                resp = await self._llm.chat(messages, purpose="eval", json_mode=True)
            except Exception as exc:  # noqa: BLE001 - 评估失败不中断检索主流程
                logger.warning("检索评估调用失败（第 %d 次），保守放行：%s", attempt + 1, exc)
                return SufficiencyVerdict(sufficient=True, reason=f"评估调用失败：{exc}")
            verdict = _parse_verdict(resp.content)
            if verdict is not None:
                return verdict
            logger.warning("检索评估输出无法解析（第 %d 次）：%r", attempt + 1, resp.content[:120])
        logger.warning("检索评估连续输出非法，保守放行（视为充足）")
        return SufficiencyVerdict(sufficient=True, reason="评估输出非法，保守放行")

    async def retrieve(
        self,
        query: str,
        kb_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> AdaptiveResult:
        """带语义缓存的检索入口：命中即整体跳过改写/召回/精排/自评。"""
        if self._cache is None:
            return await self._retrieve_uncached(query, kb_types=kb_types, top_k=top_k)
        scope = self._scope(kb_types, top_k)
        cached = await self._cache.get(query, scope)
        if cached is not None:
            return cached
        result = await self._retrieve_uncached(query, kb_types=kb_types, top_k=top_k)
        await self._cache.put(query, scope, result)
        return result

    async def _retrieve_uncached(
        self,
        query: str,
        kb_types: list[str] | None = None,
        top_k: int | None = None,
    ) -> AdaptiveResult:
        max_rounds = self._settings.adaptive_max_rewrite_rounds
        current_query = query
        best: RetrievalResult | None = None
        verdicts: list[SufficiencyVerdict] = []
        rounds = 0

        for round_idx in range(max_rounds + 1):
            rounds = round_idx + 1
            result = await self._retriever.retrieve(current_query, kb_types=kb_types, top_k=top_k)
            if best is None or _top_rerank_score(result) > _top_rerank_score(best):
                best = result
            verdict = await self._evaluate(query, result)
            verdicts.append(verdict)
            top_score = _top_rerank_score(result)
            logger.info(
                "自适应检索第 %d 轮：query=%r sufficient=%s top_score=%.4f reason=%r",
                rounds, current_query, verdict.sufficient, top_score, verdict.reason,
            )
            get_tracer().event(
                "adaptive.verdict",
                round=rounds,
                query=current_query,
                sufficient=verdict.sufficient,
                reason=(verdict.reason or "")[:200],
                improved_query=verdict.improved_query,
                top_score=top_score,
            )
            if verdict.sufficient:
                get_tracer().event(
                    "adaptive.summary",
                    query=query, rounds=rounds, stopped_reason="sufficient",
                    top_score=_top_rerank_score(best),
                )
                return AdaptiveResult(
                    retrieval=best, rounds=rounds, stopped_reason="sufficient", verdicts=verdicts
                )
            if round_idx >= max_rounds:
                break
            # 不足且未达上限：用改进查询重检索（fallback 拼接理由，保证查询文本改变）
            improved = (verdict.improved_query or "").strip()
            current_query = improved if improved and improved != current_query else (
                f"{query} {verdict.reason}".strip()
            )

        logger.info(
            "自适应检索达到最大轮次（%d 次重检索），返回当前最优结果（top_score=%.4f）",
            max_rounds, _top_rerank_score(best),
        )
        get_tracer().event(
            "adaptive.summary",
            query=query, rounds=rounds, stopped_reason="max_rounds_reached",
            top_score=_top_rerank_score(best),
        )
        return AdaptiveResult(
            retrieval=best,
            rounds=rounds,
            stopped_reason="max_rounds_reached",
            verdicts=verdicts,
        )
