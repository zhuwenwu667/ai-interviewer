"""LangGraph 面试状态机与面试引擎（Task 7）。

拓扑（START 路由器实现断点恢复，无需 LangGraph 内置 checkpointer）：

    START ─┬─ 未问候 ───────────────→ greeting → ask_question → END（等回答）
           ├─ pending_answer ───────→ receive_answer → assess_answer ─┬─ followup → END
           ├─ finish_requested ─────→ final_evaluation → report → END  ├─ ask_question → END
           └─ done ─────────────────→ END                              └─ final_evaluation

约定：
- 同题追问 ≤ 2 轮；主问题数达 num_questions 即进入评分与报告；
- 出题 / 追问必须引用至少一条 RAG 证据（AdaptiveRetriever 召回，证据 id 校验兜底）；
- 每个产生文本的节点：消息落库（SessionRepository）、SSE 事件经 event_sink 发出；
- 每次图执行结束状态快照落库，``InterviewEngine.restore`` 以 session_id 恢复续跑；
- LLM 失败兜底：出题降级为证据模板题，研判降级为"默认通过"，面试不中断。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agent.context import ContextBudget
from app.agent.state import InterviewState, initial_state
from app.api.schemas import (
    REPORT_DIMENSIONS,
    DimensionScore,
    EvidenceRef,
    InterviewReport,
    RecommendedTopic,
    SseEvent,
)
from app.config.settings import Settings
from app.core.llm import ChatMessage, DeepSeekChatClient
from app.observability.tracer import traced
from app.rag.adaptive import AdaptiveRetriever
from app.storage.db import SessionRepository

logger = logging.getLogger(__name__)

MAX_FOLLOWUPS = 2  # 同题追问上限

_GREETING_SYSTEM = (
    "你是技术面试官。用一两句话亲切地向候选人问好，并说明面试即将开始。"
    "不要提问，不要自我介绍超过一句，不要输出多余内容。"
)

_ASK_SYSTEM = (
    "你是技术面试官，正在为{track}方向候选人出第{round}道面试题。"
    "必须基于给定的证据片段出题（简历项目或题库），禁止脱离证据凭空提问，"
    "禁止与已问过的问题重复。"
    '严格输出 JSON：{{"question": "题干", "focus": "考察点", '
    '"evidence_ids": ["引用的证据id", ...]}}，evidence_ids 至少一条且必须来自给定证据。'
)
_ASK_USER = "证据片段：\n{evidence}\n\n已问过的问题：\n{asked}\n\n请出题。"

_ASSESS_SYSTEM = (
    "你是技术面试官，研判候选人的回答质量。"
    '严格输出 JSON：{{"quality": "good|partial|poor", "feedback": "一句话点评", '
    '"followup_question": "追问问题或空字符串"}}。'
    "quality=good 时 followup_question 为空；partial/poor 时给出一个精确针对回答漏洞、"
    "并结合证据或简历细节的追问问题。"
)
_ASSESS_USER = "问题：{question}\n考察点：{focus}\n参考依据：\n{evidence}\n\n候选人回答：{answer}\n\n请研判。"

EventSink = Callable[[SseEvent], Awaitable[None]]
ReportGenerator = Callable[[InterviewState], Awaitable[InterviewReport]]


# ---------------------------------------------------------------------------
# 默认报告生成器（占位实现，Task 8 以 LLM 结构化生成替换）
# ---------------------------------------------------------------------------
_QUALITY_SCORE = {"good": 4, "partial": 3, "poor": 2}


async def default_report_generator(state: InterviewState) -> InterviewReport:
    """基于研判记录的五维度骨架报告（保证 schema 合法，供 Task 7 流程闭环）。"""
    from datetime import datetime

    answers = state.get("answers", [])
    scores = [_QUALITY_SCORE.get(a.get("quality"), 3) for a in answers]
    avg = round(sum(scores) / len(scores)) if scores else 1
    qa_evidence = [
        f"问：{a['question']} 答：{a['answer'][:80]}" for a in answers[:3]
    ] or ["本次面试未完成有效问答"]
    feedbacks = [a.get("feedback", "") for a in answers if a.get("feedback")]
    return InterviewReport(
        session_id=state["session_id"],
        dimensions=[
            DimensionScore(
                dimension=name,
                score=max(1, min(5, avg)),
                evidence=qa_evidence[:1],
                comment="占位评语（Task 8 将替换为 LLM 结构化生成）",
            )
            for name in REPORT_DIMENSIONS
        ],
        highlights=feedbacks[:1] or ["（样本不足，待完整面试后生成）"],
        weaknesses=feedbacks[-1:] or ["（样本不足，待完整面试后生成）"],
        suggestions=["完成全部题目后查看针对性建议"],
        recommended_topics=[
            RecommendedTopic(
                topic=e.get("heading") or e.get("source", "题库"),
                source=e.get("source", ""),
                reason="面试涉及知识点",
            )
            for e in state.get("current_evidence", [])[:3]
        ],
        generated_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------
def _parse_question(content: str, valid_ids: list[str]) -> dict[str, Any] | None:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not str(data.get("question") or "").strip():
        return None
    ids = [str(i) for i in data.get("evidence_ids") or [] if str(i) in valid_ids]
    return {
        "question": str(data["question"]).strip(),
        "focus": str(data.get("focus") or "").strip(),
        "evidence_ids": ids or valid_ids[:1],  # 兜底：至少引用 Top-1 证据
    }


def _parse_assess(content: str) -> dict[str, Any]:
    """研判解析；失败时默认 quality=good（放行进入下一题，不中断面试）。"""
    try:
        data = json.loads(content)
        quality = str(data.get("quality") or "").strip().lower()
        if quality in _QUALITY_SCORE:
            return {
                "quality": quality,
                "feedback": str(data.get("feedback") or "").strip(),
                "followup_question": str(data.get("followup_question") or "").strip(),
            }
    except json.JSONDecodeError:
        pass
    logger.warning("研判输出无法解析，默认按 good 放行：%r", content[:120])
    return {"quality": "good", "feedback": "研判解析失败，默认通过", "followup_question": ""}


# ---------------------------------------------------------------------------
# 面试引擎
# ---------------------------------------------------------------------------
class InterviewEngine:
    """持有状态、驱动图执行、负责持久化与事件发射。"""

    def __init__(
        self,
        settings: Settings,
        repo: SessionRepository,
        retriever: AdaptiveRetriever,
        llm: DeepSeekChatClient,
        state: InterviewState,
        *,
        report_generator: ReportGenerator = default_report_generator,
        event_sink: EventSink | None = None,
        ctx_budget: ContextBudget | None = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._retriever = retriever
        self._llm = llm
        self._state = state
        self._report_generator = report_generator
        self._sink = event_sink
        self._budget = ctx_budget or ContextBudget.from_settings(settings)
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # 构造 / 恢复
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        settings: Settings,
        repo: SessionRepository,
        retriever: AdaptiveRetriever,
        llm: DeepSeekChatClient,
        session_id: str,
        track: str,
        num_questions: int,
        **kwargs: Any,
    ) -> InterviewEngine:
        return cls(
            settings, repo, retriever, llm,
            initial_state(session_id, track, num_questions), **kwargs,
        )

    @classmethod
    def restore(
        cls,
        settings: Settings,
        repo: SessionRepository,
        retriever: AdaptiveRetriever,
        llm: DeepSeekChatClient,
        session_id: str,
        **kwargs: Any,
    ) -> InterviewEngine | None:
        snapshot = repo.load_state(session_id)
        if snapshot is None:
            return None
        return cls(settings, repo, retriever, llm, InterviewState(**snapshot), **kwargs)

    @property
    def state(self) -> InterviewState:
        return self._state

    # ------------------------------------------------------------------
    # 对外驱动
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """开场 + 出第一题；幂等（已问候则不再执行）。"""
        if self._state.get("greeted"):
            return
        await self._run()

    async def submit_answer(self, answer: str) -> None:
        """提交候选人回答，推进到下一题 / 追问 / 结束。"""
        if self._state.get("done"):
            raise RuntimeError("面试已结束，无法再提交回答")
        if self._state.get("current_question") is None:
            raise RuntimeError("当前没有等待回答的问题")
        self._state["pending_answer"] = answer
        await self._run()

    async def finish(self) -> InterviewReport:
        """提前结束并生成报告；已结束则直接返回既有报告。"""
        if not self._state.get("done"):
            self._state["finish_requested"] = True
            await self._run()
        return InterviewReport(**self._state["report"])

    async def _run(self) -> None:
        self._state = await self._graph.ainvoke(self._state)
        self._persist()

    def _persist(self) -> None:
        self._repo.save_state(self._state["session_id"], dict(self._state))

    # ------------------------------------------------------------------
    # 消息 / 事件副作用
    # ------------------------------------------------------------------
    async def _say(
        self,
        role: str,
        content: str,
        stage: str,
        evidence: list[EvidenceRef] | None = None,
        round_no: int = 0,
    ) -> None:
        self._repo.append_message(
            self._state["session_id"], role, content, stage=stage, evidence=evidence
        )
        if self._sink is not None:
            await self._sink(SseEvent.stage(stage, round=round_no))  # type: ignore[arg-type]
            # 候选人回显由前端本地渲染，仅面试官文本走 token 流（避免重复推送）
            if role == "interviewer":
                for i in range(0, len(content), 12):
                    await self._sink(SseEvent.token(content[i : i + 12]))

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------
    def _build_graph(self):
        graph = StateGraph(InterviewState)
        graph.add_node("greeting", self._node_greeting)
        graph.add_node("ask_question", self._node_ask_question)
        graph.add_node("receive_answer", self._node_receive_answer)
        graph.add_node("assess_answer", self._node_assess_answer)
        graph.add_node("followup", self._node_followup)
        graph.add_node("final_evaluation", self._node_final_evaluation)
        graph.add_node("report", self._node_report)

        graph.add_conditional_edges(START, self._route_entry)
        graph.add_edge("greeting", "ask_question")
        graph.add_edge("ask_question", END)
        graph.add_edge("receive_answer", "assess_answer")
        graph.add_conditional_edges(
            "assess_answer",
            self._route_after_assess,
            {"followup": "followup", "next": "ask_question", "finish": "final_evaluation"},
        )
        graph.add_edge("followup", END)
        graph.add_edge("final_evaluation", "report")
        graph.add_edge("report", END)
        return graph.compile()

    @staticmethod
    def _route_entry(state: InterviewState) -> str:
        if state.get("done"):
            return END
        if state.get("finish_requested"):
            return "final_evaluation"
        if not state.get("greeted"):
            return "greeting"
        if state.get("pending_answer"):
            return "receive_answer"
        return END

    @staticmethod
    def _route_after_assess(state: InterviewState) -> str:
        last = state["answers"][-1]
        if (
            last.get("followup_question")
            and last.get("quality") in {"partial", "poor"}
            and state.get("followup_count", 0) < MAX_FOLLOWUPS
        ):
            return "followup"
        if len(state.get("questions_asked", [])) >= state.get("num_questions", 5):
            return "finish"
        return "next"

    # ------------------------------------------------------------------
    # 节点
    # ------------------------------------------------------------------
    @traced("node.greeting")
    async def _node_greeting(self, state: InterviewState) -> dict[str, Any]:
        name = state.get("track", "技术")
        messages: list[ChatMessage] = [
            {"role": "system", "content": _GREETING_SYSTEM},
            {"role": "user", "content": f"岗位方向：{name}。请开场。"},
        ]
        try:
            text = (await self._llm.chat(messages, purpose="question")).content.strip()
        except Exception as exc:  # noqa: BLE001 - 开场失败不阻塞面试
            logger.warning("开场白生成失败，使用兜底文案：%s", exc)
            text = "你好，欢迎参加本次模拟面试，我们正式开始。"
        await self._say("interviewer", text, "opening")
        return {"greeted": True, "stage": "opening", "node_trace": state["node_trace"] + ["greeting"]}

    @traced("node.ask_question")
    async def _node_ask_question(self, state: InterviewState) -> dict[str, Any]:
        round_no = len(state["questions_asked"]) + 1
        evidence = await self._retrieve_evidence(state)
        valid_ids = [e["chunk_id"] for e in evidence]
        snippets = "\n".join(
            f"[{e['chunk_id']}]（{e['source']}）{self._budget.clip_evidence(e['text'])}"
            for e in evidence
        )
        asked_items = self._budget.recent_asked([q["question"] for q in state["questions_asked"]])
        asked = "；".join(asked_items) or "（无）"
        messages: list[ChatMessage] = [
            {"role": "system", "content": _ASK_SYSTEM.format(track=state["track"], round=round_no)},
            {"role": "user", "content": _ASK_USER.format(evidence=snippets, asked=asked)},
        ]
        question: dict[str, Any] | None = None
        try:
            resp = await self._llm.chat(messages, purpose="question", json_mode=True)
            question = _parse_question(resp.content, valid_ids)
        except Exception as exc:  # noqa: BLE001 - 出题失败降级为证据模板题
            logger.warning("结构化出题失败，降级为模板题：%s", exc)
        if question is None and evidence:
            top = evidence[0]
            question = {
                "question": f"请结合你的经历谈谈：{top['heading'] or top['text'][:30]}？",
                "focus": top["heading"] or "综合考察",
                "evidence_ids": [top["chunk_id"]],
            }
        if question is None:  # 检索与 LLM 同时失败的最终兜底
            question = {
                "question": f"请介绍一个你在{state['track']}方向做过最有挑战的项目。",
                "focus": "项目经验",
                "evidence_ids": [],
            }
        question["round"] = round_no
        # 证据快照随题目留存，供 Task 8 报告生成引用知识来源（state 保持纯 JSON）
        question["evidence"] = [
            {
                "chunk_id": e["chunk_id"], "source": e["source"],
                "heading": e.get("heading", ""), "score": e.get("score"),
                "text": e["text"][:200],
            }
            for e in evidence
        ]
        refs = [
            EvidenceRef(
                chunk_id=e["chunk_id"], source=e["source"],
                heading=e.get("heading", ""), score=e.get("score"),
            )
            for e in evidence
            if e["chunk_id"] in question["evidence_ids"]
        ]
        await self._say("interviewer", question["question"], "question", evidence=refs, round_no=round_no)
        return {
            "stage": "question",
            "questions_asked": state["questions_asked"] + [question],
            "current_question": question,
            "current_evidence": evidence,
            "used_evidence_ids": state["used_evidence_ids"] + question["evidence_ids"],
            "followup_count": 0,
            "node_trace": state["node_trace"] + ["ask_question"],
        }

    @traced("node.receive_answer")
    async def _node_receive_answer(self, state: InterviewState) -> dict[str, Any]:
        answer = (state.get("pending_answer") or "").strip()
        await self._say("candidate", answer, "answer")
        return {
            "stage": "answer",
            "pending_answer": None,
            "current_answer": {
                "question": state["current_question"]["question"],
                "answer": answer,
            },
            "node_trace": state["node_trace"] + ["receive_answer"],
        }

    @traced("node.assess_answer")
    async def _node_assess_answer(self, state: InterviewState) -> dict[str, Any]:
        current = state["current_answer"]
        snippets = "\n".join(
            f"[{e['chunk_id']}] {self._budget.clip_evidence(e['text'])}"
            for e in state.get("current_evidence", [])
        ) or "（无）"
        messages: list[ChatMessage] = [
            {"role": "system", "content": _ASSESS_SYSTEM},
            {
                "role": "user",
                "content": _ASSESS_USER.format(
                    question=current["question"],
                    focus=state["current_question"].get("focus", ""),
                    evidence=snippets,
                    answer=current["answer"],
                ),
            },
        ]
        try:
            resp = await self._llm.chat(messages, purpose="eval", json_mode=True)
            verdict = _parse_assess(resp.content)
        except Exception as exc:  # noqa: BLE001 - 研判失败默认放行
            logger.warning("回答研判调用失败，默认按 good 放行：%s", exc)
            verdict = {"quality": "good", "feedback": f"研判失败（{exc}），默认通过", "followup_question": ""}
        record = {**current, **verdict}
        profile = state.get("candidate_profile", {"observations": []})
        observations = profile.get("observations", []) + [
            {"question": current["question"], "quality": verdict["quality"]}
        ]
        return {
            "stage": "answer",
            "answers": state["answers"] + [record],
            "current_answer": None,
            "candidate_profile": {**profile, "observations": observations},
            "node_trace": state["node_trace"] + ["assess_answer"],
        }

    @traced("node.followup")
    async def _node_followup(self, state: InterviewState) -> dict[str, Any]:
        last = state["answers"][-1]
        text = last["followup_question"]
        refs = [
            EvidenceRef(
                chunk_id=e["chunk_id"], source=e["source"],
                heading=e.get("heading", ""), score=e.get("score"),
            )
            for e in state.get("current_evidence", [])[:2]
        ]
        round_no = len(state["questions_asked"])
        await self._say("interviewer", text, "followup", evidence=refs, round_no=round_no)
        followup = {
            "question": text,
            "focus": state["current_question"].get("focus", ""),
            "evidence_ids": [r.chunk_id for r in refs],
            "round": round_no,
            "is_followup": True,
            "evidence": [
                {
                    "chunk_id": e["chunk_id"], "source": e["source"],
                    "heading": e.get("heading", ""), "score": e.get("score"),
                    "text": e["text"][:200],
                }
                for e in state.get("current_evidence", [])[:2]
            ],
        }
        return {
            "stage": "followup",
            "current_question": followup,
            "followup_count": state.get("followup_count", 0) + 1,
            "node_trace": state["node_trace"] + ["followup"],
        }

    @traced("node.final_evaluation")
    async def _node_final_evaluation(self, state: InterviewState) -> dict[str, Any]:
        answers = state.get("answers", [])
        counts = {"good": 0, "partial": 0, "poor": 0}
        for a in answers:
            if a.get("quality") in counts:
                counts[a["quality"]] += 1
        summary = {
            "total_questions": len(state.get("questions_asked", [])),
            "answered": len(answers),
            "quality_counts": counts,
            "avg_quality_score": (
                round(
                    sum(_QUALITY_SCORE[a["quality"]] for a in answers if a.get("quality") in _QUALITY_SCORE)
                    / max(1, sum(counts.values())),
                    2,
                )
            ),
        }
        logger.info("面试评估汇总 session=%s summary=%s", state["session_id"], summary)
        if self._sink is not None:  # 评分阶段无消息体，仅发阶段事件（前端阶段徽标）
            await self._sink(SseEvent.stage("scoring"))
        return {
            "stage": "scoring",
            "evaluation_summary": summary,
            "node_trace": state["node_trace"] + ["final_evaluation"],
        }

    @traced("node.report")
    async def _node_report(self, state: InterviewState) -> dict[str, Any]:
        report = await self._report_generator(state)
        self._repo.save_report(state["session_id"], report)
        self._repo.update_session_status(state["session_id"], "finished")
        if self._sink is not None:
            await self._sink(SseEvent.done("completed"))
        logger.info("面试报告已生成 session=%s", state["session_id"])
        return {
            "stage": "done",
            "done": True,
            "current_question": None,
            "report": report.model_dump(mode="json"),
            "node_trace": state["node_trace"] + ["report"],
        }

    # ------------------------------------------------------------------
    # 证据检索（去重优先）
    # ------------------------------------------------------------------
    async def _retrieve_evidence(self, state: InterviewState) -> list[dict[str, Any]]:
        query = f"{state['track']} 方向面试题"
        try:
            result = await self._retriever.retrieve(query, top_k=6)
        except Exception as exc:  # noqa: BLE001 - 检索失败不阻塞出题（走无证据兜底模板）
            logger.warning("证据检索失败：%s", exc)
            return []
        used = set(state.get("used_evidence_ids", []))
        fresh = [c for c in result.retrieval.chunks if c.chunk.chunk_id not in used]
        picked = fresh or list(result.retrieval.chunks)  # 证据耗尽时允许复用
        return [
            {
                "chunk_id": c.chunk.chunk_id,
                "source": c.chunk.source,
                "heading": c.chunk.heading,
                "score": c.rerank_score,
                "text": c.chunk.text,
            }
            for c in picked[:3]
        ]
