"""LangGraph 面试状态机测试（Task 7）。

- TR-7.1：mock LLM 队列跑完整图，断言节点序列含
  greeting→ask→receive→assess→followup→final→report，追问 ≤2 轮、题数达标即结束；
  出题失败降级模板题、研判失败默认放行。
- TR-7.2：中断后（仅落 checkpoint）用同一 session_id 恢复，历史与剩余轮次正确。
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agent.graph import InterviewEngine, _parse_assess, _parse_question
from app.api.schemas import InterviewReport
from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient
from app.rag.adaptive import AdaptiveResult, SufficiencyVerdict
from app.rag.models import Chunk
from app.rag.retriever import RetrievalResult, RetrievalTrace, RetrievedChunk
from app.storage.db import SessionRepository


def _settings(**overrides) -> Settings:
    base: dict = {
        "_env_file": None,
        "deepseek_api_key": "sk-test-ds",
        "deepseek_base_url": "https://ds.test/v1",
        "siliconflow_api_key": "sk-test-sf",
        "siliconflow_base_url": "https://sf.test/v1",
    }
    base.update(overrides)
    return Settings(**base)


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
            "model": "deepseek-chat",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    )


def _llm(settings: Settings, responses: list[str]) -> DeepSeekChatClient:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(queue.pop(0))

    return DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch):
    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.core.http_utils.asyncio.sleep", _instant)


class FakeAdaptive:
    """每次调用返回一条新证据（ev-1、ev-2……），记录查询。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(
        self, query: str, kb_types: list[str] | None = None, top_k: int | None = None
    ) -> AdaptiveResult:
        self.queries.append(query)
        n = len(self.queries)
        chunk = Chunk(
            chunk_id=f"ev-{n}", source="qa_backend.md", kb_type="qa",
            heading=f"考点{n}", text=f"证据{n}：Redis 缓存穿透可用布隆过滤器解决",
            chunk_index=0,
        )
        entry = RetrievedChunk(chunk=chunk, vector_rank=1, rerank_score=0.9)
        trace = RetrievalTrace(
            original_query=query, vector_queries=[query], hyde_answer=None,
            vector_hit_count=1, bm25_hit_count=0, fused_count=1, rerank_candidate_count=1,
        )
        return AdaptiveResult(
            retrieval=RetrievalResult(chunks=[entry], trace=trace),
            rounds=1,
            stopped_reason="sufficient",
            verdicts=[SufficiencyVerdict(sufficient=True)],
        )


@pytest.fixture
def repo(tmp_path):
    return SessionRepository(tmp_path / "test.db")


def _create_session(repo: SessionRepository, num_questions: int = 2) -> str:
    from app.api.schemas import CreateSessionRequest

    record = repo.create_session(
        CreateSessionRequest(track="backend", num_questions=num_questions)
    )
    return record.session_id


def _question_json(text: str, focus: str, ev_id: str) -> str:
    return json.dumps({"question": text, "focus": focus, "evidence_ids": [ev_id]})


def _assess_json(quality: str, feedback: str = "点评", followup: str = "") -> str:
    return json.dumps(
        {"quality": quality, "feedback": feedback, "followup_question": followup}
    )


# ---------------------------------------------------------------------------
# 解析辅助单测
# ---------------------------------------------------------------------------
def test_parse_question_validation_and_fallback():
    q = _parse_question(_question_json("Q", "F", "ev-1"), ["ev-1", "ev-2"])
    assert q == {"question": "Q", "focus": "F", "evidence_ids": ["ev-1"]}
    # 证据 id 不在召回集 / 为空时兜底 Top-1
    q = _parse_question(_question_json("Q", "F", "不存在"), ["ev-1"])
    assert q["evidence_ids"] == ["ev-1"]
    assert _parse_question("垃圾输出", ["ev-1"]) is None
    assert _parse_question('{"focus": "无题干"}', ["ev-1"]) is None


def test_parse_assess_fallback_default_good():
    assert _parse_assess(_assess_json("partial", "x", "F"))["quality"] == "partial"
    bad = _parse_assess("无法解析@@")
    assert bad["quality"] == "good" and bad["followup_question"] == ""


# ---------------------------------------------------------------------------
# TR-7.1 完整图：含一次追问，题数达标即结束
# ---------------------------------------------------------------------------
async def test_full_graph_with_one_followup(repo):
    settings = _settings()
    llm = _llm(
        settings,
        [
            "你好，欢迎参加本次面试。",  # greeting
            _question_json("Q1：Redis 缓存穿透怎么解决？", "缓存", "ev-1"),  # ask Q1
            _assess_json("partial", "缺少空值缓存", "F1：那缓存空值怎么处理？"),  # assess A1
            _assess_json("good", "补充完整"),  # assess F1-answer
            _question_json("Q2：什么是 GIL？", "GIL", "ev-2"),  # ask Q2
            _assess_json("good", "回答准确"),  # assess A2 → 达标结束
        ],
    )
    session_id = _create_session(repo, num_questions=2)
    engine = InterviewEngine.create(
        settings, repo, FakeAdaptive(), llm, session_id, "backend", 2
    )

    await engine.start()
    assert engine.state["current_question"]["question"].startswith("Q1")

    await engine.submit_answer("用布隆过滤器。")
    assert engine.state["current_question"]["question"].startswith("F1")  # 发生追问
    assert engine.state["followup_count"] == 1

    await engine.submit_answer("缓存空值并设置短过期。")
    assert engine.state["current_question"]["question"].startswith("Q2")  # 进入下一题
    assert engine.state["followup_count"] == 0

    await engine.submit_answer("GIL 是 CPython 的全局解释器锁。")
    assert engine.state["done"] is True

    trace = engine.state["node_trace"]
    assert trace == [
        "greeting", "ask_question",
        "receive_answer", "assess_answer", "followup",
        "receive_answer", "assess_answer", "ask_question",
        "receive_answer", "assess_answer", "final_evaluation", "report",
    ]
    assert len(engine.state["questions_asked"]) == 2  # 追问不占题数

    # 消息按序落库，出题消息带证据引用
    messages = repo.list_messages(session_id)
    roles = [m.role for m in messages]
    assert roles == [
        "interviewer", "interviewer", "candidate", "interviewer",
        "candidate", "interviewer", "candidate",
    ]
    q1_msg = messages[1]
    assert q1_msg.stage == "question" and q1_msg.evidence[0].chunk_id == "ev-1"
    assert messages[3].stage == "followup"

    # 报告已持久化且 schema 合法（默认生成器骨架）
    report = repo.get_report(session_id)
    assert isinstance(report, InterviewReport)
    assert {d.dimension for d in report.dimensions} == {
        "岗位匹配度", "基础知识", "项目深度", "表达逻辑", "问题解决",
    }
    assert repo.get_session(session_id).status == "finished"
    assert engine.state["evaluation_summary"]["quality_counts"] == {
        "good": 2, "partial": 1, "poor": 0,
    }


async def test_followup_capped_at_two_then_finish(repo):
    settings = _settings()
    llm = _llm(
        settings,
        [
            "开场。",
            _question_json("Q1？", "缓存", "ev-1"),
            _assess_json("partial", "r1", "F1？"),
            _assess_json("partial", "r2", "F2？"),
            _assess_json("poor", "r3", "F3-不应出现"),  # 第 3 次研判：已达追问上限
        ],
    )
    session_id = _create_session(repo, num_questions=1)
    engine = InterviewEngine.create(
        settings, repo, FakeAdaptive(), llm, session_id, "backend", 1
    )
    await engine.start()
    await engine.submit_answer("A1")
    await engine.submit_answer("A2")
    await engine.submit_answer("A3")  # 不再追问，题数达标 → 结束
    assert engine.state["done"] is True
    assert engine.state["node_trace"].count("followup") == 2  # 追问 ≤2 轮
    assert len(engine.state["questions_asked"]) == 1


async def test_ask_failure_degrades_to_template_question(repo):
    settings = _settings()
    responses = ["开场。"] + ["500 错误"] * 3 + [_assess_json("good")]  # 出题 3 次重试全失败
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        content = queue.pop(0)
        if content == "500 错误":
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return _chat_response(content)

    llm = DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))
    session_id = _create_session(repo, num_questions=1)
    engine = InterviewEngine.create(
        settings, repo, FakeAdaptive(), llm, session_id, "backend", 1
    )
    await engine.start()
    question = engine.state["current_question"]
    assert "考点1" in question["question"]  # 证据模板题兜底
    assert question["evidence_ids"] == ["ev-1"]


# ---------------------------------------------------------------------------
# TR-7.2 中断恢复：历史与剩余轮次正确
# ---------------------------------------------------------------------------
async def test_resume_after_interruption(repo):
    settings = _settings()
    llm = _llm(
        settings,
        [
            "开场。",
            _question_json("Q1？", "缓存", "ev-1"),
            _assess_json("good", "r1"),
            _question_json("Q2？", "GIL", "ev-2"),
            _assess_json("good", "r2"),
        ],
    )
    retriever = FakeAdaptive()
    session_id = _create_session(repo, num_questions=2)

    engine_a = InterviewEngine.create(
        settings, repo, retriever, llm, session_id, "backend", 2
    )
    await engine_a.start()
    await engine_a.submit_answer("A1")  # 完成第 1 题，"进程崩溃"：不再持有 engine_a

    # 用同一 session_id 恢复（模拟进程重启后新建引擎）
    engine_b = InterviewEngine.restore(settings, repo, retriever, llm, session_id)
    assert engine_b is not None
    assert engine_b.state["node_trace"] == [
        "greeting", "ask_question", "receive_answer", "assess_answer", "ask_question",
    ]
    assert len(engine_b.state["questions_asked"]) == 2  # Q1 + Q2（Q2 在恢复前已出）
    assert engine_b.state["answers"][0]["answer"] == "A1"  # 历史完整

    await engine_b.submit_answer("A2")
    assert engine_b.state["done"] is True
    assert engine_b.state["evaluation_summary"]["total_questions"] == 2
    assert len(repo.list_messages(session_id)) == 5  # 开场+Q1+A1+Q2+A2

    assert InterviewEngine.restore(settings, repo, retriever, llm, "不存在") is None


async def test_early_finish_generates_report(repo):
    settings = _settings()
    llm = _llm(settings, ["开场。", _question_json("Q1？", "缓存", "ev-1")])
    session_id = _create_session(repo, num_questions=5)
    engine = InterviewEngine.create(
        settings, repo, FakeAdaptive(), llm, session_id, "backend", 5
    )
    await engine.start()
    report = await engine.finish()  # 未答任何题提前结束
    assert isinstance(report, InterviewReport)
    assert engine.state["done"] is True
    assert engine.state["node_trace"][-2:] == ["final_evaluation", "report"]
    # 重复 finish 幂等
    again = await engine.finish()
    assert again.session_id == session_id
    with pytest.raises(RuntimeError, match="已结束"):
        await engine.submit_answer("迟到")
