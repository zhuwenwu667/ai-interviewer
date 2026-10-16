"""多维评分与报告生成测试（Task 8）。

- TR-8.2：mock LLM 首次返回残缺 JSON、第二次合法 → 重试成功；
  连续非法 → 抛 ReportGenerationError（明确错误而非脏数据）。
- schema 校验：五维度齐全、分数 1-5、每维度证据非空（Pydantic + 维度匹配检查）。
- TR-8.1（integration）：真实 LLM 跑完整场面试，报告通过 schema 校验且 Markdown 落盘。
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.agent.graph import InterviewEngine
from app.agent.report import (
    LLMReportGenerator,
    ReportGenerationError,
    render_report_markdown,
)
from app.agent.state import InterviewState
from app.api.schemas import REPORT_DIMENSIONS, CreateSessionRequest, InterviewReport
from app.config import get_settings
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


def _llm(settings: Settings, responses: list[str]):
    """MockTransport 队列 LLM；返回 (client, calls)，calls 记录每次请求体。"""
    queue = list(responses)
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return _chat_response(queue.pop(0))

    return DeepSeekChatClient(settings, transport=httpx.MockTransport(handler)), calls


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch):
    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.core.http_utils.asyncio.sleep", _instant)


def _report_payload(**overrides) -> dict:
    payload: dict = {
        "dimensions": [
            {
                "dimension": name,
                "score": 4,
                "evidence": [f"问：Q1 答：A1（{name}表现）"],
                "comment": f"{name}评语",
            }
            for name in REPORT_DIMENSIONS
        ],
        "highlights": ["缓存策略讲得清楚"],
        "weaknesses": ["GIL 理解不深"],
        "suggestions": ["补充 CPython 源码阅读"],
        "recommended_topics": [
            {"topic": "GIL", "source": "qa_python.md", "reason": "回答薄弱"}
        ],
    }
    payload.update(overrides)
    return payload


def _state(session_id: str = "s-1") -> InterviewState:
    evidence = [
        {
            "chunk_id": "ev-1", "source": "qa_backend.md", "heading": "缓存穿透",
            "score": 0.9, "text": "布隆过滤器可拦截不存在的数据",
        }
    ]
    return InterviewState(
        session_id=session_id,
        track="backend",
        num_questions=2,
        questions_asked=[
            {
                "question": "Q1：Redis 缓存穿透怎么解决？", "focus": "缓存",
                "evidence_ids": ["ev-1"], "round": 1, "evidence": evidence,
            }
        ],
        current_evidence=evidence,
        answers=[
            {
                "question": "Q1：Redis 缓存穿透怎么解决？", "answer": "用布隆过滤器。",
                "quality": "good", "feedback": "准确", "followup_question": "",
            },
            {
                "question": "Q2：什么是 GIL？", "answer": "全局解释器锁。",
                "quality": "partial", "feedback": "不够深入", "followup_question": "",
            },
        ],
        evaluation_summary={
            "total_questions": 2, "answered": 2,
            "quality_counts": {"good": 1, "partial": 1, "poor": 0},
        },
    )


# ---------------------------------------------------------------------------
# 成功路径：一次通过 + Markdown 落盘 + prompt 携带问答与知识来源
# ---------------------------------------------------------------------------
async def test_generate_success_and_markdown_persisted(tmp_path):
    settings = _settings()
    llm, calls = _llm(settings, [json.dumps(_report_payload())])
    generator = LLMReportGenerator(llm, reports_dir=tmp_path)

    report = await generator(_state())

    assert isinstance(report, InterviewReport)
    assert report.session_id == "s-1"
    assert {d.dimension for d in report.dimensions} == set(REPORT_DIMENSIONS)
    assert len(calls) == 1
    # prompt 扎根：携带问答原文与知识来源
    user_prompt = calls[0]["messages"][-1]["content"]
    assert "Q1：Redis 缓存穿透怎么解决？" in user_prompt
    assert "用布隆过滤器。" in user_prompt
    assert "qa_backend.md" in user_prompt
    # Markdown 版持久化
    md = (tmp_path / "s-1.md").read_text(encoding="utf-8")
    assert "# 面试评估报告" in md
    assert "岗位匹配度" in md and "改进建议" in md and "GIL" in md


# ---------------------------------------------------------------------------
# TR-8.2：残缺 JSON → 重试成功；维度缺失/分数越界 → 重试；连续非法 → 明确报错
# ---------------------------------------------------------------------------
async def test_retry_succeeds_after_broken_json(tmp_path):
    settings = _settings()
    llm, calls = _llm(
        settings,
        ['{"dimensions": [{"dimension": "岗位匹配度"', json.dumps(_report_payload())],
    )
    report = await LLMReportGenerator(llm, reports_dir=tmp_path)(_state())
    assert isinstance(report, InterviewReport)
    assert len(calls) == 2  # 首次残缺、第二次合法
    # 重试时把校验错误反馈给模型定向修正
    assert "未通过校验" in calls[1]["messages"][-1]["content"]


async def test_missing_dimension_retried(tmp_path):
    broken = _report_payload()
    broken["dimensions"] = broken["dimensions"][:4]  # 缺"问题解决"
    settings = _settings()
    llm, calls = _llm(settings, [json.dumps(broken), json.dumps(_report_payload())])
    report = await LLMReportGenerator(llm, reports_dir=tmp_path)(_state())
    assert len(report.dimensions) == 5
    assert len(calls) == 2


async def test_score_out_of_range_retried(tmp_path):
    broken = _report_payload()
    broken["dimensions"][0]["score"] = 9  # 越界（schema 1-5）
    settings = _settings()
    llm, calls = _llm(settings, [json.dumps(broken), json.dumps(_report_payload())])
    report = await LLMReportGenerator(llm, reports_dir=tmp_path)(_state())
    assert all(1 <= d.score <= 5 for d in report.dimensions)
    assert len(calls) == 2


async def test_persistent_invalid_raises_clear_error(tmp_path):
    settings = _settings()
    llm, calls = _llm(settings, ["垃圾输出1", "垃圾输出2", "垃圾输出3"])
    with pytest.raises(ReportGenerationError, match="连续 3 次"):
        await LLMReportGenerator(llm, reports_dir=tmp_path)(_state())
    assert len(calls) == 3  # 重试 ≤2 次（共 3 次尝试）
    assert list(tmp_path.iterdir()) == []  # 脏数据不落盘


def test_render_markdown_escapes_table_cells():
    payload = _report_payload()
    payload["dimensions"][0]["comment"] = "含|竖线\n与换行"
    report = InterviewReport.model_validate(
        {**payload, "session_id": "s-x", "generated_at": "2026-10-16T00:00:00Z"}
    )
    md = render_report_markdown(report)
    assert "含\\|竖线 与换行" in md


# ---------------------------------------------------------------------------
# TR-8.1（integration）：真实 LLM 完整面试 → 报告通过 schema 校验
# ---------------------------------------------------------------------------
class _FakeAdaptive:
    """固定证据（与 Task 7 测试同型），隔离检索变量、专注验证真实 LLM 全链路。"""

    async def retrieve(
        self, query: str, kb_types: list[str] | None = None, top_k: int | None = None
    ) -> AdaptiveResult:
        chunk = Chunk(
            chunk_id="ev-real", source="qa_backend.md", kb_type="qa",
            heading="Redis 缓存穿透", text="缓存穿透可用布隆过滤器或缓存空值解决",
            chunk_index=0,
        )
        entry = RetrievedChunk(chunk=chunk, vector_rank=1, rerank_score=0.9)
        trace = RetrievalTrace(
            original_query=query, vector_queries=[query], hyde_answer=None,
            vector_hit_count=1, bm25_hit_count=0, fused_count=1, rerank_candidate_count=1,
        )
        return AdaptiveResult(
            retrieval=RetrievalResult(chunks=[entry], trace=trace),
            rounds=1, stopped_reason="sufficient",
            verdicts=[SufficiencyVerdict(sufficient=True)],
        )


@pytest.mark.integration
async def test_real_interview_report(tmp_path):
    settings = get_settings()  # 读取真实 .env
    if not settings.deepseek_api_key:
        pytest.skip("未配置 DEEPSEEK_API_KEY")
    llm = DeepSeekChatClient(settings)
    repo = SessionRepository(tmp_path / "it.db")
    session_id = repo.create_session(
        CreateSessionRequest(track="backend", num_questions=2)
    ).session_id
    reports_dir = tmp_path / "reports"
    engine = InterviewEngine.create(
        settings, repo, _FakeAdaptive(), llm, session_id, "backend", 2,
        report_generator=LLMReportGenerator(llm, reports_dir=reports_dir),
    )

    await engine.start()
    guard = 0
    while not engine.state["done"] and guard < 12:  # 追问 ≤2/题，至多 6 次提交
        await engine.submit_answer(
            "我的理解是：这是后端常见问题，我会结合项目经验说明原理与权衡，"
            "没把握的部分愿意面试后深入学习。"
        )
        guard += 1
    assert engine.state["done"] is True

    report = repo.get_report(session_id)
    assert report is not None
    assert {d.dimension for d in report.dimensions} == set(REPORT_DIMENSIONS)
    for d in report.dimensions:
        assert 1 <= d.score <= 5
        assert d.evidence, f"{d.dimension} 缺评分证据"
    assert report.highlights and report.weaknesses and report.suggestions
    md_path = reports_dir / f"{session_id}.md"
    assert md_path.exists()
    assert "岗位匹配度" in md_path.read_text(encoding="utf-8")
    print("\n真实报告维度分：", {d.dimension: d.score for d in report.dimensions})
    print("推荐知识点：", [(t.topic, t.source) for t in report.recommended_topics])
