"""会话状态与持久化测试（Task 6）。

- TR-6.1：临时 SQLite 文件上的仓储层 CRUD 往返一致；state 快照与报告
  序列化 / 反序列化后与原对象相等。
- TR-6.2：schemas 对合法样例通过、非法样例拒绝（维度分越界、缺字段、
  空证据、题目数越界等）。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    CreateSessionRequest,
    DimensionScore,
    DoneEventData,
    ErrorEventData,
    EvidenceRef,
    InterviewReport,
    RecommendedTopic,
    SseEvent,
    StageEventData,
    TokenEventData,
)
from app.storage.db import SessionRepository


@pytest.fixture
def repo(tmp_path):
    return SessionRepository(tmp_path / "test.db")


def _create(repo: SessionRepository):
    return repo.create_session(
        CreateSessionRequest(track="backend", num_questions=5, candidate_name="张三")
    )


def _sample_report(session_id: str) -> InterviewReport:
    return InterviewReport(
        session_id=session_id,
        dimensions=[
            DimensionScore(
                dimension="基础知识", score=4,
                evidence=["问：Redis 缓存穿透如何防范？答：布隆过滤器与空值缓存"],
                comment="概念准确",
            ),
            DimensionScore(
                dimension="表达逻辑", score=3,
                evidence=["问：GIL 是什么？答：是一个锁……"],
            ),
        ],
        highlights=["八股掌握扎实"],
        weaknesses=["项目细节展开不足"],
        suggestions=["准备 STAR 法则描述项目"],
        recommended_topics=[
            RecommendedTopic(topic="Redis 缓存一致性", source="qa_backend.md", reason="追问未答全")
        ],
        generated_at=datetime(2026, 10, 16, 12, 0, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# TR-6.1 CRUD 往返一致
# ---------------------------------------------------------------------------
def test_session_crud_round_trip(repo):
    created = _create(repo)
    loaded = repo.get_session(created.session_id)
    assert loaded == created  # 全字段往返一致

    repo.update_session_status(created.session_id, "finished")
    finished = repo.get_session(created.session_id)
    assert finished.status == "finished"
    assert finished.updated_at >= created.updated_at

    assert repo.get_session("不存在") is None


def test_message_append_and_list_round_trip(repo):
    session = _create(repo)
    evidence = [
        EvidenceRef(chunk_id="qa-14", source="qa_backend.md", heading="缓存三兄弟", score=0.999)
    ]
    m1 = repo.append_message(
        session.session_id, "interviewer", "Redis 缓存穿透怎么解决？",
        stage="question", evidence=evidence,
    )
    m2 = repo.append_message(session.session_id, "candidate", "用布隆过滤器。", stage="answer")
    m3 = repo.append_message(session.session_id, "system", "占位无阶段")

    messages = repo.list_messages(session.session_id)
    assert [m.id for m in messages] == [m1.id, m2.id, m3.id]  # 按插入序
    assert messages[0] == m1 and messages[1] == m2
    assert messages[0].evidence == evidence  # 证据引用往返一致
    assert messages[1].stage == "answer"
    assert messages[2].stage is None and messages[2].evidence == []
    assert repo.list_messages("不存在") == []


def test_state_snapshot_round_trip_and_overwrite(repo):
    session = _create(repo)
    state = {
        "stage": "answer",
        "round": 2,
        "current_question": "什么是 GIL？",
        "scores": [0.999, 0.0092],
        "history": [{"role": "interviewer", "content": "中文内容✓"}],
        "followup_count": 0,
        "done": False,
    }
    repo.save_state(session.session_id, state)
    assert repo.load_state(session.session_id) == state  # 反序列化后与原对象相等

    updated = {**state, "stage": "scoring", "done": True}
    repo.save_state(session.session_id, updated)  # 覆盖式 upsert
    assert repo.load_state(session.session_id) == updated

    assert repo.load_state("不存在") is None


def test_report_save_and_get_round_trip(repo):
    session = _create(repo)
    report = _sample_report(session.session_id)
    repo.save_report(session.session_id, report)
    assert repo.get_report(session.session_id) == report

    newer = report.model_copy(update={"highlights": ["覆盖式更新"]})
    repo.save_report(session.session_id, newer)
    assert repo.get_report(session.session_id).highlights == ["覆盖式更新"]

    assert repo.get_report("不存在") is None


def test_repository_init_idempotent(tmp_path):
    db = tmp_path / "test.db"
    SessionRepository(db)
    repo = SessionRepository(db)  # 重复构造不报错
    session = _create(repo)
    assert repo.get_session(session.session_id) is not None


# ---------------------------------------------------------------------------
# TR-6.2 schema 合法 / 非法样例
# ---------------------------------------------------------------------------
def test_schemas_accept_valid_samples():
    req = CreateSessionRequest(track="ai_app")
    assert req.num_questions == 5  # 默认值

    report = _sample_report("s-1")
    dumped = report.to_json()
    assert InterviewReport.from_json(dumped) == report

    # SSE 四类事件编码格式
    cases = [
        (SseEvent.stage("question", round=2), "stage", StageEventData),
        (SseEvent.token("你好"), "token", TokenEventData),
        (SseEvent.error("E_TIMEOUT", "调用超时"), "error", ErrorEventData),
        (SseEvent.done("completed"), "done", DoneEventData),
    ]
    for event, name, data_cls in cases:
        assert event.event == name and isinstance(event.data, data_cls)
        encoded = event.encode()
        assert encoded.startswith(f"event: {name}\ndata: ") and encoded.endswith("\n\n")


def test_schemas_reject_invalid_samples():
    # 维度分越界
    with pytest.raises(ValidationError):
        DimensionScore(dimension="基础知识", score=0, evidence=["e"])
    with pytest.raises(ValidationError):
        DimensionScore(dimension="基础知识", score=6, evidence=["e"])
    # 缺字段
    with pytest.raises(ValidationError):
        DimensionScore(dimension="基础知识", evidence=["e"])  # 缺 score
    with pytest.raises(ValidationError):
        CreateSessionRequest(num_questions=5)  # 缺 track
    # 空证据 / 空文本
    with pytest.raises(ValidationError):
        DimensionScore(dimension="基础知识", score=3, evidence=[])
    with pytest.raises(ValidationError):
        CreateSessionRequest(track="", num_questions=5)
    # 题目数越界
    with pytest.raises(ValidationError):
        CreateSessionRequest(track="backend", num_questions=0)
    with pytest.raises(ValidationError):
        CreateSessionRequest(track="backend", num_questions=11)
    # 报告亮点 / 不足 / 建议不得为空
    base = _sample_report("s-1").model_dump()
    for field in ("highlights", "weaknesses", "suggestions"):
        bad = {**base, field: []}
        with pytest.raises(ValidationError):
            InterviewReport(**bad)
    # SSE done reason 非法
    with pytest.raises(ValidationError):
        SseEvent.done("未知原因")  # type: ignore[arg-type]
