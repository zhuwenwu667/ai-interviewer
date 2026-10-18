"""FastAPI 接口与 SSE 流式测试（Task 9）。

- TR-9.1：TestClient 全链路（创建 → 空 content 开场 SSE → 两轮问答 → done → 报告），
  所有响应字段与 schemas 一致；
- TR-9.2：错误分支（非法文件类型 / 超限文件 / 不存在会话 / 报告未生成 / 参数校验），
  统一错误结构 ``{"error": {code, message, details}}`` 与合理状态码；
- TR-9.3：SSE 首事件为 stage；上游全 401 时图内降级跑题、报告节点抛出后
  error 事件携带可读中文说明 + done(error)。
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agent.graph import default_report_generator
from app.agent.report import LLMReportGenerator
from app.api.deps import ServiceContainer
from app.api.schemas import REPORT_DIMENSIONS, InterviewReport
from app.config import get_settings
from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient
from app.main import create_app
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


class _FakeAdaptive:
    """固定证据（与 Task 7 测试同型）：每次调用返回 ev-N，隔离检索变量。"""

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
            rounds=1, stopped_reason="sufficient",
            verdicts=[SufficiencyVerdict(sufficient=True)],
        )


def _container(settings, llm, repo, report_generator=None) -> ServiceContainer:
    """离线测试容器：检索/摄入相关组件置 None（本文件用例不触达），adaptive 用假实现。"""
    return ServiceContainer(
        settings=settings,
        repo=repo,
        store=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        reranker=None,  # type: ignore[arg-type]
        retriever=None,  # type: ignore[arg-type]
        adaptive=_FakeAdaptive(),  # type: ignore[arg-type]
        cache=None,  # type: ignore[arg-type]
        llm=llm,
        report_generator=report_generator or default_report_generator,
    )


def _client(container: ServiceContainer) -> TestClient:
    app = create_app()
    app.state.services = container
    return TestClient(app)


def _question_json(text: str, focus: str, ev_id: str) -> str:
    return json.dumps({"question": text, "focus": focus, "evidence_ids": [ev_id]})


def _assess_json(quality: str, feedback: str = "点评", followup: str = "") -> str:
    return json.dumps(
        {"quality": quality, "feedback": feedback, "followup_question": followup}
    )


def _report_json() -> str:
    return json.dumps(
        {
            "dimensions": [
                {"dimension": n, "score": 4, "evidence": [f"问：Q 答：A（{n}）"], "comment": "评语"}
                for n in REPORT_DIMENSIONS
            ],
            "highlights": ["亮点"], "weaknesses": ["不足"], "suggestions": ["建议"],
            "recommended_topics": [
                {"topic": "缓存", "source": "qa_backend.md", "reason": "回答薄弱"}
            ],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# SSE 解析辅助
# ---------------------------------------------------------------------------
def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event, data = "", {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        events.append((event, data))
    return events


def _post_sse(client: TestClient, session_id: str, content: str) -> list[tuple[str, dict]]:
    with client.stream(
        "POST", f"/sessions/{session_id}/messages", json={"content": content}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    return _parse_sse(body)


def _stages(events: list[tuple[str, dict]]) -> list[str]:
    return [d["stage"] for e, d in events if e == "stage"]


def _tokens(events: list[tuple[str, dict]]) -> str:
    return "".join(d["content"] for e, d in events if e == "token")


# ---------------------------------------------------------------------------
# TR-9.1：全链路（创建 → 开场 → 两轮问答 → done → 会话详情 → 报告）
# ---------------------------------------------------------------------------
def test_full_interview_chain_over_sse(tmp_path):
    settings = _settings()
    llm = _llm(
        settings,
        [
            "你好，欢迎参加本次面试。",  # greeting
            _question_json("Q1：Redis 缓存穿透怎么解决？", "缓存", "ev-1"),  # ask Q1
            _assess_json("good", "准确"),  # assess A1 → ask Q2
            _question_json("Q2：什么是 GIL？", "GIL", "ev-2"),  # ask Q2
            _assess_json("good", "到位"),  # assess A2 → final → report
            _report_json(),  # LLM 报告
        ],
    )
    reports_dir = tmp_path / "reports"
    container = _container(
        settings, llm, SessionRepository(tmp_path / "api.db"),
        report_generator=LLMReportGenerator(llm, reports_dir=reports_dir),
    )
    client = _client(container)

    # 创建会话（multipart 表单，无文件）
    resp = client.post(
        "/sessions",
        data={"track": "backend", "num_questions": 2, "candidate_name": "测试候选人"},
    )
    assert resp.status_code == 201
    record = resp.json()
    sid = record["session_id"]
    assert record["status"] == "active" and record["track"] == "backend"
    assert record["num_questions"] == 2 and record["candidate_name"] == "测试候选人"

    # 1) 空 content 触发开场 + 第一题
    events = _post_sse(client, sid, "")
    assert events[0][0] == "stage" and events[0][1]["stage"] == "opening"  # 首事件即阶段事件
    assert _stages(events) == ["opening", "question"]
    assert "Q1：Redis 缓存穿透怎么解决？" in _tokens(events)
    assert [d["round"] for e, d in events if e == "stage" and d["stage"] == "question"] == [1]

    # 2) 第一轮回答 → 研判 → 第二题
    events = _post_sse(client, sid, "用布隆过滤器拦截不存在的数据。")
    assert _stages(events) == ["answer", "question"]
    tokens = _tokens(events)
    assert "布隆过滤器拦截" not in tokens  # 候选人回显不走 token 流
    assert "Q2：什么是 GIL？" in tokens

    # 3) 第二轮回答 → 评分 → 报告 → done(completed)
    events = _post_sse(client, sid, "GIL 是 CPython 的全局解释器锁。")
    assert _stages(events) == ["answer", "scoring"]
    assert events[-1][0] == "done" and events[-1][1]["reason"] == "completed"

    # 会话详情：状态 finished，消息历史与角色序正确，出题消息带证据引用
    detail = client.get(f"/sessions/{sid}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["session"]["status"] == "finished"
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["interviewer", "interviewer", "candidate", "interviewer", "candidate"]
    q1_msg = body["messages"][1]
    assert q1_msg["stage"] == "question" and q1_msg["evidence"][0]["chunk_id"] == "ev-1"

    # 报告：字段通过 InterviewReport 校验（与 schemas 严格一致）
    report_resp = client.get(f"/sessions/{sid}/report")
    assert report_resp.status_code == 200
    report = InterviewReport(**report_resp.json())
    assert {d.dimension for d in report.dimensions} == set(REPORT_DIMENSIONS)
    assert all(1 <= d.score <= 5 and d.evidence for d in report.dimensions)
    assert report.highlights and report.weaknesses and report.suggestions
    # Markdown 版同步落盘
    assert (reports_dir / f"{sid}.md").exists()


def test_finish_endpoint_early_and_idempotent(tmp_path):
    settings = _settings()
    llm = _llm(settings, ["开场。", _question_json("Q1？", "缓存", "ev-1")])
    container = _container(settings, llm, SessionRepository(tmp_path / "api.db"))
    client = _client(container)
    sid = client.post("/sessions", data={"track": "backend", "num_questions": 5}).json()[
        "session_id"
    ]
    _post_sse(client, sid, "")  # 开始（出第一题，未作答）

    resp = client.post(f"/sessions/{sid}/finish")  # 提前结束
    assert resp.status_code == 200
    report = InterviewReport(**resp.json())
    assert {d.dimension for d in report.dimensions} == set(REPORT_DIMENSIONS)
    assert client.get(f"/sessions/{sid}").json()["session"]["status"] == "finished"
    again = client.post(f"/sessions/{sid}/finish")  # 幂等返回既有报告
    assert again.status_code == 200 and again.json()["session_id"] == sid


# ---------------------------------------------------------------------------
# TR-9.2：错误分支 —— 统一错误结构与合理状态码
# ---------------------------------------------------------------------------
def _assert_error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    err = body["error"]
    assert err["code"] == code and err["message"]
    return err


def test_invalid_file_type_rejected(tmp_path):
    container = _container(_settings(), _llm(_settings(), []), SessionRepository(tmp_path / "api.db"))
    client = _client(container)
    resp = client.post(
        "/sessions",
        data={"track": "backend"},
        files={"resume": ("evil.exe", b"MZ-fake-binary", "application/octet-stream")},
    )
    err = _assert_error(resp, 400, "INVALID_FILE_TYPE")
    assert "PDF" in err["message"]


def test_oversize_file_rejected(tmp_path):
    settings = _settings(upload_max_bytes=8)  # 8 字节上限
    container = _container(settings, _llm(settings, []), SessionRepository(tmp_path / "api.db"))
    client = _client(container)
    resp = client.post(
        "/sessions",
        data={"track": "backend"},
        files={"resume": ("resume.md", "# 这份简历内容超过八字节".encode(), "text/markdown")},
    )
    _assert_error(resp, 413, "FILE_TOO_LARGE")


def test_missing_track_validation_error(tmp_path):
    container = _container(_settings(), _llm(_settings(), []), SessionRepository(tmp_path / "api.db"))
    client = _client(container)
    resp = client.post("/sessions", data={"num_questions": 3})  # 缺 track
    _assert_error(resp, 422, "VALIDATION_ERROR")


def test_unknown_session_all_endpoints_404(tmp_path):
    container = _container(_settings(), _llm(_settings(), []), SessionRepository(tmp_path / "api.db"))
    client = _client(container)
    ghost = "不存在的会话ID"
    _assert_error(client.get(f"/sessions/{ghost}"), 404, "SESSION_NOT_FOUND")
    _assert_error(client.get(f"/sessions/{ghost}/report"), 404, "SESSION_NOT_FOUND")
    _assert_error(client.post(f"/sessions/{ghost}/finish"), 404, "SESSION_NOT_FOUND")
    _assert_error(
        client.post(f"/sessions/{ghost}/messages", json={"content": "你好"}),
        404, "SESSION_NOT_FOUND",
    )


def test_report_not_ready_404(tmp_path):
    container = _container(_settings(), _llm(_settings(), []), SessionRepository(tmp_path / "api.db"))
    client = _client(container)
    sid = client.post("/sessions", data={"track": "backend"}).json()["session_id"]
    _assert_error(client.get(f"/sessions/{sid}/report"), 404, "REPORT_NOT_FOUND")


# ---------------------------------------------------------------------------
# 面试历史列表（优化项1）：GET /sessions 按创建时间倒序，标记是否已生成报告
# ---------------------------------------------------------------------------
def test_list_sessions_orders_and_flags_report(tmp_path):
    settings = _settings()
    llm = _llm(settings, ["开场。", _question_json("Q1？", "缓存", "ev-1"), _assess_json("good", "准确")])
    container = _container(settings, llm, SessionRepository(tmp_path / "api.db"))
    client = _client(container)

    assert client.get("/sessions").json() == {"sessions": []}  # 空列表

    # 会话1：仅创建（未开始、无报告）
    sid1 = client.post("/sessions", data={"track": "backend", "num_questions": 1}).json()[
        "session_id"
    ]
    # 会话2：创建 → 开场 → 作答 → 结束并落报告
    sid2 = client.post("/sessions", data={"track": "ai_app", "num_questions": 1}).json()[
        "session_id"
    ]
    _post_sse(client, sid2, "")
    _post_sse(client, sid2, "答：用布隆过滤器拦截不存在的数据。")

    body = client.get("/sessions").json()
    sessions = body["sessions"]
    assert [s["session_id"] for s in sessions] == [sid2, sid1]  # 创建时间倒序
    assert sessions[0]["has_report"] is True and sessions[0]["track"] == "ai_app"
    assert sessions[0]["status"] == "finished"
    assert sessions[1]["has_report"] is False and sessions[1]["status"] == "active"
    assert all({"session_id", "track", "num_questions", "has_report"} <= set(s) for s in sessions)

    # limit 生效
    assert len(client.get("/sessions", params={"limit": 1}).json()["sessions"]) == 1


# ---------------------------------------------------------------------------
# 效果看板（优化项3）：GET /metrics 只读聚合已入库评测产物，零 LLM
# ---------------------------------------------------------------------------
def _eval_payload(faithfulness: float = 0.9892, taken_at: str = "2026-10-18T02:14:45") -> dict:
    """最小可用的评测报告结构（字段路径与真实 data/eval/reports/*.json 一致）。"""
    return {
        "generated_at": taken_at,
        "dataset": {"total": 43, "executed": 43, "counts": {"resume": 10, "jd": 10, "qa": 16, "oos": 7}},
        "metrics": {
            "overall": {
                "faithfulness": faithfulness, "context_precision": 0.9399,
                "context_recall": 0.9931, "answer_relevancy": 0.8953,
            },
            "by_category": {
                "qa": {"n": 16, "metrics": {"faithfulness": 1.0, "context_precision": 0.9,
                                           "context_recall": 1.0, "answer_relevancy": 0.8}}
            },
        },
        "retrieval": {
            "source_hit_rate": 1.0, "source_f1": 0.98, "answer_citation_rate": 0.95,
            "oos_refusal_rate": 0.857, "mean_contexts": 6.0, "mean_rounds": 1.67,
            "by_category": {"qa": {"n": 16, "source_hit_rate": 1.0, "source_f1": 1.0}},
        },
        "thresholds": {"faithfulness": 0.85, "context_recall": 0.85, "answer_relevancy": 0.85},
        "thresholds_pass": {
            "faithfulness": faithfulness >= 0.85, "context_recall": True, "answer_relevancy": True,
        },
        "oos_refusal_target": 0.8,
        "oos_refusal_pass": True,
        "errors": [],
        "baseline": {"deltas": {"faithfulness": 0.0038, "context_precision": 0.0016}},
    }


def _write_reports(reports_dir, files: dict[str, dict]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (reports_dir / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _metrics_client(tmp_path, files: dict[str, dict], baseline: dict | None = None) -> TestClient:
    reports_dir = tmp_path / "reports"
    _write_reports(reports_dir, files)
    baseline_path = tmp_path / "baseline.json"
    if baseline is not None:
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")
    settings = _settings(eval_reports_dir=reports_dir, eval_baseline_path=baseline_path)
    return _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "m.db")))


def test_metrics_aggregates_reports_and_units(tmp_path):
    """看板：门禁结论 / 单位口径 / 分类别 / 消融 / 缓存 A/B / 复练提升率一次成形。"""
    client = _metrics_client(
        tmp_path,
        {
            # 旧报告在前、新报告在后：验证「取最新一份」
            "eval_20260101_000000.json": _eval_payload(faithfulness=0.5, taken_at="2026-01-01"),
            "eval_20261018_021445.json": _eval_payload(),
            "ablation_20261017_223939.json": {
                "generated_at": "2026-10-17T22:39:39",
                "configs": [{
                    "config": "dense", "agentic": False, "stages": {"bm25": False, "rrf": False},
                    "metrics": {"faithfulness": 0.9739, "context_precision": 0.8194,
                                "context_recall": 1.0, "answer_relevancy": 0.8818},
                    "source_hit_rate": 1.0, "mean_rounds": 1.0, "mean_contexts": 6.0,
                    "tokens": {"llm_calls": 43, "est_cost_cny_per_q": 0.00248},
                }],
            },
            "semantic_cache_20261018_093958.json": {
                "generated_at": "2026-10-18T09:39:58",
                "summary": {
                    "ask_question": {"baseline_mean_ms": 9194.1, "cached_mean_ms": 3182.2,
                                     "latency_delta_pct": -65.4},
                    "cache": {"hit_rate": 0.778},
                    "llm": {"baseline_calls": 90, "cached_calls": 45, "calls_delta_pct": -50.0,
                            "baseline_total_tokens": 1000, "cached_total_tokens": 500,
                            "tokens_delta_pct": -50.0},
                },
            },
        },
        baseline={"metrics": {"overall": {"faithfulness": 0.85, "context_precision": 0.85}}},
    )
    body = client.get("/metrics").json()

    assert body["gate_passed"] is True and body["gate_violations"] == []
    assert body["sources"]["eval"] == "eval_20261018_021445.json"  # 取最新一份，而非 0.5 那份
    assert body["dataset"] == {"total": 43, "executed": 43, "resume": 10, "jd": 10, "qa": 16, "oos": 7}

    quality = {row["key"]: row for row in body["quality"]}
    # 门槛指标保留 0-1 原值并带合格线；比例类指标统一换算为百分数
    assert quality["faithfulness"] == {
        "key": "faithfulness", "name": "忠实度", "value": 0.9892, "unit": "分",
        "threshold": 0.85, "passed": True, "delta": 0.0038, "note": "门槛指标",
    }
    assert quality["source_hit_rate"]["value"] == 100.0 and quality["source_hit_rate"]["unit"] == "%"
    assert quality["mean_rounds"]["value"] == 1.67 and quality["mean_rounds"]["unit"] == "轮"

    assert body["by_category"] == [{
        "category": "题库问答", "n": 16, "faithfulness": 1.0, "context_precision": 0.9,
        "context_recall": 1.0, "answer_relevancy": 0.8, "source_hit_rate": 100.0, "source_f1": 100.0,
    }]
    assert body["ablation"][0]["config"] == "dense" and body["ablation"][0]["stages"] == []
    assert body["ablation"][0]["est_cost_cny_per_q"] == 0.00248
    assert body["cache_hit_rate"] == 77.8
    assert body["semantic_cache"][0]["delta_pct"] == -65.4


def test_metrics_gate_reuses_ci_rule(tmp_path):
    """未达合格线时看板结论与 CI 门禁同源：gate_passed=False 且给出中文未达标说明。"""
    client = _metrics_client(
        tmp_path,
        {"eval_20261018_021445.json": _eval_payload(faithfulness=0.5)},
        baseline={"metrics": {"overall": {"faithfulness": 0.85, "context_precision": 0.85}}},
    )
    body = client.get("/metrics").json()
    assert body["gate_passed"] is False
    assert body["gate_violations"] and "忠实度" in body["gate_violations"][0]


def test_metrics_empty_when_no_reports(tmp_path):
    """尚无评测产物时接口不报错：返回空看板（前端据此提示「先跑评测」）。"""
    settings = _settings(
        eval_reports_dir=tmp_path / "none", eval_baseline_path=tmp_path / "none.json"
    )
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "m.db")))
    body = client.get("/metrics").json()
    assert body["gate_passed"] is False
    assert body["quality"] == [] and body["ablation"] == [] and body["by_category"] == []


# ---------------------------------------------------------------------------
# TR-9.3：SSE 首事件为 stage；上游全 401 → 图内降级跑题 → 报告节点 error 事件
# ---------------------------------------------------------------------------
def test_upstream_401_degrades_then_chinese_error_event(tmp_path):
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    llm = DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))
    container = _container(
        settings, llm, SessionRepository(tmp_path / "api.db"),
        report_generator=LLMReportGenerator(llm, reports_dir=tmp_path / "reports"),
    )
    client = _client(container)
    sid = client.post("/sessions", data={"track": "backend", "num_questions": 1}).json()[
        "session_id"
    ]

    # 开场：greeting/出题均 401 → 兜底开场白 + 证据模板题，仍正常下发
    events = _post_sse(client, sid, "")
    assert events[0] == ("stage", {"stage": "opening", "round": 0})
    assert _stages(events) == ["opening", "question"]
    assert "考点1" in _tokens(events)  # 出题降级为证据模板题

    # 提交回答：研判默认放行 → 报告节点 LLM 401 → error 事件（中文可读）+ done(error)
    events = _post_sse(client, sid, "答：用布隆过滤器。")
    kinds = [e for e, _ in events]
    assert _stages(events) == ["answer", "scoring"]
    err = next(d for e, d in events if e == "error")
    assert err["code"] == "UPSTREAM_AUTH_FAILED"
    assert "API Key" in err["message"] and "401" in err["message"]  # 中文排查建议
    assert kinds[-1] == "done" and events[-1][1]["reason"] == "error"
    # 脏报告不落库
    assert container.repo.get_report(sid) is None


# ---------------------------------------------------------------------------
# 真实链路（integration）：真实容器 + 真实 Qdrant(73 chunks)/SiliconFlow/DeepSeek
# 全 API 走完一场面试；同时产出真实追问样例供 TR-7.3 人工评查
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_real_full_chain_via_api():
    from app.api.deps import build_services

    settings = get_settings()  # 真实 .env
    if not (settings.deepseek_api_key and settings.siliconflow_api_key):
        pytest.skip("未配置 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY")
    app = create_app()
    app.state.services = build_services(settings)

    # 注意：必须用 with 持有 TestClient（持久 portal/事件循环）。
    # 否则每个请求一个新循环，跨请求复用的 AsyncOpenAI/httpx 连接池会报
    # "Event loop is closed"（生产 uvicorn 单循环无此问题）。
    with TestClient(app) as client:
        resp = client.post("/sessions", data={"track": "backend", "num_questions": 2})
        assert resp.status_code == 201, resp.text
        sid = resp.json()["session_id"]

        events = _post_sse(client, sid, "")
        assert events[0][0] == "stage" and events[0][1]["stage"] == "opening"
        assert "question" in _stages(events)

        # 故意回答得空泛（partial/poor），以触发真实追问（产出 TR-7.3 评查样例）
        weak_answer = "这个点我了解得不多，大概就是加缓存、加机器吧，再深就说不清了。"
        done = False
        for _ in range(10):  # 追问 ≤2/题，2 题至多 6 次提交
            events = _post_sse(client, sid, weak_answer)
            kinds = [e for e, _ in events]
            if "error" in kinds:
                pytest.fail(f"真实链路出现 error 事件：{events}")
            if kinds and kinds[-1] == "done":
                done = True
                break
        assert done, "10 轮内面试未结束"

        detail = client.get(f"/sessions/{sid}").json()
        assert detail["session"]["status"] == "finished"
        followups = [m["content"] for m in detail["messages"] if m["stage"] == "followup"]
        assert followups, "空泛回答应触发至少一次真实追问"
        questions = [
            (m["content"], [e["source"] for e in m["evidence"]])
            for m in detail["messages"] if m["stage"] == "question"
        ]
        print("\n真实出题（含证据来源）：", questions)
        print("真实追问样例（供 TR-7.3 人工评查）：", followups[:3])

        report = InterviewReport(**client.get(f"/sessions/{sid}/report").json())
        assert {d.dimension for d in report.dimensions} == set(REPORT_DIMENSIONS)
        assert all(1 <= d.score <= 5 and d.evidence for d in report.dimensions)
        print("真实报告维度分：", {d.dimension: d.score for d in report.dimensions})
