"""可观测性测试（Task 12）。

- TR-12.1：无 Langfuse key 时，跑完一次完整面试（真实图 + 真实检索管线 + mock LLM），
  本地 JSONL 应包含全部关键 span/event 字段（图节点、LLM 调用与耗时、查询改写、
  双路召回计数、rerank、检索自评结论），且每条记录 session_id/trace_id 一致；
- 单元：span/event 字段与父子层级、异常状态、JSONL 追加写、默认空 tracer 静默、
  Langfuse key 齐备但依赖缺失时降级为本地 JSONL、LangfuseSink 的记录映射；
- TR-12.2：用真实 langfuse SDK 上报到本地假 ingestion 端点（断言 trace/span 名称与
  字段），并对比"开启/关闭上报"两次完整面试，断言业务行为逐字一致。
"""

from __future__ import annotations

import contextlib
import http.server
import json
import socketserver
import sys
import threading
import types
from dataclasses import replace

import httpx
import pytest

from app.agent.graph import InterviewEngine
from app.api.schemas import CreateSessionRequest
from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient
from app.core.reranker import RerankItem
from app.observability.tracer import (
    TRACE_LOG_NAME,
    JsonlSink,
    SpanRecord,
    Tracer,
    build_tracer,
    get_tracer,
    reset_tracer,
    set_tracer,
)
from app.rag.adaptive import AdaptiveRetriever
from app.rag.models import Chunk
from app.rag.retriever import Retriever
from app.storage.db import SessionRepository


def _settings(tmp_path, **overrides) -> Settings:
    base: dict = {
        "_env_file": None,
        "deepseek_api_key": "sk-test-ds",
        "deepseek_base_url": "https://ds.test/v1",
        "siliconflow_api_key": "sk-test-sf",
        "siliconflow_base_url": "https://sf.test/v1",
        "log_dir": tmp_path,
    }
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# 轻量 sink：把记录收集到内存，便于断言
# ---------------------------------------------------------------------------
class _ListSink:
    def __init__(self) -> None:
        self.records: list[SpanRecord] = []
        self.closed = False

    def emit(self, record: SpanRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# 单元：span / event 字段、父子层级、异常状态
# ---------------------------------------------------------------------------
def test_span_event_fields_and_nesting():
    sink = _ListSink()
    tracer = Tracer(trace_id="t1", session_id="s1", sinks=[sink], trace_name="interview:s1")

    with tracer.span("node.ask_question", round=1) as outer:
        tracer.event("retrieval.summary", vector_hit_count=3, bm25_hit_count=2)
        with tracer.span("rewrite.multi_query") as inner:
            inner.set(queries=["a", "b"], count=2)
        outer.set(returned=2)
    tracer.close()

    by_name = {r.name: r for r in sink.records}
    assert set(by_name) == {"node.ask_question", "retrieval.summary", "rewrite.multi_query"}
    outer_rec = by_name["node.ask_question"]
    inner_rec = by_name["rewrite.multi_query"]
    # 顶层 span 无 parent；内层 span 与 event 挂在当前 span 之下
    assert outer_rec.parent_id is None and outer_rec.kind == "span"
    assert inner_rec.parent_id == outer_rec.span_id
    assert by_name["retrieval.summary"].parent_id == outer_rec.span_id
    assert by_name["retrieval.summary"].kind == "event"
    # set() 补充的字段、耗时、状态齐全
    assert outer_rec.fields["round"] == 1 and outer_rec.fields["returned"] == 2
    assert outer_rec.duration_ms >= 0 and outer_rec.status == "ok" and outer_rec.error is None
    assert inner_rec.fields == {"queries": ["a", "b"], "count": 2}
    assert all(r.trace_id == "t1" and r.session_id == "s1" for r in sink.records)
    assert all("duration_ms" in r.to_dict() for r in sink.records)


def test_span_marks_error_status_on_exception():
    sink = _ListSink()
    tracer = Tracer(trace_id="t1", session_id=None, sinks=[sink])
    with pytest.raises(ValueError, match="boom"):
        with tracer.span("llm.chat", purpose="question"):
            raise ValueError("boom")
    rec = sink.records[0]
    assert rec.status == "error" and "ValueError: boom" in (rec.error or "")


def test_jsonl_sink_appends_and_default_tracer_is_silent(tmp_path):
    # 默认（未设置）tracer 为空实现：调用不抛异常、不落盘
    get_tracer().event("noop", a=1)
    with get_tracer().span("noop.span"):
        pass
    assert not (tmp_path / TRACE_LOG_NAME).exists()

    sink = JsonlSink(tmp_path / TRACE_LOG_NAME)
    tracer = Tracer(trace_id="t1", session_id="s1", sinks=[sink])
    with tracer.span("node.greeting"):
        pass
    tracer.event("adaptive.verdict", sufficient=True)
    tracer.close()
    lines = [
        json.loads(line)
        for line in (tmp_path / TRACE_LOG_NAME).read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert [line["name"] for line in lines] == ["node.greeting", "adaptive.verdict"]
    assert lines[1]["fields"]["sufficient"] is True


# ---------------------------------------------------------------------------
# 装配：无 key → 仅本地 JSONL；有 key 但依赖缺失 → 自动降级
# ---------------------------------------------------------------------------
def test_build_tracer_without_keys_uses_jsonl_only(tmp_path):
    tracer = build_tracer(_settings(tmp_path), session_id="s1", trace_id="s1")
    assert len(tracer._sinks) == 1 and isinstance(tracer._sinks[0], JsonlSink)
    with tracer.span("node.greeting"):
        pass
    tracer.close()
    assert (tmp_path / TRACE_LOG_NAME).exists()


def test_build_tracer_degrades_when_langfuse_missing(tmp_path, monkeypatch):
    # 配置了 Langfuse 三件套但依赖不可用（模拟未安装）→ 降级为本地 JSONL，不抛异常
    monkeypatch.setitem(sys.modules, "langfuse", None)
    tracer = build_tracer(
        _settings(
            tmp_path,
            langfuse_secret_key="sk-lf-secret",
            langfuse_public_key="pk-lf-public",
            langfuse_host="https://cloud.langfuse.com",
        ),
        session_id="s1",
        trace_id="s1",
    )
    assert len(tracer._sinks) == 1 and isinstance(tracer._sinks[0], JsonlSink)
    with tracer.span("node.greeting"):
        pass
    tracer.close()
    assert (tmp_path / TRACE_LOG_NAME).exists()


# ---------------------------------------------------------------------------
# LangfuseSink 映射：trace/span 名称、session、字段与错误状态
# ---------------------------------------------------------------------------
class _FakeSpan:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.ended: dict | None = None
        self.children: list[_FakeSpan] = []

    def span(self, **kwargs) -> _FakeSpan:
        child = _FakeSpan(**kwargs)
        self.children.append(child)
        return child

    def end(self, **kwargs) -> None:
        self.ended = kwargs


class _FakeTrace(_FakeSpan):
    pass


class _FakeLangfuse:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.traces: dict[str, _FakeTrace] = {}
        self.flushed = False
        self.shutdown_called = False

    def trace(self, **kwargs) -> _FakeTrace:
        trace = _FakeTrace(**kwargs)
        self.traces[kwargs["id"]] = trace
        return trace

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_langfuse_sink_maps_records(monkeypatch):
    module = types.ModuleType("langfuse")
    module.Langfuse = _FakeLangfuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)

    from app.observability.langfuse_sink import LangfuseSink

    sink = LangfuseSink(secret_key="s", public_key="p", host="h")
    record = SpanRecord(
        name="retrieval.rerank", trace_id="tid", trace_name="interview:s1",
        session_id="s1", span_id="sp1", parent_id=None, kind="span",
        started_at="2026-01-01T00:00:00+00:00", duration_ms=12.5,
        status="ok", error=None, fields={"candidates": 3},
    )
    sink.emit(record)
    sink.emit(replace(record, span_id="sp2", status="error", error="boom"))
    sink.close()

    client: _FakeLangfuse = sink._client
    trace = client.traces["tid"]
    assert trace.kwargs["name"] == "interview:s1" and trace.kwargs["session_id"] == "s1"
    first, second = trace.children
    assert first.kwargs["name"] == "retrieval.rerank"
    assert first.kwargs["metadata"]["candidates"] == 3
    assert first.kwargs["metadata"]["span_id"] == "sp1"
    assert first.ended == {}
    assert second.ended["level"] == "ERROR" and second.ended["status_message"] == "boom"
    assert client.flushed is True


# ---------------------------------------------------------------------------
# 真实 Langfuse SDK 上报：本地捕获 /api/public/ingestion 载荷
# ---------------------------------------------------------------------------
class _CaptureHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 接口
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length).decode("utf-8", "replace")
        self.server.batches.append((self.path, json.loads(body)))  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"successes":[],"errors":[]}')

    def log_message(self, *args) -> None:
        return


@contextlib.contextmanager
def _capture_server():
    """本地假 Langfuse 服务：记录 SDK 上报到 /api/public/ingestion 的每一批事件。"""
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _CaptureHandler)
    server.batches = []  # type: ignore[attr-defined]
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_langfuse_sink_posts_trace_and_spans_to_ingestion_api():
    """用真实 langfuse SDK 上报，断言 trace/span 名称与字段进入 ingestion 载荷。

    等价于 TR-12.2「配置 key 时同名 trace/span 可在 Langfuse 查到」：数据确实按
    Langfuse 协议送达 ingestion 端点，UI 侧即可按 trace 名 / session 检索到。
    """
    pytest.importorskip("langfuse")
    from app.observability.langfuse_sink import LangfuseSink

    base = SpanRecord(
        name="retrieval.rerank", trace_id="tid", trace_name="interview:s1",
        session_id="s1", span_id="sp1", parent_id=None, kind="span",
        started_at="2026-01-01T00:00:00+00:00", duration_ms=12.5,
        status="ok", error=None, fields={"candidates": 3},
    )
    with _capture_server() as server:
        host = f"http://127.0.0.1:{server.server_address[1]}"
        sink = LangfuseSink(secret_key="sk-test", public_key="pk-test", host=host)
        sink.emit(base)
        sink.emit(replace(base, name="llm.chat", span_id="sp2", status="error",
                          error="boom", fields={"purpose": "eval"}))
        sink.close()

    events = [ev for _, payload in server.batches for ev in payload.get("batch", [])]

    def bodies(kind: str) -> list[dict]:
        return [ev["body"] for ev in events if ev["type"] == kind]

    traces = bodies("trace-create")
    assert any(
        b["id"] == "tid" and b["name"] == "interview:s1" and b.get("sessionId") == "s1"
        for b in traces
    )
    spans = bodies("span-create")
    rerank = next(b for b in spans if b["name"] == "retrieval.rerank")
    assert rerank["traceId"] == "tid"
    assert rerank["metadata"]["candidates"] == 3 and rerank["metadata"]["span_id"] == "sp1"
    chat = next(b for b in spans if b["name"] == "llm.chat")
    assert chat["metadata"]["purpose"] == "eval"
    assert any(u.get("level") == "ERROR" and u.get("statusMessage") == "boom"
               for u in bodies("span-update"))


# ---------------------------------------------------------------------------
# TR-12.1：无 key 跑完整面试链路 → JSONL 含全部关键 span 字段
# ---------------------------------------------------------------------------
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


def _routing_llm(settings: Settings) -> DeepSeekChatClient:
    """按 prompt 内容路由回答，覆盖改写 / 自评 / 出题 / 研判 / 开场各用途。"""

    def handler(request: httpx.Request) -> httpx.Response:
        blob = json.dumps(json.loads(request.content)["messages"], ensure_ascii=False)
        if "改写查询" in blob:
            return _chat_response(json.dumps({"queries": ["Redis 缓存穿透 布隆过滤器"]}))
        if "检索质量评估员" in blob:
            return _chat_response(
                json.dumps({"sufficient": True, "reason": "上下文充足", "improved_query": ""})
            )
        if "研判" in blob:
            return _chat_response(
                json.dumps({"quality": "good", "feedback": "回答准确", "followup_question": ""})
            )
        if "出题" in blob:
            return _chat_response(
                json.dumps({"question": "请介绍你的 Redis 缓存项目？", "focus": "缓存",
                            "evidence_ids": ["c1"]})
            )
        return _chat_response("你好，欢迎参加本次模拟面试。")

    return DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))


class _FakeStore:
    def __init__(self) -> None:
        self._chunks = [
            Chunk(chunk_id="c1", source="qa_backend.md", kb_type="qa", heading="Redis 缓存",
                  text="Redis 缓存穿透可用布隆过滤器解决", chunk_index=0),
            Chunk(chunk_id="c2", source="resume.md", kb_type="resume", heading="高并发项目",
                  text="主导高并发网关项目，QPS 提升", chunk_index=0),
        ]

    def search(self, vector, kb_types, top_k):
        return [(c, 0.9 - 0.1 * i) for i, c in enumerate(self._chunks)][: top_k or len(self._chunks)]

    def scroll_all(self):
        return list(self._chunks)


class _FakeEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


class _FakeReranker:
    async def rerank(self, query, documents, top_n=None):
        return [RerankItem(index=i, score=1.0 - 0.1 * i) for i in range(len(documents))]


async def _run_interview(directory, *, langfuse: bool) -> dict:
    """跑一次完整面试（真实图 + 真实检索管线 + mock LLM），返回行为快照。"""
    directory.mkdir(parents=True, exist_ok=True)
    overrides = (
        {"langfuse_secret_key": "sk-lf", "langfuse_public_key": "pk-lf",
         "langfuse_host": "https://cloud.langfuse.com"}
        if langfuse
        else {}
    )
    settings = _settings(directory, **overrides)
    llm = _routing_llm(settings)
    retriever = Retriever(settings, _FakeStore(), _FakeEmbedder(), _FakeReranker(), llm=llm)
    adaptive = AdaptiveRetriever(settings, retriever, llm)
    repo = SessionRepository(directory / "test.db")
    session_id = repo.create_session(
        CreateSessionRequest(track="backend", num_questions=1)
    ).session_id

    tracer = build_tracer(settings, session_id=session_id, trace_id=session_id)
    token = set_tracer(tracer)
    try:
        engine = InterviewEngine.create(
            settings, repo, adaptive, llm, session_id, "backend", 1
        )
        await engine.start()
        await engine.submit_answer("用布隆过滤器解决缓存穿透。")
        report = await engine.finish()
    finally:
        reset_tracer(token)
        tracer.close()

    state = repo.load_state(session_id) or {}
    snapshot = {k: v for k, v in state.items() if k not in {"session_id", "report"}}
    report_dump = report.model_dump(mode="json")
    return {
        "session_id": session_id,
        "state": snapshot,
        "messages": [(m.role, m.content) for m in repo.list_messages(session_id)],
        "report": {k: v for k, v in report_dump.items()
                   if k not in {"session_id", "generated_at"}},
        "records": [
            json.loads(line)
            for line in (directory / TRACE_LOG_NAME).read_text("utf-8").splitlines()
            if line.strip()
        ],
    }


async def test_full_interview_writes_all_key_spans(tmp_path):
    run = await _run_interview(tmp_path, langfuse=False)
    session_id, records = run["session_id"], run["records"]

    names = {r["name"] for r in records}
    assert {
        "node.greeting", "node.ask_question", "node.receive_answer", "node.assess_answer",
        "node.final_evaluation", "node.report",
        "llm.chat", "rewrite.multi_query", "retrieval.rerank", "retrieval.summary",
        "adaptive.verdict", "adaptive.summary",
    } <= names
    # 同一会话同一 trace，均带耗时字段
    assert all(r["session_id"] == session_id and r["trace_id"] == session_id for r in records)
    assert all(isinstance(r["duration_ms"], (int, float)) for r in records)

    chat = [r for r in records if r["name"] == "llm.chat"]
    assert {r["fields"]["purpose"] for r in chat} >= {"rewrite", "question", "eval"}
    assert all("prompt_tokens" in r["fields"] and r["duration_ms"] >= 0 for r in chat)

    summary = next(r for r in records if r["name"] == "retrieval.summary")
    assert summary["fields"]["vector_hit_count"] >= 1
    assert "bm25_hit_count" in summary["fields"] and "fused_count" in summary["fields"]

    verdict = next(r for r in records if r["name"] == "adaptive.verdict")
    assert verdict["fields"]["sufficient"] is True

    # 层级：改写 span 与 LLM 调用挂在其父节点之下
    ask = next(r for r in records if r["name"] == "node.ask_question")
    rewrite = next(r for r in records if r["name"] == "rewrite.multi_query")
    assert rewrite["parent_id"] == ask["span_id"]
    rewrite_chat = next(r for r in chat if r["fields"]["purpose"] == "rewrite")
    assert rewrite_chat["parent_id"] == rewrite["span_id"]


async def test_langfuse_enabled_keeps_business_behavior_identical(tmp_path, monkeypatch):
    """TR-12.2：开启上报后业务行为与关闭时完全一致，且同一 session 的 trace 已上报。"""
    created: list[_FakeLangfuse] = []

    class _Client(_FakeLangfuse):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            created.append(self)

    module = types.ModuleType("langfuse")
    module.Langfuse = _Client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", module)

    off = await _run_interview(tmp_path / "off", langfuse=False)
    on = await _run_interview(tmp_path / "on", langfuse=True)

    # 行为一致：对话消息、会话状态、最终报告均逐字相同
    assert on["messages"] == off["messages"]
    assert on["state"] == off["state"]
    assert on["report"] == off["report"]
    # 两种模式都完整写本地 JSONL（改用本地 sink + 上报后端，字段结构一致）
    assert off["records"] and on["records"]

    # 上报后端按 session_id 聚合出同名 trace，且包含图节点与检索 span
    client = created[0]
    assert len(created) == 1
    trace = client.traces[on["session_id"]]
    assert trace.kwargs["name"] == f"interview:{on['session_id']}"
    assert trace.kwargs["session_id"] == on["session_id"]
    child_names = {c.kwargs["name"] for c in trace.children}
    assert {"node.ask_question", "llm.chat", "retrieval.summary", "adaptive.verdict"} <= child_names
