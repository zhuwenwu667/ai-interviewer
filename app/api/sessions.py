"""面试会话路由：创建（含简历/JD 上传摄入）、SSE 消息流、提前结束、查询。

SSE 约定（FR-8）：
- ``POST /sessions/{id}/messages`` 返回 ``text/event-stream``；
- 请求体 ``content`` 为空且面试未开始 → 触发开场 + 第一题；否则提交候选人回答
  （已开始且 content 为空时不推进，直接关流）；
- 每轮图执行结束即关闭流；``done`` 事件仅报告节点发出；
- 执行异常 → ``error`` 事件（中文可读）+ ``done(error)``，不断开连接。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from app.agent.graph import EventSink, InterviewEngine
from app.agent.report import ReportGenerationError
from app.api.deps import ServiceContainer, enforce_quota, get_services
from app.api.errors import ApiError
from app.api.schemas import (
    CreateSessionRequest,
    InterviewReport,
    PostMessageRequest,
    SessionDetailResponse,
    SessionRecord,
    SseEvent,
)
from app.core.exceptions import AuthenticationError, ModelProviderError
from app.observability.tracer import build_tracer, reset_tracer, set_tracer
from app.rag.ingestion import ingest_loaded_document
from app.rag.loader import SUPPORTED_SUFFIXES, load_file
from app.rag.models import KbType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sessions"])


# ---------------------------------------------------------------------------
# 引擎装配（恢复优先）与公共校验
# ---------------------------------------------------------------------------
def _build_engine(
    services: ServiceContainer, record: SessionRecord, sink: EventSink | None = None
) -> InterviewEngine:
    engine = InterviewEngine.restore(
        services.settings, services.repo, services.adaptive, services.llm,
        record.session_id,
        report_generator=services.report_generator, event_sink=sink,
    )
    if engine is None:  # 尚无状态快照（创建后未开始）：按会话元数据新建
        engine = InterviewEngine.create(
            services.settings, services.repo, services.adaptive, services.llm,
            record.session_id, record.track, record.num_questions,
            report_generator=services.report_generator, event_sink=sink,
        )
    return engine


def _get_record(services: ServiceContainer, session_id: str) -> SessionRecord:
    record = services.repo.get_session(session_id)
    if record is None:
        raise ApiError(404, "SESSION_NOT_FOUND", f"会话 {session_id} 不存在，请检查 ID 或重新创建")
    return record


def _to_error_event(exc: Exception) -> SseEvent:
    """执行期异常 → SSE error 事件（中文可读，不泄露敏感信息）。"""
    if isinstance(exc, AuthenticationError):
        return SseEvent.error("UPSTREAM_AUTH_FAILED", str(exc))
    if isinstance(exc, ModelProviderError):
        return SseEvent.error("UPSTREAM_ERROR", str(exc))
    if isinstance(exc, ReportGenerationError):
        return SseEvent.error("REPORT_GENERATION_FAILED", str(exc))
    if isinstance(exc, RuntimeError):
        return SseEvent.error("INVALID_STATE", str(exc))
    return SseEvent.error("INTERNAL_ERROR", f"服务内部错误：{exc}")


# ---------------------------------------------------------------------------
# 上传：先校验后建会话，避免非法文件留下孤儿会话
# ---------------------------------------------------------------------------
async def _read_and_validate(services: ServiceContainer, upload: UploadFile) -> tuple[str, bytes]:
    filename = Path(upload.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ApiError(
            400, "INVALID_FILE_TYPE",
            f"不支持的文件类型 {suffix or '（无扩展名）'}，仅支持 PDF / Markdown / TXT",
        )
    data = await upload.read()
    limit = services.settings.upload_max_bytes
    if len(data) > limit:
        raise ApiError(
            413, "FILE_TOO_LARGE",
            f"文件 {filename} 超过大小限制（{limit / 1024 / 1024:.0f}MB）",
        )
    return filename, data


async def _save_and_ingest(
    services: ServiceContainer, session_id: str, filename: str, data: bytes, kb_type: KbType
) -> None:
    upload_dir = services.settings.resolve("data/uploads") / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename
    path.write_bytes(data)
    try:
        doc = load_file(path, kb_type)
    except ValueError as exc:
        raise ApiError(400, "UPLOAD_PARSE_FAILED", f"文件 {filename} 解析失败：{exc}") from exc
    doc.source = f"uploads/{session_id}/{filename}"  # source 关联会话，便于溯源与清理
    result = await ingest_loaded_document(
        services.store, services.embedder, doc,
        chunk_size=services.settings.chunk_size,
        chunk_overlap=services.settings.chunk_overlap,
    )
    logger.info(
        "上传文件已摄入 session=%s source=%s chunks=%d", session_id, doc.source, result["chunks"]
    )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@router.post("/sessions", status_code=201, dependencies=[Depends(enforce_quota)])
async def create_session(
    track: Annotated[str, Form(min_length=1, max_length=100)],
    services: Annotated[ServiceContainer, Depends(get_services)],
    num_questions: Annotated[int, Form(ge=1, le=10)] = 5,
    candidate_name: Annotated[str | None, Form(max_length=100)] = None,
    resume: Annotated[UploadFile | None, File()] = None,
    jd: Annotated[UploadFile | None, File()] = None,
) -> SessionRecord:
    uploads = [
        (kb_type, *await _read_and_validate(services, upload))
        for upload, kb_type in ((resume, KbType.RESUME), (jd, KbType.JD))
        if upload is not None and upload.filename
    ]
    record = services.repo.create_session(
        CreateSessionRequest(
            track=track, num_questions=num_questions, candidate_name=candidate_name
        )
    )
    for kb_type, filename, data in uploads:
        await _save_and_ingest(services, record.session_id, filename, data, kb_type)
    if uploads:
        count = services.retriever.refresh_bm25()
        services.cache.clear()  # 知识库已变，清空语义缓存避免检索结果陈旧
        logger.info(
            "上传摄入完成 session=%s 文件数=%d BM25 索引=%d chunks",
            record.session_id, len(uploads), count,
        )
    return record


@router.post("/sessions/{session_id}/messages", dependencies=[Depends(enforce_quota)])
async def post_message(
    session_id: str,
    req: PostMessageRequest,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> StreamingResponse:
    record = _get_record(services, session_id)
    queue: asyncio.Queue[SseEvent | None] = asyncio.Queue()

    async def sink(event: SseEvent) -> None:
        await queue.put(event)

    engine = _build_engine(services, record, sink=sink)
    answer = req.content.strip()
    started = time.perf_counter()

    async def run_engine() -> None:
        try:
            if not engine.state.get("greeted"):
                await engine.start()  # 未开始：开场 + 第一题
            elif answer:
                await engine.submit_answer(answer)
            # 已开始且 content 为空：不推进，直接关流（前端误调用安全兜底）
        except Exception as exc:  # noqa: BLE001 - 统一转为 error 事件，不中断连接
            logger.warning("面试执行异常 session=%s：%s", session_id, exc)
            await queue.put(_to_error_event(exc))
            await queue.put(SseEvent.done("error"))
        finally:
            await queue.put(None)  # 结束标记

    async def event_stream() -> AsyncIterator[str]:
        # 观测上下文：整个会话共享同一 trace_id（= session_id），下游 LLM / 检索 /
        # 图节点通过 contextvar 取用，无需逐层透传（未配 Langfuse 时写本地 JSONL）。
        tracer = build_tracer(services.settings, session_id=session_id, trace_id=session_id)
        token = set_tracer(tracer)
        task = asyncio.create_task(run_engine())
        first_logged = False
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if not first_logged:
                    first_logged = True
                    logger.info(
                        "SSE 首个事件 session=%s first_event_ms=%.1f",
                        session_id, (time.perf_counter() - started) * 1000,
                    )
                yield event.encode()
        finally:
            if not task.done():
                task.cancel()
            reset_tracer(token)
            tracer.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/finish", dependencies=[Depends(enforce_quota)])
async def finish_session(
    session_id: str, services: Annotated[ServiceContainer, Depends(get_services)]
) -> InterviewReport:
    """提前结束并生成报告；已结束会话幂等返回既有报告。"""
    record = _get_record(services, session_id)
    engine = _build_engine(services, record)
    tracer = build_tracer(services.settings, session_id=session_id, trace_id=session_id)
    token = set_tracer(tracer)
    try:
        return await engine.finish()
    finally:
        reset_tracer(token)
        tracer.close()


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str, services: Annotated[ServiceContainer, Depends(get_services)]
) -> SessionDetailResponse:
    record = _get_record(services, session_id)
    return SessionDetailResponse(
        session=record, messages=services.repo.list_messages(session_id)
    )


@router.get("/sessions/{session_id}/report")
async def get_report(
    session_id: str, services: Annotated[ServiceContainer, Depends(get_services)]
) -> InterviewReport:
    _get_record(services, session_id)
    report = services.repo.get_report(session_id)
    if report is None:
        raise ApiError(
            404, "REPORT_NOT_FOUND", "报告尚未生成，请先完成面试或调用 finish 接口提前结束"
        )
    return report
