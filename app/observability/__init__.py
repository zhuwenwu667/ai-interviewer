"""可观测性：Langfuse 上报与本地 JSONL 结构化日志的统一抽象（Task 12）。

公开 API：:func:`build_tracer` 装配 tracer，:func:`get_tracer` / :func:`set_tracer` /
:func:`reset_tracer` 在业务代码中取用，:func:`traced` 装饰器快速接入 span。
"""

from app.observability.tracer import (
    TRACE_LOG_NAME,
    JsonlSink,
    Span,
    SpanRecord,
    SpanSink,
    Tracer,
    build_tracer,
    get_tracer,
    reset_tracer,
    set_tracer,
    traced,
)

__all__ = [
    "TRACE_LOG_NAME",
    "JsonlSink",
    "Span",
    "SpanRecord",
    "SpanSink",
    "Tracer",
    "build_tracer",
    "get_tracer",
    "reset_tracer",
    "set_tracer",
    "traced",
]
