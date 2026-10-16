"""统一可观测性 tracer（Task 12）。

设计要点：
- ``Tracer`` 以 ``span``（计时区间）与 ``event``（瞬时记录）两种原语上报；
  每次上报统一成一条 :class:`SpanRecord`，本地 JSONL 与 Langfuse 共用同一字段结构；
- 默认 sink 为本地 JSONL（``<LOG_DIR>/traces.jsonl``，追加写、线程安全）；当配置了
  Langfuse 三件套且安装了 ``langfuse`` 时，额外并行上报同名 trace/span；
- 通过 ``contextvar`` 暴露"当前 tracer"，业务代码只需 ``get_tracer().span(...)``，
  无需把 tracer 逐层透传（不侵入构造签名与业务逻辑）；
- 未设置 tracer 时退化为无 sink 的空实现：不上报、不抛异常、不改变业务行为；
- 任一 sink 上报失败只记 warning，绝不影响主流程（观测是旁路）。

span 层级：``Span`` 通过 contextvar 记录"当前 span id"，据此推断 ``parent_id``，
JSONL 记录天然可重建调用树；``traced`` 装饰器可一行接入任意同步 / 异步函数。
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from app.config.settings import Settings

logger = logging.getLogger(__name__)

TRACE_LOG_NAME = "traces.jsonl"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SpanRecord:
    """一次 span / event 的完整结构化记录（JSONL 与 Langfuse 共用）。"""

    name: str
    trace_id: str
    trace_name: str
    session_id: str | None
    span_id: str
    parent_id: str | None
    kind: str  # "span" | "event"
    started_at: str
    duration_ms: float
    status: str  # "ok" | "error"
    error: str | None
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "session_id": self.session_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error": self.error,
            "fields": self.fields,
        }


class SpanSink(Protocol):
    """观测后端接口：本地 JSONL 与 Langfuse 均实现之。"""

    def emit(self, record: SpanRecord) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    """把每条记录以一行 JSON 追加写入本地文件（同字段结构，供离线复盘）。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def emit(self, record: SpanRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def close(self) -> None:
        """追加写每条即时落盘，无内部缓冲，无需额外 flush。"""
        return None


# 当前 span 上下文：(trace_id, span_id)，用于推断 parent_id
_span_context: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "observability_span_context", default=None
)


class Span:
    """span 上下文管理器，同时支持 ``with``（同步）与 ``async with``（异步）。

    ``set(**fields)`` 可在区间内补充运行期才可知的信息（计数值、top 分数等）。
    """

    def __init__(self, tracer: Tracer, name: str, fields: dict[str, Any]) -> None:
        self._tracer = tracer
        self._name = name
        self._fields = dict(fields)
        self._start = 0.0
        self._started_at = ""
        self._span_id = uuid.uuid4().hex
        self._parent_id: str | None = None
        self._token: contextvars.Token | None = None

    def set(self, **fields: Any) -> Span:
        self._fields.update(fields)
        return self

    def __enter__(self) -> Span:
        self._start = time.perf_counter()
        self._started_at = _now_iso()
        ctx = _span_context.get()
        self._parent_id = ctx[1] if ctx and ctx[0] == self._tracer.trace_id else None
        self._token = _span_context.set((self._tracer.trace_id, self._span_id))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            _span_context.reset(self._token)
        # 生成器被消费者提前关闭时抛 GeneratorExit，属正常终止而非业务错误
        failed = exc is not None and exc_type is not GeneratorExit
        error = None if not failed else f"{exc_type.__name__ if exc_type else 'Error'}: {exc}"
        self._tracer._emit(
            SpanRecord(
                name=self._name,
                trace_id=self._tracer.trace_id,
                trace_name=self._tracer.trace_name,
                session_id=self._tracer.session_id,
                span_id=self._span_id,
                parent_id=self._parent_id,
                kind="span",
                started_at=self._started_at,
                duration_ms=(time.perf_counter() - self._start) * 1000,
                status="error" if failed else "ok",
                error=error,
                fields=self._fields,
            )
        )
        return False  # 不吞异常

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)


class Tracer:
    """一次 trace（通常对应一次面试会话）的上报入口。"""

    def __init__(
        self,
        *,
        trace_id: str,
        session_id: str | None,
        sinks: list[SpanSink],
        trace_name: str = "trace",
    ) -> None:
        self._trace_id = trace_id
        self._session_id = session_id
        self._sinks = list(sinks)
        self._trace_name = trace_name

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def trace_name(self) -> str:
        return self._trace_name

    def span(self, name: str, **fields: Any) -> Span:
        return Span(self, name, fields)

    def event(self, name: str, **fields: Any) -> None:
        """瞬时记录（无耗时）：如单轮自评结论、检索计数汇总。"""
        ctx = _span_context.get()
        parent_id = ctx[1] if ctx and ctx[0] == self._trace_id else None
        self._emit(
            SpanRecord(
                name=name,
                trace_id=self._trace_id,
                trace_name=self._trace_name,
                session_id=self._session_id,
                span_id=uuid.uuid4().hex,
                parent_id=parent_id,
                kind="event",
                started_at=_now_iso(),
                duration_ms=0.0,
                status="ok",
                error=None,
                fields=dict(fields),
            )
        )

    def _emit(self, record: SpanRecord) -> None:
        for sink in self._sinks:
            try:
                sink.emit(record)
            except Exception as exc:  # noqa: BLE001 - 观测旁路，失败不得影响业务
                logger.warning("观测 sink 上报失败（%s）：%s", record.name, exc)

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - 同上
                logger.warning("观测 sink 关闭失败：%s", exc)
        self._sinks = []


# 未显式设置时的空实现：无 sink、零副作用
_NOOP_TRACER = Tracer(trace_id="-", session_id=None, sinks=[])

_current_tracer: contextvars.ContextVar[Tracer | None] = contextvars.ContextVar(
    "observability_current_tracer", default=None
)


def get_tracer() -> Tracer:
    """取当前上下文的 tracer；未设置时返回空实现（业务无感知）。"""
    return _current_tracer.get() or _NOOP_TRACER


def set_tracer(tracer: Tracer) -> contextvars.Token:
    return _current_tracer.set(tracer)


def reset_tracer(token: contextvars.Token) -> None:
    _current_tracer.reset(token)


def build_tracer(
    settings: Settings,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
    trace_name: str | None = None,
) -> Tracer:
    """按配置装配 tracer：本地 JSONL 恒开；Langfuse key 齐备且依赖可用时并行上报。

    Langfuse 未安装或初始化失败时自动降级为本地 JSONL，绝不阻断业务。
    """
    sinks: list[SpanSink] = [JsonlSink(settings.log_dir_abs / TRACE_LOG_NAME)]
    if settings.langfuse_enabled:
        try:
            from app.observability.langfuse_sink import LangfuseSink

            sinks.append(
                LangfuseSink(
                    secret_key=settings.langfuse_secret_key,
                    public_key=settings.langfuse_public_key,
                    host=settings.langfuse_host,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 缺依赖 / 初始化失败 → 降级
            logger.warning("Langfuse 已配置但初始化失败，降级为本地 JSONL：%s", exc)
    resolved_id = trace_id or uuid.uuid4().hex
    return Tracer(
        trace_id=resolved_id,
        session_id=session_id,
        sinks=sinks,
        trace_name=trace_name or (f"interview:{session_id}" if session_id else "trace"),
    )


def traced(name: str, **static_fields: Any) -> Callable[[Callable], Callable]:
    """把函数（同步或异步）包成同名 span 的装饰器（不侵入函数体）。"""

    def _decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with get_tracer().span(name, **static_fields):
                    return await fn(*args, **kwargs)

            return _async_wrapper

        @functools.wraps(fn)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with get_tracer().span(name, **static_fields):
                return fn(*args, **kwargs)

        return _sync_wrapper

    return _decorator
