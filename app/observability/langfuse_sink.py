"""Langfuse sink（可选上报后端，Task 12）。

把统一的 :class:`SpanRecord` 映射为 Langfuse 的 trace / span：同一 ``trace_id``
聚合成一条 trace（Langfuse ``session_id`` 取会话 id），每个 span 记录名称、耗时、
状态与统一字段（写入 metadata，与本地 JSONL 字段一致）。

说明与取舍：
- 本模块延迟导入 ``langfuse``，仅在 key 配置齐全且依赖可用时被 ``build_tracer`` 加载；
  未安装 / 初始化失败会由 ``build_tracer`` 捕获并降级为本地 JSONL；
- 我们只上报"完成态"记录（span 结束时生成），因此 span 在 Langfuse 侧为扁平结构，
  层级关系通过 metadata 中的 ``parent_id`` 保留——足以在 UI 中按 trace/session 检索
  与逐条核对，满足 AC-10「同名 trace/span 可在 Langfuse 查到」的验证诉求。
"""

from __future__ import annotations

import logging
from typing import Any

from app.observability.tracer import SpanRecord

logger = logging.getLogger(__name__)


class LangfuseSink:
    """Langfuse 上报后端；构造失败由调用方降级处理。"""

    def __init__(self, *, secret_key: str, public_key: str, host: str) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(
            secret_key=secret_key, public_key=public_key, host=host
        )
        self._traces: dict[str, Any] = {}

    def _trace(self, record: SpanRecord) -> Any:
        trace = self._traces.get(record.trace_id)
        if trace is None:
            trace = self._client.trace(
                id=record.trace_id,
                name=record.trace_name,
                session_id=record.session_id,
            )
            self._traces[record.trace_id] = trace
        return trace

    def emit(self, record: SpanRecord) -> None:
        trace = self._trace(record)
        metadata: dict[str, Any] = {
            "span_id": record.span_id,
            "parent_id": record.parent_id,
            "kind": record.kind,
            "session_id": record.session_id,
            "duration_ms": record.duration_ms,
            **record.fields,
        }
        span = trace.span(name=record.name, metadata=metadata)
        if record.status == "error":
            span.end(level="ERROR", status_message=record.error or "error")
        else:
            span.end()

    def close(self) -> None:
        """flush 待发批次并关闭后台消费线程（sink 与 tracer 同生命周期）。"""
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001 - 关闭失败仅告警
            logger.warning("Langfuse flush 失败：%s", exc)
        try:
            self._client.shutdown()
        except Exception as exc:  # noqa: BLE001 - 同上
            logger.warning("Langfuse shutdown 失败：%s", exc)
