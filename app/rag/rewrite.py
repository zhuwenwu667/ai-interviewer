"""查询改写：Multi-Query 与 HyDE 两种策略，均可通过配置开关启停。

- Multi-Query：一次 LLM 调用生成 N 个语义等价但表述不同的查询（JSON 模式 + 行解析兜底）；
- HyDE：生成一段"假设答案"文本，以其向量参与检索，提高问答型查询的召回；
- 可靠性约定：任一策略失败（LLM 异常 / 解析为空）均降级为不启用该策略，
  检索主流程不因此中断，失败原因写 warning 日志。
"""

from __future__ import annotations

import json
import logging
import re

from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient

logger = logging.getLogger(__name__)

_MULTI_QUERY_SYSTEM = (
    "你是检索查询改写助手。给定用户问题，生成语义等价但表述不同的检索查询，"
    "用于在知识库（简历、岗位 JD、技术题库）中召回相关内容。"
    "要求：保留关键实体与技术术语；不要回答问题；不要编号解释；"
    '严格输出 JSON，格式为 {"queries": ["查询1", "查询2", ...]}。'
)
_MULTI_QUERY_USER = "原问题：{query}\n请生成 {n} 个改写查询。"

_HYDE_SYSTEM = (
    "你是技术问答助手。给定问题，直接编写一段简短的假设性答案（120 字以内），"
    "该答案将用于向量检索相似文档。要求：使用规范技术术语；不要解释；"
    "不要输出答案以外的任何内容。"
)
_HYDE_USER = "问题：{query}\n假设答案："

# 行解析兜底时剥离常见的列表前缀标记（"- "、"1."、"1、" 等）
_LIST_PREFIX = re.compile(r"^(?:[-*•]|\d{1,2}[.、）)])\s*")


def _parse_queries(content: str) -> list[str]:
    """优先按 JSON 解析 ``{"queries": [...]}``；失败时按行解析。"""
    try:
        data = json.loads(content)
        items = data.get("queries", []) if isinstance(data, dict) else data
        if isinstance(items, list):
            return [str(x).strip() for x in items if str(x).strip()]
        return []
    except json.JSONDecodeError:
        queries = []
        for line in content.splitlines():
            line = _LIST_PREFIX.sub("", line.strip())
            if line:
                queries.append(line)
        return queries


class QueryRewriter:
    """基于 DeepSeek 的查询改写器，策略开关与数量读取 settings。"""

    def __init__(self, llm: DeepSeekChatClient, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    @property
    def multi_query_enabled(self) -> bool:
        return self._settings.retrieval_enable_multi_query

    @property
    def hyde_enabled(self) -> bool:
        return self._settings.retrieval_enable_hyde

    async def multi_query(self, query: str) -> list[str]:
        """返回含原查询在内的去重查询列表（原查询恒排首位）。

        开关关闭时直接返回 ``[query]``，不产生 LLM 调用。
        """
        n = self._settings.retrieval_multi_query_n
        if not self.multi_query_enabled or n <= 0:
            return [query]
        messages = [
            {"role": "system", "content": _MULTI_QUERY_SYSTEM},
            {"role": "user", "content": _MULTI_QUERY_USER.format(query=query, n=n)},
        ]
        try:
            result = await self._llm.chat(messages, purpose="rewrite", json_mode=True)
            variants = _parse_queries(result.content)
        except Exception as exc:  # noqa: BLE001 - 改写失败降级为原查询，不中断检索
            logger.warning("Multi-Query 改写失败，降级为原查询：%s", exc)
            return [query]
        queries = [query]
        for variant in variants:
            if variant != query and variant not in queries:
                queries.append(variant)
        return queries[: n + 1]

    async def hyde(self, query: str) -> str | None:
        """生成假设答案文本；开关关闭或失败时返回 None。"""
        if not self.hyde_enabled:
            return None
        messages = [
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": _HYDE_USER.format(query=query)},
        ]
        try:
            result = await self._llm.chat(messages, purpose="rewrite")
        except Exception as exc:  # noqa: BLE001 - 同上，降级处理
            logger.warning("HyDE 假设答案生成失败，跳过该策略：%s", exc)
            return None
        text = result.content.strip()
        return text or None
