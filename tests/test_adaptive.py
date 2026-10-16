"""Agentic RAG 自评重检索测试（Task 5）。

- TR-5.1：mock LLM 评估 no→yes，断言发生一次重检索且第二轮查询文本改变；
  首轮 yes 时只检索一次。
- TR-5.2：连续 no 达上限（共 3 次检索），断言循环终止、返回各轮中 Top-1 rerank
  分数最优的一轮，且日志记录"达到最大轮次"。
- TR-5.3：LLM 连续返回非法输出，重试一次后保守放行（视为充足、不重检索、
  /chat/completions 恰好调用 2 次），不抛未处理异常。
- _parse_verdict 单测：JSON / 关键词兜底（含 insufficient 不误判）/ 垃圾输出。
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient
from app.rag.adaptive import AdaptiveRetriever, _parse_verdict
from app.rag.models import Chunk
from app.rag.retriever import RetrievalResult, RetrievalTrace, RetrievedChunk


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
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    )


def _llm(settings: Settings, responses: list[str]) -> tuple[DeepSeekChatClient, dict]:
    """按调用顺序依次返回 responses；调用次数超限立即失败（队列越界）。"""
    calls: dict[str, int] = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _chat_response(responses[calls["n"] - 1])

    return DeepSeekChatClient(settings, transport=httpx.MockTransport(handler)), calls


class FakeRetriever:
    """按轮次返回预设 Top-1 rerank 分数的检索结果，并记录每轮查询文本。"""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.queries: list[str] = []

    async def retrieve(
        self, query: str, kb_types: list[str] | None = None, top_k: int | None = None
    ) -> RetrievalResult:
        self.queries.append(query)
        score = self._scores[len(self.queries) - 1]
        chunk = Chunk(
            chunk_id=f"c-{len(self.queries)}",
            source="qa.md",
            kb_type="qa",
            heading="题库",
            text=f"第 {len(self.queries)} 轮召回片段",
            chunk_index=0,
        )
        entry = RetrievedChunk(chunk=chunk, vector_rank=1, rerank_score=score)
        trace = RetrievalTrace(
            original_query=query,
            vector_queries=[query],
            hyde_answer=None,
            vector_hit_count=1,
            bm25_hit_count=0,
            fused_count=1,
            rerank_candidate_count=1,
        )
        return RetrievalResult(chunks=[entry], trace=trace)


def _verdict(sufficient: bool, reason: str = "", improved_query: str = "") -> str:
    return json.dumps(
        {"sufficient": sufficient, "reason": reason, "improved_query": improved_query}
    )


# ---------------------------------------------------------------------------
# _parse_verdict 单测
# ---------------------------------------------------------------------------
def test_parse_verdict_json_and_fallbacks():
    v = _parse_verdict(_verdict(False, "缺少细节", "改进查询"))
    assert v is not None and v.sufficient is False
    assert v.reason == "缺少细节" and v.improved_query == "改进查询"

    assert _parse_verdict(_verdict(True)).sufficient is True
    assert _parse_verdict("是，上下文充足").sufficient is True
    assert _parse_verdict("否，上下文不足").sufficient is False
    assert _parse_verdict("yes").sufficient is True
    assert _parse_verdict("no").sufficient is False
    # insufficient 不得被子串 "sufficient" 误判为充足；cannot 不得被 "no" 误判
    assert _parse_verdict("insufficient").sufficient is False
    assert _parse_verdict("无法判断@@") is None
    assert _parse_verdict('{"broken": ') is None


# ---------------------------------------------------------------------------
# TR-5.1 no→yes 发生一次重检索；首轮 yes 只检索一次
# ---------------------------------------------------------------------------
async def test_re_retrieve_once_when_verdict_no_then_yes():
    settings = _settings()
    llm, calls = _llm(
        settings,
        [
            _verdict(False, "缺少布隆过滤器细节", "Redis 缓存穿透 布隆过滤器"),
            _verdict(True, "上下文充足"),
        ],
    )
    retriever = FakeRetriever(scores=[0.5, 0.9])
    adaptive = AdaptiveRetriever(settings, retriever, llm)

    result = await adaptive.retrieve("Redis 缓存穿透怎么解决？")

    assert retriever.queries == ["Redis 缓存穿透怎么解决？", "Redis 缓存穿透 布隆过滤器"]
    assert result.rounds == 2 and result.stopped_reason == "sufficient"
    assert result.retrieval.chunks[0].rerank_score == 0.9  # 返回最优轮
    assert calls["n"] == 2


async def test_first_round_yes_retrieves_only_once():
    settings = _settings()
    llm, calls = _llm(settings, [_verdict(True, "充足")])
    retriever = FakeRetriever(scores=[0.7])
    adaptive = AdaptiveRetriever(settings, retriever, llm)

    result = await adaptive.retrieve("什么是 GIL？")

    assert retriever.queries == ["什么是 GIL？"]
    assert result.rounds == 1 and result.stopped_reason == "sufficient"
    assert calls["n"] == 1


async def test_fallback_query_guarantees_text_change():
    """评估未给出改进查询时，用"原查询 + 理由"拼接兜底，保证查询文本改变。"""
    settings = _settings()
    llm, _ = _llm(
        settings,
        [_verdict(False, "缺少布隆过滤器细节"), _verdict(True)],
    )
    retriever = FakeRetriever(scores=[0.4, 0.6])
    adaptive = AdaptiveRetriever(settings, retriever, llm)

    await adaptive.retrieve("Redis 缓存穿透怎么解决？")

    assert retriever.queries[1] == "Redis 缓存穿透怎么解决？ 缺少布隆过滤器细节"
    assert retriever.queries[1] != retriever.queries[0]


# ---------------------------------------------------------------------------
# TR-5.2 连续 no 达上限：终止、返回最优轮、日志记录"达到最大轮次"
# ---------------------------------------------------------------------------
async def test_max_rounds_returns_best_and_logs(caplog: pytest.LogCaptureFixture):
    settings = _settings(adaptive_max_rewrite_rounds=2)
    llm, calls = _llm(
        settings,
        [
            _verdict(False, "r0 不足", "改进1"),
            _verdict(False, "r1 不足", "改进2"),
            _verdict(False, "r2 不足", "改进3"),
        ],
    )
    retriever = FakeRetriever(scores=[0.3, 0.8, 0.5])  # 第 2 轮最优
    adaptive = AdaptiveRetriever(settings, retriever, llm)

    with caplog.at_level(logging.INFO, logger="app.rag.adaptive"):
        result = await adaptive.retrieve("原查询")

    assert retriever.queries == ["原查询", "改进1", "改进2"]  # 共 3 次检索后终止
    assert result.rounds == 3 and result.stopped_reason == "max_rounds_reached"
    assert result.retrieval.chunks[0].rerank_score == 0.8  # 各轮中最优
    assert calls["n"] == 3
    assert "达到最大轮次" in caplog.text


# ---------------------------------------------------------------------------
# TR-5.3 连续非法输出：重试一次后保守放行，恰好 2 次 chat 调用
# ---------------------------------------------------------------------------
async def test_invalid_output_retry_once_then_conservative_pass(
    caplog: pytest.LogCaptureFixture,
):
    settings = _settings()
    llm, calls = _llm(settings, ["无法判断@@", "@@仍然无法判断"])
    retriever = FakeRetriever(scores=[0.7])
    adaptive = AdaptiveRetriever(settings, retriever, llm)

    with caplog.at_level(logging.INFO, logger="app.rag.adaptive"):
        result = await adaptive.retrieve("什么是 GIL？")

    assert retriever.queries == ["什么是 GIL？"]  # 保守放行，不重检索
    assert result.rounds == 1 and result.stopped_reason == "sufficient"
    assert calls["n"] == 2  # 重试一次后放行
    assert "保守放行" in caplog.text


# ---------------------------------------------------------------------------
# 真实集成：Qdrant（73 chunks）+ SiliconFlow + DeepSeek 评估（可选证据）
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_integration_real_adaptive_retrieval(caplog: pytest.LogCaptureFixture):
    from app.config.settings import get_settings
    from app.core.embedding import SiliconFlowEmbedder
    from app.core.reranker import SiliconFlowReranker
    from app.rag.retriever import Retriever
    from app.storage.qdrant_store import QdrantStore

    settings = get_settings()  # 读取真实 .env
    if not (settings.deepseek_api_key and settings.siliconflow_api_key):
        pytest.skip("未配置真实 API key，跳过集成测试")
    store = QdrantStore(settings)
    embedder = SiliconFlowEmbedder(settings)
    reranker = SiliconFlowReranker(settings)
    llm = DeepSeekChatClient(settings)
    try:
        retriever = Retriever(settings, store, embedder, reranker, llm=llm)
        adaptive = AdaptiveRetriever(settings, retriever, llm)
        with caplog.at_level(logging.INFO, logger="app.rag.adaptive"):
            result = await adaptive.retrieve(
                "Redis 缓存穿透和缓存雪崩有什么区别？如何防范？", kb_types=["qa"], top_k=5
            )
        assert result.retrieval.chunks, "真实自适应检索应返回结果"
        assert result.stopped_reason in {"sufficient", "max_rounds_reached"}
        assert 1 <= result.rounds <= settings.adaptive_max_rewrite_rounds + 1
        assert len(result.verdicts) == result.rounds  # 每轮均有判定记录
        assert "自适应检索第 1 轮" in caplog.text
    finally:
        await embedder.close()
        await reranker.close()
        await llm.close()
        store.close()
