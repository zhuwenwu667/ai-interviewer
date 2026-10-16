"""DeepSeek / SiliconFlow 客户端测试：异常分类、重试、流式、维度与排序。

全部使用 httpx.MockTransport 注入响应，不发起真实网络请求。
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config.settings import Settings
from app.core.embedding import SiliconFlowEmbedder
from app.core.exceptions import (
    AuthenticationError,
    BadRequestError,
    ModelNotFoundError,
    RateLimitError,
    UpstreamServerError,
    UpstreamTimeoutError,
)
from app.core.llm import DeepSeekChatClient
from app.core.reranker import SiliconFlowReranker


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        deepseek_api_key="sk-test-deepseek",
        deepseek_base_url="https://ds.test/v1",
        siliconflow_api_key="sk-test-sf",
        siliconflow_base_url="https://sf.test/v1",
    )


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch):
    """跳过重试的指数退避等待。"""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.core.http_utils.asyncio.sleep", _instant)


def _chat_completion(content: str = "pong") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    )


def _error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message, "type": "api_error"}})


def _make_handler(counters: dict[str, int], routes: dict[str, list]):
    """按路径分发一组预置响应（依次弹出）；未预置时报错。"""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counters[path] = counters.get(path, 0) + 1
        queue = routes[path]
        if isinstance(queue, Exception):
            raise queue
        return queue.pop(0) if queue else httpx.Response(500, text="queue empty")

    return handler


@pytest.mark.asyncio
async def test_chat_success_returns_content():
    counters: dict[str, int] = {}
    transport = httpx.MockTransport(
        _make_handler(counters, {"/v1/chat/completions": [_chat_completion("你好")]})
    )
    client = DeepSeekChatClient(_settings(), transport=transport)
    result = await client.chat([{"role": "user", "content": "hi"}], purpose="eval")
    assert result.content == "你好"
    assert result.prompt_tokens == 5
    assert counters["/v1/chat/completions"] == 1
    await client.close()


@pytest.mark.asyncio
async def test_chat_401_does_not_retry():
    counters: dict[str, int] = {}
    transport = httpx.MockTransport(
        _make_handler(counters, {"/v1/chat/completions": [_error(401, "bad key")]})
    )
    client = DeepSeekChatClient(_settings(), transport=transport)
    with pytest.raises(AuthenticationError) as exc_info:
        await client.chat([{"role": "user", "content": "hi"}])
    assert "401" in str(exc_info.value)
    # 401 立即失败，不得重试
    assert counters["/v1/chat/completions"] == 1
    # 异常消息不得回显密钥
    assert "sk-test-deepseek" not in str(exc_info.value)
    await client.close()


@pytest.mark.asyncio
async def test_chat_429_retries_then_succeeds():
    counters: dict[str, int] = {}
    transport = httpx.MockTransport(
        _make_handler(
            counters,
            {"/v1/chat/completions": [_error(429, "limit"), _error(429, "limit"),
                                   _chat_completion("ok")]},
        )
    )
    client = DeepSeekChatClient(_settings(), transport=transport, max_retries=2)
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result.content == "ok"
    # 初始 + 2 次重试
    assert counters["/v1/chat/completions"] == 3
    await client.close()


@pytest.mark.asyncio
async def test_chat_429_exhausted_raises_rate_limit():
    counters: dict[str, int] = {}
    transport = httpx.MockTransport(
        _make_handler(
            counters,
            {"/v1/chat/completions": [_error(429, "limit") for _ in range(3)]},
        )
    )
    client = DeepSeekChatClient(_settings(), transport=transport, max_retries=2)
    with pytest.raises(RateLimitError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert counters["/v1/chat/completions"] == 3
    await client.close()


@pytest.mark.asyncio
async def test_chat_timeout_is_retried():
    counters: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        counters[path] = counters.get(path, 0) + 1
        if counters[path] < 3:
            raise httpx.ConnectTimeout("simulated timeout")
        return _chat_completion("late")

    client = DeepSeekChatClient(
        _settings(), transport=httpx.MockTransport(handler), max_retries=2
    )
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result.content == "late"
    assert counters["/v1/chat/completions"] == 3
    await client.close()


@pytest.mark.asyncio
async def test_stream_chat_yields_tokens():
    def sse_chunk(content: str) -> dict:
        return {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }

    body = (
        f"data: {json.dumps(sse_chunk('你'), ensure_ascii=False)}\n\n"
        f"data: {json.dumps(sse_chunk('好'), ensure_ascii=False)}\n\n"
        "data: [DONE]\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body.encode("utf-8")
        )

    client = DeepSeekChatClient(_settings(), transport=httpx.MockTransport(handler))
    pieces = [chunk async for chunk in client.stream_chat(
        [{"role": "user", "content": "hi"}]
    )]
    assert "".join(pieces) == "你好"
    await client.close()


@pytest.mark.asyncio
async def test_embedding_returns_1024_dim_and_keeps_order():
    settings = _settings()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        n = len(payload["input"])
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(i)] * 1024}
                    for i in range(n)
                ],
                "model": settings.embedding_model,
                "usage": {"prompt_tokens": n, "total_tokens": n},
            },
        )

    embedder = SiliconFlowEmbedder(settings, transport=httpx.MockTransport(handler))
    vectors = await embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 1024 for v in vectors)
    # 顺序与输入一致
    assert vectors[0][0] == 0.0
    assert vectors[2][0] == 2.0
    await embedder.close()


@pytest.mark.asyncio
async def test_rerank_returns_descending_scores():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.40},
            ]},
        )

    reranker = SiliconFlowReranker(_settings(), transport=httpx.MockTransport(handler))
    items = await reranker.rerank("q", ["doc0", "doc1"], top_n=2)
    assert [it.index for it in items] == [1, 0]
    assert items[0].score > items[1].score
    await reranker.close()


@pytest.mark.asyncio
async def test_rerank_401_maps_to_authentication_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return _error(401, "bad key")

    reranker = SiliconFlowReranker(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(AuthenticationError):
        await reranker.rerank("q", ["doc"])
    await reranker.close()


@pytest.mark.asyncio
async def test_timeout_on_non_openai_endpoint_is_classified():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow rerank")

    reranker = SiliconFlowReranker(
        _settings(), transport=httpx.MockTransport(handler), max_retries=0
    )
    with pytest.raises(UpstreamTimeoutError):
        await reranker.rerank("q", ["doc"])
    await reranker.close()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (404, ModelNotFoundError),
        (400, BadRequestError),
        (429, RateLimitError),
        (500, UpstreamServerError),
        (302, BadRequestError),  # 未预期的状态码也归类为参数错误
    ],
)
@pytest.mark.asyncio
async def test_rerank_status_codes_map_to_domain_errors(status, expected):
    """各 HTTP 状态码映射到统一的领域异常（max_retries=0 避免重试干扰）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return _error(status, "boom")

    reranker = SiliconFlowReranker(
        _settings(), transport=httpx.MockTransport(handler), max_retries=0
    )
    with pytest.raises(expected):
        await reranker.rerank("q", ["doc"])
    await reranker.close()


@pytest.mark.asyncio
async def test_rerank_empty_documents_returns_empty_without_request():
    """空文档列表直接返回空，不发起 HTTP 请求。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("空文档时不应发起请求")

    reranker = SiliconFlowReranker(_settings(), transport=httpx.MockTransport(handler))
    assert await reranker.rerank("q", []) == []
    await reranker.close()

