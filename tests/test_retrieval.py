"""混合检索与查询改写测试（Task 4）。

- TR-4.1：RRF 纯函数手算对照（含 k 值影响）；
- 改写器单测：JSON 解析 / 行解析兜底 / 开关 / 失败降级（httpx.MockTransport 注入）；
- TR-4.3：mock LLM 验证 Multi-Query / HyDE 开关分别生效或关闭时的检索调用次数与查询内容；
- 全链路 mock：tmp 本地 Qdrant + 确定性 FakeEmbedder + FakeReranker，
  验证融合排序、精排降序、kb_type 过滤与溯源字段；
- TR-4.2：integration 标记，真实 Qdrant + SiliconFlow（+ DeepSeek 改写）。
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.config.settings import Settings, get_settings
from app.core.embedding import SiliconFlowEmbedder
from app.core.llm import DeepSeekChatClient
from app.core.reranker import RerankItem, SiliconFlowReranker
from app.rag.fusion import rrf_fuse
from app.rag.models import Chunk
from app.rag.retriever import Retriever
from app.rag.rewrite import QueryRewriter
from app.storage.qdrant_store import QdrantStore

DIM = 8


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------
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


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch):
    """跳过重试的指数退避等待。"""

    async def _instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.core.http_utils.asyncio.sleep", _instant)


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


def _llm(settings: Settings, handler) -> DeepSeekChatClient:
    return DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))


class FakeEmbedder:
    """关键词映射坐标轴的确定性向量；记录全部输入文本供调用次数/内容断言。"""

    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.inputs.extend(texts)
        vectors = []
        for text in texts:
            vec = [0.0] * DIM
            if "GIL" in text:
                vec[0] = 1.0
            elif "Redis" in text or "缓存" in text:
                vec[1] = 1.0
            else:
                vec[2] = 1.0
            vectors.append(vec)
        return vectors


class FakeReranker:
    """含 GIL 的文档给高分，验证精排结果主导最终排序。"""

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[RerankItem]:
        items = [
            RerankItem(index=i, score=0.9 if "GIL" in doc else 0.1)
            for i, doc in enumerate(documents)
        ]
        items.sort(key=lambda item: item.score, reverse=True)
        return items[: top_n or len(documents)]


@pytest.fixture
def store(tmp_path):
    settings = _settings(
        qdrant_mode="local",
        qdrant_path=tmp_path / "qdrant",
        qdrant_collection="test_retrieval",
        embedding_dim=DIM,
    )
    s = QdrantStore(settings)
    yield s
    s.close()


async def _seed(store: QdrantStore, embedder: FakeEmbedder) -> None:
    chunks = [
        Chunk(
            chunk_id="qa-gil",
            source="qa_python.md",
            kb_type="qa",
            heading="Python 八股 > GIL",
            text="Python 的 GIL（全局解释器锁）导致多线程无法并行执行字节码",
            chunk_index=0,
        ),
        Chunk(
            chunk_id="qa-redis",
            source="qa_backend.md",
            kb_type="qa",
            heading="后端八股 > Redis",
            text="Redis 缓存穿透可用布隆过滤器与空值缓存解决",
            chunk_index=1,
        ),
        Chunk(
            chunk_id="resume-1",
            source="resume.md",
            kb_type="resume",
            heading="项目经历 > RAG 系统",
            text="基于向量检索与 LangGraph 构建智能问答系统",
            chunk_index=0,
        ),
    ]
    vectors = await embedder.embed([c.text for c in chunks])
    store.upsert_chunks(chunks, vectors)


# ---------------------------------------------------------------------------
# TR-4.1 RRF 纯函数手算对照
# ---------------------------------------------------------------------------
def test_rrf_hand_computed():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], k=10)
    # 手算：a = 1/11 + 1/13；b = 1/12 + 1/11；c = 1/13；d = 1/12
    assert [doc for doc, _ in fused] == ["b", "a", "d", "c"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 11 + 1 / 13)
    assert scores["b"] == pytest.approx(1 / 12 + 1 / 11)
    assert scores["c"] == pytest.approx(1 / 13)
    assert scores["d"] == pytest.approx(1 / 12)


def test_rrf_k_value_changes_scores():
    rankings = [["a", "b", "c"], ["b", "d", "a"]]
    scores_k1 = dict(rrf_fuse(rankings, k=1))
    assert scores_k1["a"] == pytest.approx(1 / 2 + 1 / 4)
    assert scores_k1["d"] == pytest.approx(1 / 3)
    scores_k100 = dict(rrf_fuse(rankings, k=100))
    assert scores_k100["a"] == pytest.approx(1 / 101 + 1 / 103)
    # k 越大，相邻名次间的分数差距越平缓
    assert (scores_k1["d"] - scores_k1["c"]) > (scores_k100["d"] - scores_k100["c"])


def test_rrf_empty_deterministic_and_invalid_k():
    assert rrf_fuse([], k=60) == []
    assert rrf_fuse([[]], k=60) == []
    # 两路完全相反的排名产生同分，按 ID 字典序保证确定性
    fused = rrf_fuse([["b", "a"], ["a", "b"]], k=60)
    assert [doc for doc, _ in fused] == ["a", "b"]
    with pytest.raises(ValueError, match="非负数"):
        rrf_fuse([["a"]], k=-1)


# ---------------------------------------------------------------------------
# 改写器单测（MockTransport 注入 LLM 响应）
# ---------------------------------------------------------------------------
async def test_multi_query_json_mode_dedup_and_order():
    seen: dict[str, bool] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["json_mode"] = b"json_object" in request.content
        return _chat_response(
            json.dumps({"queries": ["GIL 的原理是什么", "GIL 的原理是什么", "什么是 GIL？"]})
        )

    settings = _settings(retrieval_enable_multi_query=True, retrieval_multi_query_n=2)
    rewriter = QueryRewriter(_llm(settings, handler), settings)
    queries = await rewriter.multi_query("什么是 GIL？")
    assert seen["json_mode"] is True
    # 原查询恒居首，重复与原文相同的改写被去除
    assert queries == ["什么是 GIL？", "GIL 的原理是什么"]


async def test_multi_query_disabled_makes_no_llm_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("开关关闭时不应调用 LLM")

    settings = _settings(retrieval_enable_multi_query=False)
    rewriter = QueryRewriter(_llm(settings, handler), settings)
    assert await rewriter.multi_query("什么是 GIL？") == ["什么是 GIL？"]


async def test_multi_query_fallback_line_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response("1. GIL 原理详解\n- Python 多线程为何受限\n3、GIL 与多进程")

    settings = _settings(retrieval_enable_multi_query=True, retrieval_multi_query_n=3)
    rewriter = QueryRewriter(_llm(settings, handler), settings)
    queries = await rewriter.multi_query("什么是 GIL？")
    assert queries == ["什么是 GIL？", "GIL 原理详解", "Python 多线程为何受限", "GIL 与多进程"]


async def test_multi_query_llm_failure_degrades_to_original():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    settings = _settings(retrieval_enable_multi_query=True)
    rewriter = QueryRewriter(_llm(settings, handler), settings)
    # 5xx 重试耗尽后降级为原查询，不向外抛异常
    assert await rewriter.multi_query("什么是 GIL？") == ["什么是 GIL？"]


async def test_hyde_enabled_returns_answer_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response("GIL 是 CPython 的全局解释器锁，同一时刻只允许一个线程执行字节码。")

    settings = _settings(retrieval_enable_hyde=True)
    rewriter = QueryRewriter(_llm(settings, handler), settings)
    answer = await rewriter.hyde("什么是 GIL？")
    assert answer is not None and "GIL" in answer


async def test_hyde_disabled_makes_no_llm_call():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("开关关闭时不应调用 LLM")

    settings = _settings(retrieval_enable_hyde=False)
    rewriter = QueryRewriter(_llm(settings, handler), settings)
    assert await rewriter.hyde("什么是 GIL？") is None


# ---------------------------------------------------------------------------
# TR-4.3 检索调用次数与查询内容（mock LLM + 真实 tmp Qdrant + FakeEmbedder）
# ---------------------------------------------------------------------------
async def test_retrieve_with_rewrites_off_uses_single_vector_query(store):
    settings = _settings(retrieval_enable_multi_query=False, retrieval_enable_hyde=False)
    embedder = FakeEmbedder()
    await _seed(store, embedder)
    embedder.inputs.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("改写全关时不应调用 LLM")

    retriever = Retriever(
        settings, store, embedder, FakeReranker(), llm=_llm(settings, handler)
    )
    result = await retriever.retrieve("什么是 GIL？", kb_types=["qa"], top_k=3)
    assert embedder.inputs == ["什么是 GIL？"]  # 仅原查询一路向量检索
    assert result.trace.vector_queries == ["什么是 GIL？"]
    assert result.trace.hyde_answer is None


async def test_retrieve_multi_query_on_expands_vector_queries(store):
    settings = _settings(
        retrieval_enable_multi_query=True, retrieval_multi_query_n=2,
        retrieval_enable_hyde=False,
    )
    embedder = FakeEmbedder()
    await _seed(store, embedder)
    embedder.inputs.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(json.dumps({"queries": ["GIL 对多线程的影响", "CPython 锁机制"]}))

    retriever = Retriever(
        settings, store, embedder, FakeReranker(), llm=_llm(settings, handler)
    )
    result = await retriever.retrieve("什么是 GIL？", kb_types=["qa"], top_k=3)
    # 原查询 + 2 个改写 = 3 路向量检索，embedding 输入与之严格一致
    assert embedder.inputs == ["什么是 GIL？", "GIL 对多线程的影响", "CPython 锁机制"]
    assert result.trace.vector_queries == ["什么是 GIL？", "GIL 对多线程的影响", "CPython 锁机制"]
    top = result.chunks[0]
    assert top.chunk.chunk_id == "qa-gil"
    assert "GIL 对多线程的影响" in top.matched_queries  # 改写查询命中可溯源


async def test_retrieve_hyde_on_adds_answer_vector_path(store):
    settings = _settings(retrieval_enable_multi_query=False, retrieval_enable_hyde=True)
    embedder = FakeEmbedder()
    await _seed(store, embedder)
    embedder.inputs.clear()
    hyde_text = "GIL 是 CPython 解释器的全局锁，CPU 密集任务应改用多进程。"

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(hyde_text)

    retriever = Retriever(
        settings, store, embedder, FakeReranker(), llm=_llm(settings, handler)
    )
    result = await retriever.retrieve("什么是 GIL？", kb_types=["qa"], top_k=3)
    # 向量检索两路：原查询 + HyDE 假设答案
    assert embedder.inputs == ["什么是 GIL？", hyde_text]
    assert result.trace.hyde_used is True
    assert result.trace.hyde_answer == hyde_text
    assert result.chunks[0].chunk.chunk_id == "qa-gil"


# ---------------------------------------------------------------------------
# 全链路 mock：融合排序、精排降序、kb_type 过滤、溯源字段
# ---------------------------------------------------------------------------
async def test_full_pipeline_ordering_and_provenance(store):
    settings = _settings(retrieval_enable_multi_query=False, retrieval_enable_hyde=False)
    embedder = FakeEmbedder()
    await _seed(store, embedder)
    retriever = Retriever(settings, store, embedder, FakeReranker(), llm=None)
    result = await retriever.retrieve("什么是 GIL？", top_k=3)

    chunks = result.chunks
    assert chunks[0].chunk.chunk_id == "qa-gil"  # 精排最高分居首
    scores = [c.rerank_score for c in chunks]
    assert scores == sorted(scores, reverse=True)

    top = chunks[0]
    assert top.vector_rank == 1 and top.bm25_rank == 1  # 双路同时命中
    assert top.vector_score == pytest.approx(1.0)
    expected_rrf = 1 / (settings.retrieval_rrf_k + 1) * 2  # 两路均第 1 名
    assert top.rrf_score == pytest.approx(expected_rrf)
    assert top.matched_queries == ["什么是 GIL？"]
    assert result.trace.bm25_hit_count == 1


async def test_kb_type_filter_excludes_other_types(store):
    settings = _settings(retrieval_enable_multi_query=False, retrieval_enable_hyde=False)
    embedder = FakeEmbedder()
    await _seed(store, embedder)
    retriever = Retriever(settings, store, embedder, FakeReranker(), llm=None)
    result = await retriever.retrieve("什么是 GIL？", kb_types=["resume"], top_k=3)
    assert result.chunks, "过滤后仍应返回简历库内容"
    assert all(c.chunk.kb_type == "resume" for c in result.chunks)


# ---------------------------------------------------------------------------
# TR-4.2 集成测试：真实 Qdrant（已摄入数据）+ SiliconFlow + DeepSeek 改写
# ---------------------------------------------------------------------------
@pytest.mark.integration
async def test_integration_real_retrieval():
    settings = get_settings()  # 读取真实 .env
    if not (settings.deepseek_api_key and settings.siliconflow_api_key):
        pytest.skip("未配置真实 API key，跳过集成测试")
    store = QdrantStore(settings)
    embedder = SiliconFlowEmbedder(settings)
    reranker = SiliconFlowReranker(settings)
    llm = DeepSeekChatClient(settings)
    try:
        retriever = Retriever(settings, store, embedder, reranker, llm=llm)
        result = await retriever.retrieve(
            "Redis 缓存穿透和缓存雪崩有什么区别？如何防范？", kb_types=["qa"], top_k=5
        )
        assert result.chunks, "真实检索应返回结果"
        scores = [c.rerank_score for c in result.chunks]
        assert scores == sorted(scores, reverse=True), "Top-K 应按 rerank_score 降序"
        assert any(c.hit_by_vector and c.hit_by_bm25 for c in result.chunks), (
            "溯源中应同时看到向量路与 BM25 路的命中文档"
        )
        assert all(c.chunk.kb_type == "qa" for c in result.chunks)

        # kb_type 过滤：只查简历库，结果不得含其他类型
        filtered = await retriever.retrieve("介绍一个你做过的 RAG 项目", kb_types=["resume"], top_k=3)
        assert filtered.chunks
        assert all(c.chunk.kb_type == "resume" for c in filtered.chunks)
    finally:
        await embedder.close()
        await reranker.close()
        await llm.close()
        store.close()
