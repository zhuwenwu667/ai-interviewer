"""API 依赖装配：长生命周期服务对象集中容器。

- 首次访问时按 settings 装配一次并缓存于 ``app.state.services``（惰性装配，避免
  健康检查 / 测试路径无谓初始化 Qdrant 文件锁等资源）；
- 测试可直接赋值 ``app.state.services`` 注入 mock 容器（mock llm / FakeAdaptive）；
- 支持前端请求头 ``X-DeepSeek-Key`` / ``X-SiliconFlow-Key`` 临时覆盖密钥。
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.agent.graph import ReportGenerator
from app.agent.report import build_report_generator
from app.config import get_settings
from app.config.settings import Settings
from app.core.embedding import SiliconFlowEmbedder
from app.core.llm import DeepSeekChatClient
from app.core.reranker import SiliconFlowReranker
from app.rag.adaptive import AdaptiveRetriever
from app.rag.retriever import Retriever
from app.storage.db import SessionRepository
from app.storage.qdrant_store import QdrantStore

# 前端可选的密钥覆盖请求头（值仅存在于内存，不落盘）
DEEPSEEK_KEY_HEADER = "X-DeepSeek-Key"
SILICONFLOW_KEY_HEADER = "X-SiliconFlow-Key"


@dataclass
class ServiceContainer:
    """一次装配全程复用的服务对象（HTTP 客户端 / 向量库 / 仓储 / 检索器）。"""

    settings: Settings
    repo: SessionRepository
    store: QdrantStore
    embedder: SiliconFlowEmbedder
    reranker: SiliconFlowReranker
    retriever: Retriever
    adaptive: AdaptiveRetriever
    llm: DeepSeekChatClient
    report_generator: ReportGenerator


def build_services(settings: Settings) -> ServiceContainer:
    """生产装配：真实模型客户端 + 本地/远端 Qdrant + SQLite 仓储 + 混合检索。"""
    llm = DeepSeekChatClient(settings)
    embedder = SiliconFlowEmbedder(settings)
    reranker = SiliconFlowReranker(settings)
    store = QdrantStore(settings)
    retriever = Retriever(settings, store, embedder, reranker, llm)
    adaptive = AdaptiveRetriever(settings, retriever, llm)
    return ServiceContainer(
        settings=settings,
        repo=SessionRepository(settings.sqlite_path_abs),
        store=store,
        embedder=embedder,
        reranker=reranker,
        retriever=retriever,
        adaptive=adaptive,
        llm=llm,
        report_generator=build_report_generator(settings, llm),
    )


def _with_key_overrides(
    settings: Settings, deepseek_key: str, siliconflow_key: str
) -> Settings:
    """用请求头密钥生成临时配置副本（仅覆盖密钥，其余沿用服务端配置）。"""
    updates: dict[str, str] = {}
    if deepseek_key:
        updates["deepseek_api_key"] = deepseek_key
    if siliconflow_key:
        updates["siliconflow_api_key"] = siliconflow_key
    return settings.model_copy(update=updates) if updates else settings


def build_override_services(
    base: ServiceContainer, settings: Settings
) -> ServiceContainer:
    """按覆盖后的配置重建模型客户端与检索器；复用基础容器的向量库与仓储。"""
    llm = DeepSeekChatClient(settings)
    embedder = SiliconFlowEmbedder(settings)
    reranker = SiliconFlowReranker(settings)
    retriever = Retriever(settings, base.store, embedder, reranker, llm)
    adaptive = AdaptiveRetriever(settings, retriever, llm)
    return ServiceContainer(
        settings=settings,
        repo=base.repo,
        store=base.store,
        embedder=embedder,
        reranker=reranker,
        retriever=retriever,
        adaptive=adaptive,
        llm=llm,
        report_generator=build_report_generator(settings, llm),
    )


def get_services(request: Request) -> ServiceContainer:
    """FastAPI 依赖：取已注入容器，否则按当前 settings 惰性装配。

    前端可在请求头携带 ``X-DeepSeek-Key`` / ``X-SiliconFlow-Key`` 覆盖服务端
    ``.env`` 中的密钥（用于"页面填写 key"场景）：此时按密钥组合装配并缓存一个
    仅替换模型客户端的容器（向量库 / 仓储等有状态资源复用基础容器，避免重复
    打开 Qdrant 文件锁）。密钥只存在于进程内存，不落盘、不写日志。
    """
    services = getattr(request.app.state, "services", None)
    if services is None:
        services = build_services(get_settings())
        request.app.state.services = services

    deepseek_key = (request.headers.get(DEEPSEEK_KEY_HEADER) or "").strip()
    siliconflow_key = (request.headers.get(SILICONFLOW_KEY_HEADER) or "").strip()
    if not deepseek_key and not siliconflow_key:
        return services

    cache: dict[tuple[str, str], ServiceContainer] = getattr(
        request.app.state, "override_cache", None
    ) or {}
    request.app.state.override_cache = cache
    key = (deepseek_key, siliconflow_key)
    if key not in cache:
        overridden = _with_key_overrides(services.settings, deepseek_key, siliconflow_key)
        cache[key] = build_override_services(services, overridden)
    return cache[key]

