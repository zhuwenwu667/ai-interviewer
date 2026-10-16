"""RAGAS 适配：DeepSeek 作评判 LLM、SiliconFlow bge-m3 作评判 embedding。

ragas 0.4 的 collections 指标只接受其"现代"工厂封装：
``ragas.llms.llm_factory``（instructor 结构化输出）与
``ragas.embeddings.embedding_factory(interface="modern")``。二者均接收预构造的
``openai.AsyncOpenAI`` 客户端，因此复用项目已有的 OpenAI 兼容端点
（``deepseek_base_url`` / ``siliconflow_base_url``）。
四项指标：

- faithfulness：答案陈述能否由 contexts 推出（幻觉）；
- context_precision：相关证据是否排在前面（精排质量）；
- context_recall：ground_truth 要点是否都被 contexts 覆盖（召回完整性）；
- answer_relevancy：答案是否切题。

``oos`` 诱导题不参与 RAGAS 打分（无事实性 reference），其拒答表现由
``app.eval.refusal`` 单独统计。
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from app.config.settings import Settings
from app.eval.runner import RowOutcome

logger = logging.getLogger(__name__)

METRIC_KEYS = (
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
)
# TR-11.2 通过阈值（context_precision 仅报告，不设硬性门槛）
THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.85,
    "context_precision": 0.8,
    "context_recall": 0.8,
    "answer_relevancy": 0.85,
}


def _require_ragas() -> None:
    try:
        import ragas  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 仅缺依赖时触发
        raise RuntimeError(
            "未安装评测依赖，请先执行：uv sync --extra eval"
        ) from exc


def build_judges(settings: Settings) -> tuple[Any, Any]:
    """构建 RAGAS 使用的评判 LLM（instructor 工厂）与 embeddings 包装器。

    ragas 0.4 的 collections 指标要求基于 instructor 的现代 LLM 封装，因此用
    ``llm_factory`` 并传入指向 DeepSeek OpenAI 兼容端点的 ``AsyncOpenAI``
    （DeepSeek 支持 function calling，instructor 默认 TOOLS 模式可用）。
    """
    _require_ragas()
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    chat_client = AsyncOpenAI(
        api_key=settings.deepseek_api_key or "missing-key",
        base_url=settings.deepseek_base_url,
        timeout=120,
        max_retries=5,
    )
    judge_llm = llm_factory(
        settings.eval_model,
        provider="openai",
        client=chat_client,
        temperature=0,
        # ragas instructor 默认 1024，长答案的陈述/裁决抽取会被截断（单元格缺失）
        max_tokens=4096,
    )
    embedding_client = AsyncOpenAI(
        api_key=settings.siliconflow_api_key or "missing-key",
        base_url=settings.siliconflow_base_url,
        timeout=120,
        max_retries=5,
    )
    judge_embeddings = embedding_factory(
        "openai",
        model=settings.embedding_model,
        client=embedding_client,
        interface="modern",
    )
    return judge_llm, judge_embeddings


def _to_score(value: Any) -> float | None:
    """RAGAS 异常单元格可能为 NaN/None；统一转 None。"""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


async def run_ragas_metrics(
    rows: list[RowOutcome],
    settings: Settings,
    *,
    max_workers: int = 8,
    show_progress: bool = False,
) -> dict[str, dict[str, float | None]]:
    """对非 oos 且未报错的样例跑四项指标，返回 ``{row.id: {metric: score}}``。

    ragas 0.4 的 collections 指标是 v2 体系（``ascore(**kwargs)``），与旧的
    ``evaluate``/``aevaluate`` 入口不兼容，因此这里直接并发调用每个指标的
    ``ascore``。信号量限流；单个 (样例, 指标) 评判失败记为 None，不中断整批。
    """
    _require_ragas()
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    scored_rows = [
        r
        for r in rows
        if r.category != "oos" and r.error is None and r.answer
    ]
    if not scored_rows:
        return {}

    judge_llm, judge_embeddings = build_judges(settings)
    metric_objects = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=judge_llm, embeddings=judge_embeddings
        ),
    }

    def _kwargs(metric_key: str, row: RowOutcome) -> dict[str, Any]:
        if metric_key == "faithfulness":
            return {
                "user_input": row.question,
                "response": row.answer,
                "retrieved_contexts": row.contexts,
            }
        if metric_key == "answer_relevancy":
            return {"user_input": row.question, "response": row.answer}
        return {
            "user_input": row.question,
            "reference": row.ground_truth,
            "retrieved_contexts": row.contexts,
        }

    semaphore = asyncio.Semaphore(max_workers)

    async def _score_one(row: RowOutcome, metric_key: str) -> float | None:
        async with semaphore:
            try:
                result = await metric_objects[metric_key].ascore(
                    **_kwargs(metric_key, row)
                )
            except Exception as exc:  # noqa: BLE001 - 单格失败容错
                logger.warning("RAGAS 评判失败 %s/%s: %s", row.id, metric_key, exc)
                return None
        return _to_score(getattr(result, "value", None))

    tasks = {
        (row.id, key): _score_one(row, key)
        for row in scored_rows
        for key in METRIC_KEYS
    }
    values = await asyncio.gather(*tasks.values())
    scores: dict[str, dict[str, float | None]] = {
        row.id: {key: None for key in METRIC_KEYS} for row in scored_rows
    }
    done = 0
    for (row_id, key), value in zip(tasks.keys(), values, strict=True):
        scores[row_id][key] = value
        done += 1
        if show_progress and done % len(METRIC_KEYS) == 0:
            logger.info("RAGAS 评判进度 %d/%d 行", done // len(METRIC_KEYS), len(scored_rows))
    return scores
