"""检索管线消融对照实验（P0）。

对同一份评测集逐档叠加检索组件，输出「检索质量 + 延迟 + token 成本」对照表：

    1. dense      仅向量召回（原查询），按相似度取 Top-K
    2. +bm25      叠加关键词路（朴素拼接融合，不重排）
    3. +rrf       融合算法由朴素拼接升级为 RRF
    4. +rerank    叠加交叉编码器精排
    5. +agentic   叠加多查询改写 + 自评重检索（AdaptiveRetriever）

用法（项目根目录）：
    uv run python scripts/run_ablation.py                     # 跑全部 5 档
    uv run python scripts/run_ablation.py --configs dense,+agentic
    uv run python scripts/run_ablation.py --limit 8 --quiet

产出：
    data/eval/reports/ablation_YYYYMMDD_HHMMSS.{json,md}
    data/eval/reports/ablation_traces.jsonl（各档 span/token 明细，供核对）

说明：token 数取自本管线自身的埋点（改写 / 自评 / 生成），不含 RAGAS 评判开销
（评判仅离线评测使用，不进入产品链路）。成本按文件末尾单价常量估算。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

from app.api.deps import build_services  # noqa: E402
from app.config import validate_runtime  # noqa: E402
from app.config.settings import Settings  # noqa: E402
from app.eval.dataset import EvalQuestion, load_dataset, validate_dataset  # noqa: E402
from app.eval.metrics_ragas import (  # noqa: E402
    METRIC_KEYS,
    run_ragas_metrics,
)
from app.eval.report import mean  # noqa: E402
from app.eval.runner import RowOutcome, run_pipeline  # noqa: E402
from app.observability.tracer import JsonlSink, Tracer, reset_tracer, set_tracer  # noqa: E402
from app.rag.adaptive import AdaptiveRetriever  # noqa: E402
from app.rag.loader import SUPPORTED_SUFFIXES  # noqa: E402
from app.rag.retriever import RetrievalStages, Retriever  # noqa: E402

TRACE_FILE = "logs/ablation_traces.jsonl"

# deepseek-chat 官方定价（元 / 百万 token；缓存未命中输入价、输出价）
INPUT_PRICE_PER_M = 2.0
OUTPUT_PRICE_PER_M = 8.0

# (档名, 阶段开关, 是否用 AdaptiveRetriever)
CONFIGS: list[tuple[str, RetrievalStages, bool]] = [
    ("dense", RetrievalStages(rewrite=False, bm25=False, rrf=False, rerank=False), False),
    ("+bm25", RetrievalStages(rewrite=False, bm25=True, rrf=False, rerank=False), False),
    ("+rrf", RetrievalStages(rewrite=False, bm25=True, rrf=True, rerank=False), False),
    ("+rerank", RetrievalStages(rewrite=False, bm25=True, rrf=True, rerank=True), False),
    ("+agentic", RetrievalStages(rewrite=True, bm25=True, rrf=True, rerank=True), True),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索管线消融对照实验")
    parser.add_argument("--dataset", default="data/eval/eval_dataset.json")
    parser.add_argument("--output-dir", default="data/eval/reports")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="每档截断条数（快速试跑）")
    parser.add_argument("--concurrency", type=int, default=4, help="检索+生成并发")
    parser.add_argument("--workers", type=int, default=8, help="RAGAS 评判并发")
    parser.add_argument(
        "--configs",
        default=None,
        help="只跑指定档位，逗号分隔（如 dense,+agentic）",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _discover_known_sources(settings: Settings) -> set[str]:
    root = settings.resolve("data/knowledge")
    return {
        p.name
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    }


def _build_retriever(
    settings: Settings, services, stages: RetrievalStages, agentic: bool
):
    retriever = Retriever(
        settings,
        services.store,
        services.embedder,
        services.reranker,
        services.llm,
        stages=stages,
    )
    if agentic:
        return AdaptiveRetriever(settings, retriever, services.llm)
    return retriever


def _source_hit_rate(rows: list[RowOutcome]) -> float | None:
    hit_rows = [r for r in rows if r.source_hit is not None]
    return (
        sum(1 for r in hit_rows if r.source_hit) / len(hit_rows) if hit_rows else None
    )


def _load_spans(path) -> dict[str, list[dict]]:
    """按 trace_id（=档名）聚合埋点记录。"""
    by_trace: dict[str, list[dict]] = {}
    if not path.exists():
        return by_trace
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_trace.setdefault(rec["trace_id"], []).append(rec)
    return by_trace


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def _summarize_spans(spans: list[dict]) -> dict:
    """从一档的埋点里汇总 token 与关键阶段耗时。"""
    prompt = completion = 0
    llm_calls = 0
    llm_lat: list[float] = []
    rerank_lat: list[float] = []
    rewrite_lat: list[float] = []
    embed_calls = 0
    for rec in spans:
        name = rec["name"]
        fields = rec.get("fields", {})
        dur = rec.get("duration_ms") or 0.0
        if name == "llm.chat":
            llm_calls += 1
            prompt += int(fields.get("prompt_tokens") or 0)
            completion += int(fields.get("completion_tokens") or 0)
            llm_lat.append(dur)
        elif name == "retrieval.rerank":
            rerank_lat.append(dur)
        elif name == "rewrite.multi_query":
            rewrite_lat.append(dur)
        elif name.startswith("embedding"):
            embed_calls += 1
    return {
        "llm_calls": llm_calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "llm_ms_mean": round(statistics.fmean(llm_lat), 1) if llm_lat else None,
        "llm_ms_p95": _percentile(llm_lat, 95),
        "rerank_ms_mean": (
            round(statistics.fmean(rerank_lat), 1) if rerank_lat else None
        ),
        "rewrite_ms_mean": (
            round(statistics.fmean(rewrite_lat), 1) if rewrite_lat else None
        ),
    }


async def _run_config(
    name: str,
    stages: RetrievalStages,
    agentic: bool,
    questions: list[EvalQuestion],
    settings: Settings,
    services,
    args: argparse.Namespace,
) -> dict:
    retriever = _build_retriever(settings, services, stages, agentic)
    trace_path = settings.resolve(TRACE_FILE)
    tracer = Tracer(
        trace_id=name,
        session_id=None,
        sinks=[JsonlSink(trace_path)],
        trace_name=f"ablation:{name}",
    )
    token = set_tracer(tracer)
    print(f"\n=== 档位 {name}（stages={stages}，agentic={agentic}）===")
    try:
        t0 = time.perf_counter()
        rows = await run_pipeline(
            questions,
            retriever,
            services.llm,
            top_k=args.top_k,
            concurrency=args.concurrency,
        )
        pipeline_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        scores = await run_ragas_metrics(
            rows, settings, max_workers=args.workers, show_progress=not args.quiet
        )
        ragas_s = time.perf_counter() - t1
    finally:
        reset_tracer(token)

    in_scope = [r for r in rows if r.category != "oos" and r.error is None]
    metrics = {
        key: mean([scores.get(r.id, {}).get(key) for r in in_scope])
        for key in METRIC_KEYS
    }
    oos_rows = [r for r in rows if r.category == "oos"]
    result = {
        "config": name,
        "stages": stages.__dict__,
        "agentic": agentic,
        "n": len(rows),
        "errors": sum(1 for r in rows if r.error),
        "source_hit_rate": _source_hit_rate(rows),
        "metrics": metrics,
        "oos_refusal_rate": (
            sum(1 for r in oos_rows if r.refusal_detected) / len(oos_rows)
            if oos_rows
            else None
        ),
        "mean_rounds": mean([r.rounds for r in rows]),
        "mean_contexts": mean([len(r.contexts) for r in rows]),
        "latency": {
            "pipeline_total_s": round(pipeline_s, 1),
            "pipeline_per_q_s": round(pipeline_s / len(rows), 3) if rows else None,
            "ragas_total_s": round(ragas_s, 1),
        },
        "tokens": {},
    }
    spans = _load_spans(trace_path).get(name, [])
    tok = _summarize_spans(spans)
    cost = (
        tok["prompt_tokens"] / 1e6 * INPUT_PRICE_PER_M
        + tok["completion_tokens"] / 1e6 * OUTPUT_PRICE_PER_M
    )
    tok["est_cost_cny_total"] = round(cost, 4)
    tok["est_cost_cny_per_q"] = round(cost / len(rows), 6) if rows else None
    result["tokens"] = tok
    print(
        f"  来源命中率={_fmt_pct(result['source_hit_rate'])} "
        f"faithfulness={_fmt(result['metrics']['faithfulness'])} "
        f"ctx_recall={_fmt(result['metrics']['context_recall'])} "
        f"| 每问 {result['latency']['pipeline_per_q_s']}s "
        f"token {tok['total_tokens']} 成本约 ¥{tok['est_cost_cny_total']}"
    )
    return result


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "N/A"


def _fmt_pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _dash(value: float | None) -> str:
    """表格中缺失值渲染为「—」（该档位未启用该阶段）。"""
    return "—" if value is None else str(value)


def _render_markdown(results: list[dict], n_questions: int) -> str:
    lines = ["# 检索管线消融对照实验", ""]
    lines.append(f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(
        f"- 评测集：{n_questions} 条；并发 concurrency、RAGAS workers 见 JSON；"
        "token 仅统计本管线（改写/自评/生成），不含 RAGAS 评判开销"
    )
    lines.append(
        f"- 成本按 deepseek-chat 计价：输入 ¥{INPUT_PRICE_PER_M}/M、输出 ¥{OUTPUT_PRICE_PER_M}/M（缓存未命中）"
    )
    lines.append("")
    lines.append("## 主对照表")
    lines.append("")
    lines.append(
        "| 档位 | 来源命中率 | faithfulness | ctx_precision | ctx_recall | answer_relevancy "
        "| 诱导题拒答率 | 平均轮数 | 每问耗时(s) | LLM调用 | 总token | 成本(元/问) |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['config']} | {_fmt_pct(r['source_hit_rate'])} | {_fmt(m['faithfulness'])} "
            f"| {_fmt(m['context_precision'])} | {_fmt(m['context_recall'])} "
            f"| {_fmt(m['answer_relevancy'])} | {_fmt_pct(r['oos_refusal_rate'])} "
            f"| {_fmt(r['mean_rounds'])} | {r['latency']['pipeline_per_q_s']} "
            f"| {r['tokens']['llm_calls']} | {r['tokens']['total_tokens']} "
            f"| {r['tokens']['est_cost_cny_per_q']} |"
        )
    lines.append("")
    lines.append("## 阶段耗时（毫秒，均值 / P95）")
    lines.append("")
    lines.append("| 档位 | 改写 | rerank | LLM 调用均值 | LLM 调用 P95 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in results:
        t = r["tokens"]
        lines.append(
            f"| {r['config']} | {_dash(t['rewrite_ms_mean'])} | {_dash(t['rerank_ms_mean'])} "
            f"| {_dash(t['llm_ms_mean'])} | {_dash(t['llm_ms_p95'])} |"
        )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    args = _parse_args()
    settings = validate_runtime(require=("chat", "embedding"))
    dataset = load_dataset(settings.resolve(args.dataset))
    validate_dataset(dataset, known_sources=_discover_known_sources(settings))
    questions = dataset.filter(limit=args.limit)

    selected = CONFIGS
    if args.configs:
        wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
        selected = [c for c in CONFIGS if c[0] in wanted]
        if not selected:
            raise SystemExit(f"未知档位：{wanted}（可选 {[c[0] for c in CONFIGS]}）")
    print(
        f"评测集 {len(dataset.questions)} 条，本次每档执行 {len(questions)} 条，"
        f"档位：{[c[0] for c in selected]}"
    )

    services = build_services(settings)
    results: list[dict] = []
    try:
        for name, stages, agentic in selected:
            results.append(
                await _run_config(
                    name, stages, agentic, questions, settings, services, args
                )
            )
    finally:
        await services.llm.close()
        await services.embedder.close()
        services.store.close()

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_questions": len(questions),
        "pricing": {
            "input_cny_per_m": INPUT_PRICE_PER_M,
            "output_cny_per_m": OUTPUT_PRICE_PER_M,
            "model": settings.eval_model,
        },
        "configs": results,
    }
    output_dir = settings.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ablation_{stamp}.json"
    md_path = output_dir / f"ablation_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(results, len(questions)), encoding="utf-8")
    print(f"\n报告已写出：\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[消融实验启动失败] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
