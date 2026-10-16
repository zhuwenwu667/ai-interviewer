"""RAGAS 离线评测一键脚本（Task 11 / TR-11.1、TR-11.2）。

用法（项目根目录）：
    uv run python scripts/run_eval.py                      # 全量评测
    uv run python scripts/run_eval.py --category qa --limit 5
    uv run python scripts/run_eval.py --no-adaptive        # 消融：单轮检索
    uv run python scripts/run_eval.py --save-baseline      # 结果同时写为基线

产出：
    data/eval/reports/eval_YYYYMMDD_HHMMSS.{json,md}
    （--save-baseline 时额外写 data/eval/baseline.json，后续运行自动与其对比 Δ）

退出码：0=运行成功（门槛可能未过，见报告）；2=门槛未通过；1=环境/数据集错误。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime

sys.path.insert(0, ".")

from app.api.deps import build_services  # noqa: E402
from app.config import validate_runtime  # noqa: E402
from app.eval.dataset import (  # noqa: E402
    CATEGORIES,
    load_dataset,
    validate_dataset,
)
from app.eval.metrics_ragas import (
    METRIC_KEYS,  # noqa: E402
    run_ragas_metrics,  # noqa: E402
)
from app.eval.report import build_report, render_markdown  # noqa: E402
from app.eval.runner import (
    RowOutcome,  # noqa: E402
    run_pipeline,  # noqa: E402
)
from app.rag.loader import SUPPORTED_SUFFIXES  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS 离线评测：检索+生成+四项指标+基线对比")
    parser.add_argument("--dataset", default="data/eval/eval_dataset.json", help="评测集 JSON 路径")
    parser.add_argument("--output-dir", default="data/eval/reports", help="报告输出目录")
    parser.add_argument("--baseline", default="data/eval/baseline.json", help="基线报告路径")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖检索 final top_k")
    parser.add_argument("--limit", type=int, default=None, help="每类之外的总条数截断")
    parser.add_argument(
        "--category",
        action="append",
        choices=list(CATEGORIES),
        help="只跑指定类别，可多次传入",
    )
    parser.add_argument("--no-adaptive", action="store_true", help="使用单轮 Retriever（消融）")
    parser.add_argument("--concurrency", type=int, default=4, help="检索+生成并发数")
    parser.add_argument("--workers", type=int, default=8, help="RAGAS 评判并发数")
    parser.add_argument("--save-baseline", action="store_true", help="将本次结果另存为基线")
    parser.add_argument("--quiet", action="store_true", help="不打印逐条进度")
    return parser.parse_args()


def _discover_known_sources(settings) -> set[str]:
    root = settings.resolve("data/knowledge")
    names: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            names.add(path.name)
    return names


def _print_row(row: RowOutcome) -> None:
    flag = "拒答" if row.refusal_detected else "作答"
    hit = "" if row.source_hit is None else (" 来源✓" if row.source_hit else " 来源✗")
    status = f"ERROR {row.error}" if row.error else f"{flag}{hit} ctx={len(row.contexts)}"
    print(f"  [{row.category:<6}] {row.id:<10} {status}")


async def main() -> int:
    args = _parse_args()
    settings = validate_runtime(require=("chat", "embedding"))

    dataset = load_dataset(settings.resolve(args.dataset))
    known_sources = _discover_known_sources(settings)
    validate_dataset(dataset, known_sources=known_sources)
    questions = dataset.filter(categories=args.category, limit=args.limit)
    print(
        f"评测集校验通过：{len(dataset.questions)} 条，本次执行 {len(questions)} 条，"
        f"类别 {args.category or '全部'}"
    )

    services = build_services(settings)
    retriever = services.retriever if args.no_adaptive else services.adaptive
    try:
        print("阶段 1/2：检索 + 受控生成 ...")

        def _on_done(row: RowOutcome) -> None:
            if not args.quiet:
                _print_row(row)

        rows = await run_pipeline(
            questions,
            retriever,
            services.llm,
            top_k=args.top_k,
            concurrency=args.concurrency,
            on_row_done=_on_done,
        )

        print("阶段 2/2：RAGAS 四项指标评判（DeepSeek 评判，bge-m3 embedding）...")
        scores = await run_ragas_metrics(
            rows, settings, max_workers=args.workers, show_progress=not args.quiet
        )
    finally:
        await services.llm.close()
        await services.embedder.close()
        services.store.close()

    baseline_path = settings.resolve(args.baseline)
    baseline = None
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"已载入基线：{baseline_path}（{baseline.get('generated_at')}）")

    top_k = args.top_k or settings.retrieval_final_top_k
    report = build_report(
        dataset,
        rows,
        scores,
        config={
            "use_adaptive": not args.no_adaptive,
            "top_k": top_k,
            "concurrency": args.concurrency,
            "ragas_workers": args.workers,
            "judge_model": settings.eval_model,
            "generator_model": settings.question_model,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
        },
        baseline=baseline,
    )

    output_dir = settings.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"eval_{stamp}.json"
    md_path = output_dir / f"eval_{stamp}.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\n报告已写出：\n  {json_path}\n  {md_path}")

    if args.save_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"基线已更新：{baseline_path}")

    overall = report["metrics"]["overall"]
    print("\n== 总体指标 ==")
    for key in METRIC_KEYS:
        gated = key in report["thresholds"]
        threshold = report["thresholds"].get(key)
        delta = report["baseline"]["deltas"].get(key)
        delta_txt = f" Δ={delta:+.3f}" if delta is not None else ""
        threshold_txt = f" (阈值 {threshold:.2f})" if gated else ""
        value = overall.get(key)
        value_txt = f"{value:.3f}" if value is not None else "N/A"
        print(f"  {key:<18}{value_txt}{threshold_txt}{delta_txt}")
    retrieval = report["retrieval"]

    def _pct(value: float | None) -> str:
        return f"{value:.1%}" if value is not None else "N/A"

    print(
        f"  source_hit_rate   {_pct(retrieval['source_hit_rate'])}\n"
        f"  oos_refusal_rate  {_pct(retrieval['oos_refusal_rate'])}"
    )
    if report["passed"]:
        print("\nTR-11.2 门槛：全部通过 ✅")
        return 0
    print("\nTR-11.2 门槛：未全部通过 ❌（详见报告，必要时回 Task 4/5 调参复评）")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[评测启动失败] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
