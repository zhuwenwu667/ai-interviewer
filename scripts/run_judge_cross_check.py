"""异源 judge 交叉验证：同一批生成结果，用两个 judge 各评一次，量化自评偏置。

背景：项目默认的 RAGAS judge 与面试评分模型同源（均为 ``deepseek-chat``），
存在"自评偏置"质疑——即自评模型可能系统性抬高与自己风格一致的答案的分数。
本脚本用**异源 judge**（默认 SiliconFlow 上的 Qwen2.5-72B-Instruct，另一厂商/另一模型族）
对**完全相同**的生成结果再打一遍分，对比 per-metric 均值差与逐条一致率。

用法（项目根目录）：
    uv run python scripts/run_judge_cross_check.py                 # 全量 43 条
    uv run python scripts/run_judge_cross_check.py --limit 5       # 冒烟
    uv run python scripts/run_judge_cross_check.py --no-adaptive   # 单轮检索

产出：
    data/eval/reports/judge_cross_check_YYYYMMDD_HHMMSS.{json,md}

注意：生成结果只跑一次，两个 judge 评同一批 rows，故差异只归因于 judge 本身。
运行需独占 Qdrant 文件锁（local 模式），跑前请停掉本地后端/容器。
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
from app.eval.cross_judge import build_comparison, render_markdown  # noqa: E402
from app.eval.dataset import CATEGORIES, load_dataset, validate_dataset  # noqa: E402
from app.eval.metrics_ragas import (  # noqa: E402
    METRIC_KEYS,
    JudgeSpec,
    resolve_judge,
    run_ragas_metrics,
)
from app.eval.runner import RowOutcome, run_pipeline  # noqa: E402
from app.rag.loader import SUPPORTED_SUFFIXES  # noqa: E402

DEFAULT_CROSS_MODEL = "Qwen/Qwen2.5-72B-Instruct"


def _f(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else "N/A"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS 异源 judge 交叉验证")
    parser.add_argument("--dataset", default="data/eval/eval_dataset.json")
    parser.add_argument("--output-dir", default="data/eval/reports")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", choices=list(CATEGORIES))
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8, help="自评 judge 的并发数")
    parser.add_argument(
        "--cross-workers",
        type=int,
        default=1,
        help="异源 judge 的并发数（SiliconFlow TPM 限速严格，并发会触发 429；默认串行）",
    )
    parser.add_argument(
        "--cross-attempts",
        type=int,
        default=4,
        help="异源 judge 遇限流的最大尝试次数（指数退避；重试会重发整段 prompt）",
    )
    parser.add_argument(
        "--cross-backoff",
        type=float,
        default=6.0,
        help="异源 judge 限流退避基数（秒）",
    )
    parser.add_argument("--judge-model", default=DEFAULT_CROSS_MODEL)
    parser.add_argument("--judge-base-url", default="", help="留空用 SILICONFLOW_BASE_URL")
    parser.add_argument("--judge-api-key", default="", help="留空用 SILICONFLOW_API_KEY")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _discover_known_sources(settings) -> set[str]:
    root = settings.resolve("data/knowledge")
    names: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            names.add(path.name)
    return names


async def main() -> int:
    args = _parse_args()
    settings = validate_runtime(require=("chat", "embedding"))

    dataset = load_dataset(settings.resolve(args.dataset))
    validate_dataset(dataset, known_sources=_discover_known_sources(settings))
    questions = dataset.filter(categories=args.category, limit=args.limit)
    print(f"评测集校验通过：{len(dataset.questions)} 条，本次执行 {len(questions)} 条")

    primary = resolve_judge(settings)
    cross = JudgeSpec(
        model=args.judge_model,
        base_url=args.judge_base_url or settings.siliconflow_base_url,
        api_key=args.judge_api_key or settings.siliconflow_api_key,
    )
    print(f"自评 judge : {primary.label}")
    print(f"异源 judge : {cross.label}")

    services = build_services(settings)
    retriever = services.retriever if args.no_adaptive else services.adaptive
    try:
        print("阶段 1/3：检索 + 受控生成（只跑一次，两个 judge 评同一批结果）...")

        def _on_done(row: RowOutcome) -> None:
            if not args.quiet:
                flag = "ERROR" if row.error else ("拒答" if row.refusal_detected else "作答")
                print(f"  [{row.category:<6}] {row.id:<10} {flag} ctx={len(row.contexts)}")

        rows = await run_pipeline(
            questions,
            retriever,
            services.llm,
            top_k=args.top_k,
            concurrency=args.concurrency,
            on_row_done=_on_done,
        )
        print("阶段 2/3：自评 judge 打分 ...")
        primary_scores = await run_ragas_metrics(
            rows, settings, judge=primary, max_workers=args.workers
        )
        print("阶段 3/3：异源 judge 打分 ...")
        cross_scores = await run_ragas_metrics(
            rows,
            settings,
            judge=cross,
            max_workers=args.cross_workers,
            max_attempts=args.cross_attempts,
            retry_backoff=args.cross_backoff,
        )
    finally:
        await services.llm.close()
        await services.embedder.close()
        services.store.close()

    comparison = build_comparison(
        rows,
        primary_scores,
        cross_scores,
        primary=primary,
        cross=cross,
        config={
            "use_adaptive": not args.no_adaptive,
            "top_k": args.top_k or settings.retrieval_final_top_k,
            "generator_model": settings.question_model,
            "embedding_model": settings.embedding_model,
            "dataset": {
                "executed": len(rows),
                "counts": {
                    c: sum(1 for r in rows if r.category == c) for c in CATEGORIES
                },
            },
        },
    )

    output_dir = settings.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"judge_cross_check_{stamp}.json"
    md_path = output_dir / f"judge_cross_check_{stamp}.md"
    json_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(comparison), encoding="utf-8")
    print(f"\n报告已写出：\n  {json_path}\n  {md_path}")

    print("\n== 分指标 Δ(自评−异源) ==")
    for key in METRIC_KEYS:
        m = comparison["metrics"][key]
        d = m["delta"]
        d_txt = f"{d:+.3f}" if d is not None else "N/A"
        print(
            f"  {key:<18} 自评 {_f(m['primary_mean'])} / 异源 {_f(m['cross_mean'])} "
            f"Δ={d_txt}  平均绝对差={_f(m['mean_abs_diff'])}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[交叉验证启动失败] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
