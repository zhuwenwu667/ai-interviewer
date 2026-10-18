"""真实面试会话的成本与延迟实测（P0）。

在本机以真实模型、真实多轮面试链路跑 N 场完整面试（开场→交替出题/作答/追问→
终评→报告），再从本项目自身的埋点（``logs/traces.jsonl``）聚合：

- 单场面试：总耗时、总 token、估算成本；
- 节点级耗时：greeting / ask_question / assess_answer / followup / final_evaluation / report；
- LLM 调用：按 purpose（question / eval / rewrite）拆分次数、耗时（P50/P95）、token；
- 检索：rerank 耗时、单轮召回条数。

用法（项目根目录；需独占 Qdrant 文件锁，运行前请停掉后端与容器）：
    uv run python scripts/measure_session_cost.py --sessions 6 --questions 5

产出：
    data/eval/reports/session_cost_YYYYMMDD_HHMMSS.{json,md}
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

from app.agent.context import ContextBudget  # noqa: E402
from app.agent.graph import InterviewEngine  # noqa: E402
from app.agent.report import LLMReportGenerator  # noqa: E402
from app.api.deps import build_services  # noqa: E402
from app.api.schemas import CreateSessionRequest  # noqa: E402
from app.config import validate_runtime  # noqa: E402
from app.observability.tracer import build_tracer, reset_tracer, set_tracer  # noqa: E402

# deepseek-chat 官方定价（元 / 百万 token；缓存未命中）
INPUT_PRICE_PER_M = 2.0
OUTPUT_PRICE_PER_M = 8.0

TRACES_FILE = "logs/traces.jsonl"

TRACKS = [
    "Python 后端开发",
    "AI 应用开发",
    "数据工程",
    "全栈开发",
]

# 候选人回答池（贴近真实作答长度与信息量），按顺序循环取用
ANSWERS = [
    "在上一份工作中我用 Python + FastAPI 做后端服务，负责订单模块的接口设计与性能优化，"
    "把核心查询的 P99 从 800ms 降到 120ms，主要手段是加复合索引并把热点数据放进 Redis 缓存。",
    "我对 RAG 的理解是：先做文档切分与向量化，检索时结合向量召回和关键词检索，"
    "再用 rerank 精排，最后把命中的上下文喂给大模型生成，并在回答里保留引用来源便于溯源。",
    "做数据处理时我一般用 Airflow 编排离线任务，Spark 处理大批量，"
    "对时效要求高的链路会用 Flink 做流处理，并且对关键表建数据质量校验，失败就告警阻断下游。",
    "我在项目里用过 React + TypeScript 做前端，配合状态管理库组织复杂表单，"
    "也做过 SSR 优化首屏，把 LCP 从 3.2s 压到 1.4s，主要是拆包和图片懒加载。",
    "遇到线上故障时我会先看监控和日志定位影响面，必要时先止损回滚，"
    "再复现根因、补测试用例，最后写复盘文档并把改进项排进迭代。",
    "我对 Agent 的理解是把大模型、工具调用和记忆组合成能自主规划的系统，"
    "关键是用状态机或图来约束流程，避免模型自由发挥导致不可控，同时要做好可观测埋点。",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实面试会话成本与延迟实测")
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--output-dir", default="data/eval/reports")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def _stat(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 1),
        "p50_ms": _pct(values, 50),
        "p95_ms": _pct(values, 95),
    }


async def _run_sessions(
    args: argparse.Namespace,
    *,
    cache_enabled: bool = True,
    reset_cache: bool = False,
    budget_enabled: bool | None = None,
) -> tuple[dict, list[str]]:
    settings = validate_runtime(require=("chat", "embedding"))
    services = build_services(settings)
    services.cache.set_enabled(cache_enabled)
    if reset_cache:
        services.cache.clear()
    # P2-②：预算开关；None 表示沿用生产配置（默认开启）
    budget = ContextBudget.from_settings(settings, enabled=budget_enabled)
    report_generator = LLMReportGenerator(
        services.llm, reports_dir=settings.resolve("data/reports"), budget=budget
    )
    trace_path = settings.resolve(TRACES_FILE)
    sessions_meta: list[dict] = []
    ids: list[str] = []
    try:
        for i in range(args.sessions):
            track = TRACKS[i % len(TRACKS)]
            record = services.repo.create_session(
                CreateSessionRequest(track=track, num_questions=args.questions)
            )
            sid = record.session_id
            ids.append(sid)
            engine = InterviewEngine.create(
                settings, services.repo, services.adaptive, services.llm,
                sid, track, args.questions,
                report_generator=report_generator,
                ctx_budget=budget,
            )
            tracer = build_tracer(settings, session_id=sid, trace_id=sid)
            token = set_tracer(tracer)
            t0 = time.perf_counter()
            turn_ms: list[float] = []
            try:
                t = time.perf_counter()
                await engine.start()
                greeting_ms = (time.perf_counter() - t) * 1000

                rounds = 0
                for a in ANSWERS:
                    if engine._state.get("done"):
                        break
                    t = time.perf_counter()
                    await engine.submit_answer(a)
                    turn_ms.append((time.perf_counter() - t) * 1000)
                    rounds += 1

                t = time.perf_counter()
                report = await engine.finish()
                finish_ms = (time.perf_counter() - t) * 1000
            finally:
                reset_tracer(token)
                tracer.close()
            total_s = time.perf_counter() - t0
            dims = getattr(report, "dimensions", None) or []
            score = (
                round(statistics.fmean([d.score for d in dims]), 2) if dims else None
            )
            if not args.quiet:
                print(
                    f"  场次 {i + 1}/{args.sessions} session={sid[:8]} "
                    f"track={track} 轮数={rounds} 总耗时={total_s:.1f}s 均分={score}"
                )
            sessions_meta.append(
                {
                    "session_id": sid,
                    "track": track,
                    "rounds": rounds,
                    "mean_score": score,
                    "greeting_ms": round(greeting_ms, 1),
                    "turn_ms": [round(m, 1) for m in turn_ms],
                    "turn_mean_ms": round(statistics.fmean(turn_ms), 1) if turn_ms else None,
                    "finish_ms": round(finish_ms, 1),
                    "total_s": round(total_s, 2),
                }
            )
    finally:
        cache_stats = services.cache.stats()
        await services.llm.close()
        await services.embedder.close()
        services.store.close()

    # 从埋点聚合
    spans_by_session: dict[str, list[dict]] = {sid: [] for sid in ids}
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("trace_id") in spans_by_session:
                spans_by_session[rec["trace_id"]].append(rec)
    return {"sessions": sessions_meta, "spans": spans_by_session, "cache": cache_stats}, ids


def _aggregate(data: dict) -> dict:
    sessions = data["sessions"]
    spans_by_session: dict[str, list[dict]] = data["spans"]

    node_dur: dict[str, list[float]] = {}
    llm_by_purpose: dict[str, dict[str, int]] = {}
    llm_dur: list[float] = []
    rerank_dur: list[float] = []
    recall_ns: list[int] = []
    per_session_tokens: list[dict] = []
    node_tokens: dict[str, dict[str, int]] = {}

    for meta in sessions:
        sid = meta["session_id"]
        s_tokens = {"prompt": 0, "completion": 0}
        recs = spans_by_session.get(sid, [])
        span_names = {r["span_id"]: r["name"] for r in recs}
        for rec in recs:
            name = rec["name"]
            fields = rec.get("fields", {})
            dur = float(rec.get("duration_ms") or 0.0)
            if name.startswith("node."):
                node_dur.setdefault(name, []).append(dur)
            elif name in ("llm.chat", "llm.stream_chat"):
                purpose = fields.get("purpose", "unknown")
                bucket = llm_by_purpose.setdefault(
                    purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
                )
                bucket["calls"] += 1
                bucket["prompt_tokens"] += int(fields.get("prompt_tokens") or 0)
                bucket["completion_tokens"] += int(fields.get("completion_tokens") or 0)
                s_tokens["prompt"] += int(fields.get("prompt_tokens") or 0)
                s_tokens["completion"] += int(fields.get("completion_tokens") or 0)
                llm_dur.append(dur)
                # 归属到父节点（按 parent_id 反查节点名）
                parent = span_names.get(rec.get("parent_id") or "", "<root>")
                nb = node_tokens.setdefault(
                    parent, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
                )
                nb["calls"] += 1
                nb["prompt_tokens"] += int(fields.get("prompt_tokens") or 0)
                nb["completion_tokens"] += int(fields.get("completion_tokens") or 0)
            elif name == "retrieval.rerank":
                rerank_dur.append(dur)
            elif name == "retrieval.summary":
                if fields.get("returned") is not None:
                    recall_ns.append(int(fields["returned"]))
        cost = (
            s_tokens["prompt"] / 1e6 * INPUT_PRICE_PER_M
            + s_tokens["completion"] / 1e6 * OUTPUT_PRICE_PER_M
        )
        per_session_tokens.append(
            {
                "session_id": sid,
                "track": meta["track"],
                "rounds": meta["rounds"],
                "mean_score": meta.get("mean_score"),
                "prompt_tokens": s_tokens["prompt"],
                "completion_tokens": s_tokens["completion"],
                "total_tokens": s_tokens["prompt"] + s_tokens["completion"],
                "est_cost_cny": round(cost, 6),
                "total_s": meta["total_s"],
            }
        )

    total_costs = [s["est_cost_cny"] for s in per_session_tokens]
    total_tokens = [s["total_tokens"] for s in per_session_tokens]
    totals_s = [s["total_s"] for s in per_session_tokens]
    return {
        "per_session": per_session_tokens,
        "cost_cny": {
            "mean": round(statistics.fmean(total_costs), 6) if total_costs else None,
            "median": round(statistics.median(total_costs), 6) if total_costs else None,
            "min": round(min(total_costs), 6) if total_costs else None,
            "max": round(max(total_costs), 6) if total_costs else None,
        },
        "total_tokens": {
            "mean": round(statistics.fmean(total_tokens), 1) if total_tokens else None,
            "min": min(total_tokens) if total_tokens else None,
            "max": max(total_tokens) if total_tokens else None,
        },
        "total_s": {
            "mean": round(statistics.fmean(totals_s), 2) if totals_s else None,
            "p50": _pct(totals_s, 50),
            "max": max(totals_s) if totals_s else None,
        },
        "node": {name: _stat(v) for name, v in sorted(node_dur.items())},
        "node_tokens": {
            name: {
                "calls": b["calls"],
                "prompt_tokens": b["prompt_tokens"],
                "completion_tokens": b["completion_tokens"],
                "total_tokens": b["prompt_tokens"] + b["completion_tokens"],
            }
            for name, b in sorted(node_tokens.items())
        },
        "llm": {
            "purpose": llm_by_purpose,
            "duration": _stat(llm_dur),
        },
        "rerank": _stat(rerank_dur),
        "recall_returned": {
            "mean": round(statistics.fmean(recall_ns), 2) if recall_ns else None,
            "min": min(recall_ns) if recall_ns else None,
            "max": max(recall_ns) if recall_ns else None,
        },
    }


def _render_markdown(agg: dict, args: argparse.Namespace) -> str:
    lines = ["# 真实面试会话：成本与延迟实测", ""]
    lines.append(f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(
        f"- 样本：{args.sessions} 场真实面试（每场 {args.questions} 题），"
        "真实模型（DeepSeek chat + SiliconFlow bge-m3/rerank），数据取自本项目埋点"
    )
    lines.append(
        f"- 成本口径：deepseek-chat 输入 ¥{INPUT_PRICE_PER_M}/百万、输出 ¥{OUTPUT_PRICE_PER_M}/百万（缓存未命中）"
    )
    lines.append(
        "- 说明：本实测以**固定通用回答脚本**驱动链路，目的是测**成本与延迟**；"
        "表中「均分」是这些通用回答所得分数，不代表产品对真实候选人的评分水平。"
    )
    lines.append("")
    lines.append("## 单场面试成本与规模")
    lines.append("")
    lines.append("| 场次 | 岗位方向 | 轮数 | 均分 | 输入token | 输出token | 总token | 估算成本(元) | 总耗时(s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in agg["per_session"]:
        lines.append(
            f"| {s['session_id'][:8]} | {s['track']} | {s['rounds']} | {s.get('mean_score')} "
            f"| {s['prompt_tokens']} "
            f"| {s['completion_tokens']} | {s['total_tokens']} | {s['est_cost_cny']} | {s['total_s']} |"
        )
    c = agg["cost_cny"]
    lines.append("")
    lines.append(
        f"**单场成本**：均值 ¥{c['mean']}、中位 ¥{c['median']}、区间 ¥{c['min']}–¥{c['max']}；"
        f"**总耗时**：均值 {agg['total_s']['mean']}s、P50 {agg['total_s']['p50']}s、"
        f"最大 {agg['total_s']['max']}s"
    )
    lines.append("")
    lines.append("## 节点级耗时（毫秒）")
    lines.append("")
    lines.append("| 节点 | 次数 | 均值 | P50 | P95 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, st in agg["node"].items():
        lines.append(
            f"| {name} | {st['count']} | {st['mean_ms']} | {st['p50_ms']} | {st['p95_ms']} |"
        )
    lines.append("")
    lines.append("## LLM 调用（按用途）")
    lines.append("")
    lines.append("| 用途 | 调用次数 | 输入token | 输出token |")
    lines.append("| --- | --- | --- | --- |")
    for purpose, b in sorted(agg["llm"]["purpose"].items()):
        lines.append(f"| {purpose} | {b['calls']} | {b['prompt_tokens']} | {b['completion_tokens']} |")
    d = agg["llm"]["duration"]
    lines.append("")
    lines.append(
        f"单次 LLM 调用耗时：均值 {d['mean_ms']}ms、P50 {d['p50_ms']}ms、P95 {d['p95_ms']}ms"
    )
    r = agg["rerank"]
    lines.append(
        f"单次 rerank 耗时：均值 {r['mean_ms']}ms、P50 {r['p50_ms']}ms、P95 {r['p95_ms']}ms"
    )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    args = _parse_args()
    data, _ = await _run_sessions(args)
    agg = _aggregate(data)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sessions": args.sessions,
        "questions_per_session": args.questions,
        "pricing": {"input_cny_per_m": INPUT_PRICE_PER_M, "output_cny_per_m": OUTPUT_PRICE_PER_M},
        "raw_sessions": data["sessions"],
        "aggregate": agg,
    }
    settings = validate_runtime(require=("chat", "embedding"))
    output_dir = settings.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"session_cost_{stamp}.json"
    md_path = output_dir / f"session_cost_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(agg, args), encoding="utf-8")
    print(f"\n报告已写出：\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[会话实测启动失败] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
