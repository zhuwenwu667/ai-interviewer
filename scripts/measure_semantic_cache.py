"""P2-① 语义缓存效果实测：同一批真实面试「缓存关闭 vs 打开」对照。

在本机跑两遍相同规模的完整面试（每遍 N 场 × M 题，真实模型），仅语义缓存开关不同：

- 关闭：等价于改造前的生产行为（无缓存，每次出题都完整走改写/召回/精排/自评）；
- 打开：复用 Qdrant 语义缓存，同一查询命中即整体跳过检索管线。

从本项目埋点（``logs/traces.jsonl``）聚合 ask_question 节点耗时、各用途 LLM 调用
次数与 token，并读取缓存命中率，产出对照报告。

用法（项目根目录；需独占 Qdrant 文件锁，运行前请停掉后端与容器）：
    uv run python scripts/measure_semantic_cache.py --sessions 6 --questions 5

产出：
    data/eval/reports/semantic_cache_YYYYMMDD_HHMMSS.{json,md}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from measure_session_cost import _aggregate, _run_sessions  # noqa: E402

from app.config import validate_runtime  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="语义缓存 A/B 对照实测")
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--output-dir", default="data/eval/reports")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _pct_delta(before: float | None, after: float | None) -> float | None:
    if before in (None, 0) or after is None:
        return None
    return round((after - before) / before * 100, 1)


def _node(agg: dict, name: str) -> dict:
    return agg["node"].get(name, {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None})


def _purpose(agg: dict, purpose: str) -> dict:
    return agg["llm"]["purpose"].get(purpose, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})


def _build_summary(base: dict, cached: dict, cache_stats: dict) -> dict:
    b_q = _node(base, "node.ask_question")
    c_q = _node(cached, "node.ask_question")
    b_rewrite = _purpose(base, "rewrite")
    c_rewrite = _purpose(cached, "rewrite")
    b_eval = _purpose(base, "eval")
    c_eval = _purpose(cached, "eval")
    b_all_tokens = sum(
        p["prompt_tokens"] + p["completion_tokens"] for p in base["llm"]["purpose"].values()
    )
    c_all_tokens = sum(
        p["prompt_tokens"] + p["completion_tokens"] for p in cached["llm"]["purpose"].values()
    )
    b_calls = sum(p["calls"] for p in base["llm"]["purpose"].values())
    c_calls = sum(p["calls"] for p in cached["llm"]["purpose"].values())
    return {
        "ask_question": {
            "baseline_mean_ms": b_q["mean_ms"], "cached_mean_ms": c_q["mean_ms"],
            "baseline_p50_ms": b_q["p50_ms"], "cached_p50_ms": c_q["p50_ms"],
            "count": b_q["count"], "latency_delta_pct": _pct_delta(b_q["mean_ms"], c_q["mean_ms"]),
        },
        "session_total_s": {
            "baseline_mean": base["total_s"]["mean"], "cached_mean": cached["total_s"]["mean"],
            "delta_pct": _pct_delta(base["total_s"]["mean"], cached["total_s"]["mean"]),
        },
        "llm": {
            "baseline_calls": b_calls, "cached_calls": c_calls,
            "calls_delta_pct": _pct_delta(b_calls, c_calls),
            "baseline_total_tokens": b_all_tokens, "cached_total_tokens": c_all_tokens,
            "tokens_delta_pct": _pct_delta(b_all_tokens, c_all_tokens),
            "rewrite": {
                "baseline_calls": b_rewrite["calls"], "cached_calls": c_rewrite["calls"],
                "baseline_tokens": b_rewrite["prompt_tokens"] + b_rewrite["completion_tokens"],
                "cached_tokens": c_rewrite["prompt_tokens"] + c_rewrite["completion_tokens"],
                "calls_delta_pct": _pct_delta(b_rewrite["calls"], c_rewrite["calls"]),
            },
            "eval": {
                "baseline_calls": b_eval["calls"], "cached_calls": c_eval["calls"],
                "baseline_tokens": b_eval["prompt_tokens"] + b_eval["completion_tokens"],
                "cached_tokens": c_eval["prompt_tokens"] + c_eval["completion_tokens"],
                "calls_delta_pct": _pct_delta(b_eval["calls"], c_eval["calls"]),
            },
        },
        "cache": cache_stats,
    }


def _render_markdown(s: dict, args: argparse.Namespace) -> str:
    aq = s["ask_question"]
    tot = s["session_total_s"]
    llm = s["llm"]
    lines = ["# 语义缓存效果实测（缓存关闭 vs 打开）", ""]
    lines.append(f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    lines.append(
        f"- 样本：每遍 {args.sessions} 场真实面试（每场 {args.questions} 题），"
        "真实模型（DeepSeek chat + SiliconFlow bge-m3/rerank）；两遍仅语义缓存开关不同"
    )
    lines.append(
        "- 数据来源：本项目埋点 logs/traces.jsonl；缓存命中率取自语义缓存自身统计"
    )
    lines.append("")
    lines.append("## 关键结论")
    lines.append("")
    lines.append(
        f"- ask_question 节点平均耗时：{aq['baseline_mean_ms']}ms → **{aq['cached_mean_ms']}ms**"
        f"（{aq['latency_delta_pct']}%）"
    )
    lines.append(
        f"- 单场面试总耗时：{tot['baseline_mean']}s → **{tot['cached_mean']}s**（{tot['delta_pct']}%）"
    )
    lines.append(
        f"- LLM 调用总数：{llm['baseline_calls']} → **{llm['cached_calls']}**"
        f"（{llm['calls_delta_pct']}%）；总 token：{llm['baseline_total_tokens']} → "
        f"**{llm['cached_total_tokens']}**（{llm['tokens_delta_pct']}%）"
    )
    lines.append(
        f"- 语义缓存：查询 {s['cache']['lookups']} 次、命中 {s['cache']['hits']} 次、"
        f"命中率 **{s['cache']['hit_rate']}**（阈值 {s['cache']['threshold']}，"
        f"TTL {s['cache']['ttl_seconds']}s）"
    )
    lines.append("")
    lines.append("## 对照明细")
    lines.append("")
    lines.append("| 维度 | 缓存关闭 | 缓存打开 | 变化 |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        f"| ask_question 平均耗时(ms) | {aq['baseline_mean_ms']} | {aq['cached_mean_ms']} "
        f"| {aq['latency_delta_pct']}% |"
    )
    lines.append(
        f"| ask_question P50(ms) | {aq['baseline_p50_ms']} | {aq['cached_p50_ms']} | — |"
    )
    lines.append(
        f"| 单场总耗时均值(s) | {tot['baseline_mean']} | {tot['cached_mean']} | {tot['delta_pct']}% |"
    )
    lines.append(f"| LLM 调用总数 | {llm['baseline_calls']} | {llm['cached_calls']} | {llm['calls_delta_pct']}% |")
    lines.append(
        f"| LLM 总 token | {llm['baseline_total_tokens']} | {llm['cached_total_tokens']} "
        f"| {llm['tokens_delta_pct']}% |"
    )
    rw, ev = llm["rewrite"], llm["eval"]
    lines.append(
        f"| rewrite 调用 / token | {rw['baseline_calls']} / {rw['baseline_tokens']} "
        f"| {rw['cached_calls']} / {rw['cached_tokens']} | {rw['calls_delta_pct']}% |"
    )
    lines.append(
        f"| eval 调用 / token | {ev['baseline_calls']} / {ev['baseline_tokens']} "
        f"| {ev['cached_calls']} / {ev['cached_tokens']} | {ev['calls_delta_pct']}% |"
    )
    lines.append("")
    lines.append(
        "> 说明：出题查询在同一会话内恒定（`{track} 方向面试题`），故同一 track 的重复出题"
        "可命中缓存；命中即跳过「查询改写 + 向量化 + 双路召回 + 精排 + 充足性自评」。"
    )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    args = _parse_args()
    run_args = SimpleNamespace(
        sessions=args.sessions, questions=args.questions,
        output_dir=args.output_dir, quiet=args.quiet,
    )
    print("== 第一遍：缓存关闭（基线）==")
    base_data, _ = await _run_sessions(run_args, cache_enabled=False, reset_cache=True)
    print("== 第二遍：缓存打开 ==")
    cache_data, _ = await _run_sessions(run_args, cache_enabled=True, reset_cache=True)

    base_agg = _aggregate(base_data)
    cache_agg = _aggregate(cache_data)
    summary = _build_summary(base_agg, cache_agg, cache_data["cache"])

    settings = validate_runtime(require=("chat", "embedding"))
    output_dir = settings.resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sessions": args.sessions,
        "questions_per_session": args.questions,
        "summary": summary,
        "baseline": base_agg,
        "cached": cache_agg,
    }
    json_path = output_dir / f"semantic_cache_{stamp}.json"
    md_path = output_dir / f"semantic_cache_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(summary, args), encoding="utf-8")
    print(f"\n报告已写出：\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[语义缓存实测启动失败] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
