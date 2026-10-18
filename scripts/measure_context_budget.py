"""P2-② 上下文预算效果实测：同一批真实面试「预算关闭 vs 打开」对照。

两遍相同规模的完整面试（真实模型），仅上下文预算开关不同：

- 关闭：复刻改造前行为（证据片段 150 字、已问问题不限条数、报告知识来源不限条数/片段 120 字）；
- 打开：证据片段 100 字、已问问题最多 6 条、报告知识来源最多 8 条、片段 60 字。

两遍均**开启语义缓存**（隔离出预算单一变量），从埋点按**父节点归属 token**，
对比 ask_question / assess_answer / report 三个节点的 token 与整体成本/延迟变化。

用法（项目根目录；需独占 Qdrant 文件锁，运行前停掉后端与容器）：
    uv run python scripts/measure_context_budget.py --sessions 6 --questions 6

产出：
    data/eval/reports/context_budget_YYYYMMDD_HHMMSS.{json,md}
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

NODES = ["node.ask_question", "node.assess_answer", "node.report", "node.followup"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="上下文预算 A/B 对照实测")
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--questions", type=int, default=6)
    parser.add_argument("--output-dir", default="data/eval/reports")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _delta(before: float | None, after: float | None) -> float | None:
    if before in (None, 0) or after is None:
        return None
    return round((after - before) / before * 100, 1)


def _node_tok(agg: dict, name: str) -> dict:
    return agg["node_tokens"].get(
        name, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )


def _all_tokens(agg: dict) -> int:
    return sum(v["total_tokens"] for v in agg["node_tokens"].values())


def _calls(agg: dict) -> int:
    return sum(v["calls"] for v in agg["node_tokens"].values())


def _summarize(base: dict, on: dict) -> dict:
    nodes = {}
    for name in NODES:
        b, c = _node_tok(base, name), _node_tok(on, name)
        nodes[name] = {
            "baseline_total_tokens": b["total_tokens"],
            "budget_total_tokens": c["total_tokens"],
            "baseline_prompt": b["prompt_tokens"],
            "budget_prompt": c["prompt_tokens"],
            "calls": c["calls"] or b["calls"],
            "delta_pct": _delta(b["total_tokens"], c["total_tokens"]),
            "per_call_baseline": round(b["total_tokens"] / b["calls"]) if b["calls"] else None,
            "per_call_budget": round(c["total_tokens"] / c["calls"]) if c["calls"] else None,
        }
    return {
        "nodes": nodes,
        "total_tokens": {
            "baseline": _all_tokens(base), "budget": _all_tokens(on),
            "delta_pct": _delta(_all_tokens(base), _all_tokens(on)),
        },
        "llm_calls": {"baseline": _calls(base), "budget": _calls(on)},
        "cost_cny": {
            "baseline_mean": base["cost_cny"]["mean"], "budget_mean": on["cost_cny"]["mean"],
            "delta_pct": _delta(base["cost_cny"]["mean"], on["cost_cny"]["mean"]),
        },
        "total_s": {
            "baseline_mean": base["total_s"]["mean"], "budget_mean": on["total_s"]["mean"],
            "delta_pct": _delta(base["total_s"]["mean"], on["total_s"]["mean"]),
        },
    }


def _fmt_delta(v: float | None) -> str:
    return "—" if v is None else f"{v}%"


def _fmt_pair(before: int | None, after: int | None) -> str:
    if before is None and after is None:
        return "—"
    return f"{before} → {after}"


def _render_markdown(s: dict, args: argparse.Namespace, generated_at: str | None = None) -> str:
    lines = ["# 上下文预算效果实测（预算关闭 vs 打开）", ""]
    stamp = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    lines.append(f"- 生成时间：{stamp}")
    lines.append(
        f"- 样本：每遍 {args.sessions} 场真实面试（每场 {args.questions} 题），真实模型；"
        "**两遍均开启语义缓存**，仅上下文预算开关不同（隔离单一变量）"
    )
    lines.append(
        "- 预算（打开）：证据片段 100 字、已问问题 ≤6 条、报告知识来源 ≤8 条、来源片段 60 字；"
        "关闭即复刻改造前（150 字 / 不限 / 不限 / 120 字）"
    )
    lines.append("")
    lines.append("## 关键结论")
    lines.append("")
    t = s["total_tokens"]
    lines.append(
        f"- 全链路 LLM token：{t['baseline']} → **{t['budget']}**（{t['delta_pct']}%）"
    )
    lines.append(
        f"- 单场成本：¥{s['cost_cny']['baseline_mean']} → **¥{s['cost_cny']['budget_mean']}**"
        f"（{s['cost_cny']['delta_pct']}%）；单场耗时：{s['total_s']['baseline_mean']}s → "
        f"**{s['total_s']['budget_mean']}s**（{s['total_s']['delta_pct']}%）"
    )
    lines.append(
        f"- LLM 调用次数：{s['llm_calls']['baseline']} → {s['llm_calls']['budget']}（预算不改变调用次数，只压缩单次上下文）"
    )
    lines.append("")
    lines.append("## 按节点归属的 token（哪个节点被压下来）")
    lines.append("")
    lines.append("| 节点 | 调用次数 | 预算关闭 token | 预算打开 token | 变化 | 单次 token（关→开） |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name, n in s["nodes"].items():
        lines.append(
            f"| {name} | {n['calls']} | {n['baseline_total_tokens']} | {n['budget_total_tokens']} "
            f"| {_fmt_delta(n['delta_pct'])} | {_fmt_pair(n['per_call_baseline'], n['per_call_budget'])} |"
        )
    lines.append("")
    lines.append(
        "> 说明：预算压缩的是**喂给模型的上下文**（证据片段长度、已问问题条数、报告来源条数），"
        "不减少检索召回条数、不改变 LLM 调用次数；因此「每节点 token」是主要观测量，"
        "耗时/成本随之小幅下降。片段截短会削弱追问的简历细节粒度与报告依据丰富度，"
        "是否划算需结合出题/报告质量一并判断。本批脚本化作答未触发追问分支，"
        "故 `node.followup` 调用次数为 0。"
    )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    args = _parse_args()
    run_args = SimpleNamespace(
        sessions=args.sessions, questions=args.questions,
        output_dir=args.output_dir, quiet=args.quiet,
    )
    print("== 第一遍：预算关闭（基线，缓存开）==")
    base_data, _ = await _run_sessions(
        run_args, cache_enabled=True, reset_cache=True, budget_enabled=False
    )
    print("== 第二遍：预算打开（缓存开）==")
    on_data, _ = await _run_sessions(
        run_args, cache_enabled=True, reset_cache=True, budget_enabled=True
    )

    base_agg = _aggregate(base_data)
    on_agg = _aggregate(on_data)
    summary = _summarize(base_agg, on_agg)

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
        "budget": on_agg,
    }
    json_path = output_dir / f"context_budget_{stamp}.json"
    md_path = output_dir / f"context_budget_{stamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(summary, args), encoding="utf-8")
    print(f"\n报告已写出：\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"[上下文预算实测启动失败] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
