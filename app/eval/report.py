"""评测结果聚合与 Markdown/JSON 报告渲染（纯函数，可离线测试）。"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

from app.eval.dataset import CATEGORIES, EvalDataset
from app.eval.metrics_ragas import METRIC_KEYS, THRESHOLDS
from app.eval.runner import RowOutcome

# 题库外诱导题拒答率目标（非 TR-11.2 硬门槛，单独报告）
OOS_REFUSAL_TARGET = 0.8
GATED_METRICS = ("faithfulness", "context_recall", "answer_relevancy")

# 答案里的显式引用标记（用于诊断"生成结果是否自带引用"，进而说明引用质量的度量口径）
_CITATION_RE = re.compile(r"【资料\s*\d+】|\[\d+\]|根据资料\s*\d+|来源[:：]")

REPORT_VERSION = "1.0"


def mean(values: list[float | None]) -> float | None:
    """忽略 None 的均值；全空返回 None。"""
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def _metric_table(
    rows: list[RowOutcome], scores: dict[str, dict[str, float | None]]
) -> dict[str, float | None]:
    return {
        key: mean([scores.get(r.id, {}).get(key) for r in rows])
        for key in METRIC_KEYS
    }


def _retrieval_stats(rows: list[RowOutcome]) -> dict[str, Any]:
    hit_rows = [r for r in rows if r.source_hit is not None]
    in_scope = [r for r in rows if r.category != "oos"]
    oos = [r for r in rows if r.category == "oos"]
    cited = [r for r in in_scope if r.answer and _CITATION_RE.search(r.answer)]
    by_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        cat_rows = [r for r in rows if r.category == category]
        cat_hit_rows = [r for r in cat_rows if r.source_hit is not None]
        by_category[category] = {
            "n": len(cat_rows),
            "source_hit_rate": (
                sum(1 for r in cat_hit_rows if r.source_hit) / len(cat_hit_rows)
                if cat_hit_rows
                else None
            ),
            "source_precision": mean([r.source_precision for r in cat_rows]),
            "source_recall": mean([r.source_recall for r in cat_rows]),
            "source_f1": mean([r.source_f1 for r in cat_rows]),
            "mean_contexts": (
                sum(len(r.contexts) for r in cat_rows) / len(cat_rows)
                if cat_rows
                else None
            ),
        }
    adaptive_rounds = [r.rounds for r in rows if r.rounds and r.rounds > 0]
    return {
        "source_hit_rate": (
            sum(1 for r in hit_rows if r.source_hit) / len(hit_rows)
            if hit_rows
            else None
        ),
        "source_precision": mean([r.source_precision for r in rows]),
        "source_recall": mean([r.source_recall for r in rows]),
        "source_f1": mean([r.source_f1 for r in rows]),
        "answer_citation_rate": (
            len(cited) / len(in_scope) if in_scope else None
        ),
        "mean_contexts": (
            sum(len(r.contexts) for r in rows) / len(rows) if rows else None
        ),
        "mean_rounds": (
            sum(adaptive_rounds) / len(adaptive_rounds) if adaptive_rounds else None
        ),
        "oos_refusal_rate": (
            sum(1 for r in oos if r.refusal_detected) / len(oos) if oos else None
        ),
        "in_scope_refusal_rate": (
            sum(1 for r in in_scope if r.refusal_detected) / len(in_scope)
            if in_scope
            else None
        ),
        "by_category": by_category,
    }


def _thresholds_pass(overall: dict[str, float | None]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for key in GATED_METRICS:
        value = overall.get(key)
        result[key] = value is not None and value >= THRESHOLDS[key]
    return result


def _deltas(
    current: dict[str, float | None], baseline: dict[str, Any] | None
) -> dict[str, float | None]:
    if not baseline:
        return {key: None for key in METRIC_KEYS}
    base_metrics = baseline.get("metrics", {}).get("overall", {})
    out: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        new, old = current.get(key), base_metrics.get(key)
        out[key] = round(new - old, 4) if new is not None and old is not None else None
    return out


def build_report(
    dataset: EvalDataset,
    rows: list[RowOutcome],
    scores: dict[str, dict[str, float | None]],
    *,
    config: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """聚合全部结果为 JSON 可序列化的报告字典。"""
    counts = Counter(r.category for r in rows)
    in_scope = [r for r in rows if r.category != "oos" and r.error is None]
    overall = _metric_table(in_scope, scores)
    by_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        cat_rows = [r for r in in_scope if r.category == category]
        by_category[category] = {
            "n": len(cat_rows),
            "metrics": _metric_table(cat_rows, scores) if cat_rows else {},
        }

    passed_map = _thresholds_pass(overall)
    errors = [
        {"id": r.id, "category": r.category, "error": r.error}
        for r in rows
        if r.error
    ]
    refusal = _retrieval_stats(rows)
    oos_pass = (
        refusal["oos_refusal_rate"] is not None
        and refusal["oos_refusal_rate"] >= OOS_REFUSAL_TARGET
    )
    report = {
        "report_version": REPORT_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "version": dataset.version,
            "total": len(dataset.questions),
            "executed": len(rows),
            "counts": {c: counts.get(c, 0) for c in CATEGORIES},
        },
        "config": config,
        "metrics": {"overall": overall, "by_category": by_category},
        "thresholds": {key: THRESHOLDS[key] for key in GATED_METRICS},
        "passed": all(passed_map.values()),
        "thresholds_pass": passed_map,
        "retrieval": refusal,
        "oos_refusal_target": OOS_REFUSAL_TARGET,
        "oos_refusal_pass": oos_pass,
        "errors": errors,
        "rows": [_row_payload(r, scores.get(r.id)) for r in rows],
    }
    report["baseline"] = {
        "present": baseline is not None,
        "generated_at": (baseline or {}).get("generated_at"),
        "deltas": _deltas(overall, baseline),
    }
    return report


def _row_payload(
    row: RowOutcome, metric: dict[str, float | None] | None
) -> dict[str, Any]:
    return {
        "id": row.id,
        "category": row.category,
        "question": row.question,
        "ground_truth": row.ground_truth,
        "retrieved_sources": row.retrieved_sources,
        "expected_sources": row.expected_sources,
        "source_hit": row.source_hit,
        "source_precision": row.source_precision,
        "source_recall": row.source_recall,
        "source_f1": row.source_f1,
        "n_contexts": len(row.contexts),
        "rounds": row.rounds,
        "stopped_reason": row.stopped_reason,
        "top_rerank_score": row.top_rerank_score,
        "refusal_detected": row.refusal_detected,
        "metrics": metric,
        "answer": row.answer,
        "error": row.error,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "N/A"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) < 0.0005:
        return "0.000"
    arrow = "↑" if value > 0 else "↓"
    return f"{arrow}{abs(value):.3f}"


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def render_markdown(report: dict[str, Any]) -> str:
    """将 ``build_report`` 产出的报告渲染为 Markdown 字符串。"""
    overall = report["metrics"]["overall"]
    deltas = report["baseline"]["deltas"]
    lines: list[str] = []
    lines.append("# RAGAS 离线评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    ds = report["dataset"]
    lines.append(
        f"- 评测集：v{ds['version']}，执行 {ds['executed']}/{ds['total']} 条，"
        f"分类计数 {ds['counts']}"
    )
    cfg = report["config"]
    lines.append(
        f"- 运行配置：adaptive={cfg.get('use_adaptive')}，top_k={cfg.get('top_k')}，"
        f"评判模型={cfg.get('judge_model')}，生成模型={cfg.get('generator_model')}，"
        f"embedding={cfg.get('embedding_model')}"
    )
    lines.append(
        f"- TR-11.2 门槛：{'**全部通过**' if report['passed'] else '**未全部通过**'}"
        f"；诱导题拒答率目标 {report['oos_refusal_target']:.0%}："
        f"{'通过' if report['oos_refusal_pass'] else '未通过'}"
    )
    lines.append("")

    lines.append("## 总体指标（RAGAS，非诱导题）")
    lines.append("")
    lines.append("| 指标 | 得分 | 阈值 | 对比基线 | 是否达标 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key in METRIC_KEYS:
        gated = key in report["thresholds"]
        threshold = report["thresholds"].get(key)
        if gated:
            ok = report["thresholds_pass"][key]
            pass_cell = "✅" if ok else "❌"
            threshold_cell = f"{threshold:.2f}"
        else:
            pass_cell = "（观察项）"
            threshold_cell = "—"
        lines.append(
            f"| {key} | {_fmt(overall.get(key))} | {threshold_cell} | "
            f"{_fmt_delta(deltas.get(key))} | {pass_cell} |"
        )
    lines.append("")

    lines.append("## 分类别指标")
    lines.append("")
    lines.append("| 类别 | n | faithfulness | ctx_precision | ctx_recall | answer_relevancy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for category in CATEGORIES:
        cat = report["metrics"]["by_category"].get(category, {})
        m = cat.get("metrics", {})
        lines.append(
            f"| {category} | {cat.get('n', 0)} | {_fmt(m.get('faithfulness'))} | "
            f"{_fmt(m.get('context_precision'))} | {_fmt(m.get('context_recall'))} | "
            f"{_fmt(m.get('answer_relevancy'))} |"
        )
    lines.append("")

    retrieval = report["retrieval"]
    lines.append("## 检索侧统计")
    lines.append("")
    lines.append(f"- 期望来源命中率（二值）：{_pct(retrieval['source_hit_rate'])}")
    lines.append(
        f"- 来源精确率 P：{_fmt(retrieval['source_precision'])}"
        f"；来源召回率 R：{_fmt(retrieval['source_recall'])}"
        f"；F1：{_fmt(retrieval['source_f1'])}"
    )
    lines.append(
        f"- 答案显式引用标记率：{_pct(retrieval['answer_citation_rate'])}"
        "（生成结果很少自带引用标记，故引用质量按**检索侧**度量）"
    )
    lines.append(f"- 平均上下文条数：{_fmt(retrieval['mean_contexts'], 2)}")
    lines.append(f"- 平均自适应检索轮数：{_fmt(retrieval['mean_rounds'], 2)}")
    lines.append(
        f"- 诱导题拒答率：{_pct(retrieval['oos_refusal_rate'])}"
        f"（目标 ≥{OOS_REFUSAL_TARGET:.0%}）"
    )
    lines.append(
        f"- 非诱导题误拒答率：{_pct(retrieval['in_scope_refusal_rate'])}（应接近 0）"
    )
    lines.append("")
    lines.append("| 类别 | n | 来源命中率 | P | R | F1 | 平均上下文数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for category in CATEGORIES:
        c = retrieval["by_category"][category]
        lines.append(
            f"| {category} | {c['n']} | {_pct(c['source_hit_rate'])} | "
            f"{_fmt(c['source_precision'])} | {_fmt(c['source_recall'])} | "
            f"{_fmt(c['source_f1'])} | {_fmt(c['mean_contexts'], 2)} |"
        )
    lines.append("")

    oos_rows = [r for r in report["rows"] if r["category"] == "oos"]
    if oos_rows:
        lines.append("## 题库外诱导题明细")
        lines.append("")
        lines.append("| id | 问题 | 命中来源 | 是否拒答 |")
        lines.append("| --- | --- | --- | --- |")
        for r in oos_rows:
            sources = ", ".join(r["retrieved_sources"]) or "（无）"
            q = r["question"].replace("|", "/")
            lines.append(
                f"| {r['id']} | {q} | {sources} | {'✅' if r['refusal_detected'] else '❌'} |"
            )
        lines.append("")

    failed = [r for r in report["rows"] if r["metrics"] and any(
        v is not None and v < 0.6 for v in r["metrics"].values()
    )]
    if failed:
        lines.append("## 低分样例（任一指标 < 0.6）")
        lines.append("")
        for r in failed:
            lines.append(f"- `{r['id']}`（{r['category']}）：{r['metrics']}")
        lines.append("")

    if report["errors"]:
        lines.append("## 执行失败样例")
        lines.append("")
        for err in report["errors"]:
            lines.append(f"- `{err['id']}`（{err['category']}）：{err['error']}")
        lines.append("")

    return "\n".join(lines)
