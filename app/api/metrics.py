"""效果看板接口：把离线评测的真实数字从文档搬进产品（只读、零 LLM）。

数据来源就是仓库内**已入库的评测产物**（``data/eval/reports/*.json`` 与
``data/eval/baseline.json``）：每类取文件名排序后最新的一份，聚合成一个只读响应。

两个设计取舍：
1. **不重跑评测**——看板只读已入库结果，接口无密钥、零 LLM 成本、毫秒级返回，
   与 ``scripts/check_eval_gate.py`` 的「校验已入库报告」思路一致（评测是发布前
   手动步骤，展示与拦截都基于同一份产物）。
2. **达标判定复用 CI 门禁**——直接调 ``app.eval.gate.check_report``，保证「看板上
   显示达标」与「CI 拦门通过」是同一套规则、同一个结论，避免两处各写一套阈值。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import ServiceContainer, get_services
from app.api.schemas import (
    AbComparison,
    AblationRow,
    CategoryMetrics,
    CrossCheckRow,
    MetricRow,
    MetricsDashboard,
    TrainRetrainStats,
)
from app.api.train import compute_retrain_stats
from app.eval.gate import check_report

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

# 每类报告对应的 glob（同目录下取文件名排序后最新的一份）
_REPORT_GLOBS: dict[str, str] = {
    "eval": "eval_*.json",
    "ablation": "ablation_*.json",
    "judge_cross_check": "judge_cross_check_*.json",
    "session_cost": "session_cost_*.json",
    "semantic_cache": "semantic_cache_*.json",
    "context_budget": "context_budget_*.json",
}

# RAGAS 四项（保留 0-1 原值）与检索侧、分类比例项（换算为百分数）
_RAGAS_KEYS = ("faithfulness", "context_precision", "context_recall", "answer_relevancy")
_RETRIEVAL_RATE_KEYS = (
    "source_hit_rate",
    "source_precision",
    "source_recall",
    "source_f1",
    "answer_citation_rate",
    "oos_refusal_rate",
)
_LABELS: dict[str, str] = {
    "faithfulness": "忠实度",
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
    "answer_relevancy": "答案相关性",
    "source_hit_rate": "知识来源命中率",
    "source_precision": "来源精确率",
    "source_recall": "来源召回率",
    "source_f1": "来源 F1",
    "answer_citation_rate": "答案引用率",
    "oos_refusal_rate": "题库外诱导题拒答率",
    "mean_contexts": "平均喂入证据条数",
    "mean_rounds": "平均检索轮数",
    "cost_mean_cny": "单场面试成本（元）",
    "total_s_mean": "单场面试时长（秒）",
    "total_tokens_mean": "单场 Token 总量",
}
_CATEGORY_LABELS: dict[str, str] = {
    "resume": "简历",
    "jd": "岗位 JD",
    "qa": "题库问答",
    "oos": "题库外诱导",
}


# ---------------------------------------------------------------------------
# 载入（IO）与聚合（纯函数）分离：聚合逻辑可脱库单测
# ---------------------------------------------------------------------------
def _latest_report(reports_dir: Path, pattern: str) -> tuple[dict | None, str]:
    """目录下最新一份匹配报告；读不到返回 (None, "")，不让看板因单文件损坏整体失败。"""
    files = sorted(reports_dir.glob(pattern))
    if not files:
        return None, ""
    try:
        return json.loads(files[-1].read_text(encoding="utf-8")), files[-1].name
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("效果看板读取报告失败 %s：%s", files[-1].name, exc)
        return None, ""


def load_reports(reports_dir: Path, baseline_path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """载入各类最新报告与回归基线，返回 (报告字典, 分组→文件名)。"""
    reports: dict[str, dict] = {}
    sources: dict[str, str] = {}
    for key, pattern in _REPORT_GLOBS.items():
        data, name = _latest_report(reports_dir, pattern)
        if data is not None:
            reports[key] = data
            sources[key] = name
    try:
        reports["baseline"] = json.loads(baseline_path.read_text(encoding="utf-8"))
        sources["baseline"] = baseline_path.name
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("效果看板读取基线失败 %s：%s", baseline_path, exc)
    return reports, sources


def _pct(value: float | None) -> float | None:
    """0-1 比例 → 百分数（看板统一口径）；None 原样返回。"""
    return None if value is None else round(value * 100, 1)


def _quality_rows(report: dict) -> list[MetricRow]:
    """RAGAS 总体四项（含合格线与基线变化）+ 检索侧指标。"""
    overall = (report.get("metrics") or {}).get("overall") or {}
    thresholds = report.get("thresholds") or {}
    threshold_pass = report.get("thresholds_pass") or {}
    deltas = (report.get("baseline") or {}).get("deltas") or {}
    rows: list[MetricRow] = []
    for key in _RAGAS_KEYS:
        value = overall.get(key)
        if value is None:
            continue
        gated = key in thresholds
        rows.append(
            MetricRow(
                key=key, name=_LABELS[key], value=round(float(value), 4), unit="分",
                threshold=thresholds.get(key) if gated else None,
                passed=bool(threshold_pass.get(key)) if gated else None,
                delta=deltas.get(key),
                note="门槛指标" if gated else "观察项（不设合格线）",
            )
        )
    retrieval = report.get("retrieval") or {}
    for key in _RETRIEVAL_RATE_KEYS:
        if retrieval.get(key) is None:
            continue
        threshold = report.get("oos_refusal_target") if key == "oos_refusal_rate" else None
        passed = report.get("oos_refusal_pass") if key == "oos_refusal_rate" else None
        rows.append(
            MetricRow(
                key=key, name=_LABELS[key], value=_pct(retrieval[key]), unit="%",
                threshold=_pct(threshold), passed=passed,
                note="达标线：拒答率 ≥ 80%" if key == "oos_refusal_rate" else "",
            )
        )
    for key in ("mean_contexts", "mean_rounds"):
        if retrieval.get(key) is not None:
            rows.append(
                MetricRow(
                    key=key, name=_LABELS[key], value=round(float(retrieval[key]), 2),
                    unit="条" if key == "mean_contexts" else "轮",
                )
            )
    return rows


def _category_rows(report: dict) -> list[CategoryMetrics]:
    by_category = (report.get("metrics") or {}).get("by_category") or {}
    retrieval_by_category = (report.get("retrieval") or {}).get("by_category") or {}
    rows: list[CategoryMetrics] = []
    for name, entry in by_category.items():
        metrics = entry.get("metrics") or {}
        retrieval = retrieval_by_category.get(name) or {}
        rows.append(
            CategoryMetrics(
                category=_CATEGORY_LABELS.get(name, name),
                n=int(entry.get("n") or retrieval.get("n") or 0),
                faithfulness=metrics.get("faithfulness"),
                context_precision=metrics.get("context_precision"),
                context_recall=metrics.get("context_recall"),
                answer_relevancy=metrics.get("answer_relevancy"),
                source_hit_rate=_pct(retrieval.get("source_hit_rate")),
                source_f1=_pct(retrieval.get("source_f1")),
            )
        )
    return rows


def _ablation_rows(report: dict) -> list[AblationRow]:
    rows: list[AblationRow] = []
    for config in report.get("configs") or []:
        metrics = config.get("metrics") or {}
        tokens = config.get("tokens") or {}
        stages = config.get("stages") or {}
        rows.append(
            AblationRow(
                config=str(config.get("config", "")),
                agentic=bool(config.get("agentic")),
                stages=[k for k, on in stages.items() if on],
                faithfulness=round(float(metrics.get("faithfulness", 0.0)), 4),
                context_precision=round(float(metrics.get("context_precision", 0.0)), 4),
                context_recall=round(float(metrics.get("context_recall", 0.0)), 4),
                answer_relevancy=round(float(metrics.get("answer_relevancy", 0.0)), 4),
                source_hit_rate=_pct(config.get("source_hit_rate")) or 0.0,
                mean_rounds=float(config.get("mean_rounds", 0.0)),
                mean_contexts=float(config.get("mean_contexts", 0.0)),
                llm_calls=int(tokens.get("llm_calls", 0)),
                est_cost_cny_per_q=float(tokens.get("est_cost_cny_per_q", 0.0)),
            )
        )
    return rows


def _cross_check(report: dict) -> tuple[list[CrossCheckRow], dict[str, float], dict[str, float]]:
    rows: list[CrossCheckRow] = []
    for key, entry in (report.get("metrics") or {}).items():
        rows.append(
            CrossCheckRow(
                metric=_LABELS.get(key, key),
                primary_mean=round(float(entry.get("primary_mean", 0.0)), 4),
                cross_mean=round(float(entry.get("cross_mean", 0.0)), 4),
                delta=round(float(entry.get("delta", 0.0)), 4),
                agree_0_1=_pct(entry.get("agree_0.1")) or 0.0,
            )
        )
    cell = {k: (_pct(v) or 0.0) for k, v in (report.get("overall_cell_agreement") or {}).items()}
    inflation = {k: float(v) for k, v in (report.get("self_inflation") or {}).items()}
    return rows, cell, inflation


def _session_cost(report: dict) -> dict[str, float]:
    aggregate = report.get("aggregate") or {}
    cost = aggregate.get("cost_cny") or {}
    tokens = aggregate.get("total_tokens") or {}
    total_s = aggregate.get("total_s") or {}
    values = {
        "sessions": float(len(aggregate.get("per_session") or [])),
        "cost_mean_cny": float(cost.get("mean", 0.0)),
        "cost_max_cny": float(cost.get("max", 0.0)),
        "total_tokens_mean": float(tokens.get("mean", 0.0)),
        "total_s_mean": float(total_s.get("mean", 0.0)),
    }
    node = (aggregate.get("node") or {}).get("node.ask_question") or {}
    if node:
        values["ask_question_ms_mean"] = float(node.get("mean_ms", 0.0))
    return values


def _cache_comparisons(report: dict) -> tuple[list[AbComparison], float | None]:
    summary = report.get("summary") or {}
    ask = summary.get("ask_question") or {}
    session = summary.get("session_total_s") or {}
    llm = summary.get("llm") or {}
    rows = [
        AbComparison(
            label="单题出题延迟", baseline=float(ask.get("baseline_mean_ms", 0.0)),
            optimized=float(ask.get("cached_mean_ms", 0.0)),
            delta_pct=float(ask.get("latency_delta_pct", 0.0)), unit="ms",
        ),
        AbComparison(
            label="单场面试时长", baseline=float(session.get("baseline_mean", 0.0)),
            optimized=float(session.get("cached_mean", 0.0)),
            delta_pct=float(session.get("delta_pct", 0.0)), unit="s",
        ),
        AbComparison(
            label="LLM 调用次数", baseline=float(llm.get("baseline_calls", 0)),
            optimized=float(llm.get("cached_calls", 0)),
            delta_pct=float(llm.get("calls_delta_pct", 0.0)), unit="次",
        ),
        AbComparison(
            label="LLM Token 总量", baseline=float(llm.get("baseline_total_tokens", 0)),
            optimized=float(llm.get("cached_total_tokens", 0)),
            delta_pct=float(llm.get("tokens_delta_pct", 0.0)), unit="token",
        ),
    ]
    hit_rate = _pct((summary.get("cache") or {}).get("hit_rate"))
    return rows, hit_rate


def _budget_comparisons(report: dict) -> list[AbComparison]:
    summary = report.get("summary") or {}
    tokens = summary.get("total_tokens") or {}
    cost = summary.get("cost_cny") or {}
    total_s = summary.get("total_s") or {}
    ask = (summary.get("nodes") or {}).get("node.ask_question") or {}
    return [
        AbComparison(
            label="LLM Token 总量", baseline=float(tokens.get("baseline", 0)),
            optimized=float(tokens.get("budget", 0)),
            delta_pct=float(tokens.get("delta_pct", 0.0)), unit="token",
        ),
        AbComparison(
            label="单场面试成本", baseline=float(cost.get("baseline_mean", 0.0)),
            optimized=float(cost.get("budget_mean", 0.0)),
            delta_pct=float(cost.get("delta_pct", 0.0)), unit="元",
        ),
        AbComparison(
            label="单场面试时长", baseline=float(total_s.get("baseline_mean", 0.0)),
            optimized=float(total_s.get("budget_mean", 0.0)),
            delta_pct=float(total_s.get("delta_pct", 0.0)), unit="s",
        ),
        AbComparison(
            label="出题节点 Token", baseline=float(ask.get("baseline_total_tokens", 0)),
            optimized=float(ask.get("budget_total_tokens", 0)),
            delta_pct=float(ask.get("delta_pct", 0.0)), unit="token",
        ),
    ]


def _violation_texts(violations: list[dict]) -> list[str]:
    """把门禁的结构化违规项翻成中文（看板直接给用户看，与 CI 结论同源）。"""
    texts: list[str] = []
    for item in violations:
        kind = item.get("kind")
        metric = _LABELS.get(str(item.get("metric", "")), str(item.get("metric", "")))
        if kind == "missing_metric":
            texts.append(f"门槛指标「{metric}」在最新评测报告中缺失（可能评测未跑完）")
        elif kind == "below_threshold":
            texts.append(
                f"门槛指标「{metric}」未达合格线：{item.get('value'):.4f} < {item.get('threshold'):.2f}"
            )
        elif kind == "baseline_drop":
            texts.append(
                f"门槛指标「{metric}」相对基线下降 {item.get('drop'):.4f}"
                f"（允许 {item.get('max_drop'):.4f}）：{item.get('baseline'):.4f} → {item.get('value'):.4f}"
            )
        elif kind == "oos_refusal":
            texts.append(
                f"题库外诱导题拒答率未达目标 {item.get('threshold'):.0%}"
                f"（实测 {item.get('value')}）"
            )
        elif kind == "exec_errors":
            texts.append(str(item.get("detail", "评测存在执行失败样例")))
        else:
            texts.append(str(item.get("detail", kind)))
    return texts


def build_dashboard(
    reports: dict[str, dict],
    sources: dict[str, str],
    retrain: TrainRetrainStats,
) -> MetricsDashboard:
    """把各类报告聚合成看板响应（纯函数：不碰文件系统，便于脱库单测）。"""
    eval_report = reports.get("eval") or {}
    ablation = reports.get("ablation") or {}
    cross = reports.get("judge_cross_check") or {}
    cost = reports.get("session_cost") or {}
    cache = reports.get("semantic_cache") or {}
    budget = reports.get("context_budget") or {}

    violations = check_report(eval_report, baseline=reports.get("baseline")) if eval_report else []
    dataset_raw = eval_report.get("dataset") or {}
    dataset = {
        "total": int(dataset_raw.get("total", 0)),
        "executed": int(dataset_raw.get("executed", 0)),
    }
    dataset.update({k: int(v) for k, v in (dataset_raw.get("counts") or {}).items()})

    cross_rows, cell_agreement, self_inflation = _cross_check(cross)
    cache_rows, hit_rate = _cache_comparisons(cache)

    return MetricsDashboard(
        sources=sources,
        generated_at={
            key: str(report.get("generated_at", ""))
            for key, report in (
                ("eval", eval_report), ("ablation", ablation), ("judge_cross_check", cross),
                ("session_cost", cost), ("semantic_cache", cache), ("context_budget", budget),
            )
            if report.get("generated_at")
        },
        gate_passed=bool(eval_report) and not violations,
        gate_violations=_violation_texts(violations),
        dataset=dataset,
        quality=_quality_rows(eval_report),
        by_category=_category_rows(eval_report),
        ablation=_ablation_rows(ablation),
        cross_check=cross_rows,
        cell_agreement=cell_agreement,
        self_inflation=self_inflation,
        session_cost=_session_cost(cost),
        semantic_cache=cache_rows,
        cache_hit_rate=hit_rate,
        context_budget=_budget_comparisons(budget),
        retrain=retrain,
    )


@router.get("/metrics")
async def get_metrics(
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> MetricsDashboard:
    """效果看板：只读聚合已入库评测产物（不限速、零 LLM、无密钥）。"""
    settings = services.settings
    reports, sources = load_reports(
        settings.resolve(settings.eval_reports_dir), settings.resolve(settings.eval_baseline_path)
    )
    retrain = compute_retrain_stats(services.repo.list_train_all_attempts(settings.train_username))
    return build_dashboard(reports, sources, retrain)
