"""异源 judge 交叉验证的聚合与渲染（纯函数，可离线测试）。

量化"自评偏置"：同一批生成结果分别由**自评 judge**与**异源 judge**打分，
比较 per-metric 均值差（Δ = 自评 − 异源，>0 表示自评偏高）与逐条一致率。
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from app.eval.metrics_ragas import METRIC_KEYS, JudgeSpec
from app.eval.report import mean
from app.eval.runner import RowOutcome

# 逐条"一致"的容差：RAGAS 单项为 [0,1] 的连续/比例分，容许 ±0.1 的评判抖动
AGREE_TOL = 0.1
AGREE_TOLS = (0.05, 0.1, 0.2)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """皮尔逊相关系数；样本不足或方差为 0 时返回 None。"""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def compare_metric(
    primary: list[float | None], cross: list[float | None]
) -> dict[str, float | None]:
    """对同一指标的逐条配对比较（缺失单元格成对剔除）。"""
    pairs = [
        (a, b)
        for a, b in zip(primary, cross, strict=True)
        if a is not None and b is not None
    ]
    mean_a, mean_b = mean(primary), mean(cross)
    out: dict[str, float | None] = {
        "primary_mean": mean_a,
        "cross_mean": mean_b,
        "delta": (
            round(mean_a - mean_b, 4)
            if mean_a is not None and mean_b is not None
            else None
        ),
        "n_pairs": len(pairs),
        "n_missing": len(primary) - len(pairs),
    }
    if not pairs:
        out.update({k: None for k in ("mean_abs_diff", "max_abs_diff", "pearson")})
        out.update({f"agree_{tol}": None for tol in AGREE_TOLS})
        out["exact"] = None
        return out

    diffs = [abs(a - b) for a, b in pairs]
    out["mean_abs_diff"] = round(sum(diffs) / len(diffs), 4)
    out["max_abs_diff"] = round(max(diffs), 4)
    p = pearson([a for a, _ in pairs], [b for _, b in pairs])
    out["pearson"] = round(p, 4) if p is not None else None
    for tol in AGREE_TOLS:
        out[f"agree_{tol}"] = round(
            sum(1 for d in diffs if d <= tol) / len(pairs), 4
        )
    out["exact"] = round(
        sum(1 for a, b in pairs if round(a, 3) == round(b, 3)) / len(pairs), 4
    )
    return out


def build_comparison(
    rows: list[RowOutcome],
    primary_scores: dict[str, dict[str, float | None]],
    cross_scores: dict[str, dict[str, float | None]],
    *,
    primary: JudgeSpec,
    cross: JudgeSpec,
    config: dict[str, Any],
) -> dict[str, Any]:
    """聚合两套 judge 打分，产出 JSON 可序列化的对比报告字典。"""
    scored = [r for r in rows if r.category != "oos" and r.error is None and r.answer]
    metrics: dict[str, dict[str, float | None]] = {}
    for key in METRIC_KEYS:
        metrics[key] = compare_metric(
            [primary_scores.get(r.id, {}).get(key) for r in scored],
            [cross_scores.get(r.id, {}).get(key) for r in scored],
        )

    cell_diffs: list[float] = []
    for r in scored:
        for key in METRIC_KEYS:
            a = primary_scores.get(r.id, {}).get(key)
            b = cross_scores.get(r.id, {}).get(key)
            if a is not None and b is not None:
                cell_diffs.append(abs(a - b))
    overall_agree = {
        f"agree_{tol}": (
            round(sum(1 for d in cell_diffs if d <= tol) / len(cell_diffs), 4)
            if cell_diffs
            else None
        )
        for tol in AGREE_TOLS
    }

    row_payload: list[dict[str, Any]] = []
    for r in scored:
        cells: dict[str, dict[str, float | None]] = {}
        for key in METRIC_KEYS:
            a = primary_scores.get(r.id, {}).get(key)
            b = cross_scores.get(r.id, {}).get(key)
            cells[key] = {
                "primary": a,
                "cross": b,
                "delta": round(a - b, 4) if a is not None and b is not None else None,
            }
        max_abs = max(
            (abs(c["delta"]) for c in cells.values() if c["delta"] is not None),
            default=0.0,
        )
        row_payload.append(
            {
                "id": r.id,
                "category": r.category,
                "metrics": cells,
                "max_abs_delta": round(max_abs, 4),
                "answer": r.answer,
            }
        )

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": config.get("dataset"),
        "primary_judge": {"model": primary.model, "base_url": primary.base_url},
        "cross_judge": {"model": cross.model, "base_url": cross.base_url},
        "config": config,
        "n_scored": len(scored),
        "agree_tol": AGREE_TOL,
        "metrics": metrics,
        "overall_cell_agreement": overall_agree,
        "n_cells": len(cell_diffs),
        # 自评 judge 均值 − 异源 judge 均值；正值 = 自评偏高（疑似自评偏置）
        "self_inflation": {key: metrics[key]["delta"] for key in METRIC_KEYS},
        "rows": row_payload,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, float) else "N/A"


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def render_markdown(rep: dict[str, Any]) -> str:
    """将 ``build_comparison`` 的输出渲染为 Markdown 字符串。"""
    lines: list[str] = []
    lines.append("# RAGAS 异源 judge 交叉验证报告")
    lines.append("")
    lines.append(f"- 生成时间：{rep['generated_at']}")
    lines.append(
        f"- 自评 judge（primary）：`{rep['primary_judge']['model']}` "
        f"@ {rep['primary_judge']['base_url']}"
    )
    lines.append(
        f"- 异源 judge（cross）：`{rep['cross_judge']['model']}` "
        f"@ {rep['cross_judge']['base_url']}"
    )
    lines.append(
        f"- 打分为同一批 **{rep['n_scored']}** 条生成结果（非 oos、未报错），"
        "差异只归因于 judge 本身"
    )
    lines.append(
        f"- 判定口径：|Δ|≤{rep['agree_tol']} 记为「一致」；Δ = 自评 − 异源（>0 = 自评偏高）"
    )
    lines.append("")

    lines.append("## 分指标对比")
    lines.append("")
    lines.append(
        "| 指标 | 自评均值 | 异源均值 | Δ(自评−异源) | 平均绝对差 | 最大差 "
        "| ≤0.05 一致 | ≤0.10 一致 | ≤0.20 一致 | Pearson |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for key in METRIC_KEYS:
        m = rep["metrics"][key]
        lines.append(
            f"| {key} | {_fmt(m['primary_mean'])} | {_fmt(m['cross_mean'])} | "
            f"{_fmt(m['delta'])} | {_fmt(m['mean_abs_diff'])} | {_fmt(m['max_abs_diff'])} | "
            f"{_pct(m['agree_0.05'])} | {_pct(m['agree_0.1'])} | {_pct(m['agree_0.2'])} | "
            f"{_fmt(m['pearson'])} |"
        )
    lines.append("")

    lines.append(f"## 全体单元一致率（指标 × 样例，共 {rep['n_cells']} 格）")
    lines.append("")
    for tol in AGREE_TOLS:
        lines.append(
            f"- |Δ|≤{tol}：{_pct(rep['overall_cell_agreement'][f'agree_{tol}'])}"
        )
    lines.append("")

    worst = sorted(rep["rows"], key=lambda r: r["max_abs_delta"], reverse=True)[:8]
    lines.append("## 分歧最大的样例（Top 8）")
    lines.append("")
    lines.append("| id | 类别 | 指标 | 自评 | 异源 | Δ |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    shown = 0
    for r in worst:
        if r["max_abs_delta"] <= 0:
            break
        for key in METRIC_KEYS:
            c = r["metrics"][key]
            if c["delta"] is not None and abs(c["delta"]) >= r["max_abs_delta"] - 1e-9:
                lines.append(
                    f"| {r['id']} | {r['category']} | {key} | {_fmt(c['primary'])} | "
                    f"{_fmt(c['cross'])} | {_fmt(c['delta'])} |"
                )
                shown += 1
                break
    if shown == 0:
        lines.append("| — | — | — | — | — | — |")
    lines.append("")

    lines.append("## 结论口径")
    lines.append("")
    lines.append(
        "- 若各指标 Δ 均接近 0 且逐条一致率高，则说明指标得分**不是自评模型抬高**的结果，"
        "自评偏置可忽略；"
    )
    lines.append(
        "- 若某指标 Δ 明显为正（自评 > 异源），则该指标需谨慎解读，并考虑以异源 judge 为准。"
    )
    return "\n".join(lines)
