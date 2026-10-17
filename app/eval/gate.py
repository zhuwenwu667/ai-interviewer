"""评测回归门禁：校验已提交的最新评测报告是否仍达标。

设计取舍：**不在 CI 里重跑 RAGAS**——那需要 DeepSeek/SiliconFlow 密钥、有真实费用、
且耗时约 7 分钟、并发还受 TPM 限速。因此门禁的形态是「**校验已入库的评测报告**」：
把评测当作**发布前手动步骤**，把门禁放在 CI 里做**确定性校验**（无密钥、秒级、可复现）。

校验项（任一不过即判定失败）：
1. 门槛指标（TR-11.2）达标；
2. 题库外诱导题拒答率达标；
3. 报告中无执行失败样例；
4. 相对基线（``data/eval/baseline.json``）各指标跌幅不超过 ``max_drop``。
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from app.eval.metrics_ragas import METRIC_KEYS, THRESHOLDS
from app.eval.report import GATED_METRICS, OOS_REFUSAL_TARGET

REPORT_GLOB = "eval_*.json"


def latest_report(reports_dir: Path) -> Path | None:
    """返回目录下最新的评测报告 JSON；无报告返回 None。"""
    files = sorted(glob.glob(str(reports_dir / REPORT_GLOB)))
    return Path(files[-1]) if files else None


def check_report(
    report: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
    max_drop: float = 0.05,
) -> list[dict[str, Any]]:
    """校验报告，返回违规项列表（空列表 = 通过）。"""
    violations: list[dict[str, Any]] = []
    overall = report.get("metrics", {}).get("overall", {}) or {}

    for key in GATED_METRICS:
        value = overall.get(key)
        if value is None:
            violations.append(
                {
                    "kind": "missing_metric",
                    "metric": key,
                    "detail": f"门槛指标 {key} 在报告中缺失（可能是评测未跑完）",
                }
            )
        elif value < THRESHOLDS[key]:
            violations.append(
                {
                    "kind": "below_threshold",
                    "metric": key,
                    "value": value,
                    "threshold": THRESHOLDS[key],
                    "detail": f"{key}={value:.4f} 低于门槛 {THRESHOLDS[key]:.2f}",
                }
            )

    refusal = (report.get("retrieval") or {}).get("oos_refusal_rate")
    if refusal is None or refusal < OOS_REFUSAL_TARGET:
        violations.append(
            {
                "kind": "oos_refusal",
                "value": refusal,
                "threshold": OOS_REFUSAL_TARGET,
                "detail": f"诱导题拒答率 {refusal} 未达目标 {OOS_REFUSAL_TARGET:.0%}",
            }
        )

    errors = report.get("errors") or []
    if errors:
        ids = ", ".join(e.get("id", "?") for e in errors[:5])
        violations.append(
            {
                "kind": "exec_errors",
                "count": len(errors),
                "detail": f"评测存在 {len(errors)} 条执行失败样例：{ids}",
            }
        )

    if baseline:
        base_metrics = (baseline.get("metrics") or {}).get("overall", {}) or {}
        for key in METRIC_KEYS:
            new, old = overall.get(key), base_metrics.get(key)
            if new is None or old is None:
                continue
            drop = old - new
            if drop > max_drop:
                violations.append(
                    {
                        "kind": "baseline_drop",
                        "metric": key,
                        "value": new,
                        "baseline": old,
                        "drop": round(drop, 4),
                        "max_drop": max_drop,
                        "detail": (
                            f"{key} 相对基线下降 {drop:.4f}（> 允许 {max_drop:.4f}）："
                            f"{old:.4f} → {new:.4f}"
                        ),
                    }
                )
    return violations
