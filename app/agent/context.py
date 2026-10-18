"""P2-② 上下文预算：压缩喂入「出题 / 研判 / 报告」的上下文体量。

背景（埋点按父节点归属 token 后实测）：一次 6 场 × 6 轮的面试里，token 花在
- `node.report`：约 3.5k tok/次（知识来源清单是大头，6 次共约 20.8k）；
- `node.assess_answer`：约 480 tok/次（证据片段 + 候选人回答，36 次共约 21.6k）；
- `node.ask_question`：约 400 tok/次（证据片段 + 已问问题清单）。

于是把「喂进去的上下文」做成**显式预算**：证据片段截断长度、已问问题回带条数、
报告知识来源条数与片段长度。``enabled=False`` 时回落到改造前的原值（150 字片段、
不限条数、120 字来源片段），便于**同环境 A/B 对照**；生产默认开启。

取舍：压缩的是**喂给模型的上下文**，不是检索本身（召回条数不变）。片段截短会削弱
「追问结合简历细节」的细粒度，以及报告 recommended_topics 的依据丰富度——这是
用少量细节换 token/成本，必须用真实跑数来判定是否划算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config.settings import Settings

# 改造前的行为（作为「预算关闭」基线，保证 A/B 可比）
LEGACY_EVIDENCE_SNIPPET_CHARS = 150
LEGACY_REPORT_SOURCE_SNIPPET_CHARS = 120


@dataclass(frozen=True)
class ContextBudget:
    """上下文预算；各上限为 ``None`` 表示不限制（沿用改造前行为）。"""

    enabled: bool
    evidence_snippet_chars: int
    asked_history_max: int | None
    report_source_max: int | None
    report_source_snippet_chars: int

    @classmethod
    def from_settings(cls, settings: Settings, *, enabled: bool | None = None) -> ContextBudget:
        """按配置构造；``enabled`` 显式给出时覆盖 ``settings.ctx_budget_enabled``（供 A/B）。"""
        on = settings.ctx_budget_enabled if enabled is None else enabled
        if on:
            return cls(
                enabled=True,
                evidence_snippet_chars=settings.ctx_evidence_snippet_chars,
                asked_history_max=settings.ctx_asked_history_max,
                report_source_max=settings.ctx_report_source_max,
                report_source_snippet_chars=settings.ctx_report_source_snippet_chars,
            )
        return cls.legacy()

    @classmethod
    def legacy(cls) -> ContextBudget:
        """预算关闭：完全复刻改造前的截断行为。"""
        return cls(
            enabled=False,
            evidence_snippet_chars=LEGACY_EVIDENCE_SNIPPET_CHARS,
            asked_history_max=None,
            report_source_max=None,
            report_source_snippet_chars=LEGACY_REPORT_SOURCE_SNIPPET_CHARS,
        )

    def clip_evidence(self, text: Any) -> str:
        return str(text)[: self.evidence_snippet_chars]

    def recent_asked(self, questions: list[str]) -> list[str]:
        if self.asked_history_max is None:
            return list(questions)
        return list(questions[-self.asked_history_max:])

    def limit_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.report_source_max is None:
            return list(sources)
        return list(sources[: self.report_source_max])
