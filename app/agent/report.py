"""多维评分与报告生成（Task 8）：LLM 结构化输出 + Pydantic 校验 + Markdown 持久化。

- 输入整场面试的问答记录（含研判结论）与知识来源，输出五维度结构化报告；
- JSON mode + ``InterviewReport`` 校验 + 五维度齐全性检查，失败重试 ≤2 次，
  连续非法输出抛 ``ReportGenerationError``（拒绝脏数据落库）；
- 成功后同时产出 Markdown 版（``reports_dir/{session_id}.md``）与结构化 JSON
  （JSON 由 ``graph._node_report`` 经 ``SessionRepository.save_report`` 持久化）。
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.agent.state import InterviewState
from app.api.schemas import REPORT_DIMENSIONS, InterviewReport
from app.config.settings import Settings
from app.core.llm import ChatMessage, DeepSeekChatClient

logger = logging.getLogger(__name__)

MAX_REPORT_RETRIES = 2  # 结构化输出失败后最多重试 2 次（共 3 次尝试）

_DIMENSIONS_TEXT = "、".join(REPORT_DIMENSIONS)

_REPORT_SYSTEM = (
    "你是技术面试主考官，根据整场面试的问答记录（含逐题研判）输出结构化评估报告。"
    f"五个评分维度固定为：{_DIMENSIONS_TEXT}，每维度 1-5 分整数，"
    "评分证据必须引用具体问答原文摘录。"
    "严格输出 JSON（不要输出其他内容）："
    '{"dimensions": [{"dimension": "维度名", "score": 1-5, '
    '"evidence": ["问答证据摘录", ...], "comment": "评语"}, ...], '
    '"highlights": ["亮点", ...], "weaknesses": ["不足", ...], '
    '"suggestions": ["可执行的改进建议", ...], '
    '"recommended_topics": [{"topic": "知识点", "source": "题库来源", "reason": "推荐理由"}, ...]}。'
    "要求：dimensions 恰好包含上述五个维度；highlights/weaknesses/suggestions "
    "必须至少各一条（禁止空数组，无明显亮点时写相对较好的一点）；"
    "recommended_topics 的 source 只填知识来源列表中的来源文件名（如 qa_backend.md）；"
    "若面试无有效问答，各维度给 1 分，证据填“候选人未作答”。"
)


class ReportGenerationError(RuntimeError):
    """连续多次无法从 LLM 获得合法报告（明确失败，不产生脏数据）。"""


# ---------------------------------------------------------------------------
# prompt 构造与输出解析
# ---------------------------------------------------------------------------
def _build_messages(state: InterviewState) -> list[ChatMessage]:
    qa_lines = [
        f"{i}. 问：{a.get('question', '')}\n"
        f"   答：{a.get('answer', '')}\n"
        f"   研判：{a.get('quality', '')}；点评：{a.get('feedback', '')}"
        for i, a in enumerate(state.get("answers", []), 1)
    ]
    qa_text = "\n".join(qa_lines) or "（无有效问答记录）"

    # 知识来源按 chunk_id 去重（出题证据快照 + 当前证据）
    sources: dict[str, dict] = {}
    for q in state.get("questions_asked", []):
        for e in q.get("evidence", []):
            sources.setdefault(e["chunk_id"], e)
    for e in state.get("current_evidence", []):
        sources.setdefault(e["chunk_id"], e)
    src_text = "\n".join(
        f"- [{e['chunk_id']}] {e.get('heading') or '（无标题）'}（{e.get('source', '')}）："
        f"{str(e.get('text', ''))[:120]}"
        for e in sources.values()
    ) or "（无）"

    summary = json.dumps(state.get("evaluation_summary", {}), ensure_ascii=False)
    user = (
        f"岗位方向：{state.get('track', '')}\n"
        f"计划题数：{state.get('num_questions', 0)}\n"
        f"评估汇总：{summary}\n\n"
        f"问答记录（含追问与研判）：\n{qa_text}\n\n"
        f"知识来源（recommended_topics.source 须从此选取）：\n{src_text}\n\n"
        "请输出报告 JSON。"
    )
    return [
        {"role": "system", "content": _REPORT_SYSTEM},
        {"role": "user", "content": user},
    ]


def _parse_report(content: str, session_id: str) -> InterviewReport:
    """解析并校验 LLM 报告输出；不合法时抛 JSONDecodeError/ValidationError/ValueError。"""
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("报告输出不是 JSON 对象")
    data["session_id"] = session_id  # 不信任模型输出，以状态为准
    data["generated_at"] = datetime.now(UTC).isoformat()
    report = InterviewReport.model_validate(data)
    names = {d.dimension for d in report.dimensions}
    if names != set(REPORT_DIMENSIONS):
        raise ValueError(f"评分维度与要求的五个维度不匹配：{sorted(names)}")
    return report


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------
def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_report_markdown(report: InterviewReport) -> str:
    """将结构化报告渲染为中文 Markdown 文档。"""
    lines = [
        "# 面试评估报告",
        "",
        f"- 会话 ID：{report.session_id}",
        f"- 生成时间：{report.generated_at:%Y-%m-%d %H:%M:%S} UTC",
        "",
        "## 维度评分",
        "",
        "| 维度 | 分数 | 评语 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {d.dimension} | {d.score}/5 | {_md_cell(d.comment)} |" for d in report.dimensions
    ]
    lines += ["", "### 评分依据", ""]
    for d in report.dimensions:
        lines.append(f"- **{d.dimension}**")
        lines += [f"  - {_md_cell(ev)}" for ev in d.evidence]

    def _section(title: str, items: list[str]) -> None:
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"{i}. {_md_cell(item)}" for i, item in enumerate(items, 1))

    _section("亮点", report.highlights)
    _section("不足", report.weaknesses)
    _section("改进建议", report.suggestions)
    lines.extend(["", "## 推荐复习知识点", ""])
    if report.recommended_topics:
        for t in report.recommended_topics:
            src = f"（{t.source}）" if t.source else ""
            reason = f"：{t.reason}" if t.reason else ""
            lines.append(f"- **{t.topic}**{src}{reason}")
    else:
        lines.append("- （无）")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 生成器（符合 graph.ReportGenerator 协议：await generator(state) -> InterviewReport）
# ---------------------------------------------------------------------------
class LLMReportGenerator:
    """LLM 结构化报告生成器；``reports_dir`` 为 None 时仅返回 JSON 不写 Markdown。"""

    def __init__(
        self,
        llm: DeepSeekChatClient,
        reports_dir: Path | None = None,
        *,
        max_retries: int = MAX_REPORT_RETRIES,
    ) -> None:
        self._llm = llm
        self._reports_dir = reports_dir
        self._max_retries = max_retries

    async def __call__(self, state: InterviewState) -> InterviewReport:
        messages = _build_messages(state)
        attempts = self._max_retries + 1
        last_error = "未知错误"
        for attempt in range(1, attempts + 1):
            resp = await self._llm.chat(messages, purpose="eval", json_mode=True)
            try:
                report = _parse_report(resp.content, state["session_id"])
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = " ".join(str(exc).split())[:300]
                logger.warning(
                    "报告输出校验失败（第 %d/%d 次）：%s", attempt, attempts, last_error
                )
                # 把非法输出与校验错误反馈给模型，下一轮定向修正
                messages = [
                    *messages,
                    {"role": "assistant", "content": resp.content},
                    {
                        "role": "user",
                        "content": f"上次输出未通过校验：{last_error}。"
                        "请修正并重新输出完整 JSON（不要输出其他内容）。",
                    },
                ]
                continue
            self._write_markdown(report)
            logger.info(
                "LLM 报告生成成功 session=%s 尝试次数=%d", state["session_id"], attempt
            )
            return report
        raise ReportGenerationError(
            f"连续 {attempts} 次未能从 LLM 获得合法报告，已放弃生成"
            f"（最后错误：{last_error}）。请稍后重试或检查模型输出。"
        )

    def _write_markdown(self, report: InterviewReport) -> None:
        if self._reports_dir is None:
            return
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        path = self._reports_dir / f"{report.session_id}.md"
        path.write_text(render_report_markdown(report), encoding="utf-8")
        logger.info("Markdown 报告已写入 %s", path)


def build_report_generator(settings: Settings, llm: DeepSeekChatClient) -> LLMReportGenerator:
    """按项目配置装配报告生成器（Markdown 落 data/reports/）。"""
    return LLMReportGenerator(llm, reports_dir=settings.resolve("data/reports"))
