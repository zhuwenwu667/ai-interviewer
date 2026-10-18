"""P2-② 上下文预算单测。

- 预算关闭（legacy）必须**逐值复刻**改造前行为（证据片段 150 字 / 已问问题不限 / 报告来源不限、片段 120 字）；
- 预算打开按配置裁剪：证据片段长度、已问问题取最近 N 条、报告来源条数上限与片段长度；
- 报告 prompt 构造：开启预算后知识来源被裁剪到上限、片段被截断；关闭时保留全部。
"""

from __future__ import annotations

from app.agent.context import ContextBudget
from app.agent.report import _build_messages
from app.agent.state import InterviewState
from app.config.settings import Settings


def _settings(**overrides) -> Settings:
    base: dict = {
        "_env_file": None,
        "deepseek_api_key": "sk-test-ds",
        "deepseek_base_url": "https://ds.test/v1",
        "siliconflow_api_key": "sk-test-sf",
        "siliconflow_base_url": "https://sf.test/v1",
    }
    base.update(overrides)
    return Settings(**base)


def test_legacy_reproduces_pre_change_behavior():
    legacy = ContextBudget.legacy()
    assert legacy.enabled is False
    assert legacy.evidence_snippet_chars == 150
    assert legacy.asked_history_max is None
    assert legacy.report_source_max is None
    assert legacy.report_source_snippet_chars == 120
    assert legacy.clip_evidence("x" * 400) == "x" * 150


def test_from_settings_switches_by_enabled_flag():
    settings = _settings(
        ctx_evidence_snippet_chars=100,
        ctx_asked_history_max=6,
        ctx_report_source_max=8,
        ctx_report_source_snippet_chars=60,
    )
    on = ContextBudget.from_settings(settings, enabled=True)
    assert on.enabled and on.evidence_snippet_chars == 100
    assert on.asked_history_max == 6 and on.report_source_max == 8

    off = ContextBudget.from_settings(settings, enabled=False)
    assert off.enabled is False and off.evidence_snippet_chars == 150
    assert off.asked_history_max is None and off.report_source_max is None


def test_recent_asked_keeps_last_n_only():
    on = ContextBudget.from_settings(_settings(ctx_asked_history_max=2), enabled=True)
    assert on.recent_asked(["q1", "q2", "q3", "q4"]) == ["q3", "q4"]
    assert on.recent_asked(["q1"]) == ["q1"]
    # 关闭时返回全量
    assert ContextBudget.legacy().recent_asked(["q1", "q2", "q3"]) == ["q1", "q2", "q3"]


def test_limit_sources_caps_and_preserves_order():
    sources = [{"chunk_id": f"c{i}"} for i in range(10)]
    on = ContextBudget.from_settings(_settings(ctx_report_source_max=3), enabled=True)
    assert [s["chunk_id"] for s in on.limit_sources(sources)] == ["c0", "c1", "c2"]
    assert len(ContextBudget.legacy().limit_sources(sources)) == 10


def _state_with_many_sources(n: int, text_len: int) -> InterviewState:
    evidence = [
        {
            "chunk_id": f"ev-{i}", "source": "qa.md", "heading": f"标题{i}",
            "score": 0.9, "text": "字" * text_len,
        }
        for i in range(n)
    ]
    return InterviewState(
        session_id="s-1",
        track="backend",
        num_questions=n,
        questions_asked=[
            {
                "question": f"Q{i}", "focus": "f", "evidence_ids": [f"ev-{i}"],
                "round": i + 1, "evidence": [evidence[i]],
            }
            for i in range(n)
        ],
        current_evidence=evidence,
        answers=[
            {"question": f"Q{i}", "answer": "答", "quality": "good",
             "feedback": "好", "followup_question": ""}
            for i in range(n)
        ],
        evaluation_summary={
            "total_questions": n, "answered": n,
            "quality_counts": {"good": n, "partial": 0, "poor": 0},
        },
    )


def _report_user_prompt(state: InterviewState, budget: ContextBudget) -> str:
    messages = _build_messages(state, budget)
    return messages[1]["content"]


def test_report_prompt_limits_sources_when_budget_on():
    state = _state_with_many_sources(n=12, text_len=500)
    budget = ContextBudget.from_settings(
        _settings(ctx_report_source_max=8, ctx_report_source_snippet_chars=60), enabled=True
    )
    prompt = _report_user_prompt(state, budget)
    # 12 个来源被裁剪到 8 条：ev-8..ev-11 不应出现
    assert "[ev-7]" in prompt and "[ev-8]" not in prompt
    assert prompt.count("字" * 60) >= 8
    assert ("字" * 61) not in prompt  # 片段被截断到 60 字


def test_report_prompt_keeps_all_sources_when_budget_off():
    state = _state_with_many_sources(n=12, text_len=500)
    prompt = _report_user_prompt(state, ContextBudget.legacy())
    assert "[ev-11]" in prompt  # 未裁剪
    assert ("字" * 120) in prompt  # 片段截断到 120
