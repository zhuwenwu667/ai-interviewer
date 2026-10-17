"""前端共享 UI 组件与文案映射（三页复用）。"""

from __future__ import annotations

import streamlit as st

# 阶段徽标文案（与 schemas.InterviewStage 一一对应，无魔法字段）
STAGE_LABELS: dict[str, str] = {
    "opening": "开场",
    "question": "出题",
    "followup": "追问",
    "answer": "研判回答",
    "scoring": "评分",
    "report": "生成报告",
    "done": "已完成",
}

# 报告评分维度（与 schemas.REPORT_DIMENSIONS 一一对应，开始页据此预告产出）
REPORT_DIMENSIONS: tuple[str, ...] = (
    "岗位匹配度",
    "基础知识",
    "项目深度",
    "表达逻辑",
    "问题解决",
)

# 等待首个事件时的提示（该窗口期正在做检索 + 出题生成）
WAITING_LABEL = "检索中，正在组织问题…"

_ROLE_ALIASES = {
    "interviewer": ("assistant", "面试官"),
    "candidate": ("user", "候选人"),
    "system": ("assistant", "系统"),
}


def stage_label(stage: str | None) -> str:
    return STAGE_LABELS.get(stage or "", stage or "")


def render_message(message: dict) -> None:
    """渲染一条 MessageRecord（角色气泡 + 阶段徽标 + 证据来源 chips）。"""
    role = message.get("role", "system")
    kind, who = _ROLE_ALIASES.get(role, ("assistant", role))
    with st.chat_message(kind):
        st.markdown(message.get("content", ""))
        meta: list[str] = []
        if message.get("stage"):
            meta.append(f"阶段：{stage_label(message['stage'])}")
        evidence = message.get("evidence") or []
        if evidence:
            chips = " ".join(
                f"`{e.get('source', '')}"
                + (f"·{e['heading']}`" if e.get("heading") else "`")
                for e in evidence
            )
            meta.append(f"证据来源：{chips}")
        if meta:
            st.caption(" ｜ ".join(meta))


def render_error_banner() -> None:
    """顶部错误横幅：透出后端返回的 code 与中文 message 原文。"""
    error = st.session_state.get("api_error")
    if not error:
        return
    code = st.session_state.get("api_error_code") or ""
    suffix = f"（错误码：{code}）" if code else ""
    st.error(f"{error}{suffix}")


def set_error(message: str, code: str = "") -> None:
    st.session_state["api_error"] = message
    st.session_state["api_error_code"] = code


def clear_error() -> None:
    st.session_state["api_error"] = None
    st.session_state["api_error_code"] = ""
