"""面试页：对话式 UI + SSE 流式消费 + 阶段徽标 + 证据来源 chips + 提前结束。"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClientError, BackendClient
from ui_common import WAITING_LABEL, clear_error, render_message, set_error, stage_label


def run_turn(client: BackendClient, session_id: str, content: str) -> None:
    """消费一次 SSE 流：阶段徽标随事件更新，面试官文本逐段渲染。

    ``content`` 为空且面试未开始 → 触发开场与第一题；否则提交候选人回答。
    流结束后由调用方 ``st.rerun()``，以服务端历史为准重绘（保证证据 chips 完整）。
    """
    if content:
        with st.chat_message("user"):
            st.markdown(content)

    status = st.status(WAITING_LABEL, expanded=False)
    placeholder = st.empty()
    buffer = ""
    finished_reason: str | None = None
    try:
        for event in client.post_message(session_id, content):
            if event.event == "stage":
                stage = str(event.data.get("stage", ""))
                round_no = event.data.get("round") or 0
                label = f"当前阶段：{stage_label(stage)}"
                if round_no:
                    label += f"（第 {round_no} 题）"
                status.update(label=label)
            elif event.event == "token":
                buffer += str(event.data.get("content", ""))
                placeholder.markdown(buffer + "▌")
            elif event.event == "error":
                set_error(str(event.data.get("message", "")), str(event.data.get("code", "")))
            elif event.event == "done":
                finished_reason = str(event.data.get("reason", ""))
    except ApiClientError as exc:
        set_error(str(exc), exc.code)
        status.update(label="请求失败", state="error")
        return

    if buffer:
        placeholder.markdown(buffer)
    else:
        placeholder.empty()
    if finished_reason == "error":
        status.update(label="本轮执行失败，详见上方错误提示", state="error")
    elif finished_reason == "completed":
        status.update(label="面试结束，报告已生成", state="complete")
    else:
        status.update(label="本轮已完成", state="complete")


def render(client: BackendClient) -> None:
    session_id = st.session_state["session_id"]
    try:
        detail = client.get_session(session_id)
    except ApiClientError as exc:
        set_error(str(exc), exc.code)
        return

    session = detail["session"]
    messages = detail["messages"]
    finished = session["status"] == "finished"

    left, right = st.columns([3, 1])
    with left:
        st.subheader(f"② 面试进行中：{session['track']}（计划 {session['num_questions']} 题）")
    with right:
        st.caption(f"会话 ID：`{session_id[:8]}…`")

    for message in messages:
        render_message(message)

    if finished:
        st.success("本次面试已结束。")
        if st.button("查看评估报告 →", type="primary"):
            st.session_state["requested_page"] = "report"
            st.rerun()
        return

    if not messages:
        # 首次进入：自动发起开场 + 第一题（只执行一次，避免重跑循环）
        if not st.session_state.get("auto_started"):
            st.session_state["auto_started"] = True
            run_turn(client, session_id, "")
            st.rerun()
        return

    answer = st.chat_input("输入你的回答（Enter 提交）…")
    if answer:
        run_turn(client, session_id, answer)
        st.rerun()

    if st.button("提前结束并生成报告"):
        with st.spinner("正在汇总评估并生成报告…"):
            try:
                st.session_state["report"] = client.finish(session_id)
                clear_error()
                st.session_state["requested_page"] = "report"
                st.rerun()
            except ApiClientError as exc:
                set_error(str(exc), exc.code)
