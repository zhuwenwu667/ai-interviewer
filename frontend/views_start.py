"""开始页：API 连通状态、知识库来源（内置示例 / 上传）、岗位方向与题目数。"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClientError, BackendClient
from ui_common import clear_error, set_error

TRACK_OPTIONS = ["ai_app", "backend", "python", "llm_app"]
CUSTOM_TRACK = "其他（自定义）"
ALLOWED_UPLOAD_TYPES = ["pdf", "md", "markdown", "txt"]


def _render_health(client: BackendClient) -> None:
    try:
        health = client.health()
    except ApiClientError as exc:
        st.error(f"后端不可达：{exc}。请确认服务已启动，或检查侧边栏中的 API 地址。")
        st.session_state["server_ready"] = False
        return

    config = health.get("config", {})
    qdrant = health.get("components", {}).get("qdrant", {})
    keys_ready = bool(config.get("deepseek_configured") and config.get("siliconflow_configured"))
    st.session_state["server_ready"] = True
    st.session_state["server_keys_ready"] = keys_ready

    cols = st.columns(3)
    cols[0].metric("服务状态", "healthy" if health.get("status") == "healthy" else "degraded")
    cols[1].metric(
        "DeepSeek", "已配置" if config.get("deepseek_configured") else "未配置",
        help=f"对话模型：{config.get('deepseek_model', '')}",
    )
    cols[2].metric(
        "SiliconFlow", "已配置" if config.get("siliconflow_configured") else "未配置",
        help=f"向量：{config.get('embedding_model', '')} ｜ 重排：{config.get('reranker_model', '')}",
    )
    if qdrant.get("status") != "ok":
        st.warning(f"向量库状态异常：{qdrant.get('detail', '未知原因')}")
    if not keys_ready:
        st.info(
            "服务端未配置 API Key。可在左侧「使用我的 API Key（可选）」中临时填写"
            "（仅本次浏览器会话内存保存，不落盘），或在服务器 .env 中配置。"
        )


def _uploaded_tuple(uploaded) -> tuple[str, bytes, str]:
    return (uploaded.name, uploaded.read(), uploaded.type or "application/octet-stream")


def render(client: BackendClient) -> None:
    st.subheader("① 开始一场模拟面试")
    st.caption("知识库包含内置示例（中文简历 + 中英文 JD + 后端/AI 题库），也可上传你自己的简历与 JD。")

    _render_health(client)

    source = st.radio(
        "知识库来源",
        ["使用内置示例知识库", "上传我的简历 / JD"],
        horizontal=True,
        help="选择内置示例将直接使用服务端已摄入的示例简历、JD 与题库。",
    )
    resume_file = jd_file = None
    if source.startswith("上传"):
        up_cols = st.columns(2)
        resume_file = up_cols[0].file_uploader(
            "简历（可选，PDF / Markdown / TXT）", type=ALLOWED_UPLOAD_TYPES
        )
        jd_file = up_cols[1].file_uploader(
            "岗位 JD（可选，PDF / Markdown / TXT）", type=ALLOWED_UPLOAD_TYPES
        )
        if resume_file is None and jd_file is None:
            st.caption("未选择文件时，将等同使用内置示例知识库。")

    form_cols = st.columns([2, 1, 2])
    track_choice = form_cols[0].selectbox("岗位方向", [*TRACK_OPTIONS, CUSTOM_TRACK])
    track = track_choice
    if track_choice == CUSTOM_TRACK:
        track = form_cols[0].text_input("自定义方向（英文或中文均可）", value="")
    num_questions = form_cols[1].slider("计划题目数", 1, 10, value=3)
    candidate_name = form_cols[2].text_input("候选人称呼（可选）", value="")

    disabled = not st.session_state.get("server_ready", False) or not track.strip()
    if st.button("开始面试 →", type="primary", disabled=disabled):
        resume = _uploaded_tuple(resume_file) if resume_file else None
        jd = _uploaded_tuple(jd_file) if jd_file else None
        try:
            with st.spinner("正在创建会话并摄入上传文档…"):
                record = client.create_session(
                    track=track.strip(),
                    num_questions=num_questions,
                    candidate_name=candidate_name.strip() or None,
                    resume=resume,
                    jd=jd,
                )
        except ApiClientError as exc:
            set_error(str(exc), exc.code)
            return
        clear_error()
        st.session_state.update(
            session_id=record["session_id"],
            auto_started=False,
            report=None,
            requested_page="interview",  # 下一轮生效（避免同轮改写导航组件 key）
        )
        st.rerun()
