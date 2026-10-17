"""开始页：产品说明（Hero）+ 知识库来源 + 岗位方向与题目数 + 服务诊断。

服务诊断（健康检查、模型配置）是部署信息，默认收进折叠区，
避免首屏视觉重心被运维状态占据；仅在真的不可用时才前置提示。
"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClientError, BackendClient
from ui_common import REPORT_DIMENSIONS, clear_error, set_error

# 岗位方向：对外只展示中文标签，对内仍用后端约定的 slug 取值（format_func 负责映射）
TRACK_LABELS: dict[str, str] = {
    "ai_app": "AI 应用开发",
    "backend": "后端开发",
    "python": "Python 开发",
    "llm_app": "LLM 应用开发",
}
CUSTOM_TRACK = "其他（自定义）"
ALLOWED_UPLOAD_TYPES = ["pdf", "md", "markdown", "txt"]
MAX_UPLOAD_MB = 10

_HERO = """
<div class="iv-hero">
  <span class="iv-hero-badge">简历驱动 · 评价可溯源</span>
  <h1 class="iv-hero-title">把你的简历，变成一场模拟面试</h1>
  <p class="iv-hero-sub">
    上传简历与岗位 JD，AI 面试官据此出题、逐题追问；结束时给出五维评分、亮点与改进建议，
    每处评价都能回溯到知识库原文。
  </p>
  <p class="iv-hero-dims-label">报告维度</p>
  <ul class="iv-hero-dims">{dims}</ul>
</div>
"""


def _hero_html() -> str:
    dims = "".join(f"<li>{name}</li>" for name in REPORT_DIMENSIONS)
    return _HERO.format(dims=dims)


def _render_diagnostics(client: BackendClient) -> None:
    """读取 /health：设置 server_ready 供按钮禁用判断，并把详情收进折叠区。"""
    try:
        health = client.health()
    except ApiClientError as exc:
        st.error(
            f"后端不可达：{exc}。请确认服务已启动，"
            "或检查侧边栏「高级：连接与密钥」中的 API 地址。"
        )
        st.session_state["server_ready"] = False
        return

    config = health.get("config", {})
    qdrant = health.get("components", {}).get("qdrant", {})
    keys_ready = bool(config.get("deepseek_configured") and config.get("siliconflow_configured"))
    st.session_state["server_ready"] = True
    st.session_state["server_keys_ready"] = keys_ready

    if qdrant.get("status") != "ok":
        st.warning(f"向量库状态异常：{qdrant.get('detail', '未知原因')}")
    if not keys_ready:
        st.info(
            "服务端未配置模型 API Key，可在左侧「高级：连接与密钥」中临时填写"
            "（仅本次浏览器会话内存保存，不落盘），或在服务器 .env 中配置。"
        )

    with st.expander("服务诊断（部署信息）", expanded=False):
        cols = st.columns(3)
        cols[0].metric("服务状态", "healthy" if health.get("status") == "healthy" else "degraded")
        cols[1].metric(
            "DeepSeek",
            "已配置" if config.get("deepseek_configured") else "未配置",
            help=f"对话模型：{config.get('deepseek_model', '')}",
        )
        cols[2].metric(
            "SiliconFlow",
            "已配置" if config.get("siliconflow_configured") else "未配置",
            help=(
                f"向量：{config.get('embedding_model', '')} ｜ "
                f"重排：{config.get('reranker_model', '')}"
            ),
        )


def _uploaded_tuple(uploaded) -> tuple[str, bytes, str]:
    return (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")


def render(client: BackendClient) -> None:
    st.markdown(_hero_html(), unsafe_allow_html=True)
    _render_diagnostics(client)

    source = st.radio(
        "知识库来源",
        ["用示例题库直接体验", "上传我的简历与 JD"],
        horizontal=True,
        help="选择示例将直接使用服务端已摄入的示例简历、JD 与题库。",
    )
    resume_file = jd_file = None
    if source.startswith("上传"):
        up_cols = st.columns(2)
        resume_file = up_cols[0].file_uploader(
            "简历（可选）", type=ALLOWED_UPLOAD_TYPES
        )
        jd_file = up_cols[1].file_uploader(
            "岗位 JD（可选）", type=ALLOWED_UPLOAD_TYPES
        )
        st.caption(
            f"支持 {', '.join(t.upper() for t in ALLOWED_UPLOAD_TYPES)}，"
            f"单个文件不超过 {MAX_UPLOAD_MB} MB；两者都不传则等同使用示例知识库。"
        )

    form_cols = st.columns([3, 2, 2])
    track_choice = form_cols[0].selectbox(
        "岗位方向",
        [*TRACK_LABELS, CUSTOM_TRACK],
        format_func=lambda slug: TRACK_LABELS.get(slug, slug),
    )
    track = track_choice
    if track_choice == CUSTOM_TRACK:
        track = form_cols[0].text_input(
            "自定义方向（英文或中文均可）",
            value="",
            max_chars=60,
            placeholder="例如：数据平台 / 推荐算法",
        )
    num_questions = form_cols[1].slider("题目数量", 1, 10, value=3)
    candidate_name = form_cols[2].text_input("你的称呼（可选）", value="", max_chars=40)

    disabled = not st.session_state.get("server_ready", False) or not track.strip()
    if st.button("开始这场模拟面试 →", type="primary", disabled=disabled):
        for uploaded in (resume_file, jd_file):
            if uploaded is not None and len(uploaded.getvalue()) > MAX_UPLOAD_MB * 1024 * 1024:
                set_error(
                    f"文件「{uploaded.name}」超过 {MAX_UPLOAD_MB} MB 上限，"
                    "请压缩或截取后再上传。"
                )
                return
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
        st.query_params["session"] = record["session_id"]  # 写入 URL，刷新后仍可续接
        st.rerun()
