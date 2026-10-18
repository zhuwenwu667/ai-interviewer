"""AI 模拟面试官 Streamlit 前端入口（三页式：开始 / 面试 / 报告）。

启动：``uv run streamlit run frontend/app.py``

- 后端地址默认读环境变量 ``API_BASE_URL``（默认 http://localhost:8000），
  也可在侧边栏「高级：连接与密钥」中修改；
- 可选密钥仅保存于本次浏览器会话（session_state）与请求头，不落盘；
- URL 参数 ``?session=<id>&view=<start|interview|report>``：刷新页面、切换标签页或
  重新打开链接都能续接同一场面试（此前只存在 session_state，刷新即丢整场会话）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 允许 frontend 内部模块导入

import views_metrics  # noqa: E402
import views_train  # noqa: E402

import views_interview  # noqa: E402
import views_report  # noqa: E402
import views_start  # noqa: E402
from api_client import DEFAULT_BASE_URL, ApiClientError, BackendClient  # noqa: E402
from ui_common import clear_error, render_error_banner  # noqa: E402

PAGE_LABELS = {
    "start": "模拟面试",
    "train": "刷题训练",
    "metrics": "效果看板",
    "interview": "面试进行中",
    "report": "评估报告",
}
VALID_VIEWS = tuple(PAGE_LABELS)

# 全局样式：收窄内容宽度，Hero 用左侧竖条定重心，状态徽标与诊断卡片定版式。
# 只覆盖 .block-container 与稳定的 data-testid 容器，避免依赖易变的内部类名。
_STYLES = """
<style>
  .block-container { max-width: 1080px; padding-top: 1.8rem; padding-bottom: 3rem; }

  .iv-hero {
    padding: 0.2rem 0 0.85rem 1.05rem;
    border-left: 3px solid #0F5F59;
    margin-bottom: 0.85rem;
  }
  .iv-hero-badge {
    display: inline-block; font-size: 0.75rem; letter-spacing: 0.02em;
    color: #0F5F59; background: #E6F2F0; border-radius: 999px;
    padding: 3px 10px; margin-bottom: 10px;
  }
  .iv-hero-title {
    font-size: 1.95rem; line-height: 1.25; font-weight: 700;
    margin: 0 0 0.55rem; color: #142B2A; text-wrap: balance;
  }
  .iv-hero-sub {
    font-size: 1rem; line-height: 1.7; color: #4A5C60;
    margin: 0 0 0.85rem; max-width: 38rem;
  }
  /* 「报告维度」与 chips 同行排布：标签不再单独占一行，省首屏高度 */
  .iv-hero-dims-wrap { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .iv-dims-lead {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; color: #6B7C80;
  }
  .iv-hero-dims {
    display: flex; flex-wrap: wrap; gap: 8px;
    list-style: none; padding: 0; margin: 0;
  }
  .iv-hero-dims li {
    font-size: 0.82rem; color: #33474D; background: transparent;
    border: 1px solid #E1EAE9; border-radius: 8px; padding: 4px 10px;
  }

  /* CTA 旁的一行内联状态：服务与两个模型的配置情况 */
  .iv-status {
    display: flex; flex-wrap: wrap; gap: 4px 16px;
    font-size: 0.8rem; color: #4A5C60; margin: 0.4rem 0 0;
  }
  .iv-status span { white-space: nowrap; }
  .iv-dot {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    margin-right: 5px; vertical-align: middle;
  }
  .iv-dot.ok { background: #0F5F59; }
  .iv-dot.warn { background: #C2771A; }
  .iv-dot.bad { background: #B3261E; }

  div[data-testid="stMetric"] {
    background: #F8FAFA; border: 1px solid #E8EEEE;
    border-radius: 10px; padding: 10px 14px;
  }

  /* 窄屏（手机）：实测 390×844 下主按钮原落在折叠线下 54px，故收一档 Hero 排版把 CTA 拉回 */
  @media (max-width: 480px) {
    .block-container { padding-top: 1.0rem; }
    .iv-hero { padding-bottom: 0.5rem; margin-bottom: 0.5rem; }
    .iv-hero-badge { margin-bottom: 6px; }
    .iv-hero-title { font-size: 1.5rem; margin-bottom: 0.4rem; }
    .iv-hero-sub { font-size: 0.9rem; line-height: 1.6; margin-bottom: 0.6rem; }
    .iv-hero-dims { gap: 6px; }
    .iv-hero-dims li { font-size: 0.76rem; padding: 3px 8px; }
  }
</style>
"""


def _init_state() -> None:
    defaults: dict[str, object] = {
        "page": "start",
        "requested_page": None,
        "base_url": os.environ.get("API_BASE_URL", DEFAULT_BASE_URL),
        "deepseek_key": "",
        "siliconflow_key": "",
        "session_id": None,
        "auto_started": False,
        "report": None,
        "api_error": None,
        "api_error_code": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _build_client() -> BackendClient:
    return BackendClient(
        st.session_state["base_url"],
        deepseek_key=st.session_state["deepseek_key"],
        siliconflow_key=st.session_state["siliconflow_key"],
        train_token=st.session_state.get("train_token", ""),
    )


def _restore_from_url(client: BackendClient) -> None:
    """按 URL 参数恢复会话与页面，使刷新/换标签页/重开链接都能续接。

    仅在本次会话尚无 ``session_id``（即页面刚加载）时生效，
    以免覆盖用户在当前页面上的导航选择。
    """
    if st.session_state["session_id"]:
        return
    session_id = (st.query_params.get("session") or "").strip()
    view = (st.query_params.get("view") or "").strip()
    if session_id:
        try:
            client.get_session(session_id)  # 校验会话确实存在
        except ApiClientError:
            st.query_params.clear()  # 会话已不存在：清掉失效链接，静默回到开始页
            return
        st.session_state["session_id"] = session_id
    if view in VALID_VIEWS and (view in ("start", "train", "metrics") or st.session_state["session_id"]):
        st.session_state["page"] = view


def _sync_url() -> None:
    """把当前会话与页面写回 URL；值未变化时不写，避免无谓的重跑。"""
    session_id = st.session_state["session_id"] or ""
    page = st.session_state["page"]
    if session_id:
        if st.query_params.get("session") != session_id:
            st.query_params["session"] = session_id
        if st.query_params.get("view") != page:
            st.query_params["view"] = page
    elif st.query_params.get("session"):
        st.query_params.clear()


@st.dialog("确认放弃当前会话？")
def _confirm_restart() -> None:
    st.write("当前面试仍保留在服务端，但本页面会回到开始页，需要重新创建一场面试。")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("放弃并重新开始", type="primary"):
        st.session_state.update(session_id=None, auto_started=False, report=None)
        st.query_params.clear()
        clear_error()
        st.session_state["requested_page"] = "start"
        st.rerun()
    if cancel_col.button("继续当前面试"):
        st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("AI 模拟面试官")

        options = ["start", "train", "metrics"]
        if st.session_state["session_id"]:
            options += ["interview", "report"]
        if st.session_state["page"] not in options:
            st.session_state["page"] = "start"
        st.radio(
            "导航",
            options,
            format_func=lambda p: PAGE_LABELS[p],
            key="page",  # 绑定 session_state：其他页面可编程切换（如完成面试后跳报告页）
        )

        if st.session_state["session_id"] and st.button("重新开始（放弃当前会话）"):
            _confirm_restart()

        with st.expander("高级：连接与密钥", expanded=False):
            st.session_state["base_url"] = st.text_input(
                "后端 API 地址", value=st.session_state["base_url"]
            )
            st.caption("默认指向本服务的后端，仅自建或本地调试时才需要修改。")
            st.session_state["deepseek_key"] = st.text_input(
                "DeepSeek API Key", value=st.session_state["deepseek_key"], type="password"
            )
            st.session_state["siliconflow_key"] = st.text_input(
                "SiliconFlow API Key", value=st.session_state["siliconflow_key"], type="password"
            )
            st.caption(
                "密钥只保存在本次浏览器会话内存中（不落盘），但会随请求头发送给"
                "**本服务后端**用于调用模型，请勿填入与本次评估无关的密钥。"
            )


def main() -> None:
    st.set_page_config(page_title="AI 模拟面试官", page_icon="🎤", layout="wide")
    st.markdown(_STYLES, unsafe_allow_html=True)
    _init_state()

    client = _build_client()
    _restore_from_url(client)

    # 页面跳转采用「请求 → 下一轮生效」：必须在导航组件实例化前应用，
    # 否则会在同一轮内修改组件绑定的 session_state 键而抛
    # StreamlitWidgetAlreadyInstantiatedError（组件渲染后禁止改写其 key）。
    requested = st.session_state.pop("requested_page", None)
    if requested:
        st.session_state["page"] = requested

    _render_sidebar()
    _sync_url()

    render_error_banner()
    page = st.session_state["page"]
    if page == "start":
        views_start.render(client)
    elif page == "train":
        views_train.render(client)
    elif page == "metrics":
        views_metrics.render(client)
    elif page == "interview":
        views_interview.render(client)
    else:
        views_report.render(client)


main()
