"""AI 模拟面试官 Streamlit 前端入口（三页式：开始 / 面试 / 报告）。

启动：``uv run streamlit run frontend/app.py``
后端地址默认读环境变量 ``API_BASE_URL``（默认 http://localhost:8000），
也可在侧边栏修改；可选密钥仅保存于本次浏览器会话（session_state）与请求头，不落盘。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 允许 frontend 内部模块导入

import views_interview  # noqa: E402
import views_report  # noqa: E402
import views_start  # noqa: E402
from api_client import DEFAULT_BASE_URL, BackendClient  # noqa: E402
from ui_common import clear_error, render_error_banner  # noqa: E402

PAGE_LABELS = {"start": "① 开始面试", "interview": "② 面试进行中", "report": "③ 评估报告"}


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
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("AI 模拟面试官")
        st.caption("Agentic RAG + LangGraph · DeepSeek + SiliconFlow + Qdrant")

        st.session_state["base_url"] = st.text_input(
            "后端 API 地址", value=st.session_state["base_url"]
        )
        with st.expander("使用我的 API Key（可选）", expanded=False):
            st.caption("仅保存于本次浏览器会话内存，随请求头发送，不落盘。留空则用服务端 .env 配置。")
            st.session_state["deepseek_key"] = st.text_input(
                "DeepSeek API Key", value=st.session_state["deepseek_key"], type="password"
            )
            st.session_state["siliconflow_key"] = st.text_input(
                "SiliconFlow API Key", value=st.session_state["siliconflow_key"], type="password"
            )

        options = ["start"]
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
            st.session_state.update(
                session_id=None, auto_started=False, report=None, requested_page="start"
            )
            clear_error()
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="AI 模拟面试官", page_icon="🎤", layout="wide")
    _init_state()
    # 页面跳转采用「请求 → 下一轮生效」：必须在导航组件实例化前应用，
    # 否则会在同一轮内修改组件绑定的 session_state 键而抛
    # StreamlitWidgetAlreadyInstantiatedError（组件渲染后禁止改写其 key）。
    requested = st.session_state.pop("requested_page", None)
    if requested:
        st.session_state["page"] = requested
    _render_sidebar()
    client = _build_client()

    render_error_banner()
    page = st.session_state["page"]
    if page == "start":
        views_start.render(client)
    elif page == "interview":
        views_interview.render(client)
    else:
        views_report.render(client)


main()
