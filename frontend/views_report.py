"""报告页：雷达图 + 维度分条 + 评语 + 证据引用 + 改进建议 + 推荐复习知识点 + 下载。"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClientError, BackendClient

DIMENSION_MAX = 5


def _render_markdown(report: dict) -> str:
    """按 schemas 字段生成本地 Markdown（用于下载，与后端报告内容结构一致）。"""
    lines = [
        "# 面试评估报告",
        "",
        f"- 会话 ID：{report.get('session_id', '')}",
        f"- 生成时间：{report.get('generated_at', '')}",
        "",
        "## 维度评分",
        "",
        "| 维度 | 分数 | 评语 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {d['dimension']} | {d['score']}/{DIMENSION_MAX} | {d.get('comment', '')} |"
        for d in report.get("dimensions", [])
    ]
    lines += ["", "### 评分依据", ""]
    for d in report.get("dimensions", []):
        lines.append(f"- **{d['dimension']}**")
        lines += [f"  - {ev}" for ev in d.get("evidence", [])]
    for title, key in (("亮点", "highlights"), ("不足", "weaknesses"), ("改进建议", "suggestions")):
        lines += ["", f"## {title}", ""]
        lines += [f"{i}. {item}" for i, item in enumerate(report.get(key, []), 1)]
    lines += ["", "## 推荐复习知识点", ""]
    topics = report.get("recommended_topics") or []
    if topics:
        for t in topics:
            src = f"（{t['source']}）" if t.get("source") else ""
            reason = f"：{t['reason']}" if t.get("reason") else ""
            lines.append(f"- **{t['topic']}**{src}{reason}")
    else:
        lines.append("- （无）")
    lines.append("")
    return "\n".join(lines)


def _radar_chart(dimensions: list[dict]) -> None:
    import plotly.graph_objects as go

    labels = [d["dimension"] for d in dimensions]
    scores = [d["score"] for d in dimensions]
    fig = go.Figure(
        go.Scatterpolar(
            r=[*scores, scores[0]] if scores else [],
            theta=[*labels, labels[0]] if labels else [],
            fill="toself",
            name="维度得分",
        )
    )
    fig.update_layout(
        polar={"radialaxis": {"range": [0, DIMENSION_MAX], "dtick": 1}},
        showlegend=False,
        height=420,
        margin={"l": 60, "r": 60, "t": 40, "b": 40},
    )
    st.plotly_chart(fig, width="stretch")


def render(client: BackendClient) -> None:
    session_id = st.session_state["session_id"]
    st.subheader("③ 评估报告")

    report = st.session_state.get("report")
    if report is None:
        try:
            report = client.get_report(session_id)
        except ApiClientError as exc:
            st.info("报告尚未生成：请先完成面试，或在面试页点击「提前结束并生成报告」。")
            st.caption(f"（{exc.code}：{exc}）" if exc.code else f"（{exc}）")
            return
        st.session_state["report"] = report

    dimensions = report.get("dimensions", [])
    st.caption(f"会话 ID：`{session_id}` ｜ 生成时间：{report.get('generated_at', '')}")

    chart_col, score_col = st.columns([3, 2])
    with chart_col:
        _radar_chart(dimensions)
    with score_col:
        st.markdown("**维度得分**")
        for d in dimensions:
            st.markdown(f"{d['dimension']}：**{d['score']}/{DIMENSION_MAX}**")
            st.progress(min(1.0, d["score"] / DIMENSION_MAX))
            if d.get("comment"):
                st.caption(d["comment"])

    st.divider()
    ev_col, adv_col = st.columns(2)
    with ev_col:
        st.markdown("**评分依据（问答证据）**")
        for d in dimensions:
            with st.expander(f"{d['dimension']}（{d['score']}/{DIMENSION_MAX}）"):
                for ev in d.get("evidence", []):
                    st.markdown(f"- {ev}")
    with adv_col:
        st.markdown("**亮点**")
        for item in report.get("highlights", []):
            st.markdown(f"- {item}")
        st.markdown("**不足**")
        for item in report.get("weaknesses", []):
            st.markdown(f"- {item}")
        st.markdown("**改进建议**")
        for i, item in enumerate(report.get("suggestions", []), 1):
            st.markdown(f"{i}. {item}")

    st.divider()
    st.markdown("**推荐复习知识点（回链题库来源）**")
    topics = report.get("recommended_topics") or []
    if topics:
        st.dataframe(
            [
                {
                    "知识点": t["topic"],
                    "题库来源": t.get("source") or "（未标注）",
                    "推荐理由": t.get("reason", ""),
                }
                for t in topics
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("本次面试未给出推荐知识点。")

    st.download_button(
        "下载报告（Markdown）",
        data=_render_markdown(report).encode("utf-8"),
        file_name=f"interview_report_{session_id[:8]}.md",
        mime="text/markdown",
    )
