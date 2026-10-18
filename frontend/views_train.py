"""刷题训练页：登录 → 抽题 → 作答 → 单轮研判 + 答案要点 → 进度 / 错题本 / 历史。

与面试模式的差异（产品语义）：
- 无简历/JD、无追问、无报告——一题一评的短闭环，专注「题库八股」刻意练习；
- 题目从服务端题库抽取，错题优先（poor > partial > 未作答 > good，见后端 draw）；
- 作答提交后才展示参考答案要点（next 响应不含要点）；答不出可「显示答案」（不计进度）；
- 错题本汇总非 good 的题（题干 / 我的作答 / 点评 / 答案要点），可一键重练；
- 令牌只存 session_state（浏览器会话内存），401 时自动回落到登录态。
"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClientError, BackendClient
from ui_common import clear_error, set_error

# 研判 quality → 展示徽章（与 app.agent.assess 的 quality 取值一一对应）
_QUALITY_BADGES: dict[str, str] = {
    "good": "🟢 掌握良好",
    "partial": "🟡 部分掌握",
    "poor": "🔴 需要加强",
}


def _init_state() -> None:
    for key, default in {
        "train_token": "",
        "train_question": None,
        "train_result": None,
        "train_revealed": None,
        "train_stats": None,
    }.items():
        st.session_state.setdefault(key, default)


def _handle_error(exc: ApiClientError) -> None:
    """401 统一清令牌并立即 rerun 回登录态；其余错误透传 code 与中文 message。"""
    if exc.status == 401:
        st.session_state.update(
            train_token="", train_question=None, train_result=None, train_revealed=None
        )
        set_error("登录已失效，请重新登录。", exc.code)
        st.rerun()
    else:
        set_error(str(exc), exc.code)


def _draw_next(client: BackendClient, avoid_question_id: str | None = None) -> None:
    """抽下一题。``avoid_question_id`` 传当前题，避免「下一题」连出同一道题。"""
    try:
        with st.spinner("正在抽题…"):
            body = client.train_next(avoid_question_id)
    except ApiClientError as exc:
        _handle_error(exc)
        return
    clear_error()
    st.session_state["train_question"] = body["question"]
    st.session_state["train_stats"] = {
        "done_count": body["done_count"],
        "total_count": body["total_count"],
    }
    st.session_state["train_result"] = None
    st.session_state["train_revealed"] = None


def _render_login(client: BackendClient) -> None:
    st.markdown("### 刷题训练")
    st.caption("题库八股刻意练习：一题一评，作答后可见参考答案要点。需使用训练账号登录。")
    with st.form("train_login_form"):
        username = st.text_input("用户名", max_chars=100)
        password = st.text_input("密码", type="password", max_chars=200)
        submitted = st.form_submit_button("登录并开始刷题", type="primary")
    if not submitted:
        return
    if not username.strip() or not password:
        set_error("请输入用户名与密码。")
        return
    try:
        with st.spinner("正在登录…"):
            body = client.train_login(username.strip(), password)
    except ApiClientError as exc:
        set_error(str(exc), exc.code)
        return
    clear_error()
    st.session_state["train_token"] = body["token"]
    # client 是本次 rerun 前构建的（尚无令牌）；rerun 后由携带令牌的新 client 抽题
    st.rerun()


def _render_progress() -> None:
    stats = st.session_state.get("train_stats") or {}
    done, total = stats.get("done_count", 0), stats.get("total_count", 0)
    cols = st.columns([2, 5])
    cols[0].metric("已练习题数", f"{done} / {total}")
    if total:
        cols[1].progress(min(1.0, done / total), text=f"题库覆盖率 {done / total:.0%}")


def _render_question(client: BackendClient) -> None:
    question = st.session_state["train_question"]
    result = st.session_state["train_result"]

    st.markdown(f"#### {question['question']}")
    if question.get("tags"):
        st.caption("考点：" + "、".join(f"`{t}`" for t in question["tags"]) + f" ｜ 来源：{question['source']}")

    if result is None:
        answer = st.text_area("你的作答", height=180, key="train_answer_input")
        cols = st.columns([2, 2, 6])
        submitted = cols[0].button("提交作答", type="primary", disabled=not answer.strip())
        reveal_clicked = cols[1].button(
            "显示答案", help="答不出来时直接看答案要点；看答案不计入已练进度"
        )
        if submitted:
            try:
                with st.spinner("AI 研判中…"):
                    st.session_state["train_result"] = client.train_answer(
                        question["question_id"], answer.strip()
                    )
            except ApiClientError as exc:
                _handle_error(exc)
                return
            st.session_state["train_revealed"] = None
            clear_error()
            st.rerun()
        if reveal_clicked:
            try:
                with st.spinner("正在取答案要点…"):
                    st.session_state["train_revealed"] = client.train_reveal(
                        question["question_id"]
                    )["reference_points"]
            except ApiClientError as exc:
                _handle_error(exc)
                return
            clear_error()
            st.rerun()
        # 已看答案：展示要点，可继续作答（看完默写也算练）或直接下一题
        revealed = st.session_state["train_revealed"]
        if revealed:
            with st.expander("参考答案要点", expanded=True):
                for point in revealed:
                    st.markdown(f"- {point}")
            if st.button("下一题 →", type="primary"):
                _draw_next(client, question["question_id"])
                st.rerun()
        return

    # 作答后：研判结论 + 参考答案要点
    badge = _QUALITY_BADGES.get(result["quality"], result["quality"])
    st.markdown(f"**研判结论：{badge}**（与模拟面试同一评价口径）")
    st.info(result["feedback"] or "（无点评）")
    points = result.get("reference_points") or []
    if points:
        with st.expander("参考答案要点", expanded=True):
            for point in points:
                st.markdown(f"- {point}")
    if st.button("下一题 →", type="primary"):
        _draw_next(client, question["question_id"])
        st.rerun()


def _render_history(client: BackendClient) -> None:
    with st.expander("最近作答记录", expanded=False):
        try:
            history = client.train_history()
        except ApiClientError as exc:
            _handle_error(exc)
            return
        counts = history.get("quality_counts") or {}
        if counts:
            st.caption(
                "质量分布："
                + " ｜ ".join(f"{_QUALITY_BADGES.get(k, k)} × {v}" for k, v in counts.items())
            )
        attempts = history.get("attempts") or []
        if not attempts:
            st.caption("还没有作答记录。")
            return
        for item in attempts:
            badge = _QUALITY_BADGES.get(item["quality"], item["quality"])
            st.markdown(f"- {badge} **{item['question']}**  \n  你的作答：{item['answer'][:80]}")


def _retrain(item: dict) -> None:
    """一键重练：把错题本里的题放回作答区（清空上一轮研判与已看答案）。"""
    st.session_state["train_question"] = {
        "question_id": item["question_id"],
        "question": item["question"],
        "tags": item.get("tags") or [],
        "source": item["source"],
    }
    st.session_state["train_result"] = None
    st.session_state["train_revealed"] = None
    st.rerun()


def _render_mistakes(client: BackendClient) -> None:
    with st.expander("错题本", expanded=False):
        try:
            data = client.train_mistakes()
        except ApiClientError as exc:
            _handle_error(exc)
            return
        mistakes = data.get("mistakes") or []
        if not mistakes:
            st.caption("暂无错题——非 good 的题会自动收进这里。")
            return
        poor = sum(1 for m in mistakes if m.get("quality") == "poor")
        st.caption(f"共 {len(mistakes)} 道待巩固（其中 🔴 需要加强 {poor} 道，🔴 排在最前）")
        for item in mistakes:
            badge = _QUALITY_BADGES.get(item["quality"], item["quality"])
            st.markdown(f"**{badge}｜{item['question']}**")
            st.caption(f"你的作答：{item['answer']}")
            st.info(item["feedback"] or "（无点评）")
            points = item.get("reference_points") or []
            if points:
                with st.expander("参考答案要点", expanded=False):
                    for point in points:
                        st.markdown(f"- {point}")
            if st.button("一键重练", key=f"train_retrain_{item['question_id']}"):
                _retrain(item)


def render(client: BackendClient) -> None:
    _init_state()
    if not st.session_state["train_token"]:
        _render_login(client)
        return

    st.markdown("### 刷题训练")
    if st.session_state["train_question"] is None:
        _draw_next(client)
        if st.session_state["train_question"] is None:
            return  # 抽题失败（错误已上屏）
    _render_progress()
    _render_question(client)
    _render_mistakes(client)
    _render_history(client)

    if st.button("退出训练登录"):
        st.session_state.update(
            train_token="", train_question=None, train_result=None,
            train_revealed=None, train_stats=None,
        )
        st.rerun()
