"""效果看板页：把离线评测的真实数字从文档搬进产品（只读、零 LLM）。

页面结构（自上而下）：
1. 合格线徽标 + 评测集规模 —— 一眼看到「是否达标、跑了多少题」；
2. 核心指标卡 —— RAGAS 门槛指标（含合格线）+ 耗时 / 成本 / 复练提升率；
3. 分类别指标表 —— 简历 / JD / 题库 / 题库外四类；
4. 检索消融表 —— 各检索配置的效果 vs 成本（回答「为什么值得这么搭」）；
5. 异源 judge 交叉验证 —— 证明评分不是「自评自夸」；
6. 成本与延迟、语义缓存 / 上下文预算 A/B 实测。

数据全部来自后端 `GET /metrics`（聚合已入库产物），本页不做任何检索或 LLM 调用。
"""

from __future__ import annotations

import streamlit as st

from api_client import ApiClientError, BackendClient


def _render_gate(data: dict) -> None:
    """顶部：合格线结论 + 评测集规模。"""
    passed = bool(data.get("gate_passed"))
    if passed:
        st.success("评测回归门禁：**通过**（门槛指标均达合格线，相对基线跌幅在允许范围内）")
    else:
        st.error("评测回归门禁：**未通过**")
        for item in data.get("gate_violations") or []:
            st.markdown(f"- {item}")

    dataset = data.get("dataset") or {}
    if dataset.get("total"):
        counts = " ｜ ".join(
            f"{key} {value}"
            for key, value in dataset.items()
            if key not in ("total", "executed")
        )
        st.caption(
            f"评测集：共 {dataset.get('total', 0)} 题"
            f"（执行 {dataset.get('executed', 0)} 题）｜{counts}"
        )
    sources = data.get("sources") or {}
    if sources:
        st.caption(
            "数据来源（每类取最新一份报告）："
            + " ｜ ".join(f"{key} → `{name}`" for key, name in sources.items())
        )


def _metric_label(row: dict) -> str:
    """指标卡标题：门槛指标带合格线，观察项标注为观察项。"""
    threshold = row.get("threshold")
    if threshold is None:
        return f"{row['name']}（观察）"
    return f"{row['name']}（≥ {threshold:g}）"


def _render_quality(data: dict) -> None:
    """RAGAS 门槛指标 + 检索侧指标 + 成本/延迟/复练。"""
    quality = data.get("quality") or []
    gated = [row for row in quality if row.get("threshold") is not None]
    if gated:
        st.markdown("#### 门槛指标（RAGAS，与 CI 门禁同一套合格线）")
        columns = st.columns(len(gated))
        for column, row in zip(columns, gated, strict=False):
            value = row.get("value")
            if value is None:
                continue
            delta = row.get("delta")
            column.metric(
                _metric_label(row),
                f"{value:.4f}",
                None if delta is None else f"{delta:+.4f}",
                delta_color="normal",
            )

    st.markdown("#### 检索与答案质量")
    rows = [
        {
            "指标": row["name"],
            "数值": (
                f"{row['value']:.2f} {row['unit']}"
                if row.get("value") is not None
                else "—"
            ),
            "合格线": "—" if row.get("threshold") is None else f"{row['threshold']:g}",
            "达标": (
                "—" if row.get("passed") is None else ("✅" if row["passed"] else "❌")
            ),
            "说明": row.get("note", ""),
        }
        for row in quality
    ]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_operations(data: dict) -> None:
    """运营侧数字：单场成本 / 时长 / Token / 出题延迟 / 复练提升率。"""
    cost = data.get("session_cost") or {}
    retrain = data.get("retrain") or {}
    columns = st.columns(5)
    columns[0].metric("单场面试成本（元）", f"{cost.get('cost_mean_cny', 0.0):.3f}")
    columns[1].metric("单场面试时长（秒）", f"{cost.get('total_s_mean', 0.0):.0f}")
    columns[2].metric("单场 Token 总量", f"{cost.get('total_tokens_mean', 0.0):,.0f}")
    columns[3].metric("出题节点平均耗时（秒）", f"{cost.get('ask_question_ms_mean', 0.0) / 1000:.1f}")
    columns[4].metric(
        "复练提升率",
        f"{retrain.get('improvement_rate', 0.0):.0%}",
        f"{retrain.get('improved_count', 0)}/{retrain.get('retrained_count', 0)} 题",
    )
    st.caption(
        "成本按 DeepSeek 官方单价折算（输入 2 元 / 输出 8 元 每百万 token），"
        f"样本 {int(cost.get('sessions', 0))} 场；复练提升率来自训练作答记录，零 LLM 聚合。"
    )


def _render_category(data: dict) -> None:
    rows = data.get("by_category") or []
    if not rows:
        return
    st.markdown("#### 分类别指标")
    st.dataframe(
        [
            {
                "类别": row["category"],
                "题数": row["n"],
                "忠实度": _fixed(row.get("faithfulness")),
                "上下文精确率": _fixed(row.get("context_precision")),
                "上下文召回率": _fixed(row.get("context_recall")),
                "答案相关性": _fixed(row.get("answer_relevancy")),
                "来源命中率(%)": _fixed(row.get("source_hit_rate")),
                "来源 F1(%)": _fixed(row.get("source_f1")),
            }
            for row in rows
        ],
        hide_index=True,
        use_container_width=True,
    )


def _fixed(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _render_ablation(data: dict) -> None:
    rows = data.get("ablation") or []
    if not rows:
        return
    st.markdown("#### 检索消融：哪个配置值得这么搭")
    st.caption("对比 dense / +BM25+RRF / +重排 / Agentic 自适应等配置的质量与单题成本。")
    st.dataframe(
        [
            {
                "配置": row["config"] + ("（Agentic）" if row.get("agentic") else ""),
                "启用阶段": "+".join(row.get("stages") or []) or "—",
                "忠实度": f"{row['faithfulness']:.4f}",
                "上下文精确率": f"{row['context_precision']:.4f}",
                "上下文召回率": f"{row['context_recall']:.4f}",
                "答案相关性": f"{row['answer_relevancy']:.4f}",
                "来源命中率(%)": f"{row['source_hit_rate']:.1f}",
                "平均检索轮数": f"{row['mean_rounds']:.2f}",
                "LLM 调用": row["llm_calls"],
                "单题成本(元)": f"{row['est_cost_cny_per_q']:.5f}",
            }
            for row in rows
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_cross_check(data: dict) -> None:
    rows = data.get("cross_check") or []
    if not rows:
        return
    st.markdown("#### 异源 judge 交叉验证：评分是否可信")
    st.caption(
        "同一批答案分别由 DeepSeek 与 Qwen2.5-72B 打分：delta 为两者的均值差，"
        "一致度为容差 0.1 下评分相同的比例——用于排查「自评虚高」。"
    )
    agreement = data.get("cell_agreement") or {}
    inflation = data.get("self_inflation") or {}
    columns = st.columns(min(4, len(agreement)) or 1)
    for column, (key, value) in zip(columns, list(agreement.items())[:4], strict=False):
        column.metric(f"单元格一致度 {key}", f"{value:.1f}%")
    if inflation:
        st.caption(
            "自评虚高幅度（主 judge − 异源 judge）："
            + " ｜ ".join(f"{key} {value:+.4f}" for key, value in inflation.items())
        )
    st.dataframe(
        [
            {
                "指标": row["metric"],
                "主 judge 均值": f"{row['primary_mean']:.4f}",
                "异源 judge 均值": f"{row['cross_mean']:.4f}",
                "差值": f"{row['delta']:+.4f}",
                "一致度(±0.1)": f"{row['agree_0_1']:.1f}%",
            }
            for row in rows
        ],
        hide_index=True,
        use_container_width=True,
    )


def _render_ab_block(title: str, subtitle: str, rows: list[dict], hit_rate: float | None) -> None:
    if not rows:
        return
    st.markdown(f"#### {title}")
    st.caption(subtitle)
    if hit_rate is not None:
        st.metric("语义缓存命中率", f"{hit_rate:.1f}%")
    st.dataframe(
        [
            {
                "对比项": row["label"],
                "基线": (
                    f"{row['baseline']:,.0f} {row['unit']}"
                    if row["baseline"] >= 100
                    else f"{row['baseline']:,.2f} {row['unit']}"
                ),
                "优化后": (
                    f"{row['optimized']:,.0f} {row['unit']}"
                    if row["optimized"] >= 100
                    else f"{row['optimized']:,.2f} {row['unit']}"
                ),
                "变化": f"{row['delta_pct']:+.1f}%",
            }
            for row in rows
        ],
        hide_index=True,
        use_container_width=True,
    )


def render(client: BackendClient) -> None:
    """效果看板主入口（纯展示：一次 GET /metrics，不做检索与 LLM 调用）。"""
    st.markdown("### 效果看板")
    st.caption(
        "本页展示的是项目**已经跑出来的真实评测数字**（RAGAS 质量 / 检索消融 / 异源 judge 交叉验证 /"
        " 成本与延迟 / 缓存与预算 A/B），全部读自已入库的评测报告，页面本身零 LLM 调用。"
    )
    try:
        data = client.metrics()
    except ApiClientError as exc:
        st.error(f"效果数据获取失败：{exc}（错误码：{exc.code}）")
        return

    if not data.get("quality") and not data.get("ablation"):
        st.info("暂无评测产物：先运行评测脚本生成报告后，本页会自动展示最新数据。")
        return

    _render_gate(data)
    _render_quality(data)
    _render_operations(data)
    _render_category(data)
    _render_ablation(data)
    _render_cross_check(data)
    _render_ab_block(
        "语义缓存 A/B 实测",
        "同一批会话在「关缓存」与「开缓存」下的对比（缓存命中即复用历史检索结果，零额外 LLM）。",
        data.get("semantic_cache") or [],
        data.get("cache_hit_rate"),
    )
    _render_ab_block(
        "上下文预算 A/B 实测",
        "同一批会话在「不压缩上下文」与「开启上下文预算」下的对比。",
        data.get("context_budget") or [],
        None,
    )
