"""Streamlit 前端渲染测试（Task 10，离线确定性）。

用官方 ``streamlit.testing.v1.AppTest`` 真实执行 ``frontend/app.py`` 脚本，仅把
``BackendClient`` 的方法替换为桩函数（不打网络），验证：

- TR-10.1：三页渲染所用字段全部取自 schemas 契约（SessionRecord / MessageRecord /
  InterviewReport / SSE 事件 data），无自造字段——桩数据严格按契约构造，字段名不符即断言失败；
- TR-10.2：错误提示场景（SSE error 事件）经 st.error 透出原文 message 与错误码；
  阶段徽标、证据来源 chips、报告页雷达图/维度条/证据展开/建议清单/下载按钮均渲染。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
sys.path.insert(0, str(FRONTEND_DIR))  # 前端模块以脚本目录为根导入

from api_client import BackendClient, SseEventLite  # noqa: E402

FRONTEND_APP = FRONTEND_DIR / "app.py"

SESSION_ID = "sess-0001"
REPORT = {
    "session_id": SESSION_ID,
    "dimensions": [
        {
            "dimension": n,
            "score": 4,
            "evidence": [f"问：Q1 答：A1（{n}）"],
            "comment": f"{n} 表现良好",
        }
        for n in ("岗位匹配度", "基础知识", "项目深度", "表达逻辑", "问题解决")
    ],
    "highlights": ["项目细节扎实"],
    "weaknesses": ["分布式场景覆盖不足"],
    "suggestions": ["补充分布式锁与限流实践"],
    "recommended_topics": [
        {"topic": "缓存穿透/击穿/雪崩", "source": "qa_backend.md", "reason": "本次回答薄弱"}
    ],
    "generated_at": "2026-10-16T10:00:00+00:00",
}

MESSAGES = [
    {
        "id": 1, "session_id": SESSION_ID, "role": "interviewer",
        "content": "你好，欢迎参加本次面试。", "stage": "opening",
        "evidence": [], "created_at": "2026-10-16T09:00:00+00:00",
    },
    {
        "id": 2, "session_id": SESSION_ID, "role": "interviewer",
        "content": "Q1：Redis 缓存穿透怎么解决？", "stage": "question",
        "evidence": [
            {"chunk_id": "ev-1", "source": "qa_backend.md", "heading": "缓存", "score": 0.93}
        ],
        "created_at": "2026-10-16T09:00:00+00:00",
    },
]


def _sse(event: str, data: dict) -> SseEventLite:
    return SseEventLite(event, data)


def _ok_stream() -> Iterator[SseEventLite]:
    yield _sse("stage", {"stage": "opening", "round": 0})
    yield _sse("token", {"content": "你好，欢迎参加本次面试。"})
    yield _sse("stage", {"stage": "question", "round": 1})
    yield _sse("token", {"content": "Q1：Redis 缓存穿透怎么解决？"})
    yield _sse("done", {"reason": "completed"})


def _error_stream() -> Iterator[SseEventLite]:
    yield _sse("stage", {"stage": "answer", "round": 0})
    yield _sse(
        "error",
        {"code": "UPSTREAM_ERROR", "message": "上游模型返回 401：API Key 无效或已过期（HTTP 401），请检查 .env 中的 DEEPSEEK_API_KEY。"},
    )
    yield _sse("done", {"reason": "error"})


@pytest.fixture
def stub_backend(monkeypatch) -> dict:
    """把 BackendClient 的方法替换为桩（类对象共享，patch 类即对所有引用生效）。"""
    state: dict = {"stream": _ok_stream}

    def health(self) -> dict:
        return {
            "status": "healthy",
            "version": "0.1.0",
            "config": {
                "deepseek_configured": True, "deepseek_model": "deepseek-chat",
                "siliconflow_configured": True, "embedding_model": "BAAI/bge-m3",
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "qdrant_mode": "local", "qdrant_url": None, "langfuse_enabled": False,
            },
            "components": {"qdrant": {"status": "ok", "mode": "local", "detail": "ok"}},
        }

    def create_session(self, **kwargs) -> dict:
        return {
            "session_id": SESSION_ID, "track": kwargs.get("track", "ai_app"),
            "num_questions": kwargs.get("num_questions", 5),
            "candidate_name": kwargs.get("candidate_name"), "status": "active",
            "created_at": "2026-10-16T09:00:00+00:00", "updated_at": "2026-10-16T09:00:00+00:00",
        }

    def get_session(self, session_id: str) -> dict:
        return {
            "session": {
                "session_id": SESSION_ID, "track": "ai_app", "num_questions": 1,
                "candidate_name": None, "status": "active",
                "created_at": "2026-10-16T09:00:00+00:00", "updated_at": "2026-10-16T09:00:00+00:00",
            },
            "messages": MESSAGES,
        }

    def post_message(self, session_id: str, content: str) -> Iterator[SseEventLite]:
        yield from state["stream"]()

    def get_report(self, session_id: str) -> dict:
        return REPORT

    monkeypatch.setattr(BackendClient, "health", health)
    monkeypatch.setattr(BackendClient, "create_session", create_session)
    monkeypatch.setattr(BackendClient, "get_session", get_session)
    monkeypatch.setattr(BackendClient, "post_message", post_message)
    monkeypatch.setattr(BackendClient, "get_report", get_report)
    monkeypatch.setattr(BackendClient, "finish", lambda self, sid: REPORT)
    return state


def _button(at: AppTest, label: str):
    matches = [b for b in at.button if label in b.label]
    assert matches, f"未找到按钮：{label}（现有：{[b.label for b in at.button]}）"
    return matches[0]


def _start_session(at: AppTest) -> None:
    at.slider[0].set_value(1)
    _button(at, "开始面试").click().run()


# ---------------------------------------------------------------------------
# 开始页 → 面试页（含错误提示）→ 报告页
# ---------------------------------------------------------------------------
def test_start_page_renders_and_starts_interview(stub_backend):
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    assert not at.exception

    # 开始页：健康指标（字段取自 /health 契约）、来源选择、方向/题数控件
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["服务状态"] == "healthy"
    assert metrics["DeepSeek"] == "已配置" and metrics["SiliconFlow"] == "已配置"
    assert "知识库来源" in [r.label for r in at.radio]
    assert "岗位方向" in [s.label for s in at.selectbox]
    assert "计划题目数" in [s.label for s in at.slider]

    _start_session(at)
    assert not at.exception

    # 面试页：自动开场后渲染消息气泡、阶段徽标、证据来源 chips
    text = "\n".join(m.value for m in at.markdown)
    assert "Q1：Redis 缓存穿透怎么解决？" in text
    assert at.chat_message, "面试页应渲染对话气泡"
    captions = "\n".join(c.value for c in at.caption)
    assert "阶段：出题" in captions, "消息应带阶段徽标"
    assert "证据来源" in captions, "出题消息应展示证据来源 chips"
    assert at.chat_input, "未完成的面试应提供回答输入框"


def test_sse_error_event_shows_banner_with_code(stub_backend):
    stub_backend["stream"] = _error_stream
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _start_session(at)

    # 提交回答触发 SSE error 事件
    at.chat_input[0].set_value("答：用布隆过滤器拦截不存在的数据。").run()
    assert not at.exception, "错误事件不应让页面崩溃"
    errors = [e.value for e in at.error]
    assert errors, "SSE error 事件应经 st.error 透出"
    banner = errors[0]
    assert "API Key 无效或已过期" in banner and "401" in banner  # 原文 message 透出
    assert "错误码：UPSTREAM_ERROR" in banner  # 错误码透出


def test_report_page_renders_charts_evidence_and_download(stub_backend):
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _start_session(at)
    at.session_state["requested_page"] = "report"
    at.run()
    assert not at.exception

    text = "\n".join(m.value for m in at.markdown)
    for dimension in ("岗位匹配度", "基础知识", "项目深度", "表达逻辑", "问题解决"):
        assert f"{dimension}：**4/5**" in text, f"缺少维度分条：{dimension}"
    assert "项目细节扎实" in text and "分布式场景覆盖不足" in text
    assert "补充分布式锁与限流实践" in text

    # 维度进度条（5 条）、雷达图（plotly_chart）、证据展开、知识点表格、下载按钮
    assert len(at.progress) == 5
    assert at.get("plotly_chart"), "报告页应渲染 plotly 雷达图"
    assert at.expander, "评分依据应可展开"
    frame = at.dataframe[0].value
    cells = [str(v) for v in frame.to_numpy().ravel()]
    assert "缓存穿透/击穿/雪崩" in cells, "推荐知识点应回链题库来源"
    assert "qa_backend.md" in cells
    assert at.download_button, "应提供报告下载按钮"
