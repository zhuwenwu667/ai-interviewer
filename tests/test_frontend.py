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

from api_client import ApiClientError, BackendClient, SseEventLite  # noqa: E402

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


METRICS = {
    "sources": {"eval": "eval_20261018_021445.json", "ablation": "ablation_20261017_223939.json"},
    "generated_at": {"eval": "2026-10-18T02:14:45+08:00"},
    "gate_passed": True,
    "gate_violations": [],
    "dataset": {"total": 43, "executed": 43, "resume": 10, "jd": 10, "qa": 16, "oos": 7},
    "quality": [
        {
            "key": "faithfulness", "name": "忠实度", "value": 0.9892, "unit": "分",
            "threshold": 0.85, "passed": True, "delta": 0.0038, "note": "门槛指标",
        },
        {
            "key": "source_hit_rate", "name": "知识来源命中率", "value": 100.0, "unit": "%",
            "threshold": None, "passed": None, "delta": None, "note": "",
        },
    ],
    "by_category": [
        {"category": "简历", "n": 10, "faithfulness": 1.0, "context_precision": 0.95,
         "context_recall": 1.0, "answer_relevancy": 0.8221,
         "source_hit_rate": 100.0, "source_f1": 100.0}
    ],
    "ablation": [
        {"config": "dense", "agentic": False, "stages": [], "faithfulness": 0.9739,
         "context_precision": 0.8194, "context_recall": 1.0, "answer_relevancy": 0.8818,
         "source_hit_rate": 100.0, "mean_rounds": 1.0, "mean_contexts": 6.0,
         "llm_calls": 43, "est_cost_cny_per_q": 0.00248},
        {"config": "agentic", "agentic": True, "stages": ["rewrite", "bm25", "rrf", "rerank"],
         "faithfulness": 0.9892, "context_precision": 0.9399, "context_recall": 0.9931,
         "answer_relevancy": 0.8953, "source_hit_rate": 100.0, "mean_rounds": 1.67,
         "mean_contexts": 6.0, "llm_calls": 116, "est_cost_cny_per_q": 0.0061},
    ],
    "cross_check": [
        {"metric": "忠实度", "primary_mean": 0.973, "cross_mean": 1.0,
         "delta": -0.027, "agree_0_1": 88.9}
    ],
    "cell_agreement": {"agree_0.05": 76.4, "agree_0.1": 84.0, "agree_0.2": 93.8},
    "self_inflation": {"faithfulness": -0.027, "context_precision": 0.0147},
    "session_cost": {
        "sessions": 6.0, "cost_mean_cny": 0.05211, "cost_max_cny": 0.05952,
        "total_tokens_mean": 14304.3, "total_s_mean": 48.73, "ask_question_ms_mean": 10736.2,
    },
    "semantic_cache": [
        {"label": "单题出题延迟", "baseline": 9194.1, "optimized": 3182.2,
         "delta_pct": -65.4, "unit": "ms"}
    ],
    "cache_hit_rate": 77.8,
    "context_budget": [
        {"label": "LLM Token 总量", "baseline": 60749.0, "optimized": 54614.0,
         "delta_pct": -10.1, "unit": "token"}
    ],
    "retrain": {"retrained_count": 2, "improved_count": 1, "improvement_rate": 0.5},
}


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

    def list_sessions(self) -> dict:
        return {
            "sessions": [
                {
                    "session_id": SESSION_ID, "track": "ai_app", "num_questions": 1,
                    "candidate_name": None, "status": "finished",
                    "created_at": "2026-10-16T09:00:00+00:00",
                    "updated_at": "2026-10-16T09:30:00+00:00", "has_report": True,
                },
                {
                    "session_id": "sess-0002", "track": "backend", "num_questions": 3,
                    "candidate_name": "候选人乙", "status": "active",
                    "created_at": "2026-10-17T09:00:00+00:00",
                    "updated_at": "2026-10-17T09:05:00+00:00", "has_report": False,
                },
            ]
        }

    monkeypatch.setattr(BackendClient, "health", health)
    monkeypatch.setattr(BackendClient, "create_session", create_session)
    monkeypatch.setattr(BackendClient, "get_session", get_session)
    monkeypatch.setattr(BackendClient, "post_message", post_message)
    monkeypatch.setattr(BackendClient, "get_report", get_report)
    monkeypatch.setattr(BackendClient, "list_sessions", list_sessions)
    monkeypatch.setattr(BackendClient, "finish", lambda self, sid: REPORT)

    # 训练模式桩（Task 22）：登录 / 抽题 / 作答 / 历史
    def train_login(self, username: str, password: str) -> dict:
        if password != "train-pass":  # noqa: S105 - 测试口令
            raise ApiClientError("用户名或密码错误", code="TRAIN_AUTH_FAILED", status=401)
        return {"token": "trn1.test-token", "token_type": "Bearer", "expires_in": 3600}

    def train_next(self, avoid_question_id: str | None = None) -> dict:
        state["last_avoid"] = avoid_question_id
        return {
            "question": {
                "question_id": "bank.md#Q01", "question": "什么是 GIL？",
                "tags": ["python", "并发"], "source": "bank.md",
            },
            "done_count": 0, "total_count": 40,
        }

    def train_answer(self, question_id: str, answer: str) -> dict:
        return {
            "attempt_id": 1, "quality": "partial",
            "feedback": "答到了全局锁，但多进程规避没说清。",
            "reference_points": ["全局解释器锁，同一时刻只有一个线程执行字节码。", "CPU 密集用多进程绕过。"],
        }

    def train_reveal(self, question_id: str) -> dict:
        return {
            "question_id": question_id,
            "reference_points": ["全局解释器锁，同一时刻只有一个线程执行字节码。", "CPU 密集用多进程绕过。"],
        }

    def train_history(self) -> dict:
        return {
            "attempts": [
                {
                    "attempt_id": 1, "question_id": "bank.md#Q01", "question": "什么是 GIL？",
                    "answer": "GIL 是全局锁。", "quality": "partial",
                    "feedback": "答到了全局锁。", "created_at": "2026-10-18T10:00:00+00:00",
                }
            ],
            "done_count": 1, "total_count": 40, "quality_counts": {"partial": 1},
            "retrain_stats": {
                "retrained_count": 2, "improved_count": 1, "improvement_rate": 0.5
            },
        }

    def train_mistakes(self) -> dict:
        return {
            "mistakes": [
                {
                    "question_id": "bank.md#Q07", "question": "事务隔离级别有哪些？",
                    "tags": ["mysql"], "source": "bank.md", "answer": "读已提交和可重复读。",
                    "quality": "poor", "feedback": "漏了读未提交与串行化。",
                    "reference_points": ["读未提交 / 读已提交 / 可重复读 / 串行化。"],
                    "created_at": "2026-10-18T11:00:00+00:00",
                }
            ],
            "total_count": 40,
        }

    monkeypatch.setattr(BackendClient, "train_login", train_login)
    monkeypatch.setattr(BackendClient, "train_next", train_next)
    monkeypatch.setattr(BackendClient, "train_answer", train_answer)
    monkeypatch.setattr(BackendClient, "train_reveal", train_reveal)
    monkeypatch.setattr(BackendClient, "train_history", train_history)
    monkeypatch.setattr(BackendClient, "train_mistakes", train_mistakes)

    # 效果看板桩（优化项3）：一次 GET /metrics 的聚合结果
    monkeypatch.setattr(BackendClient, "metrics", lambda self: METRICS)
    return state


def _button(at: AppTest, label: str):
    matches = [b for b in at.button if label in b.label]
    assert matches, f"未找到按钮：{label}（现有：{[b.label for b in at.button]}）"
    return matches[0]


def _start_session(at: AppTest) -> None:
    at.slider[0].set_value(1)
    _button(at, "开始这场模拟面试").click().run()


# ---------------------------------------------------------------------------
# 开始页 → 面试页（含错误提示）→ 报告页
# ---------------------------------------------------------------------------
def test_start_page_renders_and_starts_interview(stub_backend):
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    assert not at.exception

    # 开始页：Hero 预告 + 来源选择 + 方向/题数控件 + 折叠的服务诊断
    text = "\n".join(m.value for m in at.markdown)
    assert "把你的简历，变成一场模拟面试" in text, "开始页应渲染产品说明"
    for dimension in ("岗位匹配度", "基础知识", "项目深度", "表达逻辑", "问题解决"):
        assert dimension in text, f"开始页应预告评分维度：{dimension}"
    assert "知识库来源" in [r.label for r in at.radio]
    assert "岗位方向" in [s.label for s in at.selectbox]
    assert "题目数量" in [s.label for s in at.slider]
    # 服务诊断收进折叠区，但字段仍取自 /health 契约
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["服务状态"] == "healthy"
    assert metrics["DeepSeek"] == "已配置" and metrics["SiliconFlow"] == "已配置"

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


def test_start_page_tabs_offer_train_entry(stub_backend):
    """首页三 Tab：模拟面试 / 刷题训练 / 效果看板并列，训练入口不经侧边栏即可直达。"""
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["模拟面试", "刷题训练", "效果看板"]

    # CTA 旁一行内联状态（服务 + 两个模型），健康时三项均 ok
    text = "\n".join(m.value for m in at.markdown)
    assert "服务正常" in text and "DeepSeek 已配置" in text and "SiliconFlow 已配置" in text

    _button(at, "前往刷题训练").click().run()
    assert not at.exception
    assert _button(at, "登录并开始刷题"), "首页训练入口应能直达训练页登录表单"


def test_start_page_history_opens_existing_report(stub_backend):
    """历史面试记录（优化项1）：列出会话，已生成报告的会话可点开回看。"""
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    assert not at.exception

    labels = [e.label for e in at.expander]
    assert any("历史面试记录" in label for label in labels), "开始页应提供历史记录入口"
    text = "\n".join(m.value for m in at.markdown)
    assert "ai_app" in text and "1 题" in text and "已完成" in text
    assert "进行中" in text, "未完成的会话应标注为进行中"

    # 点开既有报告 → 切到报告页并复用同一份报告展示
    _button(at, "回看报告").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "岗位匹配度：**4/5**" in text, "回看应复用现有报告展示"


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


# ---------------------------------------------------------------------------
# 训练模式页（Task 22）：登录 → 抽题 → 作答 → 研判 + 答案要点 → 历史
# ---------------------------------------------------------------------------
def _goto_train(at: AppTest) -> None:
    at.session_state["requested_page"] = "train"
    at.run()


def test_train_page_login_then_answer_flow(stub_backend):
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _goto_train(at)
    assert not at.exception

    # 登录态前：渲染登录表单
    labels = [t.label for t in at.text_input]
    assert "用户名" in labels and "密码" in labels

    # 错密码：错误原文透出（错误横幅在下一轮渲染上屏）
    at.text_input[0].set_value("trainee")
    at.text_input[1].set_value("wrong-pass")
    _button(at, "登录并开始刷题").click().run()
    at.run()
    errors = [e.value for e in at.error]
    assert any("用户名或密码错误" in e and "TRAIN_AUTH_FAILED" in e for e in errors)

    # 正确口令：登录成功 → 自动抽题渲染（含进度与考点标签）
    at.text_input[1].set_value("train-pass")
    _button(at, "登录并开始刷题").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "什么是 GIL？" in text, "登录后应渲染抽到的题目"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["已练习题数"] == "0 / 40"

    # 复练提升率（优化项2）：来自 /train/history 的 retrain_stats（零 LLM 聚合）
    captions = "\n".join(c.value for c in at.caption)
    assert "复练提升率：50%（1/2 道复练题" in captions

    # 作答 → 提交 → 研判结论 + 参考答案要点
    at.text_area[0].set_value("GIL 是全局解释器锁。")
    at.run()
    _button(at, "提交作答").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "部分掌握" in text
    infos = "\n".join(i.value for i in at.info)
    assert "多进程规避没说清" in infos, "研判点评应经 st.info 透出"
    assert "全局解释器锁，同一时刻只有一个线程执行字节码。" in text, "作答后应展示参考答案要点"
    assert _button(at, "下一题"), "研判后应提供下一题按钮"

    # 下一题（A1）：应携带当前题 id 作为 avoid，避免错题优先下连出同一题
    _button(at, "下一题").click().run()
    assert not at.exception
    assert stub_backend["last_avoid"] == "bank.md#Q01"


def test_train_page_reveal_shows_points_without_answering(stub_backend):
    """答不出来：点「显示答案」→ 要点展示 + 可直接下一题 + 作答入口保留 + 进度不前进。"""
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _goto_train(at)
    at.text_input[0].set_value("trainee")
    at.text_input[1].set_value("train-pass")
    _button(at, "登录并开始刷题").click().run()
    assert not at.exception

    _button(at, "显示答案").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "全局解释器锁，同一时刻只有一个线程执行字节码。" in text, "看答案后应展示答案要点"
    assert _button(at, "下一题"), "看答案后应可直接进入下一题"
    assert _button(at, "提交作答"), "看答案后仍应保留作答入口（看完默写也算练）"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["已练习题数"] == "0 / 40", "看答案不计入已练进度"


def test_train_page_mistakes_book_and_retrain(stub_backend):
    """错题本（A3）：展示非 good 错题（题干 / 点评 / 答案要点），一键重练把该题放回答作区。"""
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _goto_train(at)
    at.text_input[0].set_value("trainee")
    at.text_input[1].set_value("train-pass")
    _button(at, "登录并开始刷题").click().run()
    assert not at.exception

    text = "\n".join(m.value for m in at.markdown)
    assert "事务隔离级别有哪些？" in text, "错题本应列出错题题干"
    assert "串行化。" in text, "错题本应展示题库答案要点"
    infos = "\n".join(i.value for i in at.info)
    assert "漏了读未提交与串行化。" in infos, "错题本应展示最近一次研判点评"

    # 一键重练：把错题放回答作区（题面切换 + 出现作答入口）
    _button(at, "一键重练").click().run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "事务隔离级别有哪些？" in text, "重练后作答区应显示错题题面"
    assert _button(at, "提交作答"), "重练后应出现作答入口"


def test_train_page_401_falls_back_to_login(stub_backend, monkeypatch):
    def expired_next(self, avoid_question_id: str | None = None) -> dict:
        raise ApiClientError("训练令牌无效或已过期，请重新登录", code="TRAIN_UNAUTHORIZED", status=401)

    monkeypatch.setattr(BackendClient, "train_next", expired_next)
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _goto_train(at)
    at.text_input[0].set_value("trainee")
    at.text_input[1].set_value("train-pass")
    _button(at, "登录并开始刷题").click().run()
    assert not at.exception
    # 401 后清令牌回到登录表单
    labels = [t.label for t in at.text_input]
    assert "用户名" in labels and "密码" in labels
    errors = [e.value for e in at.error]
    assert any("重新登录" in e for e in errors)


# ---------------------------------------------------------------------------
# 效果看板页（优化项3）：一次 GET /metrics，把已入库评测数字渲染成看板
# ---------------------------------------------------------------------------
def _goto_metrics(at: AppTest) -> None:
    at.session_state["requested_page"] = "metrics"
    at.run()


def test_metrics_page_renders_eval_numbers(stub_backend):
    at = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _goto_metrics(at)
    assert not at.exception

    # 门禁结论（与 CI 同规则）+ 评测集规模 + 数据来源文件名（可追溯）
    assert any("评测回归门禁" in s.value and "通过" in s.value for s in at.success)
    captions = "\n".join(c.value for c in at.caption)
    assert "评测集：共 43 题" in captions
    assert "eval_20261018_021445.json" in captions, "看板应标注数据来源报告文件名"

    # 门槛指标卡（带合格线）+ 运营侧数字（成本 / 时长 / 复练提升率）
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["忠实度（≥ 0.85）"] == "0.9892"
    assert metrics["单场面试成本（元）"] == "0.052"
    assert metrics["单场面试时长（秒）"] == "49"
    assert metrics["复练提升率"] == "50%"
    assert metrics["语义缓存命中率"] == "77.8%"

    # 分类别 / 消融 / 交叉验证 / 缓存 / 预算 多张表均上屏
    assert len(at.dataframe) >= 5, f"看板应渲染多张表格（当前 {len(at.dataframe)} 张）"
    captions = "\n".join(c.value for c in at.caption)
    assert "异源 judge 交叉验证" in captions and "自评虚高幅度" in captions
    headings = "\n".join(m.value for m in at.markdown)
    assert "检索消融：哪个配置值得这么搭" in headings
    assert "语义缓存 A/B 实测" in headings and "上下文预算 A/B 实测" in headings

    # 首页第三 Tab 的入口可直达看板
    start = AppTest.from_file(str(FRONTEND_APP), default_timeout=30).run()
    _button(start, "前往效果看板").click().run()
    assert not start.exception
    assert any("评测回归门禁" in s.value for s in start.success)
