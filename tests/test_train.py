"""训练模式测试（Task 22，离线确定性）。

- 认证：scrypt 哈希往返 / 错密码 / 非法格式；令牌签发 / 校验 / 过期 / 篡改 / 错密钥；
- 题库：qa Markdown 解析（题干 / tags / 答案要点 / 稳定 question_id）、抽题排除已答、
  全部答完回退全量随机；
- API：login → next → answer → history 全链路（mock LLM 与面试同口径研判）；
  未配置凭据 503、错密码 401、无令牌 401、未知题目 404、answer 挂成本熔断 503、
  next 响应绝不泄露答案要点、next 排除已答题；
  reveal 返回答案要点且零 LLM 调用、不落库（进度不前进）、未知题 404、无令牌 401。
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.deps import ServiceContainer
from app.api.quota import QuotaManager, get_quota, reset_quota
from app.api.train import _reset_login_limiter
from app.config.settings import Settings
from app.core.llm import DeepSeekChatClient
from app.main import create_app
from app.storage.db import SessionRepository
from app.train.auth import (
    TOKEN_PREFIX,
    hash_password,
    issue_token,
    verify_password,
    verify_token,
)
from app.train.questions import QuestionBank, get_question_bank, reset_question_bank

# ---------------------------------------------------------------------------
# 公共夹具与工具
# ---------------------------------------------------------------------------

_QA_MD = """# 测试题库

## Q01: 什么是 GIL？

- tags: python, 并发
- 答案要点：
  - 全局解释器锁，同一时刻只有一个线程执行字节码。
  - CPU 密集用多进程绕过。

## Q02: 深拷贝与浅拷贝的区别？

- tags: python, 数据模型
- 答案要点：
  - 浅拷贝共享内部元素引用，深拷贝递归复制。

## Q03: 事务隔离级别？

- tags: 后端, mysql
- 答案要点：
  - 读未提交 / 读已提交 / 可重复读 / 串行化。
"""

USERNAME = "trainee"
PASSWORD = "s3cret-train-only"  # noqa: S105 - 测试口令


def _qa_dir(tmp_path) -> object:
    d = tmp_path / "qa"
    d.mkdir()
    (d / "test_bank.md").write_text(_QA_MD, encoding="utf-8")
    return d


def _settings(qa_dir, **overrides) -> Settings:
    base: dict = {
        "_env_file": None,
        "deepseek_api_key": "sk-test-ds",
        "deepseek_base_url": "https://ds.test/v1",
        "siliconflow_api_key": "sk-test-sf",
        "siliconflow_base_url": "https://sf.test/v1",
        "train_username": USERNAME,
        "train_password_hash": hash_password(PASSWORD),
        "train_token_secret": "test-token-secret-32bytes-long-xx",
        "train_qa_dir": qa_dir,
    }
    base.update(overrides)
    return Settings(**base)


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1,
            "model": "deepseek-chat",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": content},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
    )


def _llm(settings: Settings, responses: list[str]) -> DeepSeekChatClient:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_response(queue.pop(0))

    return DeepSeekChatClient(settings, transport=httpx.MockTransport(handler))


def _container(settings: Settings, llm, repo) -> ServiceContainer:
    return ServiceContainer(
        settings=settings,
        repo=repo,
        store=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        reranker=None,  # type: ignore[arg-type]
        retriever=None,  # type: ignore[arg-type]
        adaptive=None,  # type: ignore[arg-type]
        cache=None,  # type: ignore[arg-type]
        llm=llm,
        report_generator=None,  # type: ignore[arg-type]
    )


@pytest.fixture(autouse=True)
def _reset_train_state():
    """每个用例前后清空题库缓存与登录限速器，避免模块级单例串扰。"""
    reset_question_bank()
    _reset_login_limiter()
    yield
    reset_question_bank()
    _reset_login_limiter()


def _client(container: ServiceContainer) -> TestClient:
    app = create_app()
    app.state.services = container
    return TestClient(app)


def _login(client: TestClient, password: str = PASSWORD) -> str:
    resp = client.post("/train/login", json={"username": USERNAME, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 认证：哈希与令牌
# ---------------------------------------------------------------------------
def test_password_hash_roundtrip_and_rejections():
    stored = hash_password(PASSWORD)
    assert stored.startswith("scrypt$")
    assert verify_password(PASSWORD, stored)
    assert not verify_password("wrong-password", stored)
    assert not verify_password(PASSWORD, "明文不是哈希")
    assert not verify_password(PASSWORD, "bcrypt$2b$garbage")
    assert not verify_password(PASSWORD, "scrypt$abc$8$1$zz$zz")  # 参数/编码非法
    # 同密码两次哈希盐不同
    assert hash_password(PASSWORD) != hash_password(PASSWORD)


def test_token_issue_verify_expiry_and_tamper():
    secret = "s" * 32
    token = issue_token(USERNAME, secret, 60, now=1000.0)
    assert token.startswith(TOKEN_PREFIX)
    assert verify_token(token, secret, now=1030.0) == USERNAME
    assert verify_token(token, secret, now=1061.0) is None  # 过期
    assert verify_token(token, "x" * 32, now=1030.0) is None  # 错密钥
    assert verify_token("sess-abc123", secret, now=1030.0) is None  # 面试 ID 不可冒充
    parts = token.split(".")
    forged = f"{parts[0]}.{parts[1]}.{'A' * 43}"  # 篡改签名
    assert verify_token(forged, secret, now=1030.0) is None
    assert verify_token("trn1.不是base64.也不是签名", secret, now=1030.0) is None


# ---------------------------------------------------------------------------
# 题库解析与抽题
# ---------------------------------------------------------------------------
def test_question_bank_parses_qa_markdown(tmp_path):
    bank = QuestionBank.load(_qa_dir(tmp_path))
    assert bank.total == 3
    q1 = bank.get("test_bank.md#Q01")
    assert q1 is not None
    assert q1.question == "什么是 GIL？"
    assert q1.tags == ("python", "并发")
    assert len(q1.points) == 2 and "全局解释器锁" in q1.points[0]
    assert q1.source == "test_bank.md"


def test_question_bank_draw_excludes_done_and_falls_back(tmp_path):
    bank = QuestionBank.load(_qa_dir(tmp_path))
    assert bank.draw(set()) is not None  # 非空可抽
    # 排除两题后只能抽到第三题
    excluded = {"test_bank.md#Q01", "test_bank.md#Q02"}
    for _ in range(10):
        assert bank.draw(excluded).question_id == "test_bank.md#Q03"
    # 全部答完 → 回退全量随机（仍能出题）
    all_three = excluded | {"test_bank.md#Q03"}
    for _ in range(10):
        assert bank.draw(all_three).question_id in all_three
    assert QuestionBank([]).draw(set()) is None


def test_get_question_bank_cached_by_dir(tmp_path):
    qa = _qa_dir(tmp_path)
    assert get_question_bank(qa) is get_question_bank(qa)
    reset_question_bank()
    assert get_question_bank(qa) is get_question_bank(qa)


# ---------------------------------------------------------------------------
# API 全链路
# ---------------------------------------------------------------------------
def _assess_json(quality: str, feedback: str = "点评") -> str:
    return json.dumps(
        {"quality": quality, "feedback": feedback, "followup_question": ""},
        ensure_ascii=False,
    )


def test_train_full_chain_login_next_answer_history(tmp_path):
    qa = _qa_dir(tmp_path)
    settings = _settings(qa)
    llm = _llm(settings, [_assess_json("partial", "答到了 GIL，但多进程规避没说清")])
    client = _client(_container(settings, llm, SessionRepository(tmp_path / "t.db")))

    token = _login(client)
    # next：不含答案要点
    resp = client.post("/train/next", headers=_auth(token))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_count"] == 3 and body["done_count"] == 0
    view = body["question"]
    assert view["question_id"].startswith("test_bank.md#Q")
    assert view["question"] and view["tags"] and view["source"] == "test_bank.md"
    assert "points" not in view and "答案要点" not in json.dumps(body, ensure_ascii=False)

    # answer：研判与落库，作答后才返回答案要点
    resp = client.post(
        "/train/answer",
        headers=_auth(token),
        json={"question_id": view["question_id"], "answer": "GIL 是全局锁。"},
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["quality"] == "partial"
    assert "多进程" in result["feedback"]
    assert result["reference_points"], "作答后必须返回题库答案要点"
    assert result["attempt_id"] >= 1

    # history：进度与质量分布
    resp = client.get("/train/history", headers=_auth(token))
    assert resp.status_code == 200
    history = resp.json()
    assert history["done_count"] == 1 and history["total_count"] == 3
    assert history["quality_counts"] == {"partial": 1}
    attempt = history["attempts"][0]
    assert attempt["question_id"] == view["question_id"]
    assert attempt["answer"] == "GIL 是全局锁。"

    # next 排除已答题
    for _ in range(5):
        again = client.post("/train/next", headers=_auth(token)).json()
        assert again["question"]["question_id"] != view["question_id"]
        assert again["done_count"] == 1


def test_train_not_configured_503(tmp_path):
    settings = _settings(
        _qa_dir(tmp_path), train_username="", train_password_hash="", train_token_secret=""
    )
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "t.db")))
    resp = client.post("/train/login", json={"username": USERNAME, "password": PASSWORD})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "TRAIN_NOT_CONFIGURED"


def test_train_wrong_password_and_bad_token_401(tmp_path):
    settings = _settings(_qa_dir(tmp_path))
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "t.db")))

    resp = client.post("/train/login", json={"username": USERNAME, "password": "wrong"})
    assert resp.status_code == 401 and resp.json()["error"]["code"] == "TRAIN_AUTH_FAILED"
    resp = client.post("/train/login", json={"username": "nobody", "password": PASSWORD})
    assert resp.status_code == 401 and resp.json()["error"]["code"] == "TRAIN_AUTH_FAILED"

    # 无令牌 / 伪造令牌访问三个受保护接口
    for method, path, kwargs in [
        ("post", "/train/next", {}),
        ("post", "/train/answer", {"json": {"question_id": "x", "answer": "y"}}),
        ("get", "/train/history", {}),
    ]:
        resp = getattr(client, method)(path, **kwargs)
        assert resp.status_code == 401, path
        assert resp.json()["error"]["code"] == "TRAIN_UNAUTHORIZED"
        resp = getattr(client, method)(path, headers=_auth("trn1.forged.token"), **kwargs)
        assert resp.status_code == 401, path


def test_train_answer_unknown_question_404(tmp_path):
    settings = _settings(_qa_dir(tmp_path))
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "t.db")))
    token = _login(client)
    resp = client.post(
        "/train/answer",
        headers=_auth(token),
        json={"question_id": "ghost.md#Q99", "answer": "任意答案"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "TRAIN_QUESTION_NOT_FOUND"


# ---------------------------------------------------------------------------
# reveal：答不出来直接看答案要点
# ---------------------------------------------------------------------------
def test_train_reveal_returns_points_without_llm_or_progress(tmp_path):
    """reveal 返回答案要点；全程零 LLM 调用（handler 触发即失败）、不落库（进度为 0）。"""
    qa = _qa_dir(tmp_path)
    settings = _settings(qa)

    def _forbidden(request: httpx.Request) -> httpx.Response:
        raise AssertionError("reveal 不应触发任何 LLM 调用")

    llm = DeepSeekChatClient(settings, transport=httpx.MockTransport(_forbidden))
    client = _client(_container(settings, llm, SessionRepository(tmp_path / "t.db")))
    token = _login(client)
    resp = client.post(
        "/train/reveal", headers=_auth(token), json={"question_id": "test_bank.md#Q01"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["question_id"] == "test_bank.md#Q01"
    assert body["reference_points"] == [
        "全局解释器锁，同一时刻只有一个线程执行字节码。",
        "CPU 密集用多进程绕过。",
    ]
    # 不落库：历史为空、进度不前进
    history = client.get("/train/history", headers=_auth(token)).json()
    assert history["done_count"] == 0
    assert history["attempts"] == []


def test_train_reveal_unknown_question_404(tmp_path):
    settings = _settings(_qa_dir(tmp_path))
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "t.db")))
    token = _login(client)
    resp = client.post(
        "/train/reveal", headers=_auth(token), json={"question_id": "ghost.md#Q99"}
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "TRAIN_QUESTION_NOT_FOUND"


def test_train_reveal_requires_token_401(tmp_path):
    settings = _settings(_qa_dir(tmp_path))
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "t.db")))
    resp = client.post("/train/reveal", json={"question_id": "test_bank.md#Q01"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "TRAIN_UNAUTHORIZED"


def test_train_answer_blocked_when_budget_exhausted(tmp_path):
    """answer 是会消耗 LLM 的接口：日预算熔断时必须 503（P2-③ 防护覆盖训练链路）。"""
    qa = _qa_dir(tmp_path)
    settings = _settings(qa)
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "t.db")))
    token = _login(client)
    reset_quota(
        QuotaManager(
            Settings(
                _env_file=None,
                rate_limit_enabled=False,
                cost_budget_enabled=True,
                daily_token_budget=1,
                daily_cost_budget_cny=100.0,
            )
        )
    )
    get_quota().record_usage(10, 10)  # 直接把当日 token 打满
    resp = client.post(
        "/train/answer",
        headers=_auth(token),
        json={"question_id": "test_bank.md#Q01", "answer": "GIL 是全局锁。"},
    )
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "BUDGET_EXHAUSTED"


def test_train_login_rate_limited(tmp_path):
    settings = _settings(_qa_dir(tmp_path), train_login_rate_per_minute=2)
    client = _client(_container(settings, _llm(settings, []), SessionRepository(tmp_path / "t.db")))
    for _ in range(2):
        client.post("/train/login", json={"username": USERNAME, "password": "wrong"})
    resp = client.post("/train/login", json={"username": USERNAME, "password": PASSWORD})
    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "RATE_LIMITED"
    assert resp.headers["Retry-After"]


# ---------------------------------------------------------------------------
# 仓储层
# ---------------------------------------------------------------------------
def test_train_attempts_repository_roundtrip(tmp_path):
    repo = SessionRepository(tmp_path / "t.db")
    rec = repo.add_train_attempt(
        username=USERNAME,
        question_id="b.md#Q01",
        source="b.md",
        question="什么是 GIL？",
        answer="全局解释器锁。",
        quality="good",
        feedback="准确",
    )
    assert rec.attempt_id == 1 and rec.quality == "good"
    repo.add_train_attempt(
        username=USERNAME, question_id="b.md#Q01", source="b.md",
        question="什么是 GIL？", answer="再答一次", quality="partial", feedback="一般",
    )
    repo.add_train_attempt(
        username="other", question_id="b.md#Q09", source="b.md",
        question="别用户的题", answer="x", quality="poor", feedback="y",
    )
    attempts = repo.list_train_attempts(USERNAME)
    assert [a.attempt_id for a in attempts] == [2, 1]  # 倒序
    assert repo.list_train_done_question_ids(USERNAME) == {"b.md#Q01"}
    assert repo.list_train_done_question_ids("other") == {"b.md#Q09"}
    limited = repo.list_train_attempts(USERNAME, limit=1)
    assert len(limited) == 1 and limited[0].attempt_id == 2
