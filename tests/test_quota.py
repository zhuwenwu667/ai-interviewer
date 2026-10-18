"""P2-③ 公网限速与成本熔断单测。

- 固定窗口限速：窗口内超过上限即 429 + ``Retry-After``；窗口翻篇后恢复；
- 每日成本熔断：token 或成本任一达上限即 503 ``BUDGET_EXHAUSTED``；跨日自动重置；
- 开关：两项均关闭时 ``check_request`` 恒放行；
- HTTP 集成：``POST /sessions`` 被限速挡下（429 + Retry-After），熔断后 503，
  只读 ``GET`` 不受限速影响；``GET /quota`` 返回快照。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.graph import default_report_generator
from app.api.deps import ServiceContainer
from app.api.quota import DailyBudget, QuotaManager, RateLimiter, reset_quota
from app.config.settings import Settings
from app.main import create_app
from app.storage.db import SessionRepository


class FakeClock:
    """可控时钟：``now`` 可手动推进，用于确定性验证窗口/跨日行为。"""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _settings(**overrides) -> Settings:
    base: dict = {
        "_env_file": None,
        "deepseek_api_key": "sk-test-ds",
        "deepseek_base_url": "https://ds.test/v1",
        "siliconflow_api_key": "sk-test-sf",
        "siliconflow_base_url": "https://sf.test/v1",
    }
    base.update(overrides)
    return Settings(**base)


# ---------------------------------------------------------------------------
# 限速
# ---------------------------------------------------------------------------
def test_rate_limiter_blocks_over_limit_and_recovers_after_window():
    clock = FakeClock()
    limiter = RateLimiter(3, window_seconds=60.0, clock=clock)

    for _ in range(3):
        assert limiter.check("1.2.3.4").allowed is True

    denied = limiter.check("1.2.3.4")
    assert denied.allowed is False
    assert denied.status_code == 429 and denied.code == "RATE_LIMITED"
    assert denied.retry_after >= 1

    # 其他 key 不受影响（隔离到 IP）
    assert limiter.check("5.6.7.8").allowed is True

    # 窗口翻篇后恢复
    clock.advance(60.0)
    assert limiter.check("1.2.3.4").allowed is True


# ---------------------------------------------------------------------------
# 每日成本熔断
# ---------------------------------------------------------------------------
def test_daily_budget_trips_on_token_cap():
    clock = FakeClock(0.0)
    budget = DailyBudget(
        1000, 100.0, input_price_per_m=2.0, output_price_per_m=8.0, clock=clock
    )
    budget.record(400, 100)  # 500 tokens
    assert budget.check().allowed is True

    budget.record(400, 200)  # 累计 1100 >= 1000
    denied = budget.check()
    assert denied.allowed is False
    assert denied.status_code == 503 and denied.code == "BUDGET_EXHAUSTED"
    assert "token" in denied.message


def test_daily_budget_trips_on_cost_cap():
    clock = FakeClock(0.0)
    # 成本上限 ¥0.001：输入 ¥2/M、输出 ¥8/M → 1M 输出即 ¥8
    budget = DailyBudget(
        10_000_000, 0.001, input_price_per_m=2.0, output_price_per_m=8.0, clock=clock
    )
    budget.record(0, 1000)  # 1000/1e6*8 = ¥0.008 >= 0.001
    denied = budget.check()
    assert denied.allowed is False and denied.status_code == 503
    assert "成本" in denied.message
    assert budget.snapshot()["cost_cny"] == 0.008


def test_daily_budget_resets_across_days():
    clock = FakeClock(0.0)  # 1970-01-01（UTC 基准的本地日）
    budget = DailyBudget(100, 100.0, input_price_per_m=2.0, output_price_per_m=8.0, clock=clock)
    budget.record(200, 0)
    assert budget.check().allowed is False

    clock.advance(86400 * 2)  # 跨两天
    assert budget.check().allowed is True
    assert budget.snapshot()["tokens"] == 0


# ---------------------------------------------------------------------------
# 开关与组合
# ---------------------------------------------------------------------------
def test_quota_manager_disabled_always_allows():
    manager = QuotaManager(_settings(rate_limit_enabled=False, cost_budget_enabled=False))
    for _ in range(100):
        assert manager.check_request("k").allowed is True


def test_quota_manager_budget_checked_before_rate():
    clock = FakeClock()
    manager = QuotaManager(
        _settings(daily_token_budget=1, daily_cost_budget_cny=1e9),
        clock=clock,
        day_clock=clock,
    )
    manager.record_usage(5, 0)  # 已达 token 上限
    denied = manager.check_request("k")
    assert denied.status_code == 503  # 熔断优先于限速


# ---------------------------------------------------------------------------
# HTTP 集成（TestClient）
# ---------------------------------------------------------------------------
def _client(tmp_path) -> TestClient:
    settings = _settings()
    container = ServiceContainer(
        settings=settings,
        repo=SessionRepository(settings.sqlite_path_abs),
        store=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        reranker=None,  # type: ignore[arg-type]
        retriever=None,  # type: ignore[arg-type]
        adaptive=None,  # type: ignore[arg-type]
        cache=None,  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        report_generator=default_report_generator,
    )
    app = create_app()
    app.state.services = container
    return TestClient(app)


def test_create_session_is_rate_limited_with_retry_after(tmp_path):
    reset_quota(QuotaManager(_settings(rate_limit_per_minute=2, cost_budget_enabled=False)))
    client = _client(tmp_path)

    for _ in range(2):
        resp = client.post("/sessions", data={"track": "Python 后端开发", "candidate_name": "测试候选人"})
        assert resp.status_code == 201

    blocked = client.post("/sessions", data={"track": "Python 后端开发", "candidate_name": "测试候选人"})
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "RATE_LIMITED"
    assert int(blocked.headers["Retry-After"]) >= 1


def test_budget_exhaustion_returns_503(tmp_path):
    manager = QuotaManager(
        _settings(daily_token_budget=1, daily_cost_budget_cny=1e9, rate_limit_enabled=False)
    )
    manager.record_usage(10, 0)
    reset_quota(manager)
    client = _client(tmp_path)

    resp = client.post("/sessions", data={"track": "Python 后端开发", "candidate_name": "测试候选人"})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "BUDGET_EXHAUSTED"


def test_readonly_endpoints_are_not_rate_limited(tmp_path):
    reset_quota(QuotaManager(_settings(rate_limit_per_minute=1, cost_budget_enabled=False)))
    client = _client(tmp_path)

    # 只读 GET 不属于 LLM 消耗接口，超限也不会 429
    for _ in range(5):
        assert client.get("/quota").status_code == 200


def test_quota_snapshot_endpoint(tmp_path):
    reset_quota(
        QuotaManager(_settings(rate_limit_per_minute=7, daily_token_budget=500))
    )
    client = _client(tmp_path)
    body = client.get("/quota").json()
    assert body["rate_limit"]["enabled"] is True
    assert body["rate_limit"]["per_minute"] == 7
    assert body["budget"]["enabled"] is True
    assert body["budget"]["max_tokens"] == 500
    assert body["budget"]["tokens"] == 0


@pytest.mark.parametrize("path", ["/quota"])
def test_quota_endpoint_shape(path, tmp_path):
    reset_quota(QuotaManager(_settings()))
    client = _client(tmp_path)
    assert set(client.get(path).json().keys()) == {"rate_limit", "budget"}
