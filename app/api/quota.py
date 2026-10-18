"""P2-③ 公网限速与成本熔断：防止公开部署下的额度滥用。

背景：该项目以**公网 Streamlit 页面**为唯一入口，任何人都能触发「创建面试 → 出题 →
追问 → 生成报告」的全链路真实 LLM 调用。若不设防，一次异常流量即可烧掉当日额度。

两层防护（都在**应用进程内存**中，单实例部署即可，不引入 Redis）：

1. **每 IP 限速**（``RateLimiter``，固定窗口）：对「会消耗 LLM 的公共接口」
   （创建会话、SSE 消息、提前结束）按客户端 IP 限制每分钟请求数，超限返回
   ``429 RATE_LIMITED`` 并带 ``Retry-After``。
2. **每日成本熔断**（``DailyBudget``）：跨全部客户端累计 LLM token 与预估成本，
   达到日上限后返回 ``503 BUDGET_EXHAUSTED``，**停止一切新的 LLM 消耗**；跨日自动重置。

口径与取舍：
- 成本按配置单价（默认 deepseek-chat 输入 ¥2/M、输出 ¥8/M）折算，与评测脚本
  （``scripts/measure_session_cost.py``）保持同一口径，便于对照。
- 限速只作用在**会花钱**的接口上；只读查询（GET 会话 / 报告）不限速，避免误伤正常浏览。
- 熔断是**全局**的（跨客户端），因为它保护的是「账号额度」而不是「单个用户公平性」——
  单个用户能被限速挡住，但聚合流量仍需一个总量闸门。
- 反代场景提示：默认取 ``request.client.host``；若前面挂了反向代理，需在代理层透传真实
  客户端 IP（本项目公网入口为 Streamlit，api 仅回环，故默认值即可）。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.config.settings import Settings


@dataclass(frozen=True)
class QuotaDecision:
    """配额判定结果；``allowed=False`` 时携带 HTTP 状态码与给前端的可读原因。"""

    allowed: bool
    status_code: int = 200
    code: str = "OK"
    message: str = ""
    retry_after: int = 0
    details: dict = field(default_factory=dict)


class RateLimiter:
    """按 key（客户端 IP）的固定窗口限速；窗口内超过 ``limit`` 即拒绝。"""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._buckets: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> QuotaDecision:
        now = self._clock()
        with self._lock:
            count, start = self._buckets.get(key, (0, now))
            if now - start >= self._window:  # 窗口已翻篇
                count, start = 0, now
            count += 1
            self._buckets[key] = (count, start)
            over = count > self._limit
            retry_after = max(1, int(self._window - (now - start)) + 1)
            used = count
        if not over:
            return QuotaDecision(True, details={"used_in_window": used, "limit": self._limit})
        return QuotaDecision(
            False,
            status_code=429,
            code="RATE_LIMITED",
            message=(
                f"请求过于频繁：每个客户端每分钟最多 {self._limit} 次，"
                f"请 {retry_after} 秒后重试。"
            ),
            retry_after=retry_after,
            details={"used_in_window": used, "limit": self._limit},
        )


class DailyBudget:
    """每日 LLM token / 成本累计与熔断；跨日自动重置。"""

    def __init__(
        self,
        max_tokens: int,
        max_cost_cny: float,
        *,
        input_price_per_m: float,
        output_price_per_m: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._max_tokens = max_tokens
        self._max_cost = max_cost_cny
        self._in_price = input_price_per_m
        self._out_price = output_price_per_m
        self._clock = clock
        self._day: str = ""
        self._tokens = 0
        self._cost = 0.0
        self._calls = 0
        self._lock = threading.Lock()
        self._roll()

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self._clock()))

    def _roll(self) -> None:
        day = self._today()
        if day != self._day:
            self._day, self._tokens, self._cost, self._calls = day, 0, 0.0, 0

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        """记录一次 LLM 调用的用量（按配置单价折算成本）。"""
        with self._lock:
            self._roll()
            self._tokens += prompt_tokens + completion_tokens
            self._cost += (
                prompt_tokens / 1e6 * self._in_price
                + completion_tokens / 1e6 * self._out_price
            )
            self._calls += 1

    def check(self) -> QuotaDecision:
        with self._lock:
            self._roll()
            tokens, cost = self._tokens, self._cost
        over_tokens = tokens >= self._max_tokens
        over_cost = cost >= self._max_cost
        if not (over_tokens or over_cost):
            return QuotaDecision(
                True, details={"tokens": tokens, "cost_cny": round(cost, 6)}
            )
        why = []
        if over_tokens:
            why.append(f"token 已达 {tokens}/{self._max_tokens}")
        if over_cost:
            why.append(f"预估成本已达 ¥{cost:.4f}/¥{self._max_cost}")
        return QuotaDecision(
            False,
            status_code=503,
            code="BUDGET_EXHAUSTED",
            message=(
                "今日 LLM 额度已用尽（"
                + "，".join(why)
                + "），已熔断新请求；请明日再试或联系维护者调整预算。"
            ),
            details={"tokens": tokens, "cost_cny": round(cost, 6)},
        )

    def snapshot(self) -> dict:
        with self._lock:
            self._roll()
            tokens, cost, calls = self._tokens, self._cost, self._calls
        return {
            "day": self._day,
            "calls": calls,
            "tokens": tokens,
            "cost_cny": round(cost, 6),
            "max_tokens": self._max_tokens,
            "max_cost_cny": self._max_cost,
            "token_used_pct": round(tokens / self._max_tokens * 100, 2) if self._max_tokens else 0.0,
            "cost_used_pct": round(cost / self._max_cost * 100, 2) if self._max_cost else 0.0,
        }


class QuotaManager:
    """组合「每 IP 限速」与「每日成本熔断」；两者均可按配置独立关闭。"""

    def __init__(
        self,
        settings: Settings,
        *,
        clock: Callable[[], float] = time.monotonic,
        day_clock: Callable[[], float] = time.time,
    ) -> None:
        self._rate_enabled = settings.rate_limit_enabled
        self._budget_enabled = settings.cost_budget_enabled
        self._rate = (
            RateLimiter(settings.rate_limit_per_minute, clock=clock)
            if self._rate_enabled
            else None
        )
        self._budget = (
            DailyBudget(
                settings.daily_token_budget,
                settings.daily_cost_budget_cny,
                input_price_per_m=settings.llm_input_price_per_m,
                output_price_per_m=settings.llm_output_price_per_m,
                clock=day_clock,
            )
            if self._budget_enabled
            else None
        )

    def check_request(self, client_key: str) -> QuotaDecision:
        """先查全局成本熔断（影响面最大），再查每 IP 限速。"""
        if self._budget is not None:
            decision = self._budget.check()
            if not decision.allowed:
                return decision
        if self._rate is not None:
            return self._rate.check(client_key)
        return QuotaDecision(True)

    def check_rate_only(self, client_key: str) -> QuotaDecision:
        """只查每 IP 限速，不查成本熔断。

        用于「不消耗 LLM 但仍需防刷」的接口（如训练抽题）：额度熔断不该误伤
        不花钱的读取，但题库也不能被无限制抓取。
        """
        if self._rate is not None:
            return self._rate.check(client_key)
        return QuotaDecision(True)

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """LLM 用量回调（由 ``DeepSeekChatClient`` 每次调用后触发）。"""
        if self._budget is not None:
            self._budget.record(prompt_tokens, completion_tokens)

    def snapshot(self) -> dict:
        return {
            "rate_limit": {
                "enabled": self._rate_enabled,
                "per_minute": self._rate._limit if self._rate else None,  # noqa: SLF001 - 自省用
            },
            "budget": (
                {"enabled": self._budget_enabled}
                if self._budget is None
                else {"enabled": self._budget_enabled, **self._budget.snapshot()}
            ),
        }


_QUOTA: QuotaManager | None = None
_QUOTA_LOCK = threading.Lock()


def get_quota() -> QuotaManager:
    """进程级单例（按当前 settings 构建一次）；测试可用 :func:`reset_quota` 重置。"""
    global _QUOTA
    if _QUOTA is None:
        with _QUOTA_LOCK:
            if _QUOTA is None:
                from app.config import get_settings

                _QUOTA = QuotaManager(get_settings())
    return _QUOTA


def reset_quota(manager: QuotaManager | None = None) -> None:
    global _QUOTA
    with _QUOTA_LOCK:
        _QUOTA = manager
