"""训练模式路由（刷题练习）：登录 → 抽题 → 作答研判 → 历史进度 / 错题本。

与模拟面试的边界：
- 独立命名空间（``/train/*``）、独立令牌（``trn1.`` 前缀 + scope=train，见
  ``app.train.auth``），与面试 session_id 不可互换；
- 评价复用面试同源的 ``app.agent.assess``（同 prompt、同解析、同 quality 口径）；
- ``next`` 按「错题优先」抽题（poor > partial > 未作答 > good，见 ``QuestionBank.draw``），
  并以请求里的 ``avoid_question_id`` 避免连出同一题；
- ``mistakes`` 汇总最近一次研判非 good 的题（错题本：题干 / 我的作答 / 点评 / 答案要点）；
- 成本防护：login 走专用限速（防爆破），next/reveal/mistakes 走每 IP 限速（防题库与
  答案要点抓取），answer 是**唯一消耗 LLM** 的接口，挂 ``enforce_quota``（成本熔断 + 每 IP 限速）；
- 题目内容只从服务端题库取（客户端仅提交 question_id），answer 接口无法被当成
  「免费 LLM 代理」评任意文本；答案要点只在作答后（或 reveal / mistakes）下发。
"""

from __future__ import annotations

import hmac
import logging
import threading
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.agent.assess import ASSESS_SYSTEM, ASSESS_USER, parse_assess_verdict
from app.api.deps import ServiceContainer, enforce_quota, get_services
from app.api.errors import ApiError
from app.api.quota import RateLimiter, get_quota
from app.api.schemas import (
    TrainAnswerRequest,
    TrainAnswerResponse,
    TrainHistoryResponse,
    TrainLoginRequest,
    TrainLoginResponse,
    TrainMistakeItem,
    TrainMistakesResponse,
    TrainNextRequest,
    TrainNextResponse,
    TrainQuestionView,
    TrainRevealRequest,
    TrainRevealResponse,
)
from app.core.llm import ChatMessage
from app.train.auth import issue_token, verify_password, verify_token
from app.train.questions import QuestionBank, TrainQuestion, get_question_bank

logger = logging.getLogger(__name__)

router = APIRouter(tags=["train"])


# ---------------------------------------------------------------------------
# 登录限速（独立于 quota 的固定窗口：防爆破，失败计数与成功同口径消耗窗口）
# ---------------------------------------------------------------------------
_LOGIN_LIMITER: RateLimiter | None = None
_LOGIN_LIMITER_LOCK = threading.Lock()


def _login_limiter(services: ServiceContainer) -> RateLimiter:
    global _LOGIN_LIMITER
    with _LOGIN_LIMITER_LOCK:
        if _LOGIN_LIMITER is None:
            _LOGIN_LIMITER = RateLimiter(services.settings.train_login_rate_per_minute)
        return _LOGIN_LIMITER


def _reset_login_limiter() -> None:
    """测试用：清空登录限速器（下次按当前 settings 重建）。"""
    global _LOGIN_LIMITER
    with _LOGIN_LIMITER_LOCK:
        _LOGIN_LIMITER = None


def _client_ip(request: Request) -> str:
    forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    if forwarded:
        return forwarded
    return request.client.host if request.client else "unknown"


def _check_configured(services: ServiceContainer) -> None:
    s = services.settings
    if not (s.train_username and s.train_password_hash and s.train_token_secret):
        raise ApiError(
            503,
            "TRAIN_NOT_CONFIGURED",
            "训练模式未启用：服务端未配置 TRAIN_USERNAME / TRAIN_PASSWORD_HASH / "
            "TRAIN_TOKEN_SECRET（参考 .env.example）",
        )


def _get_bank(services: ServiceContainer) -> QuestionBank:
    bank = get_question_bank(services.settings.resolve(services.settings.train_qa_dir))
    if bank.total == 0:
        raise ApiError(
            503,
            "TRAIN_BANK_EMPTY",
            f"训练题库为空：{services.settings.train_qa_dir} 下未解析到任何题目",
        )
    return bank


async def require_train_user(
    request: Request, services: Annotated[ServiceContainer, Depends(get_services)]
) -> str:
    """训练接口鉴权依赖：校验 Bearer 令牌，返回令牌中的用户名。"""
    _check_configured(services)
    auth = request.headers.get("Authorization") or ""
    scheme, _, token = auth.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(401, "TRAIN_UNAUTHORIZED", "缺少训练令牌，请先登录（POST /train/login）")
    username = verify_token(token.strip(), services.settings.train_token_secret)
    if username is None or username != services.settings.train_username:
        raise ApiError(401, "TRAIN_UNAUTHORIZED", "训练令牌无效或已过期，请重新登录")
    return username


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@router.post("/train/login")
async def train_login(
    req: TrainLoginRequest,
    request: Request,
    services: Annotated[ServiceContainer, Depends(get_services)],
) -> TrainLoginResponse:
    _check_configured(services)
    decision = _login_limiter(services).check(_client_ip(request))
    if not decision.allowed:
        raise ApiError(
            decision.status_code, decision.code, decision.message,
            headers={"Retry-After": str(decision.retry_after)} if decision.retry_after else None,
        )
    s = services.settings
    # 用户名比较也用常量时间，避免先暴露「用户名是否存在」
    user_ok = hmac.compare_digest(req.username, s.train_username)
    if not (user_ok and verify_password(req.password, s.train_password_hash)):
        raise ApiError(401, "TRAIN_AUTH_FAILED", "用户名或密码错误")
    ttl = s.train_token_ttl_seconds
    return TrainLoginResponse(
        token=issue_token(s.train_username, s.train_token_secret, ttl), expires_in=ttl
    )


@router.post("/train/next")
async def train_next(
    request: Request,
    services: Annotated[ServiceContainer, Depends(get_services)],
    username: Annotated[str, Depends(require_train_user)],
    req: TrainNextRequest | None = None,
) -> TrainNextResponse:
    decision = get_quota().check_rate_only(_client_ip(request))
    if not decision.allowed:
        raise ApiError(
            decision.status_code, decision.code, decision.message,
            headers={"Retry-After": str(decision.retry_after)} if decision.retry_after else None,
        )
    bank = _get_bank(services)
    # 错题优先抽题：每题以「最近一次作答」的质量与时间为定序依据
    latest = services.repo.list_train_latest_attempts(username)
    states = {a.question_id: (a.quality, a.created_at) for a in latest}
    question = bank.draw(states, avoid_id=req.avoid_question_id if req else None)
    assert question is not None  # bank.total > 0 已校验
    return TrainNextResponse(
        question=TrainQuestionView(
            question_id=question.question_id,
            question=question.question,
            tags=list(question.tags),
            source=question.source,
        ),
        done_count=len(states),
        total_count=bank.total,
    )


@router.post("/train/answer", dependencies=[Depends(enforce_quota)])
async def train_answer(
    req: TrainAnswerRequest,
    services: Annotated[ServiceContainer, Depends(get_services)],
    username: Annotated[str, Depends(require_train_user)],
) -> TrainAnswerResponse:
    bank = _get_bank(services)
    question: TrainQuestion | None = bank.get(req.question_id)
    if question is None:
        raise ApiError(
            404, "TRAIN_QUESTION_NOT_FOUND",
            f"题目 {req.question_id} 不在训练题库中，请重新抽题（POST /train/next）",
        )

    # 研判与面试同源：参考依据直接给题库答案要点（训练场景无检索证据）
    evidence = "\n".join(f"- {p}" for p in question.points) or "（题库未提供答案要点）"
    messages: list[ChatMessage] = [
        {"role": "system", "content": ASSESS_SYSTEM},
        {
            "role": "user",
            "content": ASSESS_USER.format(
                question=question.question,
                focus="、".join(question.tags),
                evidence=evidence,
                answer=req.answer.strip(),
            ),
        },
    ]
    # 训练核心就是评价：研判失败不沿用面试「默认 good」兜底，直接抛错（全局 502）
    resp = await services.llm.chat(messages, purpose="eval", json_mode=True)
    verdict = parse_assess_verdict(resp.content)

    record = services.repo.add_train_attempt(
        username=username,
        question_id=question.question_id,
        source=question.source,
        question=question.question,
        answer=req.answer.strip(),
        quality=verdict["quality"],
        feedback=verdict["feedback"],
    )
    return TrainAnswerResponse(
        attempt_id=record.attempt_id,
        quality=verdict["quality"],
        feedback=verdict["feedback"],
        reference_points=list(question.points),
    )


@router.post("/train/reveal")
async def train_reveal(
    req: TrainRevealRequest,
    request: Request,
    services: Annotated[ServiceContainer, Depends(get_services)],
    username: Annotated[str, Depends(require_train_user)],
) -> TrainRevealResponse:
    """答不出来时直接下发答案要点：不调 LLM、不落库（看答案 ≠ 练过，进度不前进）。

    与 next 同档防护：每 IP 限速防整库抓答案，但不查成本熔断（零 LLM 开销）。
    """
    decision = get_quota().check_rate_only(_client_ip(request))
    if not decision.allowed:
        raise ApiError(
            decision.status_code, decision.code, decision.message,
            headers={"Retry-After": str(decision.retry_after)} if decision.retry_after else None,
        )
    bank = _get_bank(services)
    question: TrainQuestion | None = bank.get(req.question_id)
    if question is None:
        raise ApiError(
            404, "TRAIN_QUESTION_NOT_FOUND",
            f"题目 {req.question_id} 不在训练题库中，请重新抽题（POST /train/next）",
        )
    return TrainRevealResponse(
        question_id=question.question_id, reference_points=list(question.points)
    )


@router.get("/train/history")
async def train_history(
    services: Annotated[ServiceContainer, Depends(get_services)],
    username: Annotated[str, Depends(require_train_user)],
    limit: int = 50,
) -> TrainHistoryResponse:
    bank = _get_bank(services)
    attempts = services.repo.list_train_attempts(username, limit=max(1, min(limit, 200)))
    done_count = len(services.repo.list_train_latest_attempts(username))
    quality_counts: dict[str, int] = {}
    for attempt in attempts:
        quality_counts[attempt.quality] = quality_counts.get(attempt.quality, 0) + 1
    return TrainHistoryResponse(
        attempts=attempts,
        done_count=done_count,
        total_count=bank.total,
        quality_counts=quality_counts,
    )


@router.get("/train/mistakes")
async def train_mistakes(
    request: Request,
    services: Annotated[ServiceContainer, Depends(get_services)],
    username: Annotated[str, Depends(require_train_user)],
) -> TrainMistakesResponse:
    """错题本：最近一次研判非 good 的题（poor 在前），附我的作答 / 点评 / 答案要点。

    与 reveal 同档防护：每 IP 限速（防整库抓答案要点），零 LLM 开销、不查成本熔断。
    """
    decision = get_quota().check_rate_only(_client_ip(request))
    if not decision.allowed:
        raise ApiError(
            decision.status_code, decision.code, decision.message,
            headers={"Retry-After": str(decision.retry_after)} if decision.retry_after else None,
        )
    bank = _get_bank(services)
    latest = services.repo.list_train_latest_attempts(username)
    mistakes = [a for a in latest if a.quality != "good"]
    mistakes.sort(key=lambda a: 0 if a.quality == "poor" else 1)  # 稳定排序：poor 在前
    items: list[TrainMistakeItem] = []
    for attempt in mistakes:
        question = bank.get(attempt.question_id)
        items.append(
            TrainMistakeItem(
                question_id=attempt.question_id,
                question=question.question if question else attempt.question,
                tags=list(question.tags) if question else [],
                source=question.source if question else attempt.source,
                answer=attempt.answer,
                quality=attempt.quality,
                feedback=attempt.feedback,
                reference_points=list(question.points) if question else [],
                created_at=attempt.created_at,
            )
        )
    return TrainMistakesResponse(mistakes=items, total_count=bank.total)
