"""前后端共享的 API 契约（唯一来源）。

- 会话：创建请求 / 响应记录、消息记录（角色 / 内容 / 阶段 / 证据引用）；
- SSE 事件：stage / token / error / done 四类（FR-8 约定）；
- 报告：五个固定维度各 1-5 分、每分关联问答证据、亮点 / 不足 / 建议、
  推荐复习知识点（可回链题库来源）。

Task 9（FastAPI）与 Task 10（Streamlit）均以此文件为字段契约，禁止各自另定义。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 会话与消息
# ---------------------------------------------------------------------------

# 面试状态机的阶段名（Task 7 节点与此对齐）
InterviewStage = Literal["opening", "question", "answer", "followup", "scoring", "report", "done"]

SessionStatus = Literal["active", "finished"]


class CreateSessionRequest(BaseModel):
    """创建面试会话请求（开始页表单）。"""

    track: str = Field(min_length=1, description="岗位方向，如 backend / ai_app")
    num_questions: int = Field(default=5, ge=1, le=10, description="计划题目数")
    candidate_name: str | None = Field(
        default=None, description="候选人称呼（HTTP 层必填；仓储模型允许为空以兼容历史数据）"
    )


class SessionRecord(BaseModel):
    session_id: str
    track: str
    num_questions: int
    candidate_name: str | None = None
    status: SessionStatus = "active"
    created_at: datetime
    updated_at: datetime


class EvidenceRef(BaseModel):
    """一条消息关联的 RAG 证据（出题 / 追问 / 评分均可引用）。"""

    chunk_id: str
    source: str
    heading: str = ""
    score: float | None = None


class MessageRecord(BaseModel):
    id: int
    session_id: str
    role: Literal["interviewer", "candidate", "system"]
    content: str
    stage: InterviewStage | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)
    created_at: datetime


# ---------------------------------------------------------------------------
# SSE 事件（event: <type>\ndata: <json>\n\n）
# ---------------------------------------------------------------------------


class StageEventData(BaseModel):
    stage: InterviewStage
    round: int = Field(default=0, ge=0, description="当前题目轮次（1-based），非出题阶段为 0")


class TokenEventData(BaseModel):
    content: str


class ErrorEventData(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class DoneEventData(BaseModel):
    reason: Literal["completed", "stopped", "error"]


SseEventType = Literal["stage", "token", "error", "done"]


class SseEvent(BaseModel):
    """SSE 信封；data 按 event 类型对应上面的 *EventData。"""

    event: SseEventType
    data: StageEventData | TokenEventData | ErrorEventData | DoneEventData

    def encode(self) -> str:
        return f"event: {self.event}\ndata: {self.data.model_dump_json()}\n\n"

    @classmethod
    def stage(cls, stage: InterviewStage, round: int = 0) -> SseEvent:
        return cls(event="stage", data=StageEventData(stage=stage, round=round))

    @classmethod
    def token(cls, content: str) -> SseEvent:
        return cls(event="token", data=TokenEventData(content=content))

    @classmethod
    def error(cls, code: str, message: str) -> SseEvent:
        return cls(event="error", data=ErrorEventData(code=code, message=message))

    @classmethod
    def done(cls, reason: Literal["completed", "stopped", "error"] = "completed") -> SseEvent:
        return cls(event="done", data=DoneEventData(reason=reason))


# ---------------------------------------------------------------------------
# 评分报告（FR-7）
# ---------------------------------------------------------------------------

# 五个固定评分维度（Task 8 报告生成与此对齐）
REPORT_DIMENSIONS: tuple[str, ...] = ("岗位匹配度", "基础知识", "项目深度", "表达逻辑", "问题解决")


class DimensionScore(BaseModel):
    dimension: str = Field(min_length=1)
    score: int = Field(ge=1, le=5)
    evidence: list[str] = Field(min_length=1, description="评分关联的具体问答证据摘录")
    comment: str = ""


class RecommendedTopic(BaseModel):
    topic: str = Field(min_length=1)
    source: str = Field(default="", description="题库来源回链（source / chunk_id）")
    reason: str = ""


class InterviewReport(BaseModel):
    session_id: str
    dimensions: list[DimensionScore] = Field(min_length=1)
    highlights: list[str] = Field(min_length=1)
    weaknesses: list[str] = Field(min_length=1)
    suggestions: list[str] = Field(min_length=1)
    recommended_topics: list[RecommendedTopic] = Field(default_factory=list)
    generated_at: datetime

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> InterviewReport:
        return cls.model_validate(json.loads(raw))


# ---------------------------------------------------------------------------
# 面试消息提交 / 会话详情 / 统一错误响应
# ---------------------------------------------------------------------------


class PostMessageRequest(BaseModel):
    """提交候选人回答；``content`` 为空且面试未开始时触发开场（开始面试）。"""

    content: str = Field(default="", max_length=10000)


class SessionDetailResponse(BaseModel):
    """GET /sessions/{id} 响应：会话元数据 + 全量消息历史。"""

    session: SessionRecord
    messages: list[MessageRecord]


class SessionSummary(SessionRecord):
    """GET /sessions 列表条目：会话元数据 + 是否已生成报告（供回看报告）。"""

    has_report: bool = False


class SessionListResponse(BaseModel):
    """GET /sessions 响应：面试历史列表（按创建时间倒序）。"""

    sessions: list[SessionSummary] = Field(default_factory=list)


class ErrorBody(BaseModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict | None = None


class ErrorResponse(BaseModel):
    """统一错误响应结构（全局异常处理与业务错误均输出此结构）。"""

    error: ErrorBody


# ---------------------------------------------------------------------------
# 训练模式（刷题练习，Task 22）：单账号登录 + 抽题作答 + 单轮研判 + 进度追踪
# ---------------------------------------------------------------------------


class TrainLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class TrainLoginResponse(BaseModel):
    token: str
    token_type: str = "Bearer"
    expires_in: int = Field(description="令牌有效期（秒）")


class TrainQuestionView(BaseModel):
    """next 接口下发的题目视图：绝不含答案要点（作答后才可见）。"""

    question_id: str
    question: str
    tags: list[str] = Field(default_factory=list)
    source: str


class TrainNextRequest(BaseModel):
    """抽题请求（可空）。``avoid_question_id`` 为上一题 id，避免连出同一题。"""

    avoid_question_id: str | None = Field(default=None, max_length=300)


class TrainNextResponse(BaseModel):
    question: TrainQuestionView
    done_count: int = Field(ge=0, description="该用户已作答过的不同题目数")
    total_count: int = Field(ge=0, description="题库总题数")


class TrainAnswerRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=300)
    answer: str = Field(min_length=1, max_length=10000)


class TrainRevealRequest(BaseModel):
    """答不出来时直接看答案要点：不消耗 LLM、不落库（看答案 ≠ 练过）。"""

    question_id: str = Field(min_length=1, max_length=300)


class TrainRevealResponse(BaseModel):
    question_id: str
    reference_points: list[str] = Field(default_factory=list, description="题库答案要点")


class TrainAnswerResponse(BaseModel):
    attempt_id: int
    quality: str = Field(description="good / partial / poor（与面试研判同口径）")
    feedback: str
    reference_points: list[str] = Field(default_factory=list, description="题库答案要点")


class TrainAttemptRecord(BaseModel):
    attempt_id: int
    question_id: str
    question: str
    answer: str
    quality: str
    feedback: str
    created_at: datetime


class TrainRetrainStats(BaseModel):
    """复练提升率（零 LLM 的存量数据聚合）。

    对每题按作答先后看：某次「非 good」之后还有新作答 → 该题算「复练过」；
    其中最近一次作答转为 good 的题算「已提升」。同题多次复练只按最近一次计趋势。
    """

    retrained_count: int = Field(default=0, ge=0, description="复练过的题数（非 good 后有新作答）")
    improved_count: int = Field(default=0, ge=0, description="其中最近一次转为 good 的题数")
    improvement_rate: float = Field(
        default=0.0, ge=0, le=1, description="improved_count / retrained_count，无复练时为 0"
    )


class TrainHistoryResponse(BaseModel):
    attempts: list[TrainAttemptRecord]
    done_count: int = Field(ge=0)
    total_count: int = Field(ge=0)
    quality_counts: dict[str, int] = Field(default_factory=dict)
    retrain_stats: TrainRetrainStats = Field(default_factory=TrainRetrainStats)


class TrainMistakeItem(BaseModel):
    """错题本条目：最近一次研判非 good 的题 + 我的作答 / 点评 / 答案要点。"""

    question_id: str
    question: str
    tags: list[str] = Field(default_factory=list)
    source: str
    answer: str = Field(description="我最近一次的作答原文")
    quality: str = Field(description="partial / poor（错题本只收非 good）")
    feedback: str = Field(description="最近一次的研判点评")
    reference_points: list[str] = Field(default_factory=list, description="题库答案要点")
    created_at: datetime


class TrainMistakesResponse(BaseModel):
    mistakes: list[TrainMistakeItem]
    total_count: int = Field(ge=0, description="题库总题数")


# ---------------------------------------------------------------------------
# 效果看板（GET /metrics）：把已入库的离线评测产物暴露为只读接口，零 LLM
# ---------------------------------------------------------------------------


class MetricRow(BaseModel):
    """看板里的一行指标。

    取值单位约定（全看板统一）：RAGAS 四项分数**保留 0-1 原值**（``unit="分"``，
    便于与文档 / 合格线 0.85 直接对照）；检索侧与分类的比例类指标**换算为百分数**
    （``unit="%"``，如 ``source_hit_rate`` 1.0 → 100.0）。
    """

    key: str
    name: str
    value: float | None = None
    unit: str = ""
    threshold: float | None = Field(default=None, description="合格线（仅门槛指标有值）")
    passed: bool | None = Field(default=None, description="是否达标；观察项为 None")
    delta: float | None = Field(default=None, description="相对回归基线的变化量")
    note: str = ""


class CategoryMetrics(BaseModel):
    """分类别（简历 / JD / 题库 / 题库外）指标；比例类均已换算为百分数。"""

    category: str
    n: int
    faithfulness: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_relevancy: float | None = None
    source_hit_rate: float | None = None
    source_f1: float | None = None


class AblationRow(BaseModel):
    """检索消融的一行：某个检索配置的效果与成本。"""

    config: str
    agentic: bool = False
    stages: list[str] = Field(default_factory=list, description="启用的检索阶段，如 rewrite/bm25/rrf/rerank")
    faithfulness: float
    context_precision: float
    context_recall: float
    answer_relevancy: float
    source_hit_rate: float = Field(description="百分数")
    mean_rounds: float
    mean_contexts: float
    llm_calls: int
    est_cost_cny_per_q: float = Field(description="单题预估成本（元）")


class CrossCheckRow(BaseModel):
    """异源 judge 交叉验证的一行：两个不同模型对同一指标的评分一致度。"""

    metric: str
    primary_mean: float
    cross_mean: float
    delta: float
    agree_0_1: float = Field(description="容差 0.1 下的双 judge 一致比例（百分数）")


class AbComparison(BaseModel):
    """A/B 实测对比（语义缓存 / 上下文预算）：基线 vs 优化后。"""

    label: str
    baseline: float
    optimized: float
    delta_pct: float = Field(description="优化后相对基线的变化百分比（负值 = 下降）")
    unit: str = ""


class MetricsDashboard(BaseModel):
    """GET /metrics：效果看板（只读聚合，零 LLM）。

    数据即仓库内已入库的评测产物，取每类**最新一份**；合格线判定复用 CI 门禁
    ``check_report``，保证「看板上的达标结论」与「CI 拦门的结论」同源同规则。
    """

    sources: dict[str, str] = Field(default_factory=dict, description="各分组对应的报告文件名")
    generated_at: dict[str, str] = Field(default_factory=dict, description="各报告生成时间")
    gate_passed: bool = False
    gate_violations: list[str] = Field(default_factory=list, description="未达标项的中文说明")
    dataset: dict[str, int] = Field(default_factory=dict, description="评测集规模与分类计数")
    quality: list[MetricRow] = Field(default_factory=list, description="RAGAS 总体 + 检索侧指标")
    by_category: list[CategoryMetrics] = Field(default_factory=list)
    ablation: list[AblationRow] = Field(default_factory=list)
    cross_check: list[CrossCheckRow] = Field(default_factory=list)
    cell_agreement: dict[str, float] = Field(default_factory=dict, description="双 judge 单元格一致度（百分数）")
    self_inflation: dict[str, float] = Field(default_factory=dict, description="自评虚高幅度（主 judge - 异源 judge）")
    session_cost: dict[str, float] = Field(default_factory=dict, description="会话成本 / 延迟 / Token 汇总")
    semantic_cache: list[AbComparison] = Field(default_factory=list)
    cache_hit_rate: float | None = Field(default=None, description="语义缓存命中率（百分数）")
    context_budget: list[AbComparison] = Field(default_factory=list)
    retrain: TrainRetrainStats = Field(default_factory=TrainRetrainStats)
