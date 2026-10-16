"""面试状态机的状态定义（LangGraph StateGraph 使用，JSON 可序列化）。

状态同时作为 Task 6 仓储层 ``save_state`` 的快照对象：全部为内置类型
（dict/list/str/int/bool/None），json.dumps 后直接落库，恢复时 json.loads 等值。
"""

from __future__ import annotations

from typing import Any, TypedDict


class InterviewState(TypedDict, total=False):
    session_id: str
    track: str  # 岗位方向
    num_questions: int  # 计划主问题数（追问不占额度）
    greeted: bool
    stage: str  # 与 schemas.InterviewStage 对齐
    questions_asked: list[dict[str, Any]]  # 已问主问题 {question, focus, evidence_ids, round}
    current_question: dict[str, Any] | None  # 当前等待回答的问题（含追问）
    current_evidence: list[dict[str, Any]]  # 当前问题证据 {chunk_id, source, heading, score, text}
    used_evidence_ids: list[str]  # 证据去重
    followup_count: int  # 当前问题已追问次数（≤2）
    current_answer: dict[str, Any] | None  # 待研判的 {question, answer}
    answers: list[dict[str, Any]]  # 已研判 {question, answer, quality, feedback, followup_question}
    pending_answer: str | None  # 候选人刚提交、待 receive_answer 处理的回答
    finish_requested: bool  # 提前结束标志
    done: bool
    evaluation_summary: dict[str, Any]  # final_evaluation 聚合结果
    report: dict[str, Any] | None  # InterviewReport.model_dump(mode="json")
    node_trace: list[str]  # 经过的节点序列（测试 / trace 用）
    candidate_profile: dict[str, Any]  # 候选人画像（轻量观察记录）


def initial_state(session_id: str, track: str, num_questions: int) -> InterviewState:
    return InterviewState(
        session_id=session_id,
        track=track,
        num_questions=num_questions,
        greeted=False,
        stage="opening",
        questions_asked=[],
        current_question=None,
        current_evidence=[],
        used_evidence_ids=[],
        followup_count=0,
        current_answer=None,
        answers=[],
        pending_answer=None,
        finish_requested=False,
        done=False,
        evaluation_summary={},
        report=None,
        node_trace=[],
        candidate_profile={"observations": []},
    )
