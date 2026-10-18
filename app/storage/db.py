"""会话持久化仓储层（SQLite，标准库 sqlite3）。

四张表：
- sessions：会话元数据（方向 / 题目数 / 状态 / 时间戳）；
- messages：面试消息（角色 / 内容 / 阶段 / 证据引用 JSON）；
- states：LangGraph 兼容的会话状态快照（JSON，每会话一条，覆盖式 upsert）；
- reports：面试报告（InterviewReport JSON，每会话一条）。

每次操作独立开连接（check_same_thread 默认约束下对 FastAPI 多线程安全），
初始化时 CREATE TABLE IF NOT EXISTS，重复构造无副作用。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.api.schemas import (
    CreateSessionRequest,
    EvidenceRef,
    InterviewReport,
    MessageRecord,
    SessionRecord,
    SessionStatus,
    TrainAttemptRecord,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    track        TEXT NOT NULL,
    num_questions INTEGER NOT NULL,
    candidate_name TEXT,
    status       TEXT NOT NULL DEFAULT 'active',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    stage       TEXT,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS states (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(session_id),
    state_json  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    session_id  TEXT PRIMARY KEY REFERENCES sessions(session_id),
    report_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS train_attempts (
    attempt_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT NOT NULL,
    question_id TEXT NOT NULL,
    source      TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    quality     TEXT NOT NULL,
    feedback    TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class SessionRepository:
    """会话仓储：CRUD + 状态快照 + 报告。``db_path`` 为 SQLite 文件路径。"""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------
    def create_session(self, req: CreateSessionRequest) -> SessionRecord:
        now = _utcnow()
        record = SessionRecord(
            session_id=uuid.uuid4().hex,
            track=req.track,
            num_questions=req.num_questions,
            candidate_name=req.candidate_name,
            status="active",
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions"
                " (session_id, track, num_questions, candidate_name, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.session_id, record.track, record.num_questions,
                    record.candidate_name, record.status, _iso(now), _iso(now),
                ),
            )
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            track=row["track"],
            num_questions=row["num_questions"],
            candidate_name=row["candidate_name"],
            status=row["status"],
            created_at=_parse(row["created_at"]),
            updated_at=_parse(row["updated_at"]),
        )

    def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (status, _iso(_utcnow()), session_id),
            )

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------
    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        stage: str | None = None,
        evidence: list[EvidenceRef] | None = None,
    ) -> MessageRecord:
        now = _utcnow()
        evidence_json = json.dumps(
            [e.model_dump() for e in (evidence or [])], ensure_ascii=False
        )
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, stage, evidence_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, stage, evidence_json, _iso(now)),
            )
            message_id = cursor.lastrowid
        return MessageRecord(
            id=message_id,
            session_id=session_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            stage=stage,  # type: ignore[arg-type]
            evidence=evidence or [],
            created_at=now,
        )

    def list_messages(self, session_id: str) -> list[MessageRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
        return [
            MessageRecord(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                stage=row["stage"],
                evidence=[EvidenceRef(**e) for e in json.loads(row["evidence_json"])],
                created_at=_parse(row["created_at"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # 状态快照（LangGraph checkpoint 兼容 JSON）
    # ------------------------------------------------------------------
    def save_state(self, session_id: str, state: dict) -> None:
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO states (session_id, state_json, updated_at) VALUES (?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json,"
                " updated_at = excluded.updated_at",
                (session_id, payload, _iso(_utcnow())),
            )

    def load_state(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM states WHERE session_id = ?", (session_id,)
            ).fetchone()
        return json.loads(row["state_json"]) if row else None

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    def save_report(self, session_id: str, report: InterviewReport) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reports (session_id, report_json, created_at) VALUES (?, ?, ?)"
                " ON CONFLICT(session_id) DO UPDATE SET report_json = excluded.report_json,"
                " created_at = excluded.created_at",
                (session_id, report.to_json(), _iso(_utcnow())),
            )

    def get_report(self, session_id: str) -> InterviewReport | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT report_json FROM reports WHERE session_id = ?", (session_id,)
            ).fetchone()
        return InterviewReport.from_json(row["report_json"]) if row else None

    # ------------------------------------------------------------------
    # 训练模式作答记录（train_attempts）
    # ------------------------------------------------------------------
    def add_train_attempt(
        self,
        username: str,
        question_id: str,
        source: str,
        question: str,
        answer: str,
        quality: str,
        feedback: str,
    ) -> TrainAttemptRecord:
        now = _utcnow()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO train_attempts"
                " (username, question_id, source, question, answer, quality, feedback, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (username, question_id, source, question, answer, quality, feedback, _iso(now)),
            )
            attempt_id = cursor.lastrowid
        return TrainAttemptRecord(
            attempt_id=attempt_id,
            question_id=question_id,
            question=question,
            answer=answer,
            quality=quality,
            feedback=feedback,
            created_at=now,
        )

    def list_train_attempts(self, username: str, limit: int = 50) -> list[TrainAttemptRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM train_attempts WHERE username = ?"
                " ORDER BY attempt_id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        return [
            TrainAttemptRecord(
                attempt_id=row["attempt_id"],
                question_id=row["question_id"],
                question=row["question"],
                answer=row["answer"],
                quality=row["quality"],
                feedback=row["feedback"],
                created_at=_parse(row["created_at"]),
            )
            for row in rows
        ]

    def list_train_done_question_ids(self, username: str) -> set[str]:
        """该用户已作答过的不同 question_id（抽题排除用）。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT question_id FROM train_attempts WHERE username = ?",
                (username,),
            ).fetchall()
        return {row["question_id"] for row in rows}
