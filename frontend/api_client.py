"""后端 API 客户端：字段严格对齐 ``app/api/schemas.py`` 契约。

- 会话创建走 multipart（track / num_questions / candidate_name / resume / jd）；
- SSE 端点以生成器逐事件产出（stage / token / error / done）；
- 统一错误结构解析：后端 ``{"error": {"code", "message", "details"}}`` → ``ApiClientError``；
- 可选密钥仅经请求头传递（``X-DeepSeek-Key`` / ``X-SiliconFlow-Key``），不落盘。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://localhost:8000"
DEEPSEEK_KEY_HEADER = "X-DeepSeek-Key"
SILICONFLOW_KEY_HEADER = "X-SiliconFlow-Key"


class ApiClientError(Exception):
    """客户端统一错误：透出后端返回的 code 与中文 message 原文。"""

    def __init__(self, message: str, *, code: str = "CLIENT_ERROR", status: int | None = None):
        self.code = code
        self.status = status
        super().__init__(message)


@dataclass
class SseEventLite:
    """一条 SSE 事件（event 名 + 反序列化后的 data）。"""

    event: str
    data: dict


class BackendClient:
    """后端 HTTP 客户端（同步）。``timeout`` 用于普通请求，``stream_timeout`` 用于 SSE。"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        deepseek_key: str = "",
        siliconflow_key: str = "",
        timeout: float = 30.0,
        stream_timeout: float = 600.0,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._deepseek_key = deepseek_key.strip()
        self._siliconflow_key = siliconflow_key.strip()
        self._timeout = timeout
        self._stream_timeout = stream_timeout

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    @property
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._deepseek_key:
            headers[DEEPSEEK_KEY_HEADER] = self._deepseek_key
        if self._siliconflow_key:
            headers[SILICONFLOW_KEY_HEADER] = self._siliconflow_key
        return headers

    def _parse_error(self, resp: httpx.Response) -> ApiClientError:
        """解析统一错误结构；无法解析时退回原始文本。"""
        try:
            body = resp.json()
            err = body.get("error") or {}
            return ApiClientError(
                str(err.get("message") or resp.text[:300]),
                code=str(err.get("code") or f"HTTP_{resp.status_code}"),
                status=resp.status_code,
            )
        except (json.JSONDecodeError, AttributeError):
            return ApiClientError(
                f"后端返回 HTTP {resp.status_code}：{resp.text[:300]}",
                status=resp.status_code,
            )

    def _get_json(self, path: str) -> dict:
        try:
            resp = httpx.get(
                f"{self.base_url}{path}", timeout=self._timeout, headers=self._headers
            )
        except httpx.HTTPError as exc:
            raise ApiClientError(f"无法连接后端（{self.base_url}）：{exc}") from exc
        if resp.status_code >= 400:
            raise self._parse_error(resp)
        return resp.json()

    # ------------------------------------------------------------------
    # 接口
    # ------------------------------------------------------------------
    def health(self) -> dict:
        """GET /health：组件与配置状态（不因未配置 key 而失败）。"""
        return self._get_json("/health")

    def create_session(
        self,
        *,
        track: str,
        num_questions: int = 5,
        candidate_name: str | None = None,
        resume: tuple[str, bytes, str] | None = None,
        jd: tuple[str, bytes, str] | None = None,
    ) -> dict:
        """POST /sessions（multipart）→ SessionRecord。``resume``/``jd`` 为 (文件名, 字节, MIME)。"""
        data: dict[str, str] = {"track": track, "num_questions": str(num_questions)}
        if candidate_name:
            data["candidate_name"] = candidate_name
        files: dict[str, tuple[str, bytes, str]] = {}
        if resume is not None:
            files["resume"] = resume
        if jd is not None:
            files["jd"] = jd
        try:
            with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
                resp = client.post(
                    f"{self.base_url}/sessions", data=data, files=files or None
                )
        except httpx.HTTPError as exc:
            raise ApiClientError(f"无法连接后端（{self.base_url}）：{exc}") from exc
        if resp.status_code >= 400:
            raise self._parse_error(resp)
        return resp.json()

    def post_message(self, session_id: str, content: str) -> Iterator[SseEventLite]:
        """POST /sessions/{id}/messages（SSE）→ 逐事件产出。

        ``content`` 为空且面试未开始 → 触发开场与第一题；否则提交回答。
        """
        url = f"{self.base_url}/sessions/{session_id}/messages"
        try:
            with httpx.Client(timeout=self._stream_timeout, headers=self._headers) as client:
                with client.stream("POST", url, json={"content": content}) as resp:
                    if resp.status_code >= 400:
                        resp.read()  # 先读完响应体才能安全解析错误 JSON
                        raise self._parse_error(resp)
                    event_name = ""
                    data_lines: list[str] = []
                    for line in resp.iter_lines():
                        if line == "":
                            if event_name:
                                yield SseEventLite(
                                    event_name, json.loads("\n".join(data_lines) or "{}")
                                )
                            event_name, data_lines = "", []
                        elif line.startswith("event: "):
                            event_name = line[len("event: ") :]
                        elif line.startswith("data: "):
                            data_lines.append(line[len("data: ") :])
        except httpx.HTTPError as exc:
            raise ApiClientError(f"SSE 连接中断（{self.base_url}）：{exc}") from exc

    def finish(self, session_id: str) -> dict:
        """POST /finish：提前结束并生成报告（幂等）→ InterviewReport。"""
        path = f"/sessions/{session_id}/finish"
        try:
            with httpx.Client(timeout=self._timeout, headers=self._headers) as client:
                resp = client.post(f"{self.base_url}{path}")
        except httpx.HTTPError as exc:
            raise ApiClientError(f"无法连接后端（{self.base_url}）：{exc}") from exc
        if resp.status_code >= 400:
            raise self._parse_error(resp)
        return resp.json()

    def get_session(self, session_id: str) -> dict:
        """GET /sessions/{id} → {session: SessionRecord, messages: [MessageRecord]}。"""
        return self._get_json(f"/sessions/{session_id}")

    def get_report(self, session_id: str) -> dict:
        """GET /sessions/{id}/report → InterviewReport（未生成时抛 404 ApiClientError）。"""
        return self._get_json(f"/sessions/{session_id}/report")
