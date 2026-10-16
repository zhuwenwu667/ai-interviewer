"""/health 端点测试（TR-1.1）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config.settings import Settings
from app.main import app


def test_health_local_mode_reports_ok(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, qdrant_path=tmp_path / "qdrant")
    monkeypatch.setattr("app.api.health.get_settings", lambda: settings)

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["components"]["qdrant"]["status"] == "ok"
    assert data["components"]["qdrant"]["mode"] == "local"
    assert data["config"]["deepseek_configured"] is False
    # 未配置 key 时整体标记 degraded，但进程存活仍返回 200
    assert data["status"] == "degraded"


def test_health_server_mode_unreachable_reports_error(monkeypatch):
    settings = Settings(
        _env_file=None,
        qdrant_mode="server",
        qdrant_url="http://127.0.0.1:9",  # discard 端口：通常立即拒绝连接
        deepseek_api_key="sk-d",
        siliconflow_api_key="sk-s",
    )
    monkeypatch.setattr("app.api.health.get_settings", lambda: settings)

    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["components"]["qdrant"]["status"] == "error"
    assert data["components"]["qdrant"]["mode"] == "server"
    assert data["status"] == "degraded"
