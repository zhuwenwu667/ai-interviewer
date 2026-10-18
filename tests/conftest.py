"""pytest 公共夹具：隔离真实 .env，保证测试确定性。"""

from __future__ import annotations

import pytest

from app.api.quota import QuotaManager, reset_quota
from app.config import get_settings
from app.config.settings import Settings


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """每个用例前后清空 settings 单例缓存，避免串扰。

    同时把 ``LOG_DIR`` 重定向到用例临时目录，确保测试不会把可观测性 JSONL
    写进仓库的 ``logs/``（真实链路证据由手工/集成运行单独产出）。
    """
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _neutralize_quota():
    """默认关闭限速与成本熔断，避免「每 IP 每分钟请求数」让无关用例相互串扰。

    配额行为由 ``tests/test_quota.py`` 显式构造 ``QuotaManager`` 用例覆盖。
    """
    reset_quota(
        QuotaManager(
            Settings(_env_file=None, rate_limit_enabled=False, cost_budget_enabled=False)
        )
    )
    yield
    reset_quota(None)


@pytest.fixture
def env_without_keys(monkeypatch: pytest.MonkeyPatch):
    """显式清除两个关键 key 并返回可构造无 .env Settings 的环境。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    return monkeypatch
