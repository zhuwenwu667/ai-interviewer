"""配置加载与启动期校验测试（TR-1.2、TR-1.3）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import ConfigurationError
from app.config.settings import BASE_DIR, Settings, validate_settings


def _make_settings(**overrides) -> Settings:
    """构造跳过 .env 文件的 Settings，仅使用默认值/显式覆盖。"""
    return Settings(_env_file=None, **overrides)


def test_defaults_loaded_without_env_file(env_without_keys):
    settings = _make_settings()
    assert settings.deepseek_api_key == ""
    assert settings.siliconflow_api_key == ""
    assert settings.qdrant_mode == "local"
    assert settings.embedding_dim == 1024
    assert settings.deepseek_model == "deepseek-chat"


def test_missing_deepseek_key_raises_chinese_error(env_without_keys):
    settings = _make_settings()
    with pytest.raises(ConfigurationError) as exc_info:
        validate_settings(settings, require=("chat",))
    message = str(exc_info.value)
    # 必须点名变量名并给出中文指引，而不是 KeyError 堆栈
    assert "DEEPSEEK_API_KEY" in message
    assert ".env" in message


def test_missing_either_key_reports_all_missing(env_without_keys):
    settings = _make_settings(deepseek_api_key="sk-xxx")
    with pytest.raises(ConfigurationError) as exc_info:
        validate_settings(settings, require=("chat", "embedding"))
    assert "SILICONFLOW_API_KEY" in str(exc_info.value)
    assert "DEEPSEEK_API_KEY" not in str(exc_info.value)


def test_validate_passes_when_both_keys_present(env_without_keys):
    settings = _make_settings(
        deepseek_api_key="sk-deepseek",
        siliconflow_api_key="sk-sf",
    )
    validate_settings(settings, require=("chat", "embedding"))


def test_langfuse_enabled_requires_all_three(env_without_keys):
    assert _make_settings().langfuse_enabled is False
    partial = _make_settings(
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
    )
    assert partial.langfuse_enabled is False
    full = _make_settings(
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_host="https://cloud.langfuse.com",
    )
    assert full.langfuse_enabled is True


def test_model_overrides_fallback_to_default(env_without_keys):
    settings = _make_settings()
    assert settings.question_model == "deepseek-chat"
    customized = _make_settings(llm_model_eval="deepseek-reasoner")
    assert customized.eval_model == "deepseek-reasoner"
    assert customized.question_model == "deepseek-chat"


def test_judge_falls_back_to_eval_model_then_deepseek(env_without_keys):
    settings = _make_settings(deepseek_api_key="sk-ds")
    # 未配置 EVAL_JUDGE_* 时，judge 完全回落到 eval_model / DeepSeek（生产零行为变化）
    assert settings.judge_model == settings.eval_model == "deepseek-chat"
    assert settings.judge_base_url == settings.deepseek_base_url
    assert settings.judge_api_key == "sk-ds"

    overridden = _make_settings(
        deepseek_api_key="sk-ds",
        eval_judge_model="Qwen/Qwen2.5-72B-Instruct",
        eval_judge_base_url="https://api.siliconflow.cn/v1",
        eval_judge_api_key="sk-sf",
    )
    assert overridden.judge_model == "Qwen/Qwen2.5-72B-Instruct"
    assert overridden.judge_base_url == "https://api.siliconflow.cn/v1"
    assert overridden.judge_api_key == "sk-sf"
    # 覆盖 judge 不影响面试评分链路所用的 eval_model
    assert overridden.eval_model == "deepseek-chat"


def test_resolve_judge_tracks_settings(env_without_keys):
    from app.eval.metrics_ragas import resolve_judge

    base = _make_settings(deepseek_api_key="sk-ds")
    spec = resolve_judge(base)
    assert (spec.model, spec.base_url, spec.api_key) == (
        "deepseek-chat",
        "https://api.deepseek.com",
        "sk-ds",
    )
    assert "deepseek-chat" in spec.label

    cross = resolve_judge(
        _make_settings(
            deepseek_api_key="sk-ds",
            eval_judge_model="Qwen/Qwen2.5-72B-Instruct",
            eval_judge_api_key="sk-sf",
        )
    )
    assert cross.model == "Qwen/Qwen2.5-72B-Instruct"
    assert cross.api_key == "sk-sf"
    assert cross.base_url == "https://api.deepseek.com"  # 仅覆盖 model 时 base_url 仍回落


def test_relative_paths_resolve_to_base_dir(env_without_keys):
    settings = _make_settings(sqlite_path="data/interview.db")
    resolved = settings.sqlite_path_abs
    assert resolved.is_absolute()
    assert resolved == (BASE_DIR / "data" / "interview.db")


def test_overlap_must_be_smaller_than_chunk_size(env_without_keys):
    with pytest.raises(ValidationError):
        _make_settings(chunk_size=500, chunk_overlap=500)


def test_config_status_never_leaks_api_keys(env_without_keys):
    settings = _make_settings(deepseek_api_key="sk-secret-value")
    status = settings.config_status()
    serialized = str(status)
    assert "sk-secret-value" not in serialized
    assert status["deepseek_configured"] is True
