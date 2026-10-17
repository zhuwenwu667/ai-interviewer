"""基于 pydantic-settings 的集中配置。

设计约定：
- API Key 等敏感字段默认空字符串（应用可启动、/health 可展示配置状态），
  命令行/业务链路入口通过 ``validate_runtime`` 做启动期强校验，给出中文报错；
- 相对路径统一解析到项目根目录（本文件上两级），Windows / Linux 行为一致；
- Langfuse 未配置时 ``langfuse_enabled`` 为 False，业务自动降级为本地 JSONL 日志。
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class ConfigurationError(Exception):
    """配置缺失或非法时抛出，消息必须包含变量名与中文排查建议。"""


class Settings(BaseSettings):
    """全部运行参数。字段名与 ``.env`` 中的大写下划线变量一一对应。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- DeepSeek ----------
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_model_question: str = ""
    llm_model_followup: str = ""
    llm_model_eval: str = ""
    llm_model_rewrite: str = ""

    # ---------- 评测 judge（RAGAS 交叉验证）----------
    # 留空则回落到 eval_model / deepseek_base_url / deepseek_api_key（生产零行为变化）。
    # 仅用于离线评测的 RAGAS 评判，不影响线上面试评分链路。
    eval_judge_model: str = ""
    eval_judge_base_url: str = ""
    eval_judge_api_key: str = ""

    # ---------- SiliconFlow ----------
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 1024

    # ---------- Qdrant ----------
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_path: Path = Field(default=Path("./data/qdrant_storage"))
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "interview_kb"

    # ---------- 检索管线 ----------
    retrieval_enable_multi_query: bool = True
    retrieval_enable_hyde: bool = False
    retrieval_multi_query_n: int = 3
    retrieval_vector_top_k: int = 20
    retrieval_bm25_top_k: int = 20
    retrieval_rrf_k: int = 60
    retrieval_final_top_k: int = 6
    adaptive_max_rewrite_rounds: int = 2

    # ---------- 切块 ----------
    chunk_size: int = 500
    chunk_overlap: int = 80

    # ---------- 会话 / 上传 ----------
    sqlite_path: Path = Field(default=Path("./data/interview.db"))
    upload_max_bytes: int = 10 * 1024 * 1024

    # ---------- 面试 ----------
    interview_default_questions: int = 5
    interview_max_followups: int = 2

    # ---------- API ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:8501"

    # ---------- 可观测 ----------
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = ""
    log_dir: Path = Field(default=Path("./logs"))
    log_level: str = "INFO"

    # ----- 校验：数值合理性 -----
    @field_validator("chunk_overlap")
    @classmethod
    def _check_overlap(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 500)
        if v >= size:
            raise ValueError(f"CHUNK_OVERLAP({v}) 必须小于 CHUNK_SIZE({size})")
        return v

    @model_validator(mode="after")
    def _check_bounds(self) -> Settings:
        if self.retrieval_final_top_k > min(
            self.retrieval_vector_top_k, self.retrieval_bm25_top_k
        ) * 4:
            raise ValueError("RETRIEVAL_FINAL_TOP_K 过大，超过单路召回数的 4 倍")
        if not 1 <= self.interview_max_followups <= 5:
            raise ValueError("INTERVIEW_MAX_FOLLOWUPS 应在 1-5 之间")
        return self

    # ----- 派生配置 -----
    @property
    def question_model(self) -> str:
        return self.llm_model_question or self.deepseek_model

    @property
    def followup_model(self) -> str:
        return self.llm_model_followup or self.deepseek_model

    @property
    def eval_model(self) -> str:
        return self.llm_model_eval or self.deepseek_model

    @property
    def judge_model(self) -> str:
        return self.eval_judge_model or self.eval_model

    @property
    def judge_base_url(self) -> str:
        return self.eval_judge_base_url or self.deepseek_base_url

    @property
    def judge_api_key(self) -> str:
        return self.eval_judge_api_key or self.deepseek_api_key

    @property
    def rewrite_model(self) -> str:
        return self.llm_model_rewrite or self.deepseek_model

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_secret_key
            and self.langfuse_public_key
            and self.langfuse_host
        )

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def resolve(self, path: Path | str) -> Path:
        """将相对路径解析到项目根目录；绝对路径原样返回。"""
        p = Path(path)
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    @property
    def qdrant_path_abs(self) -> Path:
        return self.resolve(self.qdrant_path)

    @property
    def sqlite_path_abs(self) -> Path:
        return self.resolve(self.sqlite_path)

    @property
    def log_dir_abs(self) -> Path:
        return self.resolve(self.log_dir)

    def config_status(self) -> dict[str, object]:
        """供 /health 使用的配置状态（不含任何密钥明文，仅布尔/名称）。"""
        return {
            "deepseek_configured": bool(self.deepseek_api_key),
            "deepseek_model": self.deepseek_model,
            "siliconflow_configured": bool(self.siliconflow_api_key),
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "qdrant_mode": self.qdrant_mode,
            "qdrant_url": self.qdrant_url if self.qdrant_mode == "server" else None,
            "langfuse_enabled": self.langfuse_enabled,
        }


# 运行期所需的环境变量：变量名 -> 中文用途说明
_REQUIRED_KEY_GROUPS: dict[str, tuple[str, str]] = {
    "chat": ("DEEPSEEK_API_KEY", "DeepSeek 对话模型（面试官出题/评分）"),
    "embedding": ("SILICONFLOW_API_KEY", "硅基流动 bge-m3 向量模型"),
}


def validate_settings(settings: Settings, require: tuple[str, ...] = ("chat", "embedding")) -> None:
    """检查必需的 API Key，缺失时抛出含变量名与指引的中文异常。

    ``require`` 可取 "chat"、"embedding"；健康检查等非推理路径可不调用本函数。
    """
    missing: list[str] = []
    for group in require:
        env_name, desc = _REQUIRED_KEY_GROUPS[group]
        key_value = {
            "chat": settings.deepseek_api_key,
            "embedding": settings.siliconflow_api_key,
        }[group]
        if not key_value:
            missing.append(f"  - {env_name}：{desc}")
    if missing:
        raise ConfigurationError(
            "缺少必要的环境变量配置，请在项目根目录的 .env 文件中填写以下变量"
            "（可参考 .env.example）：\n" + "\n".join(missing)
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例配置；测试中可用 ``get_settings.cache_clear()`` 后重新加载。"""
    return Settings()  # type: ignore[call-arg]


def validate_runtime(require: tuple[str, ...] = ("chat", "embedding")) -> Settings:
    """命令行/业务链路入口调用：加载配置并做启动期强校验。"""
    settings = get_settings()
    validate_settings(settings, require=require)
    return settings
