"""Typed configuration contract with automatic live API vs. offline Mock detection."""

from __future__ import annotations

from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PLACEHOLDER_KEYS = frozenset(
    {"", "your_api_key_here", "sk-your-api-key", "none", "null", "mock"}
)


class StewardSettings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="your_api_key_here", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.deepseek.com/v1", alias="OPENAI_BASE_URL")
    local_slm_base_url: str = Field(default="", alias="LOCAL_SLM_BASE_URL")
    local_slm_api_key: str = Field(default="ollama", alias="LOCAL_SLM_API_KEY")

    llm_model: str = Field(default="deepseek-flash", alias="LLM_MODEL")
    slm_model: str = Field(default="deepseek-flash", alias="SLM_MODEL")
    local_privacy_model: str = Field(default="deepseek-flash", alias="LOCAL_PRIVACY_MODEL")
    vlm_model: str = Field(default="deepseek-flash", alias="VLM_MODEL")

    temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="TEMPERATURE")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    request_timeout_seconds: float = Field(default=60.0, gt=0.0, alias="REQUEST_TIMEOUT_SECONDS")

    memory_db_path: Path = Field(
        default=Path("runtime_data/memory/steward_memory.sqlite3"), alias="MEMORY_DB_PATH"
    )
    embedding_dim: int = Field(default=256, ge=64, le=2048, alias="EMBEDDING_DIM")
    memory_base_half_life_hours: float = Field(
        default=168.0, gt=0.0, alias="MEMORY_BASE_HALF_LIFE_HOURS"
    )
    memory_ppr_alpha: float = Field(default=0.85, gt=0.0, lt=1.0, alias="MEMORY_PPR_ALPHA")
    memory_abstention_threshold: float = Field(
        default=0.18, ge=0.0, le=1.0, alias="MEMORY_ABSTENTION_THRESHOLD"
    )

    router_complexity_threshold: float = Field(
        default=0.45, ge=0.05, le=0.95, alias="ROUTER_COMPLEXITY_THRESHOLD"
    )
    router_tool_budget: int = Field(default=6, ge=2, le=16, alias="ROUTER_TOOL_BUDGET")
    auto_reimburse_limit_cny: float = Field(
        default=1000.0, ge=0.0, alias="AUTO_REIMBURSE_LIMIT_CNY"
    )
    agent_max_reflect_retries: int = Field(
        default=2, ge=0, le=4, alias="AGENT_MAX_REFLECT_RETRIES"
    )

    trace_log_path: Path = Field(default=Path("logs/trace_events.jsonl"), alias="TRACE_LOG_PATH")
    sandbox_workspace_dir: Path = Field(
        default=Path("runtime_data/workspace"), alias="SANDBOX_WORKSPACE_DIR"
    )
    artifact_cache_dir: Path = Field(default=Path("artifacts_cache"), alias="ARTIFACT_CACHE_DIR")

    @field_validator("openai_base_url")
    @classmethod
    def _clean_url(cls, val: str) -> str:
        cleaned = val.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError(f"Invalid OPENAI_BASE_URL: {val!r}")
        return cleaned

    @property
    def use_mock_llm(self) -> bool:
        return self.openai_api_key.strip().lower() in _PLACEHOLDER_KEYS
