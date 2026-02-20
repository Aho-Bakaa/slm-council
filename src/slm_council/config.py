"""Centralised configuration loaded from environment / .env file."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Root application settings – populated from env vars / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ───────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8080, validation_alias="API_PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_agent_io: bool = Field(default=False, validation_alias="LOG_AGENT_IO")
    log_max_chars: int = Field(default=3000, validation_alias="LOG_MAX_CHARS")

    # ── Runtime / loop tuning ────────────────────────────────────────
    request_timeout_secs: float = Field(default=120.0, validation_alias="REQUEST_TIMEOUT_SECS")
    max_retries: int = Field(default=2, validation_alias="MAX_RETRIES")
    retry_backoff_secs: float = Field(default=1.5, validation_alias="RETRY_BACKOFF_SECS")
    max_refinement_passes: int = Field(default=3, validation_alias="MAX_REFINEMENT_PASSES")
    consensus_threshold: float = Field(default=0.8, validation_alias="CONSENSUS_THRESHOLD")

    # ── Rate limiting & caching ──────────────────────────────────────
    max_concurrent_agent_calls: int = Field(default=4, validation_alias="MAX_CONCURRENT_AGENT_CALLS")
    enable_request_cache: bool = Field(default=True, validation_alias="ENABLE_REQUEST_CACHE")
    cache_ttl_secs: int = Field(default=300, validation_alias="CACHE_TTL_SECS")

    # ── Agent memory ─────────────────────────────────────────────────
    agent_memory_window: int = Field(default=3, validation_alias="AGENT_MEMORY_WINDOW")

    # ── Synthesis token limits ───────────────────────────────────────
    synthesis_max_code_chars: int = Field(default=8000, validation_alias="SYNTHESIS_MAX_CODE_CHARS")
    synthesis_max_report_chars: int = Field(default=4000, validation_alias="SYNTHESIS_MAX_REPORT_CHARS")

    # ── Orchestrator (The One) ───────────────────────────────────────
    orchestrator_endpoint: str = Field(default="", validation_alias="ORCHESTRATOR_ENDPOINT")
    orchestrator_api_key: str = Field(default="", validation_alias="ORCHESTRATOR_API_KEY")
    orchestrator_model: str = Field(default="qwen/qwen3-235b-a22b-thinking-2507", validation_alias="ORCHESTRATOR_MODEL")

    # ── Specialist agents (The Many) ─────────────────────────────────
    researcher_endpoint: str = Field(default="http://localhost:8001/v1", validation_alias="RESEARCHER_ENDPOINT")
    researcher_api_key: str = Field(default="", validation_alias="RESEARCHER_API_KEY")
    researcher_model: str = Field(default="google/gemma-3-4b-it:free", validation_alias="RESEARCHER_MODEL")

    generator_endpoint: str = Field(default="http://localhost:8002/v1", validation_alias="GENERATOR_ENDPOINT")
    generator_api_key: str = Field(default="", validation_alias="GENERATOR_API_KEY")
    generator_model: str = Field(default="qwen/qwen3-coder:free", validation_alias="GENERATOR_MODEL")

    debugger_endpoint: str = Field(default="http://localhost:8003/v1", validation_alias="DEBUGGER_ENDPOINT")
    debugger_api_key: str = Field(default="", validation_alias="DEBUGGER_API_KEY")
    debugger_model: str = Field(default="deepseek/deepseek-r1-distill-qwen-32b", validation_alias="DEBUGGER_MODEL")

    tester_endpoint: str = Field(default="http://localhost:8004/v1", validation_alias="TESTER_ENDPOINT")
    tester_api_key: str = Field(default="", validation_alias="TESTER_API_KEY")
    tester_model: str = Field(default="phi-4-mini", validation_alias="TESTER_MODEL")

    # ── New specialist agents ────────────────────────────────────────
    planner_endpoint: str = Field(default="", validation_alias="PLANNER_ENDPOINT")
    planner_api_key: str = Field(default="", validation_alias="PLANNER_API_KEY")
    planner_model: str = Field(default="", validation_alias="PLANNER_MODEL")

    reviewer_endpoint: str = Field(default="", validation_alias="REVIEWER_ENDPOINT")
    reviewer_api_key: str = Field(default="", validation_alias="REVIEWER_API_KEY")
    reviewer_model: str = Field(default="", validation_alias="REVIEWER_MODEL")

    optimizer_endpoint: str = Field(default="", validation_alias="OPTIMIZER_ENDPOINT")
    optimizer_api_key: str = Field(default="", validation_alias="OPTIMIZER_API_KEY")
    optimizer_model: str = Field(default="", validation_alias="OPTIMIZER_MODEL")

    refactorer_endpoint: str = Field(default="", validation_alias="REFACTORER_ENDPOINT")
    refactorer_api_key: str = Field(default="", validation_alias="REFACTORER_API_KEY")
    refactorer_model: str = Field(default="", validation_alias="REFACTORER_MODEL")

    # ── Compatibility aliases for older call-sites ───────────────────
    @property
    def orchestrator_model_name(self) -> str:
        return self.orchestrator_model

    @property
    def researcher_model_name(self) -> str:
        return self.researcher_model

    @property
    def generator_model_name(self) -> str:
        return self.generator_model

    @property
    def debugger_model_name(self) -> str:
        return self.debugger_model

    @property
    def tester_model_name(self) -> str:
        return self.tester_model


settings = Settings()