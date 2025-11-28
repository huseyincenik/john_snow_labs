"""Application settings and configuration."""

from __future__ import annotations

import os
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration (OpenRouter backbone for OpenAI + Qwen)
    default_llm_provider: Literal["openai", "qwen"] = "openai"
    openrouter_api_key: str = ""
    openrouter_model_openai: str = "openai/gpt-4o-mini"
    openrouter_model_qwen: str = (
        "openrouter/qwen/qwen3-8b"  # Qwen 3 - En hızlı model (8B parameters, 8GB)
        # Not: LiteLLM model adını tanıması için "openrouter/" prefix'i gereklidir
        # Alternatif modeller (hız/performans dengesi):
        # - "openrouter/qwen/qwen3-1.7b" (hızlı, 1.7B parameters)
        # - "openrouter/qwen/qwen3-4b" (dengeli, 4B parameters)
        # - "openrouter/qwen/qwen3-8b" (iyi performans, 8B parameters)
        # Daha iyi performans (daha yavaş):
        # - "openrouter/qwen/qwen3-32b" (32B parameters)
        # - "openrouter/qwen/qwen3-next-80b-a3b-instruct" (MoE, 3B aktif parametre, uzun bağlamlarda 10x daha verimli)
        # - "openrouter/qwen/qwen3-235b-a22b" (MoE, 22B aktif parametre)
    )
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    openrouter_referer: str = "https://local.data-curation"
    openrouter_app_title: str = "Data Curation Service"
    llm_retry_attempts: int = 3

    # Database Configuration
    use_postgres: bool = False
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "data_curation"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Storage Configuration
    storage_type: Literal["json", "postgres"] = "json"
    output_dir: str = "./data/output"
    log_dir: str = "./logs"

    # Concurrency Settings
    # Optimized for high-performance parallel processing
    max_concurrent_requests: int = 30  # Increased from 20 to 30 for better LLM API throughput
    max_workers: int = 150  # Increased from 100 to 150 for better CPU utilization
    max_parallel_patients: int = (
        12  # Increased from 8 to 12 for faster patient processing (auto-scaled based on CPU count)
    )

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # DocETL Configuration
    docetl_validation_retries: int = 2  # Reduced from 3 to 2 for faster failure detection
    docetl_timeout: int = 300
    llm_request_timeout: float = (
        45.0  # Reduced from 60.0 to 45.0 for faster timeout (Qwen models are faster)
    )
    docetl_pipeline_retries: int = 2  # Reduced from 3 to 2 for faster failure recovery
    docetl_retry_backoff_seconds: float = 3.0  # Reduced from 5.0 to 3.0 for faster retries
    docetl_retry_backoff_max_seconds: float = 10.0  # Increased from 5.0 to 10.0 for better backoff
    docetl_max_threads: int = (
        200  # Increased from 128 to 200 for better parallel document processing
    )

    model_config = SettingsConfigDict(
        env_file=(".env", "config/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()


def _ensure_openrouter_env() -> None:
    """DocETL calls Litellm directly, so mirror OpenRouter creds into OpenAI env vars."""
    if not settings.openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Add it to config/.env (or export it) before running the service."
        )

    os.environ["OPENAI_API_KEY"] = settings.openrouter_api_key
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
    os.environ["OPENAI_API_BASE"] = settings.openrouter_api_base
    os.environ.setdefault("HTTP_REFERER", settings.openrouter_referer)
    os.environ.setdefault("X_TITLE", settings.openrouter_app_title)


_ensure_openrouter_env()
