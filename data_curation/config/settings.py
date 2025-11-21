"""Application settings and configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    default_llm_provider: Literal["openai", "qwen", "local"] = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"

    # Qwen / Open-source (OpenAI compatible) configuration
    qwen_api_base: str = "http://localhost:11434/v1"
    qwen_api_key: str = "qwen-local-key"
    qwen_model: str = "qwen2.5:0.5b-instruct"

    # Local LLM Configuration (Gemma/Qwen via OpenAI-compatible server)
    local_api_base: str = "http://localhost:1234/v1"
    local_api_key: str = "local-key"
    local_model_name: str = "gemma-2b-it"
    use_local_llm: bool = False
    local_model_path: str = "./models"
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
    max_concurrent_requests: int = 5
    max_workers: int = 10

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # DocETL Configuration
    docetl_validation_retries: int = 3
    docetl_timeout: int = 300
    llm_request_timeout: float = 60.0
    qwen_request_timeout: float = 18000.0  # Qwen için daha uzun timeout (5 dakika)

    model_config = SettingsConfigDict(
        env_file=(".env", "config/.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
