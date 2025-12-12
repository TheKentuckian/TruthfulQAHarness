"""Configuration management for the TruthfulQA Harness."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    anthropic_api_key: str = ""

    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "local-model"
    lm_studio_api_key: str = "not-needed"

    default_model: str = "claude-sonnet-4-5-20250929"
    default_max_tokens: int = 1024
    default_temperature: float = 1.0

    truthfulqa_sample_size: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
