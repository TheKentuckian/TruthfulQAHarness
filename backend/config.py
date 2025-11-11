"""Configuration management for the TruthfulQA Harness."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: str = ""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM Configuration
    default_model: str = "claude-sonnet-4-5-20250929"
    default_max_tokens: int = 1024
    default_temperature: float = 1.0

    # Dataset Configuration
    truthfulqa_sample_size: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
