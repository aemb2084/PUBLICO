from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Config(BaseSettings):
    """Application settings. Values can be overridden by environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


config = Config()

PROJECT_ROOT : Path = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"