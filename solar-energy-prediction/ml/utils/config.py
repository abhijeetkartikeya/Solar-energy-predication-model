"""Shared application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for the API, ML pipeline, and deployment."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    request_timeout_seconds: int = Field(default=45)
    forecast_frequency: str = Field(default="15min")
    scheduler_interval_minutes: int = Field(default=15)
    enable_internal_scheduler: bool = Field(default=True)

    postgres_host: str = Field(default="timescaledb")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="solar")
    postgres_user: str = Field(default="solar")
    postgres_password: str = Field(default="solar123")

    default_latitude: float = Field(default=22.5726)
    default_longitude: float = Field(default=88.3639)

    default_panel_area_m2: float = Field(default=16.0)
    default_panel_efficiency: float = Field(default=0.20)
    default_temperature_coefficient: float = Field(default=0.004)
    default_noct_c: float = Field(default=45.0)
    default_inverter_efficiency: float = Field(default=0.96)
    default_capacity_kw: float = Field(default=3.5)
    default_panel_tilt: float = Field(default=22.0)
    default_panel_azimuth: float = Field(default=180.0)

    model_dir: str = Field(default=str(Path("artifacts") / "models"))

    vedas_api_base_url: str = Field(default="")
    vedas_api_key: str = Field(default="")

    @property
    def postgres_dsn(self) -> str:
        """Return a psycopg-compatible DSN string."""

        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
