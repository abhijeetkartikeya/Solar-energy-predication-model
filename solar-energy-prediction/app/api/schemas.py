"""Pydantic schemas for the solar forecasting API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PanelParameters(BaseModel):
    """Optional panel configuration overrides for a forecast request."""

    area_m2: float = Field(default=16.0, gt=0)
    efficiency: float = Field(default=0.20, gt=0, le=1)
    temperature_coefficient: float = Field(default=0.004, gt=0, le=0.02)
    nominal_operating_cell_temp_c: float = Field(default=45.0, gt=20, le=80)
    inverter_efficiency: float = Field(default=0.96, gt=0, le=1)
    capacity_kw: float = Field(default=3.5, gt=0)
    tilt: float = Field(default=22.0, ge=0, le=90)
    azimuth: float = Field(default=180.0, ge=0, le=360)


class PredictRequest(BaseModel):
    """Request body for on-demand forecasting."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    panel: PanelParameters | None = None
    train_model: bool = Field(default=False)


class TrainRequest(BaseModel):
    """Request body for training a location-specific ML model."""

    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    panel: PanelParameters | None = None


class ForecastPoint(BaseModel):
    """A single forecast point returned by the API."""

    timestamp: datetime
    predicted_power_kw: float
    baseline_power_kw: float
    model_power_kw: float
    global_horizontal_irradiance: float
    global_tilted_irradiance: float
    temperature_2m: float
    wind_speed_10m: float
    cloud_cover: float
    humidity: float
    pressure: float
    generation_type: str
    solar_data_source: str
    model_name: str


class PredictResponse(BaseModel):
    """Summary returned after a forecast run."""

    latitude: float
    longitude: float
    rows_written: int
    historical_rows: int
    forecast_rows: int
    latest_prediction_kw: float
    model_name: str
    generated_at: datetime
    points: list[ForecastPoint]


class TrainResponse(BaseModel):
    """Training result payload."""

    latitude: float
    longitude: float
    model_path: str
    metrics: dict[str, float]
    selected_model: str
    cross_validation: dict[str, dict[str, float]]
    train_rows: int
    test_rows: int


class StoredForecastResponse(BaseModel):
    """Rows fetched back from TimescaleDB."""

    latitude: float
    longitude: float
    rows: list[ForecastPoint]


class HealthResponse(BaseModel):
    """Health check output."""

    status: str
    database_ok: bool
    scheduler_interval_minutes: int
