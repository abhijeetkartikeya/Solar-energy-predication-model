"""Orchestration service for solar forecast generation and persistence."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from app.api.schemas import ForecastPoint, PanelParameters
from app.services.timescaledb_service import timescaledb_service
from app.services.training_service import build_panel_config, train_location_model
from ml.data.fetch_solar_data import fetch_vedas_solar_data, merge_solar_sources
from ml.data.fetch_weather import fetch_weather_data
from ml.inference.predict_power import predict_power

LOGGER = logging.getLogger(__name__)


def _label_generation_type(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate rows as historical, realtime, or forecast for storage."""

    now = pd.Timestamp.now(tz="UTC").floor("15min")
    result = frame.copy()
    result["generation_type"] = "historical"
    result.loc[result["timestamp"] == now, "generation_type"] = "realtime"
    result.loc[result["timestamp"] > now, "generation_type"] = "forecast"
    return result


def generate_and_store_forecast(
    latitude: float,
    longitude: float,
    panel_overrides: PanelParameters | None = None,
    train_model: bool = False,
) -> dict[str, object]:
    """Fetch fresh data, generate forecasts, and upsert them into TimescaleDB."""

    panel = build_panel_config(panel_overrides)
    if train_model:
        train_location_model(latitude=latitude, longitude=longitude, panel_overrides=panel_overrides)

    weather_frame = fetch_weather_data(
        latitude=latitude,
        longitude=longitude,
        past_days=7,
        forecast_days=3,
        tilt=panel.tilt,
        azimuth=panel.azimuth,
    )
    vedas_frame = fetch_vedas_solar_data(
        latitude=latitude,
        longitude=longitude,
        start_time=weather_frame["timestamp"].min(),
        end_time=weather_frame["timestamp"].max(),
    )
    merged_frame = merge_solar_sources(weather_frame, vedas_frame)
    prediction_frame = predict_power(
        weather_frame=merged_frame,
        latitude=latitude,
        longitude=longitude,
        panel=panel,
    )
    prediction_frame = _label_generation_type(prediction_frame)
    rows_written = timescaledb_service.upsert_predictions(prediction_frame)

    latest_row = prediction_frame.sort_values("timestamp").iloc[-1]
    api_points = [
        ForecastPoint(
            timestamp=row["timestamp"].to_pydatetime(),
            predicted_power_kw=float(row["predicted_power_kw"]),
            baseline_power_kw=float(row["baseline_power_kw"]),
            model_power_kw=float(row["model_power_kw"]),
            global_horizontal_irradiance=float(row["global_horizontal_irradiance"]),
            global_tilted_irradiance=float(row["global_tilted_irradiance"]),
            temperature_2m=float(row["temperature_2m"]),
            wind_speed_10m=float(row["wind_speed_10m"]),
            cloud_cover=float(row["cloud_cover"]),
            humidity=float(row["humidity"]),
            pressure=float(row["pressure"]),
            generation_type=str(row["generation_type"]),
            solar_data_source=str(row["solar_data_source"]),
            model_name=str(row["model_name"]),
        )
        for _, row in prediction_frame.head(24).iterrows()
    ]

    return {
        "latitude": latitude,
        "longitude": longitude,
        "rows_written": rows_written,
        "historical_rows": int((prediction_frame["generation_type"] == "historical").sum()),
        "forecast_rows": int((prediction_frame["generation_type"] == "forecast").sum()),
        "latest_prediction_kw": float(latest_row["predicted_power_kw"]),
        "model_name": str(latest_row["model_name"]),
        "generated_at": datetime.now(timezone.utc),
        "points": api_points,
    }


def list_stored_predictions(latitude: float, longitude: float, hours: int) -> list[ForecastPoint]:
    """Read stored predictions and map them into API schemas."""

    rows = timescaledb_service.fetch_predictions(latitude=latitude, longitude=longitude, hours=hours)
    return [
        ForecastPoint(
            timestamp=row["timestamp"],
            predicted_power_kw=row["predicted_power_kw"],
            baseline_power_kw=row["baseline_power_kw"],
            model_power_kw=row["model_power_kw"],
            global_horizontal_irradiance=row["global_horizontal_irradiance"],
            global_tilted_irradiance=row["global_tilted_irradiance"],
            temperature_2m=row["temperature_2m"],
            wind_speed_10m=row["wind_speed_10m"],
            cloud_cover=row["cloud_cover"],
            humidity=row["humidity"],
            pressure=row["pressure"],
            generation_type=row["generation_type"],
            solar_data_source=row["solar_data_source"],
            model_name=row["model_name"],
        )
        for row in rows
    ]
