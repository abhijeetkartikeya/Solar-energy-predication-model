"""Training service for optional location-specific ML models."""

from __future__ import annotations

import logging

from app.api.schemas import PanelParameters
from ml.data.fetch_solar_data import fetch_vedas_solar_data, merge_solar_sources
from ml.data.fetch_weather import fetch_weather_data
from ml.features.feature_engineering import PanelConfig, create_features
from ml.training.train_model import train_model
from ml.utils.config import settings

LOGGER = logging.getLogger(__name__)


def build_panel_config(panel_overrides: PanelParameters | None = None) -> PanelConfig:
    """Combine default panel settings with any request-level overrides."""

    panel = panel_overrides or PanelParameters()
    return PanelConfig(
        area_m2=panel.area_m2,
        efficiency=panel.efficiency,
        temperature_coefficient=panel.temperature_coefficient,
        nominal_operating_cell_temp_c=panel.nominal_operating_cell_temp_c,
        inverter_efficiency=panel.inverter_efficiency,
        capacity_kw=panel.capacity_kw,
        tilt=panel.tilt,
        azimuth=panel.azimuth,
    )


def train_location_model(
    latitude: float,
    longitude: float,
    panel_overrides: PanelParameters | None = None,
) -> dict[str, object]:
    """Train a location-specific model using the latest seven days of weather."""

    panel = build_panel_config(panel_overrides)
    weather_frame = fetch_weather_data(
        latitude=latitude,
        longitude=longitude,
        past_days=7,
        forecast_days=0,
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
    feature_frame = create_features(merged_frame, panel)
    result = train_model(feature_frame=feature_frame, latitude=latitude, longitude=longitude)

    LOGGER.info("Completed training for lat=%s lon=%s", latitude, longitude)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "model_path": result["model_path"],
        "metrics": result["metrics"],
        "selected_model": result["selected_model"],
        "cross_validation": result["cross_validation"],
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
    }
