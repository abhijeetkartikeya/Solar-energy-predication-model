"""Feature engineering for solar power estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "global_horizontal_irradiance",
    "global_tilted_irradiance",
    "temperature_2m",
    "wind_speed_10m",
    "cloud_cover",
    "humidity",
    "pressure",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_daylight",
    "cell_temperature_c",
    "temperature_loss_factor",
    "clearness_proxy",
]
TARGET_COLUMN = "target_power_kw"


@dataclass(frozen=True)
class PanelConfig:
    """Physical properties used by the baseline power model."""

    area_m2: float
    efficiency: float
    temperature_coefficient: float
    nominal_operating_cell_temp_c: float
    inverter_efficiency: float
    capacity_kw: float
    tilt: float
    azimuth: float


def align_to_15_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample an arbitrary time series to a uniform 15-minute grid."""

    if frame.empty:
        return frame.copy()

    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result = result.sort_values("timestamp").set_index("timestamp")
    result = result.resample("15min").mean(numeric_only=True)
    result = result.interpolate(method="time").ffill().bfill().reset_index()
    return result


def compute_baseline_power(frame: pd.DataFrame, panel: PanelConfig) -> pd.DataFrame:
    """Estimate solar output with a physics-inspired baseline formula."""

    result = frame.copy()
    irradiance = result["global_tilted_irradiance"].clip(lower=0)
    ambient_temp = result["temperature_2m"].astype(float)
    wind_speed = result["wind_speed_10m"].fillna(0).astype(float)

    cell_temperature = ambient_temp + (
        (panel.nominal_operating_cell_temp_c - 20.0) / 800.0
    ) * irradiance - (0.6 * wind_speed)
    temperature_delta = (cell_temperature - 25.0).clip(lower=0)
    temperature_loss_factor = 1.0 - (panel.temperature_coefficient * temperature_delta)
    temperature_loss_factor = temperature_loss_factor.clip(lower=0.7, upper=1.05)

    dc_power_kw = (irradiance * panel.area_m2 * panel.efficiency) / 1000.0
    ac_power_kw = dc_power_kw * temperature_loss_factor * panel.inverter_efficiency
    capped_power_kw = ac_power_kw.clip(lower=0, upper=panel.capacity_kw)

    result["cell_temperature_c"] = cell_temperature
    result["temperature_loss_factor"] = temperature_loss_factor
    result["baseline_power_kw"] = capped_power_kw.round(4)
    return result


def create_features(frame: pd.DataFrame, panel: PanelConfig) -> pd.DataFrame:
    """Create model-ready features while preserving the forecasting context."""

    result = align_to_15_minutes(frame)
    result = compute_baseline_power(result, panel)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    hour_fraction = result["timestamp"].dt.hour + (result["timestamp"].dt.minute / 60.0)
    day_of_year = result["timestamp"].dt.dayofyear

    result["hour_sin"] = np.sin((2 * np.pi * hour_fraction) / 24.0)
    result["hour_cos"] = np.cos((2 * np.pi * hour_fraction) / 24.0)
    result["day_of_year_sin"] = np.sin((2 * np.pi * day_of_year) / 365.25)
    result["day_of_year_cos"] = np.cos((2 * np.pi * day_of_year) / 365.25)
    result["is_daylight"] = (
        result["global_horizontal_irradiance"].fillna(0).astype(float) > 20
    ).astype(int)
    result["clearness_proxy"] = 1.0 - (result["cloud_cover"].fillna(0).astype(float) / 100.0)
    result["target_power_kw"] = result.get("actual_power_kw", result["baseline_power_kw"])

    numeric_columns = [
        "global_horizontal_irradiance",
        "global_tilted_irradiance",
        "temperature_2m",
        "wind_speed_10m",
        "cloud_cover",
        "humidity",
        "pressure",
        "cell_temperature_c",
        "temperature_loss_factor",
        "baseline_power_kw",
        "target_power_kw",
        "clearness_proxy",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    LOGGER.info("Engineered %s rows with %s features", len(result), len(FEATURE_COLUMNS))
    return result
