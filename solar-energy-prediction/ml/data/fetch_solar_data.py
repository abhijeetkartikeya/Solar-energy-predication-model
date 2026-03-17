"""Optional VEDAS solar data integration helpers."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from ml.utils.config import settings

LOGGER = logging.getLogger(__name__)


def fetch_vedas_solar_data(
    latitude: float,
    longitude: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch VEDAS solar irradiance data when the API endpoint is configured.

    The public VEDAS API centre requires registration and an API key. This
    client is therefore opt-in through environment variables and returns an
    empty frame when no VEDAS endpoint has been configured.
    """

    if not settings.vedas_api_base_url:
        LOGGER.info("VEDAS endpoint not configured; skipping satellite solar fetch")
        return pd.DataFrame()

    headers: dict[str, str] = {}
    if settings.vedas_api_key:
        headers["x-api-key"] = settings.vedas_api_key

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start": start_time.isoformat(),
        "end": end_time.isoformat(),
        "interval": settings.forecast_frequency,
    }

    LOGGER.info("Fetching VEDAS solar data for lat=%s lon=%s", latitude, longitude)
    try:
        response = requests.get(
            settings.vedas_api_base_url,
            params=params,
            headers=headers,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        LOGGER.warning("VEDAS solar fetch failed: %s", exc)
        return pd.DataFrame()

    payload = response.json()
    records = payload.get("data", payload if isinstance(payload, list) else [])
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    rename_map = {
        "time": "timestamp",
        "timestamp": "timestamp",
        "ghi": "vedas_global_horizontal_irradiance",
        "global_horizontal_irradiance": "vedas_global_horizontal_irradiance",
        "gti": "vedas_global_tilted_irradiance",
        "global_tilted_irradiance": "vedas_global_tilted_irradiance",
    }
    frame = frame.rename(columns=rename_map)
    if "timestamp" not in frame.columns:
        LOGGER.warning("VEDAS response missing timestamp column; ignoring payload")
        return pd.DataFrame()

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for column in (
        "vedas_global_horizontal_irradiance",
        "vedas_global_tilted_irradiance",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])


def merge_solar_sources(weather_frame: pd.DataFrame, vedas_frame: pd.DataFrame) -> pd.DataFrame:
    """Merge Open-Meteo weather data with optional VEDAS irradiance values."""

    frame = weather_frame.copy()
    if vedas_frame.empty:
        frame["solar_data_source"] = "open_meteo_only"
        return frame

    frame = frame.merge(vedas_frame, on="timestamp", how="left")
    if "vedas_global_horizontal_irradiance" in frame.columns:
        frame["global_horizontal_irradiance"] = frame[
            "vedas_global_horizontal_irradiance"
        ].combine_first(frame["global_horizontal_irradiance"])
    if "vedas_global_tilted_irradiance" in frame.columns:
        frame["global_tilted_irradiance"] = frame[
            "vedas_global_tilted_irradiance"
        ].combine_first(frame["global_tilted_irradiance"])

    frame["solar_data_source"] = frame.apply(
        lambda row: "vedas+open_meteo"
        if pd.notna(row.get("vedas_global_horizontal_irradiance"))
        or pd.notna(row.get("vedas_global_tilted_irradiance"))
        else "open_meteo_only",
        axis=1,
    )
    return frame
