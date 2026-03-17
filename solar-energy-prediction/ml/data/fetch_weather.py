"""Weather ingestion helpers for the solar forecasting pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ml.utils.config import settings

LOGGER = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_COLUMNS = {
    "shortwave_radiation": "global_horizontal_irradiance",
    "global_tilted_irradiance": "global_tilted_irradiance",
    "temperature_2m": "temperature_2m",
    "wind_speed_10m": "wind_speed_10m",
    "cloud_cover": "cloud_cover",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
}


def _build_retrying_session() -> Session:
    """Create a resilient HTTP session for external API calls."""

    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _raise_for_api_error(response: Response) -> None:
    """Raise a descriptive exception for unsuccessful Open-Meteo responses."""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        details = response.text[:500]
        raise requests.HTTPError(
            f"Open-Meteo request failed with status {response.status_code}: {details}"
        ) from exc


def _normalise_weather_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert Open-Meteo minutely data into the project schema."""

    minutely_data = payload.get("minutely_15", {})
    timestamps = pd.to_datetime(minutely_data.get("time", []), utc=True)
    frame = pd.DataFrame({"timestamp": timestamps})

    for api_name, internal_name in WEATHER_COLUMNS.items():
        frame[internal_name] = pd.to_numeric(
            minutely_data.get(api_name, []),
            errors="coerce",
        )

    if frame.empty:
        return pd.DataFrame(columns=["timestamp", *WEATHER_COLUMNS.values()])

    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    frame = frame.set_index("timestamp").resample(settings.forecast_frequency).mean()
    frame = frame.interpolate(method="time").ffill().bfill().reset_index()
    frame["latitude"] = payload.get("latitude")
    frame["longitude"] = payload.get("longitude")
    frame["data_source"] = "open_meteo"
    return frame


def fetch_weather_data(
    latitude: float,
    longitude: float,
    past_days: int = 7,
    forecast_days: int = 3,
    tilt: float | None = None,
    azimuth: float | None = None,
) -> pd.DataFrame:
    """Fetch a unified 15-minute weather time series from Open-Meteo.

    The response contains the required forecasting features covering a rolling
    historical window plus the requested forecast horizon.
    """

    tilt_value = settings.default_panel_tilt if tilt is None else tilt
    azimuth_value = settings.default_panel_azimuth if azimuth is None else azimuth

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "UTC",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "tilt": tilt_value,
        "azimuth": azimuth_value,
        "minutely_15": ",".join(WEATHER_COLUMNS.keys()),
    }

    LOGGER.info(
        "Fetching Open-Meteo weather for lat=%s lon=%s past_days=%s forecast_days=%s",
        latitude,
        longitude,
        past_days,
        forecast_days,
    )

    session = _build_retrying_session()
    response = session.get(
        OPEN_METEO_FORECAST_URL,
        params=params,
        timeout=settings.request_timeout_seconds,
    )
    _raise_for_api_error(response)

    frame = _normalise_weather_frame(response.json())
    LOGGER.info("Fetched %s weather rows from Open-Meteo", len(frame))
    return frame
