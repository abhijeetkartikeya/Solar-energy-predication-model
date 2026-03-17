"""Inference helpers for baseline and ML-assisted solar forecasts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features.feature_engineering import FEATURE_COLUMNS, PanelConfig, create_features
from ml.training.train_model import build_model_key
from ml.utils.config import settings

LOGGER = logging.getLogger(__name__)


def load_model(latitude: float, longitude: float):
    """Load the location-specific model if one has been trained."""

    model_key = build_model_key(latitude, longitude)
    model_path = Path(settings.model_dir) / f"{model_key}.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def load_model_metadata(latitude: float, longitude: float) -> dict[str, object] | None:
    """Load saved metadata for a trained model if present."""

    model_key = build_model_key(latitude, longitude)
    metadata_path = Path(settings.model_dir) / f"{model_key}.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def predict_power(
    weather_frame: pd.DataFrame,
    latitude: float,
    longitude: float,
    panel: PanelConfig,
) -> pd.DataFrame:
    """Generate baseline and optional ML-enhanced power predictions."""

    features = create_features(weather_frame, panel)
    if "solar_data_source" in weather_frame.columns:
        source_frame = weather_frame[["timestamp", "solar_data_source"]].copy()
        source_frame["timestamp"] = pd.to_datetime(source_frame["timestamp"], utc=True)
        features = features.merge(source_frame, on="timestamp", how="left")
        features["solar_data_source"] = features["solar_data_source"].ffill().bfill()
    else:
        features["solar_data_source"] = "open_meteo_only"

    features["latitude"] = latitude
    features["longitude"] = longitude

    model = load_model(latitude, longitude)
    if model is None:
        features["model_power_kw"] = features["baseline_power_kw"]
        features["predicted_power_kw"] = features["baseline_power_kw"]
        features["model_name"] = "baseline_formula"
        return features

    raw_prediction = model.predict(features[FEATURE_COLUMNS])
    corrected_prediction = np.clip(raw_prediction, 0, panel.capacity_kw)
    features["model_power_kw"] = corrected_prediction.round(4)
    features["predicted_power_kw"] = (
        0.6 * features["baseline_power_kw"] + 0.4 * features["model_power_kw"]
    ).round(4)
    features["model_name"] = "baseline_plus_gradient_boosting"
    LOGGER.info("Generated ML-assisted predictions for lat=%s lon=%s", latitude, longitude)
    return features
