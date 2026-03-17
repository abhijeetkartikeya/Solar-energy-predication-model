"""Train an optional ML model that complements the formula baseline."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from ml.features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN
from ml.training.evaluate import evaluate_model
from ml.utils.config import settings

LOGGER = logging.getLogger(__name__)

MODEL_CANDIDATES = {
    "gradient_boosting": GradientBoostingRegressor(random_state=42),
    "hist_gradient_boosting": HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        l2_regularization=0.05,
    ),
    "random_forest": RandomForestRegressor(
        random_state=42,
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=-1,
    ),
}


def build_model_key(latitude: float, longitude: float) -> str:
    """Create a stable identifier for location-specific model artifacts."""

    digest = hashlib.sha256(f"{latitude:.6f}:{longitude:.6f}".encode("utf-8")).hexdigest()
    return digest[:16]


def train_model(
    feature_frame: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> dict[str, object]:
    """Train a gradient boosting regressor on the available power targets."""

    if len(feature_frame) < 96:
        raise ValueError("At least 96 samples are required to train the ML model.")

    model_key = build_model_key(latitude, longitude)
    model_dir = Path(settings.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_key}.joblib"
    metadata_path = model_dir / f"{model_key}.json"

    train_cutoff = max(int(len(feature_frame) * 0.8), 1)
    train_frame = feature_frame.iloc[:train_cutoff]
    test_frame = feature_frame.iloc[train_cutoff:]
    if test_frame.empty:
        raise ValueError("The evaluation split is empty; increase the training window.")

    best_name, model, cv_metrics = _select_best_model(train_frame)
    model.fit(train_frame[FEATURE_COLUMNS], train_frame[TARGET_COLUMN])

    predictions = model.predict(test_frame[FEATURE_COLUMNS])
    metrics = evaluate_model(test_frame[TARGET_COLUMN].to_numpy(), predictions)

    joblib.dump(model, model_path)
    metadata = {
        "model_key": model_key,
        "latitude": latitude,
        "longitude": longitude,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "cross_validation": cv_metrics,
        "selected_model": best_name,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "target_source": (
            "actual_power_kw" if "actual_power_kw" in feature_frame.columns else "baseline_power_kw"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Saved ML model for lat=%s lon=%s at %s", latitude, longitude, model_path)
    return {
        "model_key": model_key,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "metrics": metrics,
        "selected_model": best_name,
        "cross_validation": cv_metrics,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
    }


def _select_best_model(train_frame: pd.DataFrame) -> tuple[str, object, dict[str, dict[str, float]]]:
    """Evaluate candidate regressors with time-series cross-validation."""

    splitter = TimeSeriesSplit(n_splits=3)
    X = train_frame[FEATURE_COLUMNS]
    y = train_frame[TARGET_COLUMN]
    cv_results: dict[str, dict[str, float]] = {}
    best_name = ""
    best_model = None
    best_mae = float("inf")

    for name, candidate in MODEL_CANDIDATES.items():
        fold_metrics = []
        for train_index, validation_index in splitter.split(X):
            fold_model = clone(candidate)
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_validation = X.iloc[validation_index]
            y_validation = y.iloc[validation_index]

            fold_model.fit(X_train, y_train)
            predictions = fold_model.predict(X_validation)
            fold_metrics.append(evaluate_model(y_validation.to_numpy(), predictions))

        mae = sum(metric["mae"] for metric in fold_metrics) / len(fold_metrics)
        rmse = sum(metric["rmse"] for metric in fold_metrics) / len(fold_metrics)
        r2 = sum(metric["r2"] for metric in fold_metrics) / len(fold_metrics)
        cv_results[name] = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
        }

        if mae < best_mae:
            best_mae = mae
            best_name = name
            best_model = clone(candidate)

    LOGGER.info("Selected model '%s' using time-series cross-validation", best_name)
    return best_name, best_model, cv_results
