"""Backward-compatible wrapper for the renamed feature module."""

from ml.features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, PanelConfig, create_features

__all__ = ["FEATURE_COLUMNS", "TARGET_COLUMN", "PanelConfig", "create_features"]
