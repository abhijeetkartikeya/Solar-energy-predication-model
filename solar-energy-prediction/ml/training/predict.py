"""Backward-compatible wrapper for the new inference module."""

from ml.inference.predict_power import load_model, load_model_metadata, predict_power

__all__ = ["load_model", "load_model_metadata", "predict_power"]
