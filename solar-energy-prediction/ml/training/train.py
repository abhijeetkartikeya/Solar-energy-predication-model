"""Backward-compatible wrapper for the renamed training module."""

from ml.training.train_model import build_model_key, train_model

__all__ = ["build_model_key", "train_model"]
