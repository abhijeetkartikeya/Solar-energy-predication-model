"""
==============================================================================
Shared Metric Helpers
==============================================================================
Pure NumPy implementations of common regression metrics.
No scikit-learn dependency needed for these basic calculations.

==============================================================================
"""

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).
    Measures average magnitude of errors without considering direction.

    MAE = (1/n) * Σ|y_true - y_pred|
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (RMSE).
    Penalizes larger errors more heavily than MAE.

    RMSE = √((1/n) * Σ(y_true - y_pred)²)
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² Score (Coefficient of Determination).
    1.0 = perfect prediction, 0.0 = as good as predicting the mean.
    Can be negative if model is worse than mean.

    R² = 1 - (SS_res / SS_tot)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0  # Avoid division by zero (constant target)

    return float(1.0 - (ss_res / ss_tot))
