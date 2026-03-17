"""
==============================================================================
Model Evaluation Module
==============================================================================
Computes regression metrics for model evaluation:
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R²   (Coefficient of Determination)

==============================================================================
"""

import numpy as np
from ml.utils.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluate model predictions against ground truth.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values

    Returns:
        Dictionary with 'mae', 'rmse', 'r2' (all rounded to 6 decimal places)
    """
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 6),
        "r2": round(float(r2_score(y_true, y_pred)), 6),
    }
