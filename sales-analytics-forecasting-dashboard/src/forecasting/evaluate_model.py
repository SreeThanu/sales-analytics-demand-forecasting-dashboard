"""Model evaluation metrics for the ARIMA forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.helpers import load_config, resolve_path, ensure_dir


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return MAE, RMSE, and MAPE for two aligned arrays."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    non_zero = actual != 0
    mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2)}


def evaluate_forecast(
    actual_series: pd.Series,
    fitted_model,
    test_days: int = 90,
    config: dict | None = None,
) -> dict:
    """
    Hold out the last *test_days* from actual_series, generate in-sample
    forecasts for that window, and compute evaluation metrics.
    """
    train = actual_series.iloc[:-test_days]
    test = actual_series.iloc[-test_days:]

    pred = fitted_model.predict(start=len(train), end=len(actual_series) - 1)
    metrics = compute_metrics(test.values, pred.values)

    print(f"[evaluate] Test window = {test_days} days")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def save_metrics(metrics: dict, config: dict | None = None) -> str:
    """Save evaluation metrics to CSV. Returns the file path."""
    config = config or load_config()
    out_dir = resolve_path(config["outputs"]["metrics_dir"])
    ensure_dir(out_dir)
    path = out_dir / "model_metrics.csv"
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"[evaluate] Saved metrics → {path}")
    return str(path)
