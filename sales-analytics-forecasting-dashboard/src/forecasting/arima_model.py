"""ARIMA demand forecasting model."""

from __future__ import annotations

import warnings

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.utils.helpers import load_config, resolve_path, ensure_dir


def aggregate_daily_sales(df: pd.DataFrame) -> pd.Series:
    """Sum sales across all stores/items per day → indexed by date."""
    daily = df.groupby("date")["sales"].sum()
    daily = daily.asfreq("D")
    daily = daily.fillna(method="ffill")
    return daily


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test.
    Returns dict with test statistic, p-value, and whether series is stationary.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < significance,
        "critical_values": result[4],
    }


def train_arima(
    series: pd.Series,
    order: tuple[int, int, int] = (5, 1, 2),
) -> ARIMA:
    """Fit an ARIMA model on the full training series."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=order)
        fitted = model.fit()
    print(f"[arima] Fitted ARIMA{order}  AIC={fitted.aic:.1f}  BIC={fitted.bic:.1f}")
    return fitted


def forecast_future(
    fitted_model,
    horizon: int = 90,
    last_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate out-of-sample forecasts for *horizon* days."""
    pred = fitted_model.forecast(steps=horizon)
    if last_date is None:
        last_date = fitted_model.model.data.dates[-1]
    dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    forecast_df = pd.DataFrame({"date": dates, "forecast": pred.values})
    forecast_df["forecast"] = forecast_df["forecast"].clip(lower=0).round(0)
    return forecast_df


def save_forecast(forecast_df: pd.DataFrame, config: dict | None = None) -> str:
    """Persist forecast to CSV. Returns file path."""
    config = config or load_config()
    out_dir = resolve_path(config["outputs"]["forecasts_dir"])
    ensure_dir(out_dir)
    path = out_dir / "sales_forecast.csv"
    forecast_df.to_csv(path, index=False)
    print(f"[arima] Saved forecast → {path}")
    return str(path)
