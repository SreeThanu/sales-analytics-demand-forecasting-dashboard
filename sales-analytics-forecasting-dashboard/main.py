#!/usr/bin/env python3
"""
Sales Analytics & Demand Forecasting — Full Pipeline
=====================================================
Runs every stage end-to-end and optionally launches the dashboard.

Usage:
    python main.py                # run pipeline + launch dashboard
    python main.py --no-dash      # run pipeline only (skip dashboard)
"""

from __future__ import annotations

import argparse
import sys

from src.utils.helpers import load_config
from src.data_processing.load_data import load_train_data
from src.data_processing.clean_data import clean_data, save_processed
from src.feature_engineering.create_time_features import add_time_features
from src.analysis.exploratory_analysis import (
    daily_sales_trend,
    monthly_sales_trend,
    store_level_analysis,
    item_level_demand,
    seasonality_pattern,
    print_summary,
)
from src.visualization.plots import (
    plot_sales_trend,
    plot_seasonal_pattern,
    plot_store_comparison,
    plot_forecast_vs_actual,
)
from src.forecasting.arima_model import (
    aggregate_daily_sales,
    check_stationarity,
    train_arima,
    forecast_future,
    save_forecast,
)
from src.forecasting.evaluate_model import evaluate_forecast, save_metrics
from src.monitoring.forecast_monitor import build_comparison_table
from src.monitoring.performance_metrics import summarize_performance


def main(launch_dashboard: bool = True) -> None:
    config = load_config()

    # ── 1. Load Data ──────────────────────────────────────────────────
    print("\n▶ Stage 1 — Loading raw data")
    df = load_train_data(config)

    # ── 2. Clean Data ─────────────────────────────────────────────────
    print("\n▶ Stage 2 — Cleaning data")
    df = clean_data(df)

    # ── 3. Feature Engineering ────────────────────────────────────────
    print("\n▶ Stage 3 — Feature engineering")
    df = add_time_features(df)
    save_processed(df, config)

    # ── 4. Exploratory Analysis ───────────────────────────────────────
    print("\n▶ Stage 4 — Exploratory analysis")
    print_summary(df)
    daily = daily_sales_trend(df)
    monthly = monthly_sales_trend(df)
    stores = store_level_analysis(df)
    items = item_level_demand(df)
    season = seasonality_pattern(df)

    # ── 5. Visualization ─────────────────────────────────────────────
    print("\n▶ Stage 5 — Generating charts")
    plot_sales_trend(daily, config)
    plot_seasonal_pattern(season, config)
    plot_store_comparison(stores, config)

    # ── 6. Forecasting ───────────────────────────────────────────────
    print("\n▶ Stage 6 — ARIMA demand forecasting")
    daily_series = aggregate_daily_sales(df)
    stationarity = check_stationarity(daily_series)
    print(f"  Stationarity test: p={stationarity['p_value']:.4f}  "
          f"stationary={stationarity['is_stationary']}")

    order = tuple(config["forecasting"]["arima_order"])
    fitted = train_arima(daily_series, order=order)

    horizon = config["forecasting"]["forecast_horizon"]
    forecast_df = forecast_future(fitted, horizon=horizon)
    save_forecast(forecast_df, config)

    # ── 7. Model Evaluation ──────────────────────────────────────────
    print("\n▶ Stage 7 — Evaluating model")
    test_days = config["forecasting"]["test_split_days"]
    metrics = evaluate_forecast(daily_series, fitted, test_days=test_days, config=config)
    save_metrics(metrics, config)

    # ── 8. Forecast Monitoring ───────────────────────────────────────
    print("\n▶ Stage 8 — Forecast monitoring")
    comparison = build_comparison_table(daily_series, fitted, test_days=test_days)
    perf = summarize_performance(comparison)
    for k, v in perf.items():
        print(f"  {k}: {v}")

    # Forecast-vs-actual chart (uses the test window)
    actual_tail = daily.tail(test_days).reset_index(drop=True)
    pred_tail = comparison[["date", "predicted"]].rename(columns={"predicted": "forecast"})
    plot_forecast_vs_actual(actual_tail, pred_tail, config)

    # ── 9. Dashboard ─────────────────────────────────────────────────
    if launch_dashboard:
        print("\n▶ Stage 9 — Launching interactive dashboard")
        from src.dashboard.dash_app import run_dashboard
        run_dashboard(config)
    else:
        print("\n✓ Pipeline complete (dashboard skipped).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Analytics Pipeline")
    parser.add_argument(
        "--no-dash",
        action="store_true",
        help="Run the pipeline without launching the dashboard",
    )
    args = parser.parse_args()
    main(launch_dashboard=not args.no_dash)
