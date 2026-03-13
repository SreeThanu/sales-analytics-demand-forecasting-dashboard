"""Generate and save static matplotlib charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.helpers import load_config, resolve_path, ensure_dir


def _figures_dir(config: dict | None = None) -> Path:
    config = config or load_config()
    d = resolve_path(config["outputs"]["figures_dir"])
    ensure_dir(d)
    return d


def plot_sales_trend(daily: pd.DataFrame, config: dict | None = None) -> str:
    """Line chart of daily total sales. Returns saved file path."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"], daily["total_sales"], linewidth=0.6, color="#2563eb")
    ax.set_title("Daily Total Sales Trend", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = _figures_dir(config) / "sales_trend.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return str(path)


def plot_seasonal_pattern(seasonality: pd.DataFrame, config: dict | None = None) -> str:
    """Bar chart of average sales by month."""
    fig, ax = plt.subplots(figsize=(10, 5))
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    ax.bar(seasonality["month"], seasonality["avg_sales"], color="#7c3aed")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_title("Average Sales by Month (Seasonality)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Sales")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = _figures_dir(config) / "seasonal_pattern.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return str(path)


def plot_store_comparison(store_agg: pd.DataFrame, config: dict | None = None) -> str:
    """Horizontal bar chart of total sales by store."""
    fig, ax = plt.subplots(figsize=(10, 6))
    store_agg = store_agg.sort_values("total_sales")
    ax.barh(
        store_agg["store"].astype(str),
        store_agg["total_sales"],
        color="#059669",
    )
    ax.set_title("Total Sales by Store", fontsize=14, fontweight="bold")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Store")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = _figures_dir(config) / "store_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return str(path)


def plot_forecast_vs_actual(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    config: dict | None = None,
) -> str:
    """Overlay chart of actual vs forecasted sales."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual["date"], actual["total_sales"], label="Actual", linewidth=1)
    ax.plot(
        forecast["date"],
        forecast["forecast"],
        label="Forecast",
        linewidth=1,
        linestyle="--",
        color="#dc2626",
    )
    ax.set_title("Forecast vs Actual Sales", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = _figures_dir(config) / "forecast_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] Saved → {path}")
    return str(path)
