"""Rolling and summary performance metrics for forecast monitoring."""

from __future__ import annotations

import pandas as pd
import numpy as np


def rolling_mae(comparison: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Compute a rolling-window MAE from the comparison table."""
    rolling = comparison.copy()
    rolling["rolling_mae"] = (
        rolling["abs_error"].rolling(window=window, min_periods=1).mean().round(2)
    )
    return rolling[["date", "rolling_mae"]]


def summarize_performance(comparison: pd.DataFrame) -> dict:
    """Aggregate performance summary from the comparison table."""
    return {
        "mean_abs_error": round(comparison["abs_error"].mean(), 2),
        "median_abs_error": round(comparison["abs_error"].median(), 2),
        "max_abs_error": round(comparison["abs_error"].max(), 2),
        "mean_pct_error": round(comparison["pct_error"].mean(), 2),
        "within_5pct": round(
            (comparison["pct_error"] <= 5).mean() * 100, 1
        ),
        "within_10pct": round(
            (comparison["pct_error"] <= 10).mean() * 100, 1
        ),
    }
