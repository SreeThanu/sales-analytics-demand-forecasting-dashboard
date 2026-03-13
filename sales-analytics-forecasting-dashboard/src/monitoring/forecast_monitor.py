"""Compare predicted vs actual values over a monitoring window."""

from __future__ import annotations

import pandas as pd
import numpy as np


def build_comparison_table(
    actual_series: pd.Series,
    fitted_model,
    test_days: int = 90,
) -> pd.DataFrame:
    """
    Create a date-indexed DataFrame with actual, predicted, and error columns
    for the last *test_days* of the series.
    """
    train_len = len(actual_series) - test_days
    test = actual_series.iloc[-test_days:]

    pred = fitted_model.predict(start=train_len, end=len(actual_series) - 1)

    comparison = pd.DataFrame({
        "date": test.index,
        "actual": test.values,
        "predicted": pred.values.round(0),
    })
    comparison["error"] = comparison["actual"] - comparison["predicted"]
    comparison["abs_error"] = comparison["error"].abs()
    comparison["pct_error"] = np.where(
        comparison["actual"] != 0,
        (comparison["abs_error"] / comparison["actual"] * 100).round(2),
        0.0,
    )
    return comparison
