"""Create time-based features from the date column."""

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive calendar features from the 'date' column:
      year, month, quarter, day_of_week, day_of_month, week_of_year, is_weekend
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon … 6=Sun
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    print(
        f"[feature_engineering] Added time features: "
        f"year, month, quarter, day_of_week, day_of_month, week_of_year, is_weekend"
    )
    return df
