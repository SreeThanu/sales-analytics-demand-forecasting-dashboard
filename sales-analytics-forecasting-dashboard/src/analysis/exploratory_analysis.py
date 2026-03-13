"""Exploratory Data Analysis — compute summary statistics and trend tables."""

from __future__ import annotations

import pandas as pd


def daily_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales per day."""
    trend = df.groupby("date")["sales"].sum().reset_index()
    trend.columns = ["date", "total_sales"]
    return trend


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales per year-month."""
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    trend = df.groupby("year_month")["sales"].sum().reset_index()
    trend.columns = ["year_month", "total_sales"]
    return trend


def store_level_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Total and average daily sales per store."""
    agg = (
        df.groupby("store")["sales"]
        .agg(total_sales="sum", avg_daily_sales="mean", max_daily_sales="max")
        .reset_index()
    )
    return agg.sort_values("total_sales", ascending=False)


def item_level_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Total demand per item across all stores."""
    agg = (
        df.groupby("item")["sales"]
        .agg(total_demand="sum", avg_daily_demand="mean")
        .reset_index()
    )
    return agg.sort_values("total_demand", ascending=False)


def seasonality_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Average sales by month (across all years) to reveal seasonality."""
    pattern = df.groupby("month")["sales"].mean().reset_index()
    pattern.columns = ["month", "avg_sales"]
    return pattern


def weekday_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Average sales by day of week."""
    pattern = df.groupby("day_of_week")["sales"].mean().reset_index()
    pattern.columns = ["day_of_week", "avg_sales"]
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pattern["day_name"] = pattern["day_of_week"].map(lambda x: day_names[x])
    return pattern


def print_summary(df: pd.DataFrame) -> None:
    """Print a quick dataset summary to stdout."""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Rows            : {len(df):,}")
    print(f"  Date range      : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Stores          : {df['store'].nunique()}")
    print(f"  Items           : {df['item'].nunique()}")
    print(f"  Total sales     : {df['sales'].sum():,.0f}")
    print(f"  Avg daily sales : {df['sales'].mean():.2f}")
    print("=" * 60 + "\n")
