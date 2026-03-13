"""Data cleaning: handle missing values, remove duplicates, validate types."""

import pandas as pd

from src.utils.helpers import load_config, resolve_path, ensure_dir


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw sales DataFrame:
      1. Drop exact duplicate rows
      2. Fill missing numeric values with 0
      3. Drop rows where date is null
      4. Ensure correct dtypes
    """
    initial = len(df)

    df = df.drop_duplicates()
    dropped = initial - len(df)
    if dropped:
        print(f"[clean_data] Removed {dropped:,} duplicate rows")

    if df["date"].isna().any():
        null_dates = df["date"].isna().sum()
        df = df.dropna(subset=["date"])
        print(f"[clean_data] Dropped {null_dates} rows with null dates")

    if "sales" in df.columns:
        df["sales"] = df["sales"].fillna(0).astype(int)

    for col in ["store", "item"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df = df.sort_values("date").reset_index(drop=True)
    print(f"[clean_data] Cleaned data: {df.shape[0]:,} rows")
    return df


def save_processed(df: pd.DataFrame, config: dict | None = None) -> str:
    """Save cleaned DataFrame to the processed directory. Returns the file path."""
    config = config or load_config()
    out_dir = resolve_path(config["data"]["processed_dir"])
    ensure_dir(out_dir)
    out_path = out_dir / config["data"]["processed_file"]
    df.to_csv(out_path, index=False)
    print(f"[clean_data] Saved processed data → {out_path}")
    return str(out_path)
