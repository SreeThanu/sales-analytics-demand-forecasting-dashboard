"""Load raw Kaggle dataset files."""

import pandas as pd

from src.utils.helpers import load_config, resolve_path


def load_train_data(config: dict | None = None) -> pd.DataFrame:
    """Load training CSV with date parsing."""
    config = config or load_config()
    path = resolve_path(config["data"]["raw_dir"]) / config["data"]["train_file"]
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[load_data] Loaded train data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_test_data(config: dict | None = None) -> pd.DataFrame:
    """Load test CSV with date parsing."""
    config = config or load_config()
    path = resolve_path(config["data"]["raw_dir"]) / config["data"]["test_file"]
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[load_data] Loaded test data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_processed_data(config: dict | None = None) -> pd.DataFrame:
    """Load the processed dataset (after cleaning + feature engineering)."""
    config = config or load_config()
    path = resolve_path(config["data"]["processed_dir"]) / config["data"]["processed_file"]
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[load_data] Loaded processed data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df
