# Sales Analytics & Demand Forecasting Dashboard

A production-style Python project that analyzes historical retail sales data, extracts insights, forecasts demand using ARIMA, and presents results through an interactive Plotly Dash dashboard.

## Dataset

This project uses the [Kaggle Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) dataset containing 5 years of daily sales data across 10 stores and 50 items.

## Project Structure

```
sales-analytics-forecasting-dashboard/
├── config/config.yaml              # Project configuration
├── data/
│   ├── raw/                        # Original Kaggle CSV files
│   └── processed/                  # Cleaned & feature-engineered data
├── notebooks/                      # Exploratory Jupyter notebooks
├── src/
│   ├── data_processing/            # Data loading and cleaning
│   ├── feature_engineering/        # Time-based feature creation
│   ├── analysis/                   # Exploratory analysis
│   ├── visualization/              # Plotting utilities
│   ├── forecasting/                # ARIMA model & evaluation
│   ├── monitoring/                 # Forecast monitoring & metrics
│   ├── dashboard/                  # Plotly Dash interactive app
│   └── utils/                      # Helper functions
├── outputs/
│   ├── forecasts/                  # Forecast CSV outputs
│   ├── metrics/                    # Model evaluation metrics
│   └── figures/                    # Generated charts
├── main.py                         # Full pipeline entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Clone or download the project
cd sales-analytics-forecasting-dashboard

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle
#    Place train.csv, test.csv, and sample_submission.csv into data/raw/
```

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This will execute: data loading → cleaning → feature engineering → exploratory analysis → forecasting → evaluation → monitoring, and then launch the interactive dashboard.

### Run Only the Dashboard

```bash
python -m src.dashboard.dash_app
```

### Explore Notebooks

```bash
jupyter notebook notebooks/
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_processing/` | Load CSV, parse dates, handle missing values, remove duplicates |
| 2 | `feature_engineering/` | Add year, month, quarter, day_of_week features |
| 3 | `analysis/` | Daily/monthly trends, store & item analysis, seasonality |
| 4 | `visualization/` | Revenue trend, store comparison, demand distribution charts |
| 5 | `forecasting/` | ARIMA model training and future demand prediction |
| 6 | `monitoring/` | Predicted vs actual comparison, MAE/RMSE metrics |
| 7 | `dashboard/` | Interactive Plotly Dash web application |

## Key Technologies

- **pandas / numpy** — data manipulation
- **matplotlib / plotly** — static and interactive visualization
- **statsmodels** — ARIMA time-series forecasting
- **scikit-learn** — evaluation metrics
- **Dash** — interactive web dashboard
- **PyYAML** — configuration management
