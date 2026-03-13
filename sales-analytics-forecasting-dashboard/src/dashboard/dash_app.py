"""Interactive Plotly Dash dashboard for Sales Analytics & Demand Forecasting."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from src.utils.helpers import load_config, resolve_path


def _load_dashboard_data(config: dict | None = None):
    """Load all data assets required by the dashboard."""
    config = config or load_config()

    processed_path = (
        resolve_path(config["data"]["processed_dir"]) / config["data"]["processed_file"]
    )
    forecast_path = resolve_path(config["outputs"]["forecasts_dir"]) / "sales_forecast.csv"
    metrics_path = resolve_path(config["outputs"]["metrics_dir"]) / "model_metrics.csv"

    df = pd.read_csv(
        processed_path, parse_dates=["date"],
        usecols=["date", "store", "item", "sales", "month"],
        dtype={"store": "int16", "item": "int16", "sales": "int32", "month": "int8"},
    )
    forecast_df = pd.read_csv(forecast_path, parse_dates=["date"])
    if "predicted_sales" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"predicted_sales": "forecast"})
    if "actual_sales" in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"actual_sales": "actual"})
    metrics_df = pd.read_csv(metrics_path)

    return df, forecast_df, metrics_df


COLORS = {
    "bg": "#0f172a",
    "card": "#1e293b",
    "accent": "#3b82f6",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "green": "#22c55e",
    "red": "#ef4444",
    "purple": "#a855f7",
}


def build_app(config: dict | None = None) -> Dash:
    """Construct and return the Dash application (does NOT start the server)."""
    config = config or load_config()
    df, forecast_df, metrics_df = _load_dashboard_data(config)

    daily = df.groupby("date")["sales"].sum().reset_index()
    daily.columns = ["date", "total_sales"]

    stores = sorted(df["store"].unique())
    items = sorted(df["item"].unique())

    app = Dash(
        __name__,
        title="Sales Analytics Dashboard",
        suppress_callback_exceptions=True,
    )

    # ------------------------------------------------------------------ layout
    app.layout = html.Div(
        style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "24px",
               "fontFamily": "'Inter', 'Segoe UI', sans-serif", "color": COLORS["text"]},
        children=[
            # Header
            html.Div(
                style={"textAlign": "center", "marginBottom": "32px"},
                children=[
                    html.H1(
                        "Sales Analytics & Demand Forecasting",
                        style={"fontSize": "2rem", "fontWeight": "700", "margin": "0"},
                    ),
                    html.P(
                        "Interactive dashboard — historical trends, store & item analysis, ARIMA forecasts",
                        style={"color": COLORS["muted"], "marginTop": "8px"},
                    ),
                ],
            ),

            # KPI cards
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "28px",
                       "flexWrap": "wrap", "justifyContent": "center"},
                children=[
                    _kpi_card("Total Sales", f"{df['sales'].sum():,.0f}"),
                    _kpi_card("Stores", str(df["store"].nunique())),
                    _kpi_card("Items", str(df["item"].nunique())),
                    _kpi_card("MAE", str(metrics_df["MAE"].iloc[0])),
                    _kpi_card("RMSE", str(metrics_df["RMSE"].iloc[0])),
                ],
            ),

            # Tabs
            dcc.Tabs(
                id="tabs",
                value="trends",
                colors={"border": COLORS["card"], "primary": COLORS["accent"],
                        "background": COLORS["card"]},
                style={"marginBottom": "20px"},
                children=[
                    dcc.Tab(label="Sales Trends", value="trends",
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label="Store Performance", value="store",
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label="Item Performance", value="item",
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label="Forecast", value="forecast",
                            style=_tab_style(), selected_style=_tab_selected_style()),
                    dcc.Tab(label="Forecast vs Actual", value="compare",
                            style=_tab_style(), selected_style=_tab_selected_style()),
                ],
            ),

            # Dynamic content
            html.Div(id="tab-content"),

            # Hidden stores for shared data
            dcc.Store(id="store-dropdown-value", data=stores[0]),
        ],
    )

    # -------------------------------------------------------------- callbacks
    @app.callback(Output("tab-content", "children"), Input("tabs", "value"))
    def render_tab(tab):
        if tab == "trends":
            return _trends_tab(daily, df)
        if tab == "store":
            return _store_tab(df, stores)
        if tab == "item":
            return _item_tab(df, items)
        if tab == "forecast":
            return _forecast_tab(daily, forecast_df)
        if tab == "compare":
            return _compare_tab(daily, forecast_df)
        return html.P("Select a tab")

    return app


# ---------------------------------------------------------------- tab builders

def _trends_tab(daily: pd.DataFrame, df: pd.DataFrame):
    fig_daily = px.line(
        daily, x="date", y="total_sales",
        title="Daily Total Sales",
        template="plotly_dark",
    )
    fig_daily.update_traces(line_color=COLORS["accent"])

    monthly = df.groupby(df["date"].dt.to_period("M").astype(str))["sales"].sum().reset_index()
    monthly.columns = ["month", "total_sales"]
    fig_monthly = px.bar(
        monthly, x="month", y="total_sales",
        title="Monthly Total Sales",
        template="plotly_dark",
    )
    fig_monthly.update_traces(marker_color=COLORS["purple"])

    seasonality = df.groupby("month")["sales"].mean().reset_index()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    seasonality["month_name"] = seasonality["month"].map(lambda m: month_names[m - 1])
    fig_season = px.bar(
        seasonality, x="month_name", y="sales",
        title="Seasonality — Avg Sales by Month",
        template="plotly_dark",
    )
    fig_season.update_traces(marker_color=COLORS["green"])

    return html.Div([
        _chart_card(fig_daily),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            html.Div(_chart_card(fig_monthly), style={"flex": "1 1 48%"}),
            html.Div(_chart_card(fig_season), style={"flex": "1 1 48%"}),
        ]),
    ])


def _store_tab(df: pd.DataFrame, stores):
    store_agg = (
        df.groupby("store")["sales"]
        .agg(total_sales="sum", avg_sales="mean")
        .reset_index()
        .sort_values("total_sales", ascending=True)
    )
    fig_bar = px.bar(
        store_agg, x="total_sales", y=store_agg["store"].astype(str),
        orientation="h", title="Total Sales by Store",
        template="plotly_dark", color="total_sales",
        color_continuous_scale="Viridis",
    )
    fig_bar.update_layout(yaxis_title="Store", coloraxis_showscale=False)

    store_monthly = df.copy()
    store_monthly["ym"] = store_monthly["date"].dt.to_period("M").astype(str)
    pivot = store_monthly.groupby(["ym", "store"])["sales"].sum().reset_index()
    fig_line = px.line(
        pivot, x="ym", y="sales", color=pivot["store"].astype(str),
        title="Monthly Sales per Store",
        template="plotly_dark",
    )

    return html.Div([
        _chart_card(fig_bar),
        _chart_card(fig_line),
    ])


def _item_tab(df: pd.DataFrame, items):
    item_agg = (
        df.groupby("item")["sales"]
        .agg(total_demand="sum", avg_demand="mean")
        .reset_index()
        .sort_values("total_demand", ascending=False)
    )
    fig_bar = px.bar(
        item_agg, x=item_agg["item"].astype(str), y="total_demand",
        title="Total Demand by Item",
        template="plotly_dark",
    )
    fig_bar.update_traces(marker_color=COLORS["accent"])
    fig_bar.update_layout(xaxis_title="Item", xaxis_tickangle=-45)

    fig_dist = px.box(
        df, x=df["item"].astype(str), y="sales",
        title="Sales Distribution by Item",
        template="plotly_dark",
    )
    fig_dist.update_layout(xaxis_title="Item")

    return html.Div([
        _chart_card(fig_bar),
        _chart_card(fig_dist),
    ])


def _forecast_tab(daily: pd.DataFrame, forecast_df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["total_sales"],
        name="Historical", mode="lines",
        line=dict(color=COLORS["accent"], width=1),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        name="Forecast", mode="lines",
        line=dict(color=COLORS["red"], width=2, dash="dash"),
    ))
    fig.update_layout(
        title="ARIMA Demand Forecast",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Total Sales",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _chart_card(fig)


def _compare_tab(daily: pd.DataFrame, forecast_df: pd.DataFrame):
    test_days = len(forecast_df)
    actual_tail = daily.tail(test_days).copy().reset_index(drop=True)

    if len(actual_tail) == len(forecast_df):
        compare = pd.DataFrame({
            "date": actual_tail["date"],
            "actual": actual_tail["total_sales"],
            "predicted": forecast_df["forecast"].values,
        })
    else:
        compare = forecast_df.copy()
        compare["actual"] = None

    fig = go.Figure()
    if compare["actual"].notna().any():
        fig.add_trace(go.Scatter(
            x=compare["date"], y=compare["actual"],
            name="Actual", mode="lines",
            line=dict(color=COLORS["accent"]),
        ))
    fig.add_trace(go.Scatter(
        x=compare["date"], y=compare["predicted"],
        name="Predicted", mode="lines",
        line=dict(color=COLORS["red"], dash="dash"),
    ))
    fig.update_layout(
        title="Forecast vs Actual Comparison",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Total Sales",
    )
    return _chart_card(fig)


# -------------------------------------------------------------- UI helpers

def _kpi_card(title: str, value: str):
    return html.Div(
        style={
            "backgroundColor": COLORS["card"],
            "borderRadius": "12px",
            "padding": "20px 28px",
            "minWidth": "150px",
            "textAlign": "center",
        },
        children=[
            html.P(title, style={"margin": "0", "fontSize": "0.85rem",
                                  "color": COLORS["muted"], "letterSpacing": "0.5px"}),
            html.H2(value, style={"margin": "8px 0 0", "fontSize": "1.6rem",
                                   "fontWeight": "700"}),
        ],
    )


def _chart_card(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return html.Div(
        style={
            "backgroundColor": COLORS["card"],
            "borderRadius": "12px",
            "padding": "16px",
            "marginBottom": "20px",
        },
        children=[dcc.Graph(figure=fig, config={"displayModeBar": False})],
    )


def _tab_style():
    return {
        "backgroundColor": COLORS["card"],
        "color": COLORS["muted"],
        "border": "none",
        "borderRadius": "8px 8px 0 0",
        "padding": "10px 20px",
        "fontWeight": "500",
    }


def _tab_selected_style():
    return {
        "backgroundColor": COLORS["accent"],
        "color": "#fff",
        "border": "none",
        "borderRadius": "8px 8px 0 0",
        "padding": "10px 20px",
        "fontWeight": "600",
    }


def run_dashboard(config: dict | None = None):
    """Build and launch the Dash server."""
    config = config or load_config()
    app = build_app(config)
    app.run(
        host=config["dashboard"]["host"],
        port=config["dashboard"]["port"],
        debug=False,
    )


if __name__ == "__main__":
    run_dashboard()
