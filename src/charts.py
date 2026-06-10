"""Chart data loading and Plotly visualization."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import (
    CHART_RANGES,
    END_DATE,
    PRICE_COLUMNS,
    RETURN_FEATURE_COLUMNS,
    START_DATE,
    processed_file_path,
    raw_file_path,
)
from src.data_io import list_available_primary_tickers, load_ticker_raw_data
from src.features import calculate_adjusted_ohlc, calculate_benchmark_returns


def load_processed_ticker(ticker: str) -> pd.DataFrame | None:
    """Load processed market features for a ticker."""
    path = processed_file_path(ticker.upper(), "parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _get_benchmark_adj_close(benchmark: str) -> pd.DataFrame | None:
    """Load benchmark adj_close from raw data (computed on the fly)."""
    raw = load_ticker_raw_data(benchmark)
    if raw is None or raw.empty:
        return None
    adj = calculate_adjusted_ohlc(raw)
    return adj[["date", "adj_close"]].copy()


def _get_benchmark_return_series(benchmark: str, column: str) -> pd.DataFrame | None:
    """Load benchmark return column from raw data (computed on the fly)."""
    raw = load_ticker_raw_data(benchmark)
    if raw is None or raw.empty:
        return None
    returns_df = calculate_benchmark_returns(raw, benchmark)
    if column not in returns_df.columns:
        return None
    return returns_df[["date", column]].rename(columns={column: "value"})


def filter_by_chart_range(df: pd.DataFrame, chart_range: str) -> pd.DataFrame:
    """Filter dataframe to the selected chart range."""
    if df is None or df.empty:
        return df

    work = df.copy()
    work = work.sort_values("date")

    if chart_range == "1D":
        latest = work["date"].max()
        return work[work["date"] == latest]

    if chart_range == "5D":
        return work.tail(5)

    if chart_range == "1M":
        latest = work["date"].max()
        cutoff = latest - timedelta(days=30)
        return work[work["date"] >= cutoff]

    if chart_range == "1Y":
        start = pd.Timestamp(START_DATE)
        end = pd.Timestamp(END_DATE)
        return work[(work["date"] >= start) & (work["date"] <= end)]

    return work


def _parse_overlay_key(overlay_key: str) -> tuple[str, str]:
    """
    Parse overlay key like 'AAPL:adj_close' or 'PPA:ppa_return_1d'.
    Returns (source_id, column_name).
    """
    parts = overlay_key.split(":", 1)
    if len(parts) != 2:
        return overlay_key, overlay_key
    return parts[0], parts[1]


def _get_series_for_overlay(
    overlay_key: str,
    primary_ticker: str,
    chart_range: str,
) -> tuple[str, pd.DataFrame] | None:
    """
    Resolve overlay key to a labeled series dataframe with date and value columns.
    Returns (label, df with columns date, value) or None.
    """
    source_id, column = _parse_overlay_key(overlay_key)

    # Benchmark adj_close from raw
    if column == "adj_close" and source_id.upper() in ("PPA", "SPY"):
        raw_adj = _get_benchmark_adj_close(source_id.upper())
        if raw_adj is None:
            return None
        raw_adj = raw_adj.rename(columns={"adj_close": "value"})
        raw_adj = filter_by_chart_range(raw_adj, chart_range)
        label = f"{source_id.upper()} adj_close"
        return label, raw_adj

    # Benchmark returns computed from raw benchmark data
    if source_id.upper() in ("PPA", "SPY") and column in (
        "ppa_return_1d",
        "ppa_return_5d",
        "spy_return_1d",
    ):
        series_df = _get_benchmark_return_series(source_id.upper(), column)
        if series_df is None:
            return None
        series_df = filter_by_chart_range(series_df, chart_range)
        label = f"{source_id.upper()} {column}"
        return label, series_df

    # Another primary ticker's processed data
    other_df = load_processed_ticker(source_id)
    if other_df is None or column not in other_df.columns:
        return None
    series_df = other_df[["date", column]].rename(columns={column: "value"})
    series_df = filter_by_chart_range(series_df, chart_range)
    label = f"{source_id.upper()} {column}"
    return label, series_df


def build_overlay_options(primary_ticker: str) -> list[tuple[str, str]]:
    """
    Build list of (display_label, overlay_key) for multiselect.
    """
    options: list[tuple[str, str]] = []
    all_tickers = list_available_primary_tickers()

    for t in all_tickers:
        if t == primary_ticker.upper():
            continue
        for col in ("adj_close", "stock_return_1d", "stock_return_5d"):
            key = f"{t}:{col}"
            options.append((f"{t} {col}", key))

    for col in ("ppa_return_1d", "ppa_return_5d", "spy_return_1d"):
        bench = "PPA" if col.startswith("ppa") else "SPY"
        key = f"{bench}:{col}"
        options.append((f"{bench} {col}", key))

    for col in ("relative_return_1d", "relative_return_5d", "market_adjusted_return_1d"):
        key = f"{primary_ticker}:{col}"
        options.append((f"{primary_ticker} {col} (overlay)", key))

    options.append(("PPA adj_close", "PPA:adj_close"))
    options.append(("SPY adj_close", "SPY:adj_close"))
    options.append((f"{primary_ticker} volume", f"{primary_ticker}:volume"))

    return options


def load_chart_data(
    primary_ticker: str,
    primary_metric: str,
    chart_range: str,
    overlay_keys: list[str],
) -> dict:
    """
    Load primary series and overlay series for charting.
    Returns dict with primary_df, overlays list, and metadata.
    """
    primary_df = load_processed_ticker(primary_ticker)
    if primary_df is None:
        return {"primary_df": None, "overlays": [], "error": "Processed data unavailable."}

    # Benchmark return metrics may not be stored in processed file; compute from raw
    if primary_metric in ("ppa_return_1d", "ppa_return_5d"):
        primary_series = _get_benchmark_return_series("PPA", primary_metric)
    elif primary_metric == "spy_return_1d":
        primary_series = _get_benchmark_return_series("SPY", primary_metric)
    elif primary_metric not in primary_df.columns:
        return {"primary_df": None, "overlays": [], "error": f"Metric '{primary_metric}' not available."}
    else:
        primary_series = primary_df[["date", primary_metric]].rename(columns={primary_metric: "value"})

    if primary_series is None or primary_series.empty:
        return {"primary_df": None, "overlays": [], "error": "No data for selected metric."}

    primary_series = filter_by_chart_range(primary_series, chart_range)

    overlays = []
    for key in overlay_keys:
        resolved = _get_series_for_overlay(key, primary_ticker, chart_range)
        if resolved:
            overlays.append({"key": key, "label": resolved[0], "df": resolved[1]})

    return {
        "primary_df": primary_series,
        "primary_metric": primary_metric,
        "overlays": overlays,
        "error": None,
    }


def _normalize_price_series(
    series_list: list[tuple[str, pd.DataFrame]],
) -> list[tuple[str, pd.DataFrame]]:
    """Rebase price series to 100 at first common starting value per series."""
    normalized = []
    for label, df in series_list:
        if df is None or df.empty or df["value"].isna().all():
            normalized.append((label, df))
            continue
        work = df.copy()
        first_valid_idx = work["value"].first_valid_index()
        if first_valid_idx is None:
            normalized.append((label, work))
            continue
        base = work.loc[first_valid_idx, "value"]
        if base and base != 0:
            work["value"] = (work["value"] / base) * 100
        normalized.append((label, work))
    return normalized


def build_stock_chart(
    primary_label: str,
    primary_df: pd.DataFrame,
    primary_metric: str,
    overlays: list[dict],
    normalize_prices: bool = False,
    chart_range: str = "1Y",
) -> go.Figure:
    """
    Build interactive Plotly chart with zoom, pan, hover, legend toggle, range slider.
    """
    if primary_df is None or primary_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig

    is_primary_price = primary_metric in PRICE_COLUMNS
    overlay_price_series: list[tuple[str, pd.DataFrame]] = []
    overlay_feature_series: list[tuple[str, pd.DataFrame]] = []

    primary_work = primary_df.copy()

    for ov in overlays:
        col = _parse_overlay_key(ov["key"])[1]
        if col in PRICE_COLUMNS:
            overlay_price_series.append((ov["label"], ov["df"]))
        else:
            overlay_feature_series.append((ov["label"], ov["df"]))

    price_series_to_plot: list[tuple[str, pd.DataFrame]] = []
    feature_series_to_plot: list[tuple[str, pd.DataFrame]] = []

    if is_primary_price:
        price_series_to_plot.append((primary_label, primary_work))
    else:
        feature_series_to_plot.append((primary_label, primary_work))

    price_series_to_plot.extend(overlay_price_series)
    feature_series_to_plot.extend(overlay_feature_series)

    # Normalize only price series when requested and 2+ price series
    if normalize_prices and len(price_series_to_plot) >= 2:
        price_series_to_plot = _normalize_price_series(price_series_to_plot)

    use_secondary = (is_primary_price and len(feature_series_to_plot) > 0) or (
        not is_primary_price and len(price_series_to_plot) > 0
    )

    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

    marker_mode = "markers" if chart_range == "1D" else "lines"

    def _add_trace(label: str, df: pd.DataFrame, secondary: bool) -> None:
        if df is None or df.empty:
            return
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["value"],
                mode=marker_mode,
                name=label,
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.4f}<extra></extra>",
            ),
            secondary_y=secondary,
        )

    for label, df in price_series_to_plot:
        _add_trace(label, df, secondary=use_secondary and not is_primary_price)

    for label, df in feature_series_to_plot:
        _add_trace(label, df, secondary=use_secondary and is_primary_price)

    if normalize_prices and len(price_series_to_plot) >= 2:
        y_title_primary = "Normalized Price (base=100)"
    elif is_primary_price:
        y_title_primary = "Price"
    else:
        y_title_primary = primary_metric

    fig.update_layout(
        title=f"{primary_label} — {chart_range}",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
        ),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=y_title_primary, secondary_y=False)
    if use_secondary:
        fig.update_yaxes(title_text="Feature / Return", secondary_y=True)

    return fig


def default_normalize_enabled(primary_metric: str, overlay_keys: list[str]) -> bool:
    """Return True if normalization checkbox should default to on."""
    if primary_metric not in PRICE_COLUMNS:
        return False
    price_overlay_count = sum(
        1 for k in overlay_keys if _parse_overlay_key(k)[1] in PRICE_COLUMNS
    )
    return price_overlay_count >= 1
