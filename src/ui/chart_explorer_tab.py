"""Stock Chart Explorer tab UI."""

import streamlit as st

from src.charts import (
    build_overlay_options,
    build_stock_chart,
    default_normalize_enabled,
    load_chart_data,
)
from src.config import CHART_RANGES, PRIMARY_METRICS
from src.data_io import list_available_primary_tickers


def render_chart_explorer_tab() -> None:
    """Render Tab 2: Stock Chart Explorer."""
    st.header("Stock Chart Explorer")
    st.caption("Explore processed market features with interactive Plotly charts.")

    available_tickers = list_available_primary_tickers()

    if not available_tickers:
        st.warning(
            "No processed data available. Please run **Data Collection** first "
            "to pull data for at least one ticker."
        )
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        primary_ticker = st.selectbox("Primary Ticker", options=available_tickers)
    with col2:
        chart_range = st.radio("Chart Range", options=CHART_RANGES, horizontal=True)
    with col3:
        primary_metric = st.selectbox("Primary Series", options=PRIMARY_METRICS)

    overlay_options = build_overlay_options(primary_ticker)
    overlay_labels = [o[0] for o in overlay_options]
    overlay_key_map = {o[0]: o[1] for o in overlay_options}

    selected_overlay_labels = st.multiselect(
        "Overlay Series (optional)",
        options=overlay_labels,
        help="Add benchmark returns, other tickers, or additional features.",
    )
    overlay_keys = [overlay_key_map[label] for label in selected_overlay_labels]

    default_norm = default_normalize_enabled(primary_metric, overlay_keys)
    normalize_prices = st.checkbox(
        "Normalize price series to common starting value (base=100)",
        value=default_norm,
        help="Applies when plotting 2+ price series (e.g. adj_close comparisons).",
    )

    chart_data = load_chart_data(primary_ticker, primary_metric, chart_range, overlay_keys)

    if chart_data.get("error"):
        st.warning(chart_data["error"])
        return

    primary_df = chart_data["primary_df"]
    overlays = chart_data["overlays"]

    primary_label = f"{primary_ticker} {primary_metric}"

    fig = build_stock_chart(
        primary_label=primary_label,
        primary_df=primary_df,
        primary_metric=primary_metric,
        overlays=overlays,
        normalize_prices=normalize_prices,
        chart_range=chart_range,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"**Ticker:** {primary_ticker} | "
        f"**Range:** {chart_range} | "
        f"**Metric:** {primary_metric} | "
        f"**Overlays:** {', '.join(selected_overlay_labels) if selected_overlay_labels else 'None'}"
    )
