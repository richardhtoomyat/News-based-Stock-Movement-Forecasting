"""Streamlit web app for news-based stock movement forecasting — market data module."""

import streamlit as st

from src.config import END_DATE, START_DATE, ensure_data_dirs
from src.ui.chart_explorer_tab import render_chart_explorer_tab
from src.ui.data_collection_tab import render_data_collection_tab


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Stock Market Data Collector",
        page_icon="📈",
        layout="wide",
    )

    ensure_data_dirs()

    st.title("News-Based Stock Movement Forecasting")
    st.markdown(
        f"Market data collection and exploration | Project period: "
        f"**{START_DATE}** to **{END_DATE}**"
    )

    tab_data, tab_chart = st.tabs(["Data Collection", "Stock Chart Explorer"])

    with tab_data:
        render_data_collection_tab()

    with tab_chart:
        render_chart_explorer_tab()


if __name__ == "__main__":
    main()
