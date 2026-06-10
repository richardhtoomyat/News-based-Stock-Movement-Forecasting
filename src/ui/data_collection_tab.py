"""Data Collection tab UI."""

import streamlit as st

from src.config import COMBINED_PROCESSED_CSV, END_DATE, START_DATE
from src.pipeline import run_pipeline_for_selected_ticker


def render_data_collection_tab() -> None:
    """Render Tab 1: Data Collection."""
    st.header("Data Collection")
    st.caption("Pull daily market data from yfinance and generate processed features.")

    col1, col2 = st.columns(2)
    with col1:
        ticker_input = st.text_input(
            "Primary Ticker Symbol",
            placeholder="e.g. AAPL",
            help="Enter any valid stock ticker. Not hardcoded.",
        ).strip().upper()
    with col2:
        st.text_input("Start Date (fixed)", value=START_DATE.isoformat(), disabled=True)
        st.text_input("End Date (fixed)", value=END_DATE.isoformat(), disabled=True)

    if st.button("Pull / Update Data", type="primary"):
        if not ticker_input:
            st.error("Please enter a ticker symbol.")
            return

        with st.spinner(f"Downloading and processing {ticker_input}..."):
            result = run_pipeline_for_selected_ticker(ticker_input)

        if not result.success:
            for msg in result.messages:
                if "missing" in msg.lower() or "no data" in msg.lower():
                    st.error(msg)
                else:
                    st.error(msg)
            return

        if not result.sufficient_warmup:
            st.warning(
                f"Insufficient warmup history before {START_DATE}. "
                "Rolling features on early project dates may be missing."
            )

        for msg in result.messages:
            st.warning(msg)

        st.success(f"Data pull completed for {result.ticker}")

        st.subheader("Run Summary")
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Selected Ticker", result.ticker)
            st.metric("Raw Row Count (warmup + project)", result.raw_row_count)
            st.metric("Warmup Rows (before project start)", result.warmup_row_count)
            st.metric("Processed Row Count (project period)", result.processed_row_count)
        with summary_col2:
            st.metric("Project Date Range", f"{START_DATE} to {END_DATE}")
            if result.raw_date_min and result.raw_date_max:
                st.write(
                    f"**Raw date range:** {result.raw_date_min} to {result.raw_date_max} "
                    f"(includes {result.warmup_row_count}-day warmup)"
                )
            if result.processed_df is not None:
                missing = result.processed_df.isna().sum()
                missing_nonzero = missing[missing > 0]
                if not missing_nonzero.empty:
                    st.write("**Missing Value Counts (non-zero columns):**")
                    missing_df = missing_nonzero.reset_index()
                    missing_df.columns = ["column", "missing_count"]
                    st.dataframe(missing_df, hide_index=True)

        st.subheader("Output Files")
        st.write(f"**Raw (Parquet):** `{result.raw_parquet_path}`")
        st.write(f"**Raw (CSV):** `{result.raw_csv_path}`")
        st.write(f"**Processed (Parquet):** `{result.processed_parquet_path}`")
        st.write(f"**Processed (CSV):** `{result.processed_csv_path}`")
        if result.combined_parquet_path:
            st.write(f"**Combined (Parquet):** `{result.combined_parquet_path}`")
            st.write(f"**Combined (CSV):** `{result.combined_csv_path or COMBINED_PROCESSED_CSV}`")

        if result.processed_df is not None and not result.processed_df.empty:
            st.subheader("Latest 10 Processed Rows")
            st.dataframe(result.processed_df.tail(10), hide_index=True)
