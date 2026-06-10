"""Application configuration: dates, paths, column schemas, and metric lists."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Fixed project date range (user cannot change)
START_DATE = date(2024, 5, 1)
END_DATE = date(2025, 5, 31)

# Warmup lookback for rolling feature calculation (stored in raw files, excluded from processed)
WARMUP_TRADING_DAYS = 20
WARMUP_CALENDAR_DAYS = 60
WARMUP_RETRY_CALENDAR_DAYS = 30
WARMUP_MAX_RETRIES = 3

# Benchmark tickers
BENCHMARK_TICKERS = ["PPA", "SPY"]
DEFENSE_BENCHMARK = "PPA"
MARKET_BENCHMARK = "SPY"

# Data source settings
DATA_SOURCE = "yfinance"
INTERVAL = "1d"
AUTO_ADJUST = False
ACTIONS = True

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_PRICES_DIR = DATA_DIR / "raw" / "prices"
PROCESSED_DIR = DATA_DIR / "processed"
RUN_MANIFEST_PATH = RAW_PRICES_DIR / "run_manifest.csv"
COMBINED_PROCESSED_PARQUET = PROCESSED_DIR / "combined_market_features.parquet"
COMBINED_PROCESSED_CSV = PROCESSED_DIR / "combined_market_features.csv"

# Raw column schema
RAW_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "dividends",
    "stock_splits",
    "source",
    "fetched_at_utc",
]

# Processed column schema (final output order)
PROCESSED_COLUMNS = [
    "date",
    "ticker",
    "company_name",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "volume",
    "stock_return_1d",
    "stock_return_3d",
    "stock_return_5d",
    "volume_change_1d",
    "volume_ratio_5d_avg",
    "volatility_5d",
    "price_vs_ma20",
    "drawdown_20d",
    "overnight_gap_return",
    "open_to_close_return",
    "ppa_return_1d",
    "relative_return_1d",
    "relative_return_5d",
    "spy_return_1d",
    "market_adjusted_return_1d",
    "next_stock_return",
    "next_relative_return",
    "target_up_next_day",
    "target_outperform_ppa_next_day",
    "next_day_open_gap",
    "next_day_open_to_close_return",
]

# Chart primary metrics
PRIMARY_METRICS = [
    "adj_close",
    "stock_return_1d",
    "stock_return_5d",
    "relative_return_1d",
    "relative_return_5d",
    "market_adjusted_return_1d",
    "ppa_return_1d",
    "ppa_return_5d",
    "spy_return_1d",
    "volume",
]

# Price-based columns (eligible for normalization)
PRICE_COLUMNS = {"adj_close"}

# Return/feature columns (never normalized)
RETURN_FEATURE_COLUMNS = {
    "stock_return_1d",
    "stock_return_5d",
    "relative_return_1d",
    "relative_return_5d",
    "market_adjusted_return_1d",
    "ppa_return_1d",
    "ppa_return_5d",
    "spy_return_1d",
}

# Chart range options
CHART_RANGES = ["1D", "5D", "1M", "1Y"]

# Manifest columns
MANIFEST_COLUMNS = [
    "run_id",
    "source",
    "selected_ticker",
    "benchmark_tickers",
    "start_date",
    "end_date",
    "interval",
    "auto_adjust",
    "actions",
    "fetched_at_utc",
    "raw_output_file",
    "processed_output_file",
    "row_count_raw",
    "row_count_processed",
    "status",
    "notes",
]


def raw_file_path(ticker: str, ext: str = "parquet") -> Path:
    """Return path for a ticker's raw data file."""
    return RAW_PRICES_DIR / f"{ticker.upper()}_raw.{ext}"


def processed_file_path(ticker: str, ext: str = "parquet") -> Path:
    """Return path for a ticker's processed market features file."""
    return PROCESSED_DIR / f"{ticker.upper()}_market_features.{ext}"


def ensure_data_dirs() -> None:
    """Create data directories if they do not exist."""
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_download_start_date() -> date:
    """Return download start date with calendar buffer for warmup trading days."""
    return START_DATE - timedelta(days=WARMUP_CALENDAR_DAYS)


def trim_to_project_period(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to the fixed project period [START_DATE, END_DATE]."""
    if df is None or df.empty:
        return df
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)
    mask = (work["date"] >= start_ts) & (work["date"] <= end_ts)
    return work.loc[mask].sort_values("date").reset_index(drop=True)


def get_warmup_cutoff_date(df: pd.DataFrame) -> date | None:
    """Return earliest warmup date kept (20th trading day before START_DATE, or earliest available)."""
    if df is None or df.empty:
        return None
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    start_ts = pd.Timestamp(START_DATE)
    warmup_rows = work.loc[work["date"] < start_ts].sort_values("date")
    if warmup_rows.empty:
        return None
    if len(warmup_rows) >= WARMUP_TRADING_DAYS:
        cutoff_ts = warmup_rows.iloc[-WARMUP_TRADING_DAYS]["date"]
    else:
        cutoff_ts = warmup_rows.iloc[0]["date"]
    return cutoff_ts.date()


def trim_raw_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Keep last WARMUP_TRADING_DAYS rows before START_DATE plus full project period."""
    if df is None or df.empty:
        return df
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    start_ts = pd.Timestamp(START_DATE)
    end_ts = pd.Timestamp(END_DATE)

    warmup = work.loc[work["date"] < start_ts].sort_values("date").tail(WARMUP_TRADING_DAYS)
    project = work.loc[(work["date"] >= start_ts) & (work["date"] <= end_ts)]
    combined = pd.concat([warmup, project], ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)
