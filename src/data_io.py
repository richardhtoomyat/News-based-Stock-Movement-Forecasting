"""Data download, standardization, raw file I/O, and run manifest."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import (
    ACTIONS,
    AUTO_ADJUST,
    BENCHMARK_TICKERS,
    DATA_SOURCE,
    END_DATE,
    INTERVAL,
    MANIFEST_COLUMNS,
    PROCESSED_DIR,
    RAW_COLUMNS,
    RAW_PRICES_DIR,
    RUN_MANIFEST_PATH,
    START_DATE,
    WARMUP_MAX_RETRIES,
    WARMUP_RETRY_CALENDAR_DAYS,
    WARMUP_TRADING_DAYS,
    ensure_data_dirs,
    get_download_start_date,
    raw_file_path,
)


def download_yfinance_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """
    Download daily yfinance data for a single ticker.

    Uses auto_adjust=False and actions=True per spec.
    End date is exclusive in yfinance, so we add one day.
    """
    end_exclusive = end + timedelta(days=1)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end_exclusive.isoformat(),
        interval=INTERVAL,
        auto_adjust=AUTO_ADJUST,
        actions=ACTIONS,
        progress=False,
    )
    return df


def standardize_yfinance_output(
    df: pd.DataFrame,
    ticker: str,
    fetched_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """
    Convert yfinance download output to long-format raw schema.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    work = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [col[0] if isinstance(col, tuple) else col for col in work.columns]

    work = work.reset_index()

    # Normalize date column name
    date_col = None
    for candidate in ("Date", "Datetime", "date"):
        if candidate in work.columns:
            date_col = candidate
            break
    if date_col is None:
        return pd.DataFrame(columns=RAW_COLUMNS)

    work = work.rename(columns={date_col: "date"})
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()

    # Map yfinance column names to our schema
    col_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
        "Dividends": "dividends",
        "Stock Splits": "stock_splits",
    }
    work = work.rename(columns=col_map)

    for col in ("open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits"):
        if col not in work.columns:
            work[col] = pd.NA

    work["ticker"] = ticker.upper()
    work["source"] = DATA_SOURCE
    work["fetched_at_utc"] = (fetched_at_utc or datetime.now(timezone.utc)).isoformat()

    result = work[RAW_COLUMNS].copy()
    result = result.sort_values("date").reset_index(drop=True)
    return result


def get_company_name(ticker: str) -> str:
    """Fetch company name from yfinance metadata; fallback to ticker symbol."""
    try:
        info = yf.Ticker(ticker).info
        if info:
            name = info.get("longName") or info.get("shortName")
            if name:
                return str(name)
    except Exception:
        pass
    return ticker.upper()


def validate_raw_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate raw data before saving.
    Returns (is_valid, list of warning/error messages).
    """
    messages: list[str] = []
    if df is None or df.empty:
        messages.append("No data was returned by yfinance.")
        return False, messages

    if START_DATE > END_DATE:
        messages.append("The date range is invalid.")
        return False, messages

    if "adj_close" not in df.columns or df["adj_close"].isna().all():
        messages.append("Adj Close is missing.")
        return False, messages

    if "volume" not in df.columns or df["volume"].isna().all():
        messages.append("Volume is missing.")
        return False, messages

    return True, messages


def save_ticker_raw_data(df: pd.DataFrame, ticker: str) -> tuple[Path, Path]:
    """
    Save raw data as both Parquet and CSV.
    Returns (parquet_path, csv_path).
    """
    ensure_data_dirs()
    ticker = ticker.upper()
    parquet_path = raw_file_path(ticker, "parquet")
    csv_path = raw_file_path(ticker, "csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return parquet_path, csv_path


def load_ticker_raw_data(ticker: str) -> pd.DataFrame | None:
    """Load raw data from parquet if it exists."""
    path = raw_file_path(ticker.upper(), "parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def benchmark_raw_exists(ticker: str) -> bool:
    """Check if benchmark raw parquet file exists."""
    return raw_file_path(ticker.upper(), "parquet").exists()


def has_sufficient_warmup(df: pd.DataFrame) -> bool:
    """Return True if dataframe has enough trading days before START_DATE for rolling features."""
    if df is None or df.empty:
        return False
    dates = pd.to_datetime(df["date"])
    start_ts = pd.Timestamp(START_DATE)
    rows_before = int((dates < start_ts).sum())
    return rows_before >= WARMUP_TRADING_DAYS


def count_warmup_rows(df: pd.DataFrame) -> int:
    """Count trading rows strictly before START_DATE."""
    if df is None or df.empty:
        return 0
    dates = pd.to_datetime(df["date"])
    return int((dates < pd.Timestamp(START_DATE)).sum())


def download_extended_raw_with_warmup(ticker: str, fetched_at: datetime) -> pd.DataFrame:
    """
    Download daily data with enough pre-START_DATE history for rolling features.

    Retries with an earlier start date if the first download lacks 20 warmup trading days.
    """
    start = get_download_start_date()
    result = pd.DataFrame(columns=RAW_COLUMNS)

    for attempt in range(WARMUP_MAX_RETRIES + 1):
        raw_download = download_yfinance_data(ticker, start, END_DATE)
        result = standardize_yfinance_output(raw_download, ticker, fetched_at)
        if has_sufficient_warmup(result):
            return result
        if attempt < WARMUP_MAX_RETRIES:
            start = start - timedelta(days=WARMUP_RETRY_CALENDAR_DAYS)

    return result


def create_or_update_run_manifest(
    selected_ticker: str,
    raw_output_file: str | None,
    processed_output_file: str | None,
    row_count_raw: int,
    row_count_processed: int,
    status: str,
    notes: str = "",
    fetched_at_utc: datetime | None = None,
) -> None:
    """Append a new row to the run manifest CSV."""
    ensure_data_dirs()
    fetched_at = (fetched_at_utc or datetime.now(timezone.utc)).isoformat()

    row = {
        "run_id": str(uuid.uuid4()),
        "source": DATA_SOURCE,
        "selected_ticker": selected_ticker.upper(),
        "benchmark_tickers": ",".join(BENCHMARK_TICKERS),
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "interval": INTERVAL,
        "auto_adjust": AUTO_ADJUST,
        "actions": ACTIONS,
        "fetched_at_utc": fetched_at,
        "raw_output_file": raw_output_file or "",
        "processed_output_file": processed_output_file or "",
        "row_count_raw": row_count_raw,
        "row_count_processed": row_count_processed,
        "status": status,
        "notes": notes,
    }

    if RUN_MANIFEST_PATH.exists():
        manifest = pd.read_csv(RUN_MANIFEST_PATH)
    else:
        manifest = pd.DataFrame(columns=MANIFEST_COLUMNS)

    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest.to_csv(RUN_MANIFEST_PATH, index=False)


def list_available_primary_tickers() -> list[str]:
    """List primary tickers that have processed market feature files."""
    ensure_data_dirs()
    tickers: list[str] = []
    for path in sorted(PROCESSED_DIR.glob("*_market_features.parquet")):
        if path.name.startswith("combined_"):
            continue
        name = path.stem.replace("_market_features", "")
        if name.upper() not in {b.upper() for b in BENCHMARK_TICKERS}:
            tickers.append(name.upper())
    return sorted(set(tickers))
