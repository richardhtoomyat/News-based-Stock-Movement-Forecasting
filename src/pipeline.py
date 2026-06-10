"""End-to-end data pipeline for selected tickers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import (
    BENCHMARK_TICKERS,
    COMBINED_PROCESSED_CSV,
    COMBINED_PROCESSED_PARQUET,
    END_DATE,
    PROCESSED_DIR,
    START_DATE,
    ensure_data_dirs,
    processed_file_path,
    trim_raw_for_storage,
    trim_to_project_period,
)
from src.data_io import (
    benchmark_raw_exists,
    count_warmup_rows,
    create_or_update_run_manifest,
    download_extended_raw_with_warmup,
    get_company_name,
    has_sufficient_warmup,
    load_ticker_raw_data,
    save_ticker_raw_data,
    validate_raw_data,
)
from src.features import build_processed_dataset, calculate_benchmark_returns


@dataclass
class PipelineResult:
    """Result of running the pipeline for a selected ticker."""

    success: bool
    ticker: str
    messages: list[str]
    raw_row_count: int = 0
    processed_row_count: int = 0
    warmup_row_count: int = 0
    sufficient_warmup: bool = True
    raw_date_min: str | None = None
    raw_date_max: str | None = None
    raw_parquet_path: str | None = None
    raw_csv_path: str | None = None
    processed_parquet_path: str | None = None
    processed_csv_path: str | None = None
    combined_parquet_path: str | None = None
    combined_csv_path: str | None = None
    processed_df: pd.DataFrame | None = None


def _warmup_warning_message() -> str:
    return (
        f"Warning: Fewer than 20 trading days of history before {START_DATE}; "
        "rolling features on early project dates may be missing."
    )


def _ensure_benchmark_raw(ticker: str, messages: list[str]) -> pd.DataFrame | None:
    """
    Return extended benchmark raw data for feature calculation.

    Raw files store 20-day warmup plus project period. Cached files are reused when valid.
    """
    fetched_at = datetime.now(timezone.utc)

    if benchmark_raw_exists(ticker):
        cached = load_ticker_raw_data(ticker)
        if cached is not None and has_sufficient_warmup(cached):
            return cached

    extended_raw = download_extended_raw_with_warmup(ticker, fetched_at)
    if extended_raw is None or extended_raw.empty:
        messages.append(f"No data was returned by yfinance for benchmark {ticker}.")
        if benchmark_raw_exists(ticker):
            return load_ticker_raw_data(ticker)
        return None

    if not has_sufficient_warmup(extended_raw):
        messages.append(f"{_warmup_warning_message()} (benchmark {ticker})")

    project_slice = trim_to_project_period(extended_raw)
    is_valid, validation_msgs = validate_raw_data(project_slice)
    if not is_valid:
        messages.extend(validation_msgs)
        if has_sufficient_warmup(extended_raw):
            return extended_raw
        if benchmark_raw_exists(ticker):
            return load_ticker_raw_data(ticker)
        return None

    raw_to_save = trim_raw_for_storage(extended_raw)
    save_ticker_raw_data(raw_to_save, ticker)
    return extended_raw


def save_ticker_processed_data(df: pd.DataFrame, ticker: str) -> tuple[Path, Path]:
    """Save processed market features as Parquet and CSV."""
    ensure_data_dirs()
    ticker = ticker.upper()
    parquet_path = processed_file_path(ticker, "parquet")
    csv_path = processed_file_path(ticker, "csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return parquet_path, csv_path


def update_combined_processed_dataset() -> tuple[Path | None, Path | None, int]:
    """
    Rebuild combined dataset by stacking all per-ticker processed files.
    Excludes combined file itself and benchmark-only files if any.
    """
    ensure_data_dirs()
    frames: list[pd.DataFrame] = []

    for path in sorted(PROCESSED_DIR.glob("*_market_features.parquet")):
        if path.name.startswith("combined_"):
            continue
        name = path.stem.replace("_market_features", "")
        if name.upper() in {b.upper() for b in BENCHMARK_TICKERS}:
            continue
        df = pd.read_parquet(path)
        frames.append(df)

    if not frames:
        return None, None, 0

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    combined.to_parquet(COMBINED_PROCESSED_PARQUET, index=False)
    combined.to_csv(COMBINED_PROCESSED_CSV, index=False)

    return COMBINED_PROCESSED_PARQUET, COMBINED_PROCESSED_CSV, len(combined)


def run_pipeline_for_selected_ticker(ticker: str) -> PipelineResult:
    """
    Full pipeline: download primary + cached benchmarks, process, save, update combined.
    """
    ticker = ticker.strip().upper()
    messages: list[str] = []
    fetched_at = datetime.now(timezone.utc)

    if not ticker:
        return PipelineResult(success=False, ticker="", messages=["Please enter a valid ticker symbol."])

    if START_DATE > END_DATE:
        return PipelineResult(
            success=False,
            ticker=ticker,
            messages=["The date range is invalid."],
        )

    extended_raw_df = download_extended_raw_with_warmup(ticker, fetched_at)
    if extended_raw_df is None or extended_raw_df.empty:
        msgs = ["No data was returned by yfinance."]
        create_or_update_run_manifest(
            selected_ticker=ticker,
            raw_output_file=None,
            processed_output_file=None,
            row_count_raw=0,
            row_count_processed=0,
            status="failure",
            notes="; ".join(msgs),
            fetched_at_utc=fetched_at,
        )
        return PipelineResult(success=False, ticker=ticker, messages=msgs)

    sufficient_warmup = has_sufficient_warmup(extended_raw_df)
    if not sufficient_warmup:
        messages.append(_warmup_warning_message())

    trimmed_raw_df = trim_to_project_period(extended_raw_df)
    is_valid, validation_msgs = validate_raw_data(trimmed_raw_df)
    if not is_valid:
        create_or_update_run_manifest(
            selected_ticker=ticker,
            raw_output_file=None,
            processed_output_file=None,
            row_count_raw=0,
            row_count_processed=0,
            status="failure",
            notes="; ".join(validation_msgs),
            fetched_at_utc=fetched_at,
        )
        return PipelineResult(success=False, ticker=ticker, messages=validation_msgs)

    raw_to_save = trim_raw_for_storage(extended_raw_df)

    try:
        raw_parquet, raw_csv = save_ticker_raw_data(raw_to_save, ticker)
    except Exception as exc:
        msg = f"The selected ticker file could not be saved: {exc}"
        create_or_update_run_manifest(
            selected_ticker=ticker,
            raw_output_file=None,
            processed_output_file=None,
            row_count_raw=len(raw_to_save),
            row_count_processed=0,
            status="failure",
            notes=msg,
            fetched_at_utc=fetched_at,
        )
        return PipelineResult(success=False, ticker=ticker, messages=[msg])

    ppa_raw = _ensure_benchmark_raw("PPA", messages)
    spy_raw = _ensure_benchmark_raw("SPY", messages)

    ppa_returns = calculate_benchmark_returns(ppa_raw, "PPA") if ppa_raw is not None else None
    spy_returns = calculate_benchmark_returns(spy_raw, "SPY") if spy_raw is not None else None

    if ppa_raw is None:
        messages.append("Warning: PPA benchmark data unavailable; benchmark features will be missing.")
    if spy_raw is None:
        messages.append("Warning: SPY benchmark data unavailable; benchmark features will be missing.")

    company_name = get_company_name(ticker)
    processed_full = build_processed_dataset(
        extended_raw_df, ticker, company_name, ppa_returns, spy_returns
    )
    processed_df = trim_to_project_period(processed_full)

    try:
        proc_parquet, proc_csv = save_ticker_processed_data(processed_df, ticker)
    except Exception as exc:
        msg = f"Processed file could not be saved: {exc}"
        create_or_update_run_manifest(
            selected_ticker=ticker,
            raw_output_file=str(raw_parquet),
            processed_output_file=None,
            row_count_raw=len(raw_to_save),
            row_count_processed=0,
            status="failure",
            notes=msg,
            fetched_at_utc=fetched_at,
        )
        return PipelineResult(
            success=False,
            ticker=ticker,
            messages=[msg],
            raw_row_count=len(raw_to_save),
            raw_parquet_path=str(raw_parquet),
            raw_csv_path=str(raw_csv),
        )

    combined_parquet, combined_csv, combined_rows = update_combined_processed_dataset()

    warmup_count = count_warmup_rows(raw_to_save)
    raw_dates = pd.to_datetime(raw_to_save["date"])

    create_or_update_run_manifest(
        selected_ticker=ticker,
        raw_output_file=str(raw_parquet),
        processed_output_file=str(proc_parquet),
        row_count_raw=len(raw_to_save),
        row_count_processed=len(processed_df),
        status="success",
        notes="",
        fetched_at_utc=fetched_at,
    )

    return PipelineResult(
        success=True,
        ticker=ticker,
        messages=messages,
        raw_row_count=len(raw_to_save),
        processed_row_count=len(processed_df),
        warmup_row_count=warmup_count,
        sufficient_warmup=sufficient_warmup,
        raw_date_min=raw_dates.min().strftime("%Y-%m-%d"),
        raw_date_max=raw_dates.max().strftime("%Y-%m-%d"),
        raw_parquet_path=str(raw_parquet),
        raw_csv_path=str(raw_csv),
        processed_parquet_path=str(proc_parquet),
        processed_csv_path=str(proc_csv),
        combined_parquet_path=str(combined_parquet) if combined_parquet else None,
        combined_csv_path=str(combined_csv) if combined_csv else None,
        processed_df=processed_df,
    )
