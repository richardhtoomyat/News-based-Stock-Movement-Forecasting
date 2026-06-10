"""Feature engineering: adjusted OHLC, market features, benchmarks, targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_COLUMNS


def calculate_adjusted_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create adjusted OHLC from raw close/adj_close.

    adjustment_factor = adj_close / close
    adj_open = open * adjustment_factor
    adj_high = high * adjustment_factor
    adj_low = low * adjustment_factor
    adj_close = adj_close (already adjusted)
    """
    work = df.copy()
    work = work.sort_values("date").reset_index(drop=True)

    close = work["close"].astype(float)
    adj_close_raw = work["adj_close"].astype(float)

    # Avoid division by zero
    adjustment_factor = np.where(close != 0, adj_close_raw / close, np.nan)

    work["adj_open"] = work["open"].astype(float) * adjustment_factor
    work["adj_high"] = work["high"].astype(float) * adjustment_factor
    work["adj_low"] = work["low"].astype(float) * adjustment_factor
    work["adj_close"] = adj_close_raw

    return work


def calculate_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate price, volume, volatility, and session features from adjusted OHLCV.
    Rolling features use only previous/current trading days within the ticker.
    """
    work = df.copy()
    work = work.sort_values("date").reset_index(drop=True)

    adj_close = work["adj_close"]
    adj_open = work["adj_open"]
    volume = work["volume"].astype(float)

    prev_adj_close = adj_close.shift(1)
    prev_volume = volume.shift(1)

    # Price return features
    work["stock_return_1d"] = adj_close / prev_adj_close - 1
    work["stock_return_3d"] = adj_close / adj_close.shift(3) - 1
    work["stock_return_5d"] = adj_close / adj_close.shift(5) - 1

    # Volume features
    work["volume_change_1d"] = volume / prev_volume - 1
    rolling_vol_5d = volume.rolling(window=5, min_periods=5).mean()
    work["volume_ratio_5d_avg"] = volume / rolling_vol_5d

    # Volatility and trend features
    work["volatility_5d"] = work["stock_return_1d"].rolling(window=5, min_periods=5).std()
    work["moving_average_5d"] = adj_close.rolling(window=5, min_periods=5).mean()
    work["moving_average_20d"] = adj_close.rolling(window=20, min_periods=20).mean()
    work["price_vs_ma20"] = adj_close / work["moving_average_20d"] - 1
    rolling_20d_high = adj_close.rolling(window=20, min_periods=20).max()
    work["drawdown_20d"] = adj_close / rolling_20d_high - 1

    # Session-style daily features
    work["overnight_gap_return"] = adj_open / prev_adj_close - 1
    work["open_to_close_return"] = adj_close / adj_open - 1

    return work


def calculate_benchmark_returns(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Calculate benchmark return columns from raw benchmark data.

    For PPA: ppa_return_1d, ppa_return_5d
    For SPY: spy_return_1d
    """
    work = calculate_adjusted_ohlc(raw_df)
    work = work.sort_values("date").reset_index(drop=True)

    adj_close = work["adj_close"]
    returns_1d = adj_close / adj_close.shift(1) - 1
    returns_5d = adj_close / adj_close.shift(5) - 1

    result = pd.DataFrame({"date": work["date"]})

    ticker_upper = ticker.upper()
    if ticker_upper == "PPA":
        result["ppa_return_1d"] = returns_1d
        result["ppa_return_5d"] = returns_5d
    elif ticker_upper == "SPY":
        result["spy_return_1d"] = returns_1d

    return result


def merge_benchmark_features(
    primary_df: pd.DataFrame,
    ppa_returns: pd.DataFrame | None,
    spy_returns: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge PPA and SPY benchmark returns into primary ticker data by date."""
    work = primary_df.copy()

    if ppa_returns is not None and not ppa_returns.empty:
        work = work.merge(ppa_returns, on="date", how="left")
    else:
        work["ppa_return_1d"] = np.nan
        work["ppa_return_5d"] = np.nan

    if spy_returns is not None and not spy_returns.empty:
        work = work.merge(spy_returns, on="date", how="left")
    else:
        work["spy_return_1d"] = np.nan

    # Relative features
    work["relative_return_1d"] = work["stock_return_1d"] - work["ppa_return_1d"]
    work["relative_return_5d"] = work["stock_return_5d"] - work["ppa_return_5d"]
    work["market_adjusted_return_1d"] = work["stock_return_1d"] - work["spy_return_1d"]

    return work


def create_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create future target labels. Missing next-day values yield null targets, not 0.
    """
    work = df.copy()
    work = work.sort_values("date").reset_index(drop=True)

    adj_close = work["adj_close"]
    adj_open = work["adj_open"]
    ppa_return_1d = work["ppa_return_1d"]

    next_adj_close = adj_close.shift(-1)
    next_adj_open = adj_open.shift(-1)
    next_ppa_return = ppa_return_1d.shift(-1)

    work["next_stock_return"] = next_adj_close / adj_close - 1
    work["next_relative_return"] = work["next_stock_return"] - next_ppa_return
    work["next_day_open_gap"] = next_adj_open / adj_close - 1
    work["next_day_open_to_close_return"] = next_adj_close / next_adj_open - 1

    # Binary targets with nullable integer type
    work["target_up_next_day"] = pd.array(
        [pd.NA if pd.isna(v) else (1 if v > 0 else 0) for v in work["next_stock_return"]],
        dtype="Int64",
    )
    work["target_outperform_ppa_next_day"] = pd.array(
        [pd.NA if pd.isna(v) else (1 if v > 0 else 0) for v in work["next_relative_return"]],
        dtype="Int64",
    )

    return work


def build_processed_dataset(
    raw_df: pd.DataFrame,
    ticker: str,
    company_name: str,
    ppa_returns: pd.DataFrame | None,
    spy_returns: pd.DataFrame | None,
) -> pd.DataFrame:
    """Full feature pipeline for a single primary ticker.

    Expects extended raw history (warmup rows before START_DATE). Caller should
    trim the result to the project period before saving.
    """
    work = calculate_adjusted_ohlc(raw_df)
    work = calculate_market_features(work)
    work = merge_benchmark_features(work, ppa_returns, spy_returns)
    work = create_target_labels(work)

    work["ticker"] = ticker.upper()
    work["company_name"] = company_name

    # Select and order final columns
    available = [c for c in PROCESSED_COLUMNS if c in work.columns]
    result = work[available].copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)

    return result
