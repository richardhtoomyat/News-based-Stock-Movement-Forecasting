# News-Based Stock Movement Forecasting

Research prototype for collecting and processing daily stock market data from yfinance.

## Features

- **Data Collection**: Pull daily OHLCV data for any user-selected primary ticker plus benchmark tickers (PPA, SPY)
- **Feature Engineering**: Calculate returns, volatility, volume, benchmark-relative features, and target labels
- **Stock Chart Explorer**: Interactive Plotly charts with overlays, normalization, and range selection

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Project Structure

```text
app.py                  # Streamlit entrypoint
src/
  config.py             # Fixed dates, paths, schemas
  data_io.py            # Download, raw I/O, manifest
  features.py           # Feature calculations
  pipeline.py           # End-to-end pipeline
  charts.py             # Plotly chart builder
  ui/                   # Streamlit tab components
data/
  raw/prices/           # Raw yfinance data
  processed/            # Processed market features
```

## Fixed Date Range

- Start: 2024-05-01
- End: 2025-05-31

## Benchmark Tickers

- **PPA** — Defense ETF benchmark
- **SPY** — Broad market benchmark

Benchmark raw data is cached after first download and reused on subsequent pulls.
