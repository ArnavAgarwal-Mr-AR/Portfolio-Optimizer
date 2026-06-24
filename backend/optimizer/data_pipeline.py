import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from fredapi import Fred

from optimizer.config import config
from optimizer.exceptions import DataDownloadError

logger = logging.getLogger(__name__)

def fetch_prices(tickers, start_date, end_date, use_cache=True):
    """
    Fetch daily asset closing prices from Yahoo Finance.
    
    Parameters:
    - tickers: list of tickers
    - start_date: str, YYYY-MM-DD
    - end_date: str, YYYY-MM-DD
    - use_cache: bool, whether to use locally cached data if available
    
    Returns:
    - pd.DataFrame: historical prices for specified tickers
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty.")
        
    cache_file = config.cache_dir / "raw_prices.csv"
    
    # Try to load from cache
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Check if cache contains all requested tickers and date range
            ticker_match = all(t in df.columns for t in tickers)
            date_match = df.index.min() <= pd.to_datetime(start_date) and df.index.max() >= pd.to_datetime(end_date)
            
            if ticker_match and date_match:
                logger.info(f"Loaded asset prices from cache: {cache_file}")
                # Slice date range and select relevant tickers
                sliced_df = df.loc[start_date:end_date, tickers]
                return sliced_df
            else:
                logger.info("Cache missed (missing tickers or dates). Downloading new data...")
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}. Proceeding to download...")

    # Download from yfinance
    logger.info(f"Downloading historical price data for {tickers} from {start_date} to {end_date}...")
    try:
        # yfinance download
        # auto_adjust=True merges dividends and splits into the close price
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
        
        if data.empty:
            raise DataDownloadError("Yahoo Finance returned an empty DataFrame.")
            
        # If only one ticker is downloaded, yfinance might structure it differently
        if len(tickers) == 1:
            prices = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            prices = data["Close"]
            
        # Verify all tickers are present and contain valid data
        for t in tickers:
            if t not in prices.columns:
                raise DataDownloadError(f"Ticker {t} was not returned by Yahoo Finance.")
            if prices[t].isnull().all():
                raise DataDownloadError(f"Ticker {t} contains only NaN values. Download failed or ticker is invalid.")
            
        # Clean up column names/index
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        
        # Save to cache
        if use_cache:
            try:
                # Merge with existing cache if it exists, to avoid losing historical data
                if cache_file.exists():
                    try:
                        existing = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        combined = prices.combine_first(existing)
                        combined.to_csv(cache_file)
                    except Exception:
                        prices.to_csv(cache_file)
                else:
                    prices.to_csv(cache_file)
                logger.info(f"Saved asset prices to cache: {cache_file}")
            except Exception as e:
                logger.warning(f"Could not save to cache: {e}")
                
        return prices[tickers]
    except Exception as e:
        if not isinstance(e, DataDownloadError):
            raise DataDownloadError(f"Failed to download asset prices from Yahoo Finance: {e}") from e
        raise

def fetch_risk_free_rate(start_date, end_date, use_cache=True):
    """
    Fetch annualized risk-free interest rates.
    Attempts to download from FRED using FRED_API_KEY.
    Falls back to Yahoo Finance ^IRX (13-Week Treasury Bill yield) if FRED fails or key is missing.
    
    Parameters:
    - start_date: str, YYYY-MM-DD
    - end_date: str, YYYY-MM-DD
    - use_cache: bool, whether to use cached data
    
    Returns:
    - pd.Series: daily annualized risk-free rate in decimal form (e.g. 0.04 for 4%)
    """
    cache_file = config.cache_dir / "raw_rf_rate.csv"
    
    # Try load from cache
    if use_cache and cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            date_match = df.index.min() <= pd.to_datetime(start_date) and df.index.max() >= pd.to_datetime(end_date)
            if date_match:
                logger.info(f"Loaded risk-free rate from cache: {cache_file}")
                return df.loc[start_date:end_date, "rf_rate"]
        except Exception as e:
            logger.warning(f"Failed to read risk-free rate cache: {e}. Downloading...")

    # Fetch from FRED if API key is provided
    rf_series = None
    if config.fred_api_key:
        logger.info(f"Attempting to fetch risk-free rate ({config.fred_series_id}) from FRED...")
        try:
            fred = Fred(api_key=config.fred_api_key)
            # Retrieve from FRED (yield is in percentages, e.g. 4.5 for 4.5%)
            raw_rf = fred.get_series(config.fred_series_id, observation_start=start_date, observation_end=end_date)
            if raw_rf is not None and not raw_rf.empty:
                rf_series = pd.Series(raw_rf, name="rf_rate") / 100.0  # Convert to decimal
                rf_series.index = pd.to_datetime(rf_series.index)
                logger.info("Successfully fetched risk-free rate from FRED.")
        except Exception as e:
            logger.warning(f"Failed to fetch from FRED API (Error: {e}). Falling back to Yahoo Finance...")

    # Fallback to Yahoo Finance (^IRX) if FRED failed or was skipped
    if rf_series is None:
        fallback_ticker = config.fred_fallback_ticker
        logger.info(f"FRED unavailable. Fetching risk-free rate fallback ({fallback_ticker}) from Yahoo Finance...")
        try:
            data = yf.download(fallback_ticker, start=start_date, end=end_date, auto_adjust=True)
            if data.empty:
                raise DataDownloadError(f"Yahoo Finance returned empty data for fallback ticker {fallback_ticker}")
            
            # Squeeze to handle potential MultiIndex or 1-column DataFrame from yfinance
            close_series = data["Close"].squeeze()
            rf_series = pd.Series(close_series, name="rf_rate") / 100.0
            rf_series.index = pd.to_datetime(rf_series.index)
            logger.info(f"Successfully fetched fallback risk-free rate from Yahoo Finance using {fallback_ticker}.")
        except Exception as e:
            logger.warning(f"Failed to fetch risk-free rate from Yahoo Finance fallback: {e}")
            
    # Final fallback to static default if both FRED and Yahoo fail
    if rf_series is None or rf_series.empty:
        logger.warning(f"Could not retrieve risk-free rate. Using static default of {config.default_risk_free_rate * 100}%")
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        rf_series = pd.Series(config.default_risk_free_rate, index=dates, name="rf_rate")

    # Clean up and ensure sorted index
    rf_series = rf_series.sort_index()
    
    # Save cache
    if use_cache:
        try:
            df_to_cache = pd.DataFrame({"rf_rate": rf_series})
            if cache_file.exists():
                try:
                    existing = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    combined = df_to_cache.combine_first(existing)
                    combined.to_csv(cache_file)
                except Exception:
                    df_to_cache.to_csv(cache_file)
            else:
                df_to_cache.to_csv(cache_file)
        except Exception as e:
            logger.warning(f"Could not cache risk-free rate: {e}")
            
    return rf_series

def clean_data(prices_df):
    """
    Clean historical asset prices DataFrame by handling missing values.
    
    Parameters:
    - prices_df: pd.DataFrame
    
    Returns:
    - pd.DataFrame: cleaned prices
    """
    # Drop rows where all elements are NaN
    cleaned = prices_df.dropna(how="all")
    # Forward fill missing values (price remains constant if market is closed or no trade)
    # Then backward fill for any leading NaNs
    cleaned = cleaned.ffill().bfill()
    
    # Check if there are still NaNs
    if cleaned.isnull().values.any():
        logger.warning("Prices still contain NaN values after cleaning.")
        
    return cleaned

def align_data(prices, rf_rate):
    """
    Align asset prices and risk-free rate onto the same index (asset price trading days).
    
    Parameters:
    - prices: pd.DataFrame
    - rf_rate: pd.Series
    
    Returns:
    - tuple: (cleaned_prices, aligned_rf_rate)
    """
    prices = clean_data(prices)
    # Reindex risk-free rate to match prices index, forward filling missing daily rates
    rf_aligned = rf_rate.reindex(prices.index).ffill().bfill()
    return prices, rf_aligned
