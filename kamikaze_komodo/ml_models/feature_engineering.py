# FILE: kamikaze_komodo/ml_models/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Optional, List

from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
    """Adds lag features for returns."""
    # FIX: Added lag 20 to the default list to match model expectations.
    for lag in lags:
        df[f'log_return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
        df[f'close_change_lag_{lag}'] = df['close'].pct_change(lag)
    return df

def add_rolling_window_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Adds rolling window features like volatility."""
    if 'log_return_lag_1' not in df.columns:
        df = add_lag_features(df, lags=[1])
        
    for window in windows:
        df[f'volatility_{window}'] = df['log_return_lag_1'].rolling(window=window).std()
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds common technical indicators using pandas_ta."""
    try:
        import pandas_ta as ta
        df.ta.rsi(length=14, append=True, col_names=('RSI_14',))
        df.ta.macd(append=True, col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))
        df.ta.atr(length=14, append=True, col_names=('ATR_14',))
        bbands = ta.bbands(df['close'], length=20, std=2.0)
        if bbands is not None:
            df['bb_width'] = bbands['BBB_20_2.0']
            df['bb_percent'] = bbands['BBP_20_2.0']
    except ImportError:
        logger.warning("pandas_ta not installed. Skipping technical indicator features.")
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}", exc_info=True)
    return df

def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features based on sentiment score."""
    if 'sentiment_score' in df.columns:
        df['sentiment_sma_5'] = df['sentiment_score'].rolling(window=5).mean()
        df['sentiment_cumulative'] = df['sentiment_score'].cumsum()
    else:
        # Ensure columns exist even if no sentiment data is provided
        df['sentiment_score'] = 0.0
        df['sentiment_sma_5'] = 0.0
        df['sentiment_cumulative'] = 0.0
    return df

def add_funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features based on funding rates."""
    if 'funding_rate' in df.columns:
        df['funding_rate_sma_8'] = df['funding_rate'].rolling(window=8).mean()
    else:
        df['funding_rate'] = 0.0
        df['funding_rate_sma_8'] = 0.0
    return df

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sine/cosine transformations for cyclical time-based features."""
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not a DatetimeIndex. Cannot create cyclical time features.")
        return df
        
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df

def get_weights_ffd(d: float, thres: float) -> np.ndarray:
    """Helper to generate weights for fractional differentiation."""
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def fractional_differentiation(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """Computes fractionally differentiated series."""
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    df = pd.DataFrame({'original': series})
    df['frac_diff'] = np.nan
    
    for i in range(width, len(df)):
        series_slice = df.iloc[i - width:i + 1, 0].values.reshape(-1, 1)
        df.iloc[i, df.columns.get_loc('frac_diff')] = np.dot(w.T, series_slice)[0, 0]
        
    return df['frac_diff']

def add_fractional_diff_features(df: pd.DataFrame, d: float = 0.5, thres: float = 1e-4, column: str = 'close') -> pd.DataFrame:
    """Wrapper to apply fractional differentiation and add as a new column."""
    if column in df.columns:
        df[f'{column}_frac_diff'] = fractional_differentiation(df[column], d, thres)
        logger.info(f"Added fractional differentiation feature for column '{column}'.")
    else:
        logger.warning(f"Column '{column}' not found for fractional differentiation.")
    return df

def market_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features based on market microstructure (VWAP, OFI proxy)."""
    if not all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
        logger.warning("Microstructure features require 'high', 'low', 'close', 'volume' columns.")
        return df

    # VWAP approximation (true VWAP requires tick data)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tpv'] = df['typical_price'] * df['volume']
    
    # Use a rolling window for a more stable VWAP in a bar context
    rolling_tpv = df['tpv'].rolling(window=20).sum()
    rolling_volume = df['volume'].rolling(window=20).sum()
    df['vwap_proxy'] = rolling_tpv / rolling_volume
    
    df['vwap_deviation'] = ((df['close'] - df['vwap_proxy']) / df['vwap_proxy']) * 100

    # Order Flow Imbalance (OFI) proxy
    price_change = df['close'].diff()
    df['ofi_proxy'] = df['volume'] * np.sign(price_change)
    df['ofi_proxy_sma_10'] = df['ofi_proxy'].rolling(window=10).mean()

    df.drop(['typical_price', 'tpv'], axis=1, inplace=True, errors='ignore')
    return df

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wrapper to add all market microstructure features."""
    df = market_microstructure_features(df)
    logger.info("Added market microstructure features.")
    return df