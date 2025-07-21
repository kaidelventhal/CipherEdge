# FILE: kamikaze_komodo/ml_models/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Optional, List

from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
    """Adds lag features for returns."""
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