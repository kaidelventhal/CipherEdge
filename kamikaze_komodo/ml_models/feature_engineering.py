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

def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds more sophisticated technical indicators using pandas_ta."""
    try:
        import pandas_ta as ta
        # Donchian Channels
        df.ta.donchian(append=True)
        # Ichimoku Cloud - returns multiple columns, we can keep the main ones
        ichimoku_df = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku_df is not None and isinstance(ichimoku_df, tuple) and len(ichimoku_df) > 0:
            # The result is often a tuple of the dataframe and the span text
            ichimoku_df = ichimoku_df[0] 
            df[['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26']] = ichimoku_df[['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26']]
        # Vortex Indicator
        df.ta.vortex(append=True)
        # On-Balance Volume (OBV)
        df.ta.obv(append=True)
        # Volume-Weighted Average Price (VWAP)
        df.ta.vwap(append=True)
    except ImportError:
        logger.warning("pandas_ta not installed. Skipping advanced technical indicator features.")
    except Exception as e:
        logger.error(f"Error calculating advanced technical indicators: {e}", exc_info=True)
    return df

def add_market_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features that describe the market's recent structure and behavior."""
    try:
        import pandas_ta as ta
        # Price vs. Moving Average
        sma_50 = ta.sma(df['close'], length=50)
        if sma_50 is not None and sma_50.gt(0).all():
            df['price_vs_sma50'] = df['close'] / sma_50
        
        # High-Low Range as a Percentage of Close
        df['high_low_range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
        
        # Distance from Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2.0)
        if bbands is not None:
            upper_band_col = f'BBU_{20}_{2.0}'
            lower_band_col = f'BBL_{20}_{2.0}'
            df['dist_from_upper_bb'] = df['close'] - bbands[upper_band_col]
            df['dist_from_lower_bb'] = df['close'] - bbands[lower_band_col]
    except ImportError:
        logger.warning("pandas_ta not installed. Skipping market structure features.")
    except Exception as e:
        logger.error(f"Error calculating market structure features: {e}", exc_info=True)
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