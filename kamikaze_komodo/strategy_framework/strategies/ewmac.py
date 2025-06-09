# FILE: kamikaze_komodo/strategy_framework/strategies/ewmac.py
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class EWMACStrategy(BaseStrategy):
    """
    Implements a stateless Exponential Weighted Moving Average Crossover (EWMAC) strategy.
    Generates entry signals only. Exits are handled by the StopManager.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        self.short_window = int(self.params.get('shortwindow', 12))
        self.long_window = int(self.params.get('longwindow', 26))
        self.atr_period = int(self.params.get('atr_period', 14))
        
        logger.info(
            f"Initialized EWMACStrategy for {symbol} ({timeframe}) "
            f"with Short EMA: {self.short_window}, Long EMA: {self.long_window}. "
            f"Shorting Enabled: {self.enable_shorting}."
        )

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if data_df.empty or len(data_df) < self.long_window :
            return pd.DataFrame()
        df = data_df.copy()
        df[f'ema_short'] = ta.ema(df['close'], length=self.short_window)
        df[f'ema_long'] = ta.ema(df['close'], length=self.long_window)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if len(df) >= self.atr_period:
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        return df

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        logger.warning("generate_signals is not the primary method for this strategy; logic is in on_bar_data.")
        return pd.Series(index=data.index, dtype='object').fillna(SignalType.HOLD)

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        if len(self.data_history) < self.long_window + 1:
            return SignalType.HOLD

        df = self._calculate_indicators(self.data_history)
        if df.empty or len(df) < 2 or 'ema_short' not in df.columns or 'ema_long' not in df.columns:
            return SignalType.HOLD

        if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]):
            bar_data.atr = df['atr'].iloc[-1]

        latest_ema_short = df['ema_short'].iloc[-1]
        prev_ema_short = df['ema_short'].iloc[-2]
        latest_ema_long = df['ema_long'].iloc[-1]
        prev_ema_long = df['ema_long'].iloc[-2]

        if any(pd.isna(v) for v in [latest_ema_short, prev_ema_short, latest_ema_long, prev_ema_long]):
            return SignalType.HOLD

        is_golden_cross = latest_ema_short > latest_ema_long and prev_ema_short <= prev_ema_long
        is_death_cross = latest_ema_short < latest_ema_long and prev_ema_short >= prev_ema_long

        if is_golden_cross:
            return SignalType.LONG
        elif is_death_cross and self.enable_shorting:
            return SignalType.SHORT
            
        return SignalType.HOLD