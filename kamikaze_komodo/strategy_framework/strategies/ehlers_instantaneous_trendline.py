# kamikaze_komodo/strategy_framework/strategies/ehlers_instantaneous_trendline.py
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class EhlersInstantaneousTrendlineStrategy(BaseStrategy):
    """
    Implements a stateless Ehlers' Instantaneous Trendline strategy.
    It generates entry signals only. Exits are handled by the StopManager.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        self.it_lag_trigger = int(self.params.get('it_lag_trigger', 1))
        self.atr_period = int(self.params.get('atr_period', 14))

        logger.info(
            f"Initialized EhlersInstantaneousTrendlineStrategy for {symbol} ({timeframe}) "
            f"with IT Lag Trigger: {self.it_lag_trigger}. Shorting Enabled: {self.enable_shorting}"
        )

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if data_df.empty or 'close' not in data_df.columns or len(data_df) < 3:
            return pd.DataFrame()

        df = data_df.copy()
        close_prices = df['close']
        # The core IT is a 3-bar filter, but we apply it over the series
        it_values = (close_prices + 2 * close_prices.shift(1) + close_prices.shift(2)) / 4
        df['it'] = it_values
        df['it_trigger'] = df['it'].shift(self.it_lag_trigger)
        
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if len(df) >= self.atr_period:
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        return df

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        logger.warning("generate_signals is not the primary method for this strategy; logic is in on_bar_data.")
        return pd.Series(index=data.index, dtype='object').fillna(SignalType.HOLD)

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        if len(self.data_history) < 3 + self.it_lag_trigger:
            return SignalType.HOLD

        df = self._calculate_indicators(self.data_history)
        if df.empty or len(df) < 2 or 'it' not in df.columns or 'it_trigger' not in df.columns:
            return SignalType.HOLD
            
        if 'atr' in df.columns and pd.notna(df['atr'].iloc[-1]):
            bar_data.atr = df['atr'].iloc[-1]

        latest_it = df['it'].iloc[-1]
        prev_it = df['it'].iloc[-2]
        latest_it_trigger = df['it_trigger'].iloc[-1]
        prev_it_trigger = df['it_trigger'].iloc[-2]

        if any(pd.isna(v) for v in [latest_it, prev_it, latest_it_trigger, prev_it_trigger]):
            return SignalType.HOLD

        is_bullish_cross = latest_it > latest_it_trigger and prev_it <= prev_it_trigger
        is_bearish_cross = latest_it < latest_it_trigger and prev_it >= prev_it_trigger

        if is_bullish_cross:
            return SignalType.LONG
        elif is_bearish_cross and self.enable_shorting:
            return SignalType.SHORT
        
        return SignalType.HOLD