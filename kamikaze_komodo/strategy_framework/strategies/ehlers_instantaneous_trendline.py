# FILE: kamikaze_komodo/strategy_framework/strategies/ehlers_instantaneous_trendline.py
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
    Implements Ehlers' Instantaneous Trendline strategy.
    
    NOTE ON PERFORMANCE: This is a very fast-reacting (low-lag) trend indicator.
    While this is good for catching trends early, it makes the strategy extremely
    sensitive and prone to "whipsaws" (numerous false signals) in sideways or
    choppy markets. "Too good to be true" backtest results are often a sign of
    overfitting to a specific historical period with smooth trends or, more commonly,
    unrealistically low transaction cost (slippage/commission) assumptions.
    Robustness testing, such as Walk-Forward Optimization, is crucial for this strategy.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        self.it_lag_trigger = int(self.params.get('it_lag_trigger', 1))
        self.atr_period = int(self.params.get('atr_period', 14))

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if df.empty or len(df) < 3:
            return pd.DataFrame()

        # Ehlers' Instantaneous Trendline calculation
        close = df['close']
        df['it'] = (close + 2 * close.shift(1) + close.shift(2)) / 4
        df['it_trigger'] = df['it'].shift(self.it_lag_trigger)
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        # Vectorized Signal Conditions
        df['bullish_cross'] = (df['it'] > df['it_trigger']) & (df['it'].shift(1) <= df['it_trigger'].shift(1))
        df['bearish_cross'] = (df['it'] < df['it_trigger']) & (df['it'].shift(1) >= df['it_trigger'].shift(1))
        
        return df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        signal = SignalType.HOLD

        is_bullish_cross = getattr(current_bar, 'bullish_cross', False)
        is_bearish_cross = getattr(current_bar, 'bearish_cross', False)

        if self.current_position_status == SignalType.LONG:
            if is_bearish_cross:
                signal = SignalType.CLOSE_LONG
                self.current_position_status = None
        elif self.current_position_status == SignalType.SHORT:
            if is_bullish_cross:
                signal = SignalType.CLOSE_SHORT
                self.current_position_status = None
        else: # No position
            if is_bullish_cross:
                signal = SignalType.LONG
                self.current_position_status = SignalType.LONG
            elif is_bearish_cross and self.enable_shorting:
                signal = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
                
        return signal