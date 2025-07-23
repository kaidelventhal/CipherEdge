import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from cipher_edge.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from cipher_edge.core.enums import SignalType
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class FundingRateStrategy(BaseStrategy):
    """
    Implements a contrarian strategy based on perpetual futures funding rates.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.lookback_period = int(self.params.get('lookback_period', 14))
        self.short_threshold = float(self.params.get('short_threshold', 0.0005))
        self.long_threshold = float(self.params.get('long_threshold', -0.0005))
        self.exit_threshold_short = float(self.params.get('exit_threshold_short', 0.0001))
        self.exit_threshold_long = float(self.params.get('exit_threshold_long', -0.0001))
        self.atr_period = int(self.params.get('atr_period', 14))

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if 'funding_rate' not in df.columns or df.empty:
            return pd.DataFrame()
            
        df['funding_rate_ma'] = df['funding_rate'].rolling(window=self.lookback_period).mean()

        if len(df) >= self.atr_period:
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        df['long_entry'] = df['funding_rate_ma'] < self.long_threshold
        df['short_entry'] = df['funding_rate_ma'] > self.short_threshold
        
        df['long_exit'] = df['funding_rate_ma'] > self.exit_threshold_long
        df['short_exit'] = df['funding_rate_ma'] < self.exit_threshold_short
        
        return df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        signal = SignalType.HOLD

        if self.current_position_status == SignalType.LONG:
            if getattr(current_bar, 'long_exit', False):
                signal = SignalType.CLOSE_LONG
                self.current_position_status = None
        elif self.current_position_status == SignalType.SHORT:
            if getattr(current_bar, 'short_exit', False):
                signal = SignalType.CLOSE_SHORT
                self.current_position_status = None
        else: # No position
            if getattr(current_bar, 'long_entry', False):
                signal = SignalType.LONG
                self.current_position_status = SignalType.LONG
            elif getattr(current_bar, 'short_entry', False) and self.enable_shorting:
                signal = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
                
        return signal