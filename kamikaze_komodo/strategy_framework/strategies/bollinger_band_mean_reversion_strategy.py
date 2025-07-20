# FILE: kamikaze_komodo/strategy_framework/strategies/bollinger_band_mean_reversion_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class BollingerBandMeanReversionStrategy(BaseStrategy):
    """
    Implements a Bollinger Band Mean Reversion strategy, ideal for ranging markets.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.bb_period = int(self.params.get('bb_period', 20))
        self.bb_std_dev = float(self.params.get('bb_std_dev', 2.0))
        self.atr_period = int(self.params.get('atr_period', 14))

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if df.empty or len(df) < self.bb_period:
            return pd.DataFrame()

        bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
        df['bb_lower'] = bbands[f'BBL_{self.bb_period}_{self.bb_std_dev:.1f}']
        df['bb_middle'] = bbands[f'BBM_{self.bb_period}_{self.bb_std_dev:.1f}']
        df['bb_upper'] = bbands[f'BBU_{self.bb_period}_{self.bb_std_dev:.1f}']
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        
        # Vectorized Signal Conditions
        df['long_entry'] = (df['close'].shift(1) >= df['bb_lower'].shift(1)) & (df['close'] < df['bb_lower'])
        df['short_entry'] = (df['close'].shift(1) <= df['bb_upper'].shift(1)) & (df['close'] > df['bb_upper'])
        
        df['long_exit'] = (df['close'].shift(1) <= df['bb_middle'].shift(1)) & (df['close'] > df['bb_middle'])
        df['short_exit'] = (df['close'].shift(1) >= df['bb_middle'].shift(1)) & (df['close'] < df['bb_middle'])

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