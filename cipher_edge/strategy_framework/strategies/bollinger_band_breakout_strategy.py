import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from cipher_edge.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from cipher_edge.core.enums import SignalType
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class BollingerBandBreakoutStrategy(BaseStrategy):
    """
    Implements a Bollinger Band Breakout strategy.
    Enters on price breakouts from Bollinger Bands, potentially filtered by volume or momentum.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.bb_period = int(self.params.get('bb_period', 20))
        self.bb_std_dev = float(self.params.get('bb_std_dev', 2.0))
        self.atr_period = int(self.params.get('atr_period', 14))
        self.volume_filter_enabled = str(self.params.get('volume_filter_enabled', 'false')).lower() == 'true'
        self.volume_sma_period = int(self.params.get('volume_sma_period', 20))
        self.volume_factor_above_sma = float(self.params.get('volume_factor_above_sma', 1.5))
        self.min_breakout_atr_multiple = float(self.params.get('min_breakout_atr_multiple', 0.0))

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        if df.empty or len(df) < self.bb_period:
            return pd.DataFrame()

        bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
        if bbands is not None and not bbands.empty:
            df['bb_lower'] = bbands.iloc[:, 0] # Lower band
            df['bb_upper'] = bbands.iloc[:, 2] # Upper band
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        # Volume SMA
        if self.volume_filter_enabled:
            df['volume_sma'] = ta.sma(df['volume'], length=self.volume_sma_period)
        
        # Filters
        volume_ok = True
        if self.volume_filter_enabled:
            volume_ok = df['volume'] > (df['volume_sma'] * self.volume_factor_above_sma)
        
        candle_size_ok = True
        if self.min_breakout_atr_multiple > 0:
            candle_size_ok = (df['high'] - df['low']) > (df['atr'] * self.min_breakout_atr_multiple)

        # Entry Signals
        df['long_entry'] = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1)) & volume_ok & candle_size_ok
        df['short_entry'] = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1)) & volume_ok & candle_size_ok
        
        # Exit Signals (simple version: close when price re-enters the band)
        df['long_exit'] = (df['close'] < df['bb_upper']) & (df['close'].shift(1) >= df['bb_upper'].shift(1))
        df['short_exit'] = (df['close'] > df['bb_lower']) & (df['close'].shift(1) <= df['bb_lower'].shift(1))

        return df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        signal = SignalType.HOLD
        
        if self.current_position_status is None: # No position
            if getattr(current_bar, 'long_entry', False):
                signal = SignalType.LONG
                self.current_position_status = SignalType.LONG
            elif getattr(current_bar, 'short_entry', False) and self.enable_shorting:
                signal = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
        elif self.current_position_status == SignalType.LONG:
            if getattr(current_bar, 'long_exit', False):
                signal = SignalType.CLOSE_LONG
                self.current_position_status = None
        elif self.current_position_status == SignalType.SHORT:
            if getattr(current_bar, 'short_exit', False):
                signal = SignalType.CLOSE_SHORT
                self.current_position_status = None
                
        return signal