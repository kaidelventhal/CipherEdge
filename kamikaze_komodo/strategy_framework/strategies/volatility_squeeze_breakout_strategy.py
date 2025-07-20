# FILE: kamikaze_komodo/strategy_framework/strategies/volatility_squeeze_breakout_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class VolatilitySqueezeBreakoutStrategy(BaseStrategy):
    """
    Implements a Volatility Squeeze Breakout strategy (inspired by TTM Squeeze).
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.bb_period = int(self.params.get('bb_period', 20))
        self.bb_std_dev = float(self.params.get('bb_std_dev', 2.0))
        self.kc_period = int(self.params.get('kc_period', 20))
        self.kc_atr_period = int(self.params.get('kc_atr_period', 10))
        self.kc_atr_multiplier = float(self.params.get('kc_atr_multiplier', 1.5))
        self.atr_period = self.kc_atr_period

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        min_len = max(self.bb_period, self.kc_period, self.kc_atr_period)
        if df.empty or len(df) < min_len:
            return pd.DataFrame()

        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
        df['bb_lower'] = bbands[f'BBL_{self.bb_period}_{self.bb_std_dev:.1f}']
        df['bb_upper'] = bbands[f'BBU_{self.bb_period}_{self.bb_std_dev:.1f}']

        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=self.kc_period, atr_length=self.kc_atr_period, multiplier=self.kc_atr_multiplier)
        df['kc_lower'] = kc.iloc[:, 0]
        df['kc_upper'] = kc.iloc[:, 2]
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        # --- Vectorized Signal Conditions ---
        df['squeeze_on'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])
        
        # Squeeze has just been released
        squeeze_released = (df['squeeze_on'].shift(1)) & (~df['squeeze_on'])

        # Entry Signals
        df['long_entry'] = squeeze_released & (df['close'] > df['bb_upper'])
        df['short_entry'] = squeeze_released & (df['close'] < df['bb_lower'])
        
        # Exit Signals (simple: cross back inside the bands)
        df['long_exit'] = df['close'] < df['bb_upper']
        df['short_exit'] = df['close'] > df['bb_lower']

        return df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        signal = SignalType.HOLD
        
        if self.current_position_status is None:
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