# FILE: kamikaze_komodo/strategy_framework/strategies/ewmac.py
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class EWMACStrategy(BaseStrategy):
    """
    Exponential Weighted Moving Average Crossover (EWMAC) strategy.
    Phase 1 Refactor: Logic moved to vectorized prepare_data method.
    on_bar_data is now stateless and reads pre-calculated signal columns.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        self.short_window = int(self.params.get('shortwindow', 12))
        self.long_window = int(self.params.get('longwindow', 26))
        self.atr_period = int(self.params.get('atr_period', 14))

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates EMAs, ATR, and the crossover signal conditions.
        """
        df = data.copy()
        if df.empty or len(df) < self.long_window:
            logger.warning("Not enough data to calculate EWMAC indicators.")
            return df
            
        ema_short_col = f'ema_short'
        ema_long_col = f'ema_long'

        df[ema_short_col] = ta.ema(df['close'], length=self.short_window)
        df[ema_long_col] = ta.ema(df['close'], length=self.long_window)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)

        # Vectorized signal generation
        df['golden_cross'] = (df[ema_short_col].shift(1) <= df[ema_long_col].shift(1)) & \
                             (df[ema_short_col] > df[ema_long_col])
        
        df['death_cross'] = (df[ema_short_col].shift(1) >= df[ema_long_col].shift(1)) & \
                            (df[ema_short_col] < df[ema_long_col])
        
        logger.info(f"EWMAC data prepared. Columns added: {ema_short_col}, {ema_long_col}, atr, golden_cross, death_cross")
        return df

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        """
        Generates trade signals based on pre-calculated crossover columns in the BarData.
        """
        is_golden_cross = getattr(current_bar, 'golden_cross', False)
        is_death_cross = getattr(current_bar, 'death_cross', False)

        signal = SignalType.HOLD
        
        if self.current_position_status == SignalType.LONG:
            if is_death_cross:
                signal = SignalType.CLOSE_LONG
                self.current_position_status = None
        elif self.current_position_status == SignalType.SHORT:
            if is_golden_cross:
                signal = SignalType.CLOSE_SHORT
                self.current_position_status = None
        else: # No position
            if is_golden_cross:
                signal = SignalType.LONG
                self.current_position_status = SignalType.LONG
            elif is_death_cross and self.enable_shorting:
                signal = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
                
        return signal