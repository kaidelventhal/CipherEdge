# kamikaze_komodo/strategy_framework/strategies/ehlers_instantaneous_trendline.py
import pandas as pd
import pandas_ta as ta # pandas_ta might not have Ehlers' IT directly, may need custom impl.
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
    This is a simplified version focusing on the trendline crossover.
    The Instantaneous Trendline is a 2-bar lookback Finite Impulse Response (FIR) filter.
    A common way to use it is with a trigger line (e.g., delayed trendline).
    Source: "Rocket Science for Traders" by John Ehlers, Chapter 3.
    Phase 6: Added shorting capability.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        # Alpha for smoothing, if desired (not part of core IT calculation directly but often used with Ehlers' indicators)
        # Default to a common value if not specified, e.g. for smoothing price before IT calc or the IT line itself.
        # However, the core IT is (Close + 2*Close[-1] + Close[-2]) / 4
        self.alpha = float(self.params.get('alpha', 0.07)) # Example if smoothing was used for other Ehlers indicators
        self.it_lag_trigger = int(self.params.get('it_lag_trigger', 1)) # Lag for the trigger line (e.g. IT lagged by 1 bar)
        
        logger.info(
            f"Initialized EhlersInstantaneousTrendlineStrategy for {symbol} ({timeframe}) "
            f"with IT Lag Trigger: {self.it_lag_trigger}. (Alpha: {self.alpha} is for general Ehlers context, not core IT). "
            f"Shorting Enabled: {self.enable_shorting}"
        )
        # self.current_position_status is inherited from BaseStrategy

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if data_df.empty: return data_df
        df = data_df.copy()

        if 'close' not in df.columns or len(df) < 3: # Need at least 3 bars for IT calculation
            return df

        # Ehlers' Instantaneous Trendline calculation:
        # IT_t = (Close_t + 2*Close_t-1 + Close_t-2) / 4
        close_prices = df['close']
        it_values = np.full(len(df), np.nan)
        if len(close_prices) >= 3:
            for i in range(2, len(close_prices)):
                it_values[i] = (close_prices.iloc[i] + 2 * close_prices.iloc[i-1] + close_prices.iloc[i-2]) / 4
        
        df['it'] = it_values
        df['it_trigger'] = df['it'].shift(self.it_lag_trigger) # Lagged IT as a trigger

        # ATR for BarData object (optional, but good practice)
        if all(col in df.columns for col in ['high', 'low', 'close']):
                atr_period = int(self.params.get('atr_period', 14))
                if len(df) >= atr_period:
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
                else:
                    df['atr'] = pd.NA
        else:
            df['atr'] = pd.NA
            
        return df

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        if data.empty or len(data) < 3 + self.it_lag_trigger: # Need enough for IT and its lag
            logger.warning(f"Not enough historical data for Ehlers IT signals. Need {3 + self.it_lag_trigger}, got {len(data)}.")
            return pd.Series(index=data.index, dtype='object')

        df_processed = self._calculate_indicators(data)

        if 'it' not in df_processed.columns or 'it_trigger' not in df_processed.columns:
            return pd.Series(index=data.index, dtype='object')

        signals = pd.Series(index=df_processed.index, dtype='object').fillna(SignalType.HOLD)
        current_pos_state: Optional[SignalType] = None

        for i in range(1, len(df_processed)): # Start from 1 to compare with previous
            # Ensure all values are non-NaN for comparison
            if pd.isna(df_processed['it'].iloc[i]) or \
               pd.isna(df_processed['it_trigger'].iloc[i]) or \
               pd.isna(df_processed['it'].iloc[i-1]) or \
               pd.isna(df_processed['it_trigger'].iloc[i-1]):
                signals.iloc[i] = SignalType.HOLD
                continue

            is_bullish_cross = df_processed['it'].iloc[i] > df_processed['it_trigger'].iloc[i] and \
                                 df_processed['it'].iloc[i-1] <= df_processed['it_trigger'].iloc[i-1]
            
            is_bearish_cross = df_processed['it'].iloc[i] < df_processed['it_trigger'].iloc[i] and \
                                 df_processed['it'].iloc[i-1] >= df_processed['it_trigger'].iloc[i-1]

            if current_pos_state == SignalType.LONG:
                if is_bearish_cross:
                    signals.iloc[i] = SignalType.CLOSE_LONG
                    current_pos_state = None
                else:
                    signals.iloc[i] = SignalType.HOLD
            elif current_pos_state == SignalType.SHORT:
                if is_bullish_cross:
                    signals.iloc[i] = SignalType.CLOSE_SHORT
                    current_pos_state = None
                else:
                    signals.iloc[i] = SignalType.HOLD
            else: # No current position
                if is_bullish_cross:
                    signals.iloc[i] = SignalType.LONG
                    current_pos_state = SignalType.LONG
                elif is_bearish_cross and self.enable_shorting:
                    signals.iloc[i] = SignalType.SHORT
                    current_pos_state = SignalType.SHORT
                else:
                    signals.iloc[i] = SignalType.HOLD
        
        logger.info(f"Generated Ehlers IT signals. Longs: {signals.eq(SignalType.LONG).sum()}, Shorts: {signals.eq(SignalType.SHORT).sum()}, CloseLongs: {signals.eq(SignalType.CLOSE_LONG).sum()}, CloseShorts: {signals.eq(SignalType.CLOSE_SHORT).sum()}")
        return signals

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        if len(self.data_history) < 3 + self.it_lag_trigger: # Need enough data for IT and its lag
            return SignalType.HOLD

        df_with_indicators = self._calculate_indicators(self.data_history)
        
        if 'atr' in df_with_indicators.columns and pd.notna(df_with_indicators['atr'].iloc[-1]):
                bar_data.atr = df_with_indicators['atr'].iloc[-1] # Update BarData with ATR

        if df_with_indicators.empty or 'it' not in df_with_indicators.columns or \
           'it_trigger' not in df_with_indicators.columns or len(df_with_indicators) < 2:
            return SignalType.HOLD # Not enough data or indicator calculation failed

        # Get latest and previous values for IT and its trigger
        latest_it = df_with_indicators['it'].iloc[-1]
        prev_it = df_with_indicators['it'].iloc[-2]
        latest_it_trigger = df_with_indicators['it_trigger'].iloc[-1]
        prev_it_trigger = df_with_indicators['it_trigger'].iloc[-2]

        if pd.isna(latest_it) or pd.isna(prev_it) or \
           pd.isna(latest_it_trigger) or pd.isna(prev_it_trigger):
            return SignalType.HOLD # Not enough data for signal generation

        signal_to_return = SignalType.HOLD

        is_bullish_cross = latest_it > latest_it_trigger and prev_it <= prev_it_trigger
        is_bearish_cross = latest_it < latest_it_trigger and prev_it >= prev_it_trigger

        if self.current_position_status == SignalType.LONG:
            if is_bearish_cross:
                signal_to_return = SignalType.CLOSE_LONG
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - Ehlers IT CLOSE_LONG for {self.symbol}. IT: {latest_it:.2f}, Trigger: {latest_it_trigger:.2f}")
        elif self.current_position_status == SignalType.SHORT:
            if is_bullish_cross:
                signal_to_return = SignalType.CLOSE_SHORT
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - Ehlers IT CLOSE_SHORT for {self.symbol}. IT: {latest_it:.2f}, Trigger: {latest_it_trigger:.2f}")
        else: # No current position
            if is_bullish_cross:
                signal_to_return = SignalType.LONG
                self.current_position_status = SignalType.LONG
                logger.info(f"{bar_data.timestamp} - Ehlers IT LONG for {self.symbol}. IT: {latest_it:.2f}, Trigger: {latest_it_trigger:.2f}. ATR: {bar_data.atr if bar_data.atr else 'N/A'}")
            elif is_bearish_cross and self.enable_shorting:
                signal_to_return = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
                logger.info(f"{bar_data.timestamp} - Ehlers IT SHORT for {self.symbol}. IT: {latest_it:.2f}, Trigger: {latest_it_trigger:.2f}. ATR: {bar_data.atr if bar_data.atr else 'N/A'}")
            
        return signal_to_return