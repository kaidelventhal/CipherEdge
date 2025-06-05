# kamikaze_komodo/strategy_framework/strategies/ehlers_instantaneous_trendline.py
import pandas as pd
import pandas_ta as ta # pandas_ta might not have Ehlers' IT directly, may need custom impl.
import numpy as np
from typing import Dict, Any, Optional

from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
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
            f"with IT Lag Trigger: {self.it_lag_trigger}. (Alpha: {self.alpha} is for general Ehlers context, not core IT)."
        )
        self.current_position_status: Optional[SignalType] = None

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if data_df.empty: return data_df
        df = data_df.copy()

        if 'close' not in df.columns or len(df) < 3: # Need at least 3 bars for IT calculation
            return df

        # Ehlers' Instantaneous Trendline calculation:
        # IT_t = (Close_t + 2*Close_t-1 + Close_t-2) / 4
        # This formula is from some interpretations. Ehlers' book might use slightly different coeffs for specific filters.
        # A common simplified IT is also: (price + 2*price.shift(1) + price.shift(2))/4
        # Let's use a direct calculation.
        # Simpler version: FIR filter (0.25, 0.5, 0.25) applied effectively
        # Or often: (Close + 2 * Close.shift(1) + Close.shift(2)) / 4
        # For an adaptive version, one would use dominant cycle period.
        # For a fixed version, as described in some sources:
        # IT(n) = (Price(n) + 2*Price(n-1) + Price(n-2))/4
        # If we want a less lagging version, this is effectively a very short filter.

        # Using a simplified approach often cited: alpha * Price + (1-alpha) * IT_prev
        # This is more like an EMA. Let's use the FIR definition if possible.
        # The "Instantaneous Trendline" from Rocket Science for Traders, Chapter 3, is actually:
        # Trendline = (Price + 2*Price[1] + Price[2]) / 4 where Price[1] is previous bar, Price[2] is bar before previous.
        # This is a 2-pole FIR.

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

        # Incorporate sentiment if provided (similar to EWMAC)
        # For this example, Ehlers IT will operate independently of sentiment unless explicitly modified.

        signals = pd.Series(index=df_processed.index, dtype='object').fillna(SignalType.HOLD)
        current_pos_state = None

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

            if current_pos_state != SignalType.LONG:
                if is_bullish_cross:
                    signals.iloc[i] = SignalType.LONG
                    current_pos_state = SignalType.LONG
                else:
                    signals.iloc[i] = SignalType.HOLD
            
            elif current_pos_state == SignalType.LONG:
                if is_bearish_cross:
                    signals.iloc[i] = SignalType.CLOSE_LONG
                    current_pos_state = None
                else:
                    signals.iloc[i] = SignalType.HOLD
        
        logger.info(f"Generated Ehlers IT signals. Longs: {signals.eq(SignalType.LONG).sum()}, CloseLongs: {signals.eq(SignalType.CLOSE_LONG).sum()}")
        return signals

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None) -> Optional[SignalType]:
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

        # Long Entry: IT crosses above its trigger line
        is_bullish_cross = latest_it > latest_it_trigger and prev_it <= prev_it_trigger
        if is_bullish_cross and self.current_position_status != SignalType.LONG:
            signal_to_return = SignalType.LONG
            self.current_position_status = SignalType.LONG
            logger.info(f"{bar_data.timestamp} - Ehlers IT LONG for {self.symbol}. IT: {latest_it:.2f}, Trigger: {latest_it_trigger:.2f}. ATR: {bar_data.atr if bar_data.atr else 'N/A'}")

        # Long Exit: IT crosses below its trigger line
        is_bearish_cross = latest_it < latest_it_trigger and prev_it >= prev_it_trigger
        if is_bearish_cross and self.current_position_status == SignalType.LONG:
            signal_to_return = SignalType.CLOSE_LONG
            self.current_position_status = None
            logger.info(f"{bar_data.timestamp} - Ehlers IT CLOSE_LONG for {self.symbol}. IT: {latest_it:.2f}, Trigger: {latest_it_trigger:.2f}")
            
        return signal_to_return