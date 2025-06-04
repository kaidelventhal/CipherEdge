# kamikaze_komodo/strategy_framework/strategies/ewmac.py
import pandas as pd
import pandas_ta as ta # For EMA calculations
from typing import Dict, Any, Optional
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class EWMACStrategy(BaseStrategy):
    """
    Exponential Weighted Moving Average Crossover (EWMAC) Strategy.
    Generates LONG signals when the short-term EMA crosses above the long-term EMA.
    Generates SHORT signals when the short-term EMA crosses below the long-term EMA.
    (Note: For this basic implementation, SHORT signals might imply selling a long position
     or going short if the system supports it. For now, we'll focus on LONG and CLOSE_LONG.)
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        self.short_window = self.params.get('short_window', 12)
        self.long_window = self.params.get('long_window', 26)
        
        if not isinstance(self.short_window, int) or not isinstance(self.long_window, int):
            raise ValueError("EWMACStrategy: 'short_window' and 'long_window' parameters must be integers.")
        if self.short_window >= self.long_window:
            raise ValueError("EWMACStrategy: 'short_window' must be less than 'long_window'.")
            
        logger.info(
            f"Initialized EWMACStrategy for {symbol} ({timeframe}) "
            f"with Short EMA: {self.short_window}, Long EMA: {self.long_window}"
        )
        self.current_position: Optional[SignalType] = None # None, LONG (no shorting in this simple version)


    def _calculate_emas(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Calculates EMAs and adds them to the DataFrame."""
        if data_df.empty or len(data_df) < self.long_window:
            # logger.warning(f"Not enough data ({len(data_df)}) to calculate EMAs requiring {self.long_window} periods.")
            return data_df # Return original df if not enough data

        df = data_df.copy()
        df[f'ema_short'] = ta.ema(df['close'], length=self.short_window)
        df[f'ema_long'] = ta.ema(df['close'], length=self.long_window)
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generates trading signals based on historical data.
        This is typically used for backtesting over a full dataset.
        """
        if data.empty or len(data) < self.long_window:
            logger.warning(f"Not enough historical data ({len(data)}) for EWMAC signals generation. Need at least {self.long_window} periods.")
            return pd.Series(index=data.index, dtype='object') # Return empty signals

        df_with_emas = self._calculate_emas(data)
        signals = pd.Series(index=df_with_emas.index, dtype='object') # Using object to store Enum or None

        # Conditions for signals
        # Golden Cross (Buy Signal): Short EMA crosses above Long EMA
        buy_condition = (df_with_emas['ema_short'] > df_with_emas['ema_long']) & \
                        (df_with_emas['ema_short'].shift(1) <= df_with_emas['ema_long'].shift(1))
        
        # Death Cross (Sell Signal): Short EMA crosses below Long EMA
        sell_condition = (df_with_emas['ema_short'] < df_with_emas['ema_long']) & \
                         (df_with_emas['ema_short'].shift(1) >= df_with_emas['ema_long'].shift(1))

        # Apply signals based on conditions
        # This simple version doesn't track state (current position) across the Series directly for generate_signals.
        # It just marks the crossover points. A backtester would interpret these.
        signals[buy_condition] = SignalType.LONG
        signals[sell_condition] = SignalType.CLOSE_LONG # For simplicity, a sell condition means close any long.

        # To avoid look-ahead bias, signals should be actionable on the next bar's open.
        # However, for simplicity in this `generate_signals`, we mark the bar where crossover happens.
        # The backtester should handle how these signals are translated into trades (e.g., next bar open).

        # Fill forward HOLD signals after an initial signal, until a counter-signal.
        # This part is more complex for generate_signals and better handled by a stateful on_bar_data or backtester logic.
        # For now, generate_signals will just mark the crossover events.
        # If you need HOLD signals here, you'd iterate and maintain state:
        # current_sig_state = SignalType.HOLD
        # for i in range(len(df_with_emas)):
        #     if buy_condition.iloc[i]:
        #         current_sig_state = SignalType.LONG
        #     elif sell_condition.iloc[i]:
        #         current_sig_state = SignalType.CLOSE_LONG # Or HOLD if no position
        #     signals.iloc[i] = current_sig_state if current_sig_state != SignalType.CLOSE_LONG else SignalType.HOLD
        # This requires careful state management.

        logger.info(f"Generated EWMAC signals. Longs: {signals.eq(SignalType.LONG).sum()}, CloseLongs: {signals.eq(SignalType.CLOSE_LONG).sum()}")
        return signals


    def on_bar_data(self, bar_data: BarData) -> Optional[SignalType]:
        """
        Processes a new bar of data and decides on a trading action for live/simulated trading.
        This method maintains the state of `self.current_position`.
        """
        self.update_data_history(bar_data) # Add new bar to history

        if len(self.data_history) < self.long_window + 1: # Need enough data for current + previous EMAs
            # logger.debug(f"Not enough data in history ({len(self.data_history)}) for EWMAC on_bar_data. Need at least {self.long_window + 1}.")
            return SignalType.HOLD # Not enough data to make a decision

        # Calculate EMAs on the current history
        df_with_emas = self._calculate_emas(self.data_history)

        if df_with_emas.empty or 'ema_short' not in df_with_emas.columns or 'ema_long' not in df_with_emas.columns or len(df_with_emas) < 2:
            # logger.debug("EMA calculation failed or not enough data points after EMA calculation.")
            return SignalType.HOLD

        # Get the latest two values for crossover detection
        latest_ema_short = df_with_emas['ema_short'].iloc[-1]
        prev_ema_short = df_with_emas['ema_short'].iloc[-2]
        latest_ema_long = df_with_emas['ema_long'].iloc[-1]
        prev_ema_long = df_with_emas['ema_long'].iloc[-2]

        if pd.isna(latest_ema_short) or pd.isna(prev_ema_short) or \
           pd.isna(latest_ema_long) or pd.isna(prev_ema_long):
            # logger.debug("EMA values are NaN, cannot make a decision.")
            return SignalType.HOLD # EMAs not yet calculated (NaNs during warmup)

        signal = SignalType.HOLD # Default action

        # Entry Condition (Golden Cross)
        is_golden_cross = latest_ema_short > latest_ema_long and prev_ema_short <= prev_ema_long
        if is_golden_cross and self.current_position != SignalType.LONG:
            signal = SignalType.LONG
            self.current_position = SignalType.LONG
            logger.info(f"{bar_data.timestamp} - EWMAC LONG signal for {self.symbol}. Short EMA: {latest_ema_short:.2f}, Long EMA: {latest_ema_long:.2f}")
            return signal

        # Exit Condition (Death Cross)
        is_death_cross = latest_ema_short < latest_ema_long and prev_ema_short >= prev_ema_long
        if is_death_cross and self.current_position == SignalType.LONG:
            signal = SignalType.CLOSE_LONG
            self.current_position = None # Position closed
            logger.info(f"{bar_data.timestamp} - EWMAC CLOSE_LONG signal for {self.symbol}. Short EMA: {latest_ema_short:.2f}, Long EMA: {latest_ema_long:.2f}")
            return signal
            
        # If already in a position and no exit signal, hold.
        if self.current_position == SignalType.LONG:
            return SignalType.HOLD

        return signal # Default to HOLD if no other conditions met