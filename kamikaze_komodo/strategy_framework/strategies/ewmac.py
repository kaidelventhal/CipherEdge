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
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        # Access params using lowercase keys, as configparser.items() typically lowercases them
        self.short_window = int(self.params.get('shortwindow', 12)) # Changed to lowercase
        self.long_window = int(self.params.get('longwindow', 26))# Changed to lowercase
        self.atr_period = int(self.params.get('atr_period', 14)) # Already lowercase in config by convention, but ensure consistency
    
        # Sentiment thresholds are often passed directly into params by main.py, preserving case.
        # If they were meant to be read from strategy's own config section, they'd also be lowercase.
        self.sentiment_filter_long_threshold = self.params.get('sentimentfilter_long_threshold') # ensure lowercase for get
        if isinstance(self.sentiment_filter_long_threshold, str):
            try:
                self.sentiment_filter_long_threshold = None if self.sentiment_filter_long_threshold.lower() == 'none' else float(self.sentiment_filter_long_threshold)
            except ValueError:
                logger.warning(f"Could not parse sentiment_filter_long_threshold '{self.params.get('sentimentfilter_long_threshold')}' to float. Defaulting to None.")
                self.sentiment_filter_long_threshold = None

        self.sentiment_filter_short_threshold = self.params.get('sentimentfilter_short_threshold') # ensure lowercase for get
        if isinstance(self.sentiment_filter_short_threshold, str):
            try:
                self.sentiment_filter_short_threshold = None if self.sentiment_filter_short_threshold.lower() == 'none' else float(self.sentiment_filter_short_threshold)
            except ValueError:
                logger.warning(f"Could not parse sentiment_filter_short_threshold '{self.params.get('sentimentfilter_short_threshold')}' to float. Defaulting to None.")
                self.sentiment_filter_short_threshold = None

        if not isinstance(self.short_window, int) or not isinstance(self.long_window, int):
            raise ValueError("EWMACStrategy: 'short_window' and 'long_window' must be integers.")
        if self.short_window >= self.long_window:
            raise ValueError("EWMACStrategy: 'short_window' must be less than 'long_window'.")

        logger.info(
            f"Initialized EWMACStrategy for {symbol} ({timeframe}) "
            f"with Short EMA: {self.short_window}, Long EMA: {self.long_window}, ATR Period: {self.atr_period}. "
            f"Sentiment Long Thresh: {self.sentiment_filter_long_threshold}, Short Thresh: {self.sentiment_filter_short_threshold}. "
            f"Shorting Enabled: {self.enable_shorting}"
        )
        # self.current_position_status is inherited from BaseStrategy

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if data_df.empty: return data_df
        df = data_df.copy()

        if 'close' not in df.columns or len(df) < self.long_window :
            return df

        df[f'ema_short'] = ta.ema(df['close'], length=self.short_window)
        df[f'ema_long'] = ta.ema(df['close'], length=self.long_window)

        if all(col in df.columns for col in ['high', 'low', 'close']):
            if len(df) >= self.atr_period:
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
            else:
                df['atr'] = pd.NA
        else:
            df['atr'] = pd.NA
        return df

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        if data.empty or len(data) < self.long_window:
            logger.warning(f"Not enough historical data for EWMAC signals. Need {self.long_window}, got {len(data)}.")
            return pd.Series(index=data.index, dtype='object')

        df_processed = self._calculate_indicators(data)

        if 'ema_short' not in df_processed.columns or 'ema_long' not in df_processed.columns:
            return pd.Series(index=data.index, dtype='object')

        if sentiment_series is not None and not sentiment_series.empty:
            df_processed = df_processed.join(sentiment_series.rename('sentiment_score'), how='left')
            df_processed['sentiment_score'] = df_processed['sentiment_score'].fillna(0.0)
        elif 'sentiment_score' not in df_processed.columns:
            df_processed['sentiment_score'] = 0.0

        signals = pd.Series(index=df_processed.index, dtype='object').fillna(SignalType.HOLD)
        current_pos_state: Optional[SignalType] = None

        for i in range(1, len(df_processed)):
            prev_short_ema = df_processed['ema_short'].iloc[i-1]
            curr_short_ema = df_processed['ema_short'].iloc[i]
            prev_long_ema = df_processed['ema_long'].iloc[i-1]
            curr_long_ema = df_processed['ema_long'].iloc[i]
        
            current_sentiment = df_processed['sentiment_score'].iloc[i]

            if pd.isna(curr_short_ema) or pd.isna(curr_long_ema) or pd.isna(prev_short_ema) or pd.isna(prev_long_ema):
                signals.iloc[i] = SignalType.HOLD
                continue

            is_golden_cross = curr_short_ema > curr_long_ema and prev_short_ema <= prev_long_ema
            is_death_cross = curr_short_ema < curr_long_ema and prev_short_ema >= prev_long_ema

            if current_pos_state == SignalType.LONG:
                if is_death_cross:
                    signals.iloc[i] = SignalType.CLOSE_LONG
                    current_pos_state = None
                else:
                    signals.iloc[i] = SignalType.HOLD
            elif current_pos_state == SignalType.SHORT:
                if is_golden_cross:
                    signals.iloc[i] = SignalType.CLOSE_SHORT
                    current_pos_state = None
                else:
                    signals.iloc[i] = SignalType.HOLD
            else: # No current position
                if is_golden_cross:
                    if self.sentiment_filter_long_threshold is None or current_sentiment >= self.sentiment_filter_long_threshold:
                        signals.iloc[i] = SignalType.LONG
                        current_pos_state = SignalType.LONG
                    else:
                        signals.iloc[i] = SignalType.HOLD
                elif is_death_cross and self.enable_shorting:
                    if self.sentiment_filter_short_threshold is None or current_sentiment <= self.sentiment_filter_short_threshold:
                        signals.iloc[i] = SignalType.SHORT
                        current_pos_state = SignalType.SHORT
                    else:
                        signals.iloc[i] = SignalType.HOLD
                else:
                    signals.iloc[i] = SignalType.HOLD
    
        logger.info(f"Generated EWMAC signals (vectorized). Longs: {signals.eq(SignalType.LONG).sum()}, Shorts: {signals.eq(SignalType.SHORT).sum()}, CloseLongs: {signals.eq(SignalType.CLOSE_LONG).sum()}, CloseShorts: {signals.eq(SignalType.CLOSE_SHORT).sum()}")
        return signals

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        if len(self.data_history) < self.long_window + 1:
            return SignalType.HOLD

        df_with_indicators = self._calculate_indicators(self.data_history)
    
        if 'atr' in df_with_indicators.columns and pd.notna(df_with_indicators['atr'].iloc[-1]):
            bar_data.atr = df_with_indicators['atr'].iloc[-1]
    
        current_sentiment = sentiment_score if sentiment_score is not None else bar_data.sentiment_score
        # FIX: Robustly handle None and NaN values from the sentiment score
        if current_sentiment is None or pd.isna(current_sentiment): 
            current_sentiment = 0.0

        if df_with_indicators.empty or 'ema_short' not in df_with_indicators.columns or \
            'ema_long' not in df_with_indicators.columns or len(df_with_indicators) < 2:
            return SignalType.HOLD

        latest_ema_short = df_with_indicators['ema_short'].iloc[-1]
        prev_ema_short = df_with_indicators['ema_short'].iloc[-2]
        latest_ema_long = df_with_indicators['ema_long'].iloc[-1]
        prev_ema_long = df_with_indicators['ema_long'].iloc[-2]

        if pd.isna(latest_ema_short) or pd.isna(prev_ema_short) or \
            pd.isna(latest_ema_long) or pd.isna(prev_ema_long):
            return SignalType.HOLD

        signal_to_return = SignalType.HOLD
        is_golden_cross = latest_ema_short > latest_ema_long and prev_ema_short <= prev_ema_long
        is_death_cross = latest_ema_short < latest_ema_long and prev_ema_short >= prev_ema_long

        if self.current_position_status == SignalType.LONG:
            if is_death_cross:
                signal_to_return = SignalType.CLOSE_LONG
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - EWMAC CLOSE_LONG for {self.symbol}. EMA_S: {latest_ema_short:.2f}, EMA_L: {latest_ema_long:.2f}")
        elif self.current_position_status == SignalType.SHORT:
            if is_golden_cross:
                signal_to_return = SignalType.CLOSE_SHORT
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - EWMAC CLOSE_SHORT for {self.symbol}. EMA_S: {latest_ema_short:.2f}, EMA_L: {latest_ema_long:.2f}")
        else: # No current position
            if is_golden_cross:
                if self.sentiment_filter_long_threshold is None or current_sentiment >= self.sentiment_filter_long_threshold:
                    signal_to_return = SignalType.LONG
                    self.current_position_status = SignalType.LONG
                    logger.info(f"{bar_data.timestamp} - EWMAC LONG for {self.symbol}. Sent: {current_sentiment:.2f}. EMA_S: {latest_ema_short:.2f}, EMA_L: {latest_ema_long:.2f}. ATR: {bar_data.atr if bar_data.atr else 'N/A'}")
                else:
                    logger.info(f"{bar_data.timestamp} - EWMAC LONG for {self.symbol} SUPPRESSED by sentiment ({current_sentiment:.2f} < {self.sentiment_filter_long_threshold}).")
            elif is_death_cross and self.enable_shorting:
                if self.sentiment_filter_short_threshold is None or current_sentiment <= self.sentiment_filter_short_threshold:
                    signal_to_return = SignalType.SHORT
                    self.current_position_status = SignalType.SHORT
                    logger.info(f"{bar_data.timestamp} - EWMAC SHORT for {self.symbol}. Sent: {current_sentiment:.2f}. EMA_S: {latest_ema_short:.2f}, EMA_L: {latest_ema_long:.2f}. ATR: {bar_data.atr if bar_data.atr else 'N/A'}")
                else:
                    logger.info(f"{bar_data.timestamp} - EWMAC SHORT for {self.symbol} SUPPRESSED by sentiment ({current_sentiment:.2f} > {self.sentiment_filter_short_threshold}).")
    
        return signal_to_return