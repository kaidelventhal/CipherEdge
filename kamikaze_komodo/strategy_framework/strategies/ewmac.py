# FILE: kamikaze_komodo/strategy_framework/strategies/ewmac.py
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class EWMACStrategy(BaseStrategy):
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        self.short_window = int(self.params.get('shortwindow', 12))
        self.long_window = int(self.params.get('longwindow', 26))
        self.atr_period = int(self.params.get('atr_period', 14))
        
        self.use_regime_filter = str(self.params.get('useregimefilter', 'false')).lower() == 'true'
        
        favorable_regimes_val = self.params.get('favorableregimes', '')
        if isinstance(favorable_regimes_val, str):
            self.favorable_regimes = [int(r.strip()) for r in favorable_regimes_val.split(',') if r.strip().isdigit()]
        elif isinstance(favorable_regimes_val, int):
            self.favorable_regimes = [favorable_regimes_val]
        elif isinstance(favorable_regimes_val, list):
            self.favorable_regimes = favorable_regimes_val
        else:
            self.favorable_regimes = []

        self.sentiment_filter_long_threshold = self.params.get('sentimentfilter_long_threshold')
        if isinstance(self.sentiment_filter_long_threshold, str):
            try:
                self.sentiment_filter_long_threshold = None if self.sentiment_filter_long_threshold.lower() == 'none' else float(self.sentiment_filter_long_threshold)
            except ValueError:
                self.sentiment_filter_long_threshold = None

        self.sentiment_filter_short_threshold = self.params.get('sentimentfilter_short_threshold')
        if isinstance(self.sentiment_filter_short_threshold, str):
            try:
                self.sentiment_filter_short_threshold = None if self.sentiment_filter_short_threshold.lower() == 'none' else float(self.sentiment_filter_short_threshold)
            except ValueError:
                self.sentiment_filter_short_threshold = None

        if not isinstance(self.short_window, int) or not isinstance(self.long_window, int):
            raise ValueError("EWMACStrategy: 'short_window' and 'long_window' must be integers.")
        if self.short_window >= self.long_window:
            raise ValueError("EWMACStrategy: 'short_window' must be less than 'long_window'.")

        logger.info(
            f"Initialized EWMACStrategy for {symbol} ({timeframe}) "
            f"with Short EMA: {self.short_window}, Long EMA: {self.long_window}, ATR Period: {self.atr_period}. "
            f"Sentiment Long Thresh: {self.sentiment_filter_long_threshold}, Short Thresh: {self.sentiment_filter_short_threshold}. "
            f"Shorting Enabled: {self.enable_shorting}. "
            f"Regime Filter Enabled: {self.use_regime_filter}, Favorable Regimes: {self.favorable_regimes}"
        )

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
        # This method is not the primary one used by the live portfolio manager but is kept for optimization/analysis tools.
        logger.warning("generate_signals for EWMACStrategy is designed for vectorized analysis and may differ from on_bar_data logic.")
        return super().generate_signals(data, sentiment_series) # Fallback to base implementation

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        if len(self.data_history) < self.long_window + 1:
            return SignalType.HOLD

        is_favorable_regime = True
        if self.use_regime_filter:
            if bar_data.market_regime is not None:
                if bar_data.market_regime not in self.favorable_regimes:
                    is_favorable_regime = False
            else:
                logger.debug(f"Regime for {bar_data.symbol} is None; filter will be ignored for this bar.")
                is_favorable_regime = True

        df_with_indicators = self._calculate_indicators(self.data_history)
        
        if 'atr' in df_with_indicators.columns and pd.notna(df_with_indicators['atr'].iloc[-1]):
            bar_data.atr = df_with_indicators['atr'].iloc[-1]
        
        current_sentiment = bar_data.sentiment_score
        
        if df_with_indicators.empty or 'ema_short' not in df_with_indicators.columns or \
            'ema_long' not in df_with_indicators.columns or len(df_with_indicators) < 2:
            return SignalType.HOLD

        latest_ema_short = df_with_indicators['ema_short'].iloc[-1]
        prev_ema_short = df_with_indicators['ema_short'].iloc[-2]
        latest_ema_long = df_with_indicators['ema_long'].iloc[-1]
        prev_ema_long = df_with_indicators['ema_long'].iloc[-2]

        if pd.isna(latest_ema_short) or pd.isna(prev_ema_short) or pd.isna(latest_ema_long) or pd.isna(prev_ema_long):
            return SignalType.HOLD

        signal_to_return = SignalType.HOLD
        is_golden_cross = latest_ema_short > latest_ema_long and prev_ema_short <= prev_ema_long
        is_death_cross = latest_ema_short < latest_ema_long and prev_ema_short >= prev_ema_long

        if self.current_position_status == SignalType.LONG:
            if is_death_cross:
                signal_to_return = SignalType.CLOSE_LONG
                self.current_position_status = None
        elif self.current_position_status == SignalType.SHORT:
            if is_golden_cross:
                signal_to_return = SignalType.CLOSE_SHORT
                self.current_position_status = None
        else: # No current position
            sentiment_ok = True
            if is_golden_cross:
                if self.sentiment_filter_long_threshold is not None and current_sentiment is not None:
                    if current_sentiment < self.sentiment_filter_long_threshold:
                        sentiment_ok = False
                if sentiment_ok and is_favorable_regime:
                    signal_to_return = SignalType.LONG
                    self.current_position_status = SignalType.LONG
                else:
                    logger.info(f"{bar_data.timestamp} - EWMAC LONG for {self.symbol} SUPPRESSED. Sentiment OK: {sentiment_ok} (Score: {current_sentiment}), Regime OK: {is_favorable_regime} (Regime: {bar_data.market_regime})")

            elif is_death_cross and self.enable_shorting:
                if self.sentiment_filter_short_threshold is not None and current_sentiment is not None:
                    if current_sentiment > self.sentiment_filter_short_threshold:
                        sentiment_ok = False
                if sentiment_ok and is_favorable_regime:
                    signal_to_return = SignalType.SHORT
                    self.current_position_status = SignalType.SHORT
                else:
                    logger.info(f"{bar_data.timestamp} - EWMAC SHORT for {self.symbol} SUPPRESSED. Sentiment OK: {sentiment_ok} (Score: {current_sentiment}), Regime OK: {is_favorable_regime} (Regime: {bar_data.market_regime})")
    
        return signal_to_return