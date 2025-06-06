# kamikaze_komodo/strategy_framework/strategies/bollinger_band_breakout_strategy.py
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, Union, List
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class BollingerBandBreakoutStrategy(BaseStrategy):
    """
    Implements a Bollinger Band Breakout strategy.
    Enters on price breakouts from Bollinger Bands, potentially filtered by volume or momentum.
    Adaptable for long and short positions.
    """
    def __init__(self, symbol: str, timeframe: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, timeframe, params)
        
        self.bb_period = int(self.params.get('bb_period', 20))
        self.bb_std_dev = float(self.params.get('bb_std_dev', 2.0))
        self.atr_period = int(self.params.get('atr_period', 14)) # For ATR-based filters or stops

        # Optional filters
        self.volume_filter_enabled = str(self.params.get('volume_filter_enabled', 'false')).lower() == 'true'
        self.volume_sma_period = int(self.params.get('volume_sma_period', 20))
        self.volume_factor_above_sma = float(self.params.get('volume_factor_above_sma', 1.5))
        self.min_breakout_atr_multiple = float(self.params.get('min_breakout_atr_multiple', 0.0)) # 0 means no filter

        logger.info(
            f"Initialized BollingerBandBreakoutStrategy for {symbol} ({timeframe}) "
            f"with BB Period: {self.bb_period}, StdDev: {self.bb_std_dev}, ATR Period: {self.atr_period}. "
            f"Volume Filter: {self.volume_filter_enabled} (SMA {self.volume_sma_period}, Factor {self.volume_factor_above_sma}). "
            f"Min Breakout ATR Multiple: {self.min_breakout_atr_multiple}. Shorting Enabled: {self.enable_shorting}"
        )

    def _calculate_indicators(self, data_df: pd.DataFrame) -> pd.DataFrame:
        if data_df.empty: return data_df
        df = data_df.copy()

        if 'close' not in df.columns or len(df) < self.bb_period:
            return df

        # Bollinger Bands
        try:
            bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
            if bbands is not None and not bbands.empty:
                df['bb_lower'] = bbands[f'BBL_{self.bb_period}_{self.bb_std_dev:.1f}'] # pandas_ta uses .1f for std in col name
                df['bb_middle'] = bbands[f'BBM_{self.bb_period}_{self.bb_std_dev:.1f}']
                df['bb_upper'] = bbands[f'BBU_{self.bb_period}_{self.bb_std_dev:.1f}']
            else:
                df['bb_lower'] = pd.NA
                df['bb_middle'] = pd.NA
                df['bb_upper'] = pd.NA
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            df['bb_lower'] = pd.NA
            df['bb_middle'] = pd.NA
            df['bb_upper'] = pd.NA


        # ATR
        if all(col in df.columns for col in ['high', 'low', 'close']):
            if len(df) >= self.atr_period:
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
            else:
                df['atr'] = pd.NA
        else:
            df['atr'] = pd.NA
        
        # Volume SMA (if filter enabled)
        if self.volume_filter_enabled and 'volume' in df.columns:
            if len(df) >= self.volume_sma_period:
                df['volume_sma'] = ta.sma(df['volume'], length=self.volume_sma_period)
            else:
                df['volume_sma'] = pd.NA
        elif 'volume' not in df.columns and self.volume_filter_enabled:
            logger.warning("Volume column not found for volume filter in BollingerBandBreakoutStrategy.")
            df['volume_sma'] = pd.NA

        return df

    def generate_signals(self, data: pd.DataFrame, sentiment_series: Optional[pd.Series] = None) -> pd.Series:
        if data.empty or len(data) < self.bb_period:
            logger.warning(f"Not enough historical data for Bollinger Band Breakout signals. Need {self.bb_period}, got {len(data)}.")
            return pd.Series(index=data.index, dtype='object')

        df_processed = self._calculate_indicators(data)

        if 'bb_upper' not in df_processed.columns or 'bb_lower' not in df_processed.columns:
            logger.error("Bollinger Band columns not found after calculation. Cannot generate signals.")
            return pd.Series(index=data.index, dtype='object')

        signals = pd.Series(index=df_processed.index, dtype='object').fillna(SignalType.HOLD)
        current_pos_state: Optional[SignalType] = None

        for i in range(1, len(df_processed)): # Start from 1 to check previous bar conditions
            close = df_processed['close'].iloc[i]
            prev_close = df_processed['close'].iloc[i-1]
            bb_upper = df_processed['bb_upper'].iloc[i]
            prev_bb_upper = df_processed['bb_upper'].iloc[i-1] if i > 0 and 'bb_upper' in df_processed.columns and pd.notna(df_processed['bb_upper'].iloc[i-1]) else bb_upper

            bb_lower = df_processed['bb_lower'].iloc[i]
            prev_bb_lower = df_processed['bb_lower'].iloc[i-1] if i > 0 and 'bb_lower' in df_processed.columns and pd.notna(df_processed['bb_lower'].iloc[i-1]) else bb_lower
            
            atr = df_processed['atr'].iloc[i] if 'atr' in df_processed.columns and pd.notna(df_processed['atr'].iloc[i]) else None

            if pd.isna(close) or pd.isna(prev_close) or pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(prev_bb_upper) or pd.isna(prev_bb_lower):
                continue

            # Volume Filter Check
            volume_condition_met = True
            if self.volume_filter_enabled and 'volume' in df_processed.columns and 'volume_sma' in df_processed.columns:
                current_volume = df_processed['volume'].iloc[i]
                volume_sma = df_processed['volume_sma'].iloc[i]
                if pd.notna(current_volume) and pd.notna(volume_sma) and volume_sma > 0: # ensure sma is not zero
                    if current_volume < volume_sma * self.volume_factor_above_sma:
                        volume_condition_met = False
                # else: volume_condition_met remains True if data insufficient for filter
            
            # Breakout Candle Size Filter
            breakout_candle_filter_met = True
            if self.min_breakout_atr_multiple > 0 and atr is not None and atr > 0:
                candle_range = abs(df_processed['high'].iloc[i] - df_processed['low'].iloc[i])
                if candle_range < self.min_breakout_atr_multiple * atr:
                    breakout_candle_filter_met = False


            # Long entry: Close breaks above upper band
            long_breakout = close > bb_upper and prev_close <= prev_bb_upper
            
            # Short entry: Close breaks below lower band
            short_breakout = close < bb_lower and prev_close >= prev_bb_lower

            if current_pos_state == SignalType.LONG:
                # Exit Long: Price closes back inside, e.g., below upper band or hits middle band
                if close < bb_upper: # Simple exit: re-enters band
                    signals.iloc[i] = SignalType.CLOSE_LONG
                    current_pos_state = None
            elif current_pos_state == SignalType.SHORT:
                 # Exit Short: Price closes back inside, e.g., above lower band or hits middle band
                if close > bb_lower: # Simple exit: re-enters band
                    signals.iloc[i] = SignalType.CLOSE_SHORT
                    current_pos_state = None
            else: # No position
                if long_breakout and volume_condition_met and breakout_candle_filter_met:
                    signals.iloc[i] = SignalType.LONG
                    current_pos_state = SignalType.LONG
                elif short_breakout and self.enable_shorting and volume_condition_met and breakout_candle_filter_met:
                    signals.iloc[i] = SignalType.SHORT
                    current_pos_state = SignalType.SHORT
        
        logger.info(f"Generated Bollinger Band Breakout signals. Longs: {signals.eq(SignalType.LONG).sum()}, Shorts: {signals.eq(SignalType.SHORT).sum()}, CloseLongs: {signals.eq(SignalType.CLOSE_LONG).sum()}, CloseShorts: {signals.eq(SignalType.CLOSE_SHORT).sum()}")
        return signals

    def on_bar_data(self, bar_data: BarData, sentiment_score: Optional[float] = None, market_regime_data: Optional[Any] = None) -> Union[Optional[SignalType], List[SignalCommand]]:
        self.update_data_history(bar_data)
        if len(self.data_history) < self.bb_period + 1: # Need enough data for BBands and prev close check
            return SignalType.HOLD

        df_with_indicators = self._calculate_indicators(self.data_history)
        
        if 'atr' in df_with_indicators.columns and pd.notna(df_with_indicators['atr'].iloc[-1]):
            bar_data.atr = df_with_indicators['atr'].iloc[-1]

        if df_with_indicators.empty or 'bb_upper' not in df_with_indicators.columns or \
           'bb_lower' not in df_with_indicators.columns or len(df_with_indicators) < 2:
            return SignalType.HOLD

        latest_close = df_with_indicators['close'].iloc[-1]
        prev_close = df_with_indicators['close'].iloc[-2]
        latest_bb_upper = df_with_indicators['bb_upper'].iloc[-1]
        prev_bb_upper = df_with_indicators['bb_upper'].iloc[-2] if pd.notna(df_with_indicators['bb_upper'].iloc[-2]) else latest_bb_upper
        latest_bb_lower = df_with_indicators['bb_lower'].iloc[-1]
        prev_bb_lower = df_with_indicators['bb_lower'].iloc[-2] if pd.notna(df_with_indicators['bb_lower'].iloc[-2]) else latest_bb_lower
        latest_atr = df_with_indicators['atr'].iloc[-1] if 'atr' in df_with_indicators.columns and pd.notna(df_with_indicators['atr'].iloc[-1]) else None

        if pd.isna(latest_close) or pd.isna(prev_close) or \
           pd.isna(latest_bb_upper) or pd.isna(prev_bb_upper) or \
           pd.isna(latest_bb_lower) or pd.isna(prev_bb_lower):
            return SignalType.HOLD

        signal_to_return = SignalType.HOLD

        # Volume Filter Check
        volume_condition_met = True
        if self.volume_filter_enabled and 'volume' in df_with_indicators.columns and 'volume_sma' in df_with_indicators.columns:
            current_volume = df_with_indicators['volume'].iloc[-1]
            volume_sma = df_with_indicators['volume_sma'].iloc[-1]
            if pd.notna(current_volume) and pd.notna(volume_sma) and volume_sma > 0:
                if current_volume < volume_sma * self.volume_factor_above_sma:
                    volume_condition_met = False
            # else: volume condition remains true if data insufficient

        # Breakout Candle Size Filter
        breakout_candle_filter_met = True
        if self.min_breakout_atr_multiple > 0 and latest_atr is not None and latest_atr > 0:
            candle_range = abs(df_with_indicators['high'].iloc[-1] - df_with_indicators['low'].iloc[-1])
            if candle_range < self.min_breakout_atr_multiple * latest_atr:
                breakout_candle_filter_met = False

        # Long entry: Close breaks above upper band (current close vs current upper, prev close vs prev upper)
        long_breakout_condition = latest_close > latest_bb_upper and prev_close <= prev_bb_upper
        
        # Short entry: Close breaks below lower band
        short_breakout_condition = latest_close < latest_bb_lower and prev_close >= prev_bb_lower

        if self.current_position_status == SignalType.LONG:
            # Exit Long: Price closes back inside (e.g., below current upper band)
            if latest_close < latest_bb_upper: 
                signal_to_return = SignalType.CLOSE_LONG
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - Bollinger CLOSE_LONG for {self.symbol}. Close: {latest_close:.2f}, BB_Upper: {latest_bb_upper:.2f}")
        elif self.current_position_status == SignalType.SHORT:
            # Exit Short: Price closes back inside (e.g., above current lower band)
            if latest_close > latest_bb_lower:
                signal_to_return = SignalType.CLOSE_SHORT
                self.current_position_status = None
                logger.info(f"{bar_data.timestamp} - Bollinger CLOSE_SHORT for {self.symbol}. Close: {latest_close:.2f}, BB_Lower: {latest_bb_lower:.2f}")
        else: # No current position
            if long_breakout_condition and volume_condition_met and breakout_candle_filter_met:
                signal_to_return = SignalType.LONG
                self.current_position_status = SignalType.LONG
                logger.info(f"{bar_data.timestamp} - Bollinger LONG for {self.symbol}. Close: {latest_close:.2f}, BB_Upper: {latest_bb_upper:.2f}. Vol Met: {volume_condition_met}, Candle Met: {breakout_candle_filter_met}. ATR: {bar_data.atr if bar_data.atr else 'N/A'}")
            elif short_breakout_condition and self.enable_shorting and volume_condition_met and breakout_candle_filter_met:
                signal_to_return = SignalType.SHORT
                self.current_position_status = SignalType.SHORT
                logger.info(f"{bar_data.timestamp} - Bollinger SHORT for {self.symbol}. Close: {latest_close:.2f}, BB_Lower: {latest_bb_lower:.2f}. Vol Met: {volume_condition_met}, Candle Met: {breakout_candle_filter_met}. ATR: {bar_data.atr if bar_data.atr else 'N/A'}")
                
        return signal_to_return