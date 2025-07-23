from typing import Optional, Dict, Any
import pandas as pd
import pandas_ta as ta
from cipher_edge.risk_control_module.stop_manager import BaseStopManager
from cipher_edge.core.models import BarData, Trade
from cipher_edge.core.enums import OrderSide
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class VolatilityBandStopManager(BaseStopManager):
    """
    Manages stops based on volatility bands like Bollinger Bands or Keltner Channels.
    Can be used for trailing stops along the bands.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.band_type = self.params.get('volatilitybandstop_band_type', 'bollinger').lower() # 'bollinger' or 'keltner'
        
        self.bb_period = int(self.params.get('volatilitybandstop_bb_period', 20))
        self.bb_std_dev = float(self.params.get('volatilitybandstop_bb_stddev', 2.0))
        
        self.kc_period = int(self.params.get('volatilitybandstop_kc_period', 20)) # EMA period
        self.kc_atr_period = int(self.params.get('volatilitybandstop_kc_atr_period', 10))
        self.kc_atr_multiplier = float(self.params.get('volatilitybandstop_kc_atr_multiplier', 1.5))

        self.trail_type = self.params.get('volatilitybandstop_trailtype', 'none').lower() # e.g., 'trailing_bb_upper', 'trailing_bb_lower', 'none'
        
        self.current_trailing_stop_price: Optional[float] = None

        logger.info(f"VolatilityBandStopManager initialized. Band: {self.band_type}, Trail: {self.trail_type}")

    def _calculate_bands(self, data_history: pd.DataFrame) -> pd.DataFrame:
        df = data_history.copy()
        if df.empty or len(df) < max(self.bb_period, self.kc_period, self.kc_atr_period):
            return df # Not enough data

        if self.band_type == 'bollinger':
            if 'close' in df.columns and len(df) >= self.bb_period:
                try:
                    bbands = ta.bbands(df['close'], length=self.bb_period, std=self.bb_std_dev)
                    if bbands is not None and not bbands.empty:
                        # pandas_ta typically names columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
                        df['band_lower'] = bbands[f'BBL_{self.bb_period}_{self.bb_std_dev:.1f}']
                        df['band_middle'] = bbands[f'BBM_{self.bb_period}_{self.bb_std_dev:.1f}']
                        df['band_upper'] = bbands[f'BBU_{self.bb_period}_{self.bb_std_dev:.1f}']
                except Exception as e:
                    logger.error(f"Error calculating Bollinger Bands for VolStopManager: {e}")
                    df['band_lower'] = df['band_middle'] = df['band_upper'] = pd.NA
            else:
                 df['band_lower'] = df['band_middle'] = df['band_upper'] = pd.NA

        elif self.band_type == 'keltner':
            if all(c in df.columns for c in ['high', 'low', 'close']) and len(df) >= max(self.kc_period, self.kc_atr_period):
                try:
                    kc = ta.kc(df['high'], df['low'], df['close'], length=self.kc_period, atr_length=self.kc_atr_period, mamode="EMA", multiplier=self.kc_atr_multiplier)
                    if kc is not None and not kc.empty:
                        df['band_lower'] = kc.iloc[:,0] # Lower band often first column
                        df['band_middle'] = kc.iloc[:,1] # Middle band
                        df['band_upper'] = kc.iloc[:,2] # Upper band
                except Exception as e:
                    logger.error(f"Error calculating Keltner Channels for VolStopManager: {e}")
                    df['band_lower'] = df['band_middle'] = df['band_upper'] = pd.NA
            else:
                 df['band_lower'] = df['band_middle'] = df['band_upper'] = pd.NA
        else:
            logger.warning(f"Unsupported band_type: {self.band_type} in VolatilityBandStopManager.")
            df['band_lower'] = df['band_middle'] = df['band_upper'] = pd.NA
        return df

    def check_stop_loss(
        self,
        current_trade: Trade,
        latest_bar: BarData,
        data_history_for_bands: Optional[pd.DataFrame] = None 
    ) -> Optional[float]:
        if not data_history_for_bands or data_history_for_bands.empty:
            logger.warning("Data history for bands not provided to VolatilityBandStopManager.")
            return None

        df_with_bands = self._calculate_bands(data_history_for_bands)
        if df_with_bands.empty or 'band_lower' not in df_with_bands.columns or 'band_upper' not in df_with_bands.columns:
            logger.debug("Bands not available for stop loss check.")
            return None
        
        latest_band_lower = df_with_bands['band_lower'].iloc[-1]
        latest_band_upper = df_with_bands['band_upper'].iloc[-1]

        if pd.isna(latest_band_lower) or pd.isna(latest_band_upper):
            logger.debug("Latest band values are NaN.")
            return None

        stop_price = None

        if self.trail_type == 'none': 
            if current_trade.side == OrderSide.BUY:
                stop_price = latest_band_lower # Simplistic: stop at current lower band
                if latest_bar.low <= stop_price:
                    logger.info(f"VOL_BAND STOP (BUY, fixed on current) for {current_trade.symbol} at {stop_price:.4f}")
                    return stop_price
            elif current_trade.side == OrderSide.SELL:
                stop_price = latest_band_upper # Simplistic: stop at current upper band
                if latest_bar.high >= stop_price:
                    logger.info(f"VOL_BAND STOP (SELL, fixed on current) for {current_trade.symbol} at {stop_price:.4f}")
                    return stop_price
        else: # Trailing stop logic
            if current_trade.side == OrderSide.BUY:
                # Trail stop along the lower band (or middle band)
                potential_stop = latest_band_lower # Default to lower band for long
                if self.trail_type == 'trailing_bb_middle' or self.trail_type == 'trailing_kc_middle':
                     if 'band_middle' in df_with_bands.columns and pd.notna(df_with_bands['band_middle'].iloc[-1]):
                        potential_stop = df_with_bands['band_middle'].iloc[-1]
                
                if self.current_trailing_stop_price is None or potential_stop > self.current_trailing_stop_price:
                    self.current_trailing_stop_price = potential_stop
                
                if self.current_trailing_stop_price and latest_bar.low <= self.current_trailing_stop_price:
                    logger.info(f"VOL_BAND TRAILING STOP (BUY) for {current_trade.symbol} at {self.current_trailing_stop_price:.4f}")
                    stop_price_to_return = self.current_trailing_stop_price
                    self.current_trailing_stop_price = None # Reset after hit
                    return stop_price_to_return

            elif current_trade.side == OrderSide.SELL:
                potential_stop = latest_band_upper # Default to upper band for short
                if self.trail_type == 'trailing_bb_middle' or self.trail_type == 'trailing_kc_middle':
                     if 'band_middle' in df_with_bands.columns and pd.notna(df_with_bands['band_middle'].iloc[-1]):
                        potential_stop = df_with_bands['band_middle'].iloc[-1]

                if self.current_trailing_stop_price is None or potential_stop < self.current_trailing_stop_price:
                    self.current_trailing_stop_price = potential_stop

                if self.current_trailing_stop_price and latest_bar.high >= self.current_trailing_stop_price:
                    logger.info(f"VOL_BAND TRAILING STOP (SELL) for {current_trade.symbol} at {self.current_trailing_stop_price:.4f}")
                    stop_price_to_return = self.current_trailing_stop_price
                    self.current_trailing_stop_price = None # Reset after hit
                    return stop_price_to_return
        return None

    def check_take_profit(
        self,
        current_trade: Trade,
        latest_bar: BarData,
    ) -> Optional[float]:
        if self.current_trailing_stop_price is not None and current_trade.exit_timestamp is not None:
             self.current_trailing_stop_price = None
        return None

    def reset_trailing_stop(self):
        """Called when a new trade is initiated or an old one is closed by other means."""
        self.current_trailing_stop_price = None