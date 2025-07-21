# FILE: kamikaze_komodo/risk_control_module/parabolic_sar_stop.py
from typing import Optional, Dict, Any
import pandas as pd
import pandas_ta as ta
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager
from kamikaze_komodo.core.models import BarData, Trade
from kamikaze_komodo.core.enums import OrderSide
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class ParabolicSARStop(BaseStopManager):
    """
    Implements a dynamic stop-loss based on the Parabolic SAR (Stop and Reverse) indicator.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.acceleration_factor = float(self.params.get('parabolicsar_accelerationfactor', 0.02))
        self.max_acceleration = float(self.params.get('parabolicsar_maxacceleration', 0.2))
         
        logger.info(f"ParabolicSARStop initialized. AF={self.acceleration_factor}, Max AF={self.max_acceleration}")

    def check_stop_loss(
        self,
        current_trade: Trade,
        latest_bar: BarData,
        bar_index: int,
        **kwargs 
    ) -> Optional[float]:
        """
        Checks if the Parabolic SAR stop-loss is triggered for the current trade.
        """
        data_history_for_sar = kwargs.get('data_history_for_sar') 
        if data_history_for_sar is None or data_history_for_sar.empty:
            logger.warning("Data history not provided for Parabolic SAR calculation.")
            return None
         
        if len(data_history_for_sar) < 3: # SAR needs at least a few bars
            return None

        try:
            # **FIX**: Corrected the function call from ta.sar to ta.psar
            sar_values = ta.psar(
                high=data_history_for_sar['high'],
                low=data_history_for_sar['low'],
                af=self.acceleration_factor,
                max_af=self.max_acceleration
            )
             
            if sar_values is None or sar_values.empty:
                return None
             
            # Get the last value from the first column of the resulting DataFrame
            latest_sar = sar_values.iloc[-1, 0]

            if pd.isna(latest_sar):
                return None
             
            stop_price = float(latest_sar)
             
            if current_trade.side == OrderSide.BUY and latest_bar.low <= stop_price:
                logger.info(f"Parabolic SAR stop (LONG) triggered for {current_trade.symbol} at {stop_price:.4f}")
                return stop_price
            elif current_trade.side == OrderSide.SELL and latest_bar.high >= stop_price:
                logger.info(f"Parabolic SAR stop (SHORT) triggered for {current_trade.symbol} at {stop_price:.4f}")
                return stop_price

        except Exception as e:
            logger.error(f"Error calculating Parabolic SAR: {e}", exc_info=True)
         
        return None

    def check_take_profit(self, current_trade: Trade, latest_bar: BarData, **kwargs) -> Optional[float]:
        return None