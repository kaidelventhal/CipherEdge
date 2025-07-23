from typing import Optional, Dict, Any
from datetime import timedelta
from enum import Enum

from cipher_edge.risk_control_module.stop_manager import BaseStopManager
from cipher_edge.core.models import BarData, Trade
from cipher_edge.core.enums import OrderSide
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class StopTriggerType(Enum):
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TIME_LIMIT = "TIME_LIMIT"

class TripleBarrierStop(BaseStopManager):
    """
    Implements De Prado's Triple-Barrier Method (stop-loss, take-profit, time limit).
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.profit_barrier_multiplier = float(self.params.get('triplebarrier_profitmultiplier', 2.0))
        self.loss_barrier_multiplier = float(self.params.get('triplebarrier_lossmultiplier', 1.0))
        self.time_barrier_days = int(self.params.get('triplebarrier_timelimitdays', 10))
        
        self.active_trade_barriers: Dict[str, Dict[str, Any]] = {} 

        logger.info(
            f"TripleBarrierStop initialized. "
            f"Profit Multiplier: {self.profit_barrier_multiplier}, "
            f"Loss Multiplier: {self.loss_barrier_multiplier}, "
            f"Time Limit: {self.time_barrier_days} days."
        )

    def calculate_barriers(self, trade: Trade, current_bar: BarData):
        if trade.id in self.active_trade_barriers:
            return

        atr_at_entry = trade.custom_fields.get("atr_at_entry")
        if atr_at_entry is None or atr_at_entry <= 0:
            logger.warning(f"ATR at entry not found for trade {trade.id}. Using 1% of entry price as risk unit.")
            risk_unit = trade.entry_price * 0.01
        else:
            risk_unit = atr_at_entry
        
        entry_price = trade.entry_price
        
        if trade.side == OrderSide.BUY:
            sl_price = entry_price - (self.loss_barrier_multiplier * risk_unit)
            tp_price = entry_price + (self.profit_barrier_multiplier * risk_unit)
        else: # SELL
            sl_price = entry_price + (self.loss_barrier_multiplier * risk_unit)
            tp_price = entry_price - (self.profit_barrier_multiplier * risk_unit)
            
        time_barrier_timestamp = trade.entry_timestamp + timedelta(days=self.time_barrier_days)

        self.active_trade_barriers[trade.id] = {
            "sl_price": sl_price,
            "tp_price": tp_price,
            "time_barrier_timestamp": time_barrier_timestamp
        }
        logger.info(f"Barriers calculated for trade {trade.id}: SL={sl_price:.4f}, TP={tp_price:.4f}, Time={time_barrier_timestamp}")

    def check_bar(self, trade: Trade, current_bar: BarData) -> Optional[StopTriggerType]:
        barriers = self.active_trade_barriers.get(trade.id)
        if not barriers:
            return None

        if trade.side == OrderSide.BUY:
            if current_bar.low <= barriers['sl_price']:
                return StopTriggerType.STOP_LOSS
            if current_bar.high >= barriers['tp_price']:
                return StopTriggerType.TAKE_PROFIT
        else: # SELL
            if current_bar.high >= barriers['sl_price']:
                return StopTriggerType.STOP_LOSS
            if current_bar.low <= barriers['tp_price']:
                return StopTriggerType.TAKE_PROFIT
        
        if current_bar.timestamp >= barriers['time_barrier_timestamp']:
            return StopTriggerType.TIME_LIMIT
            
        return None
    
    def reset_for_trade(self, trade_id: str):
        """Clears stored barriers for a specific trade."""
        if trade_id in self.active_trade_barriers:
            del self.active_trade_barriers[trade_id]

    def check_stop_loss(self, current_trade: Trade, latest_bar: BarData, bar_index: int, **kwargs) -> Optional[float]:
        trigger = self.check_bar(current_trade, latest_bar)
        if trigger in [StopTriggerType.STOP_LOSS, StopTriggerType.TIME_LIMIT]:
            return self.active_trade_barriers.get(current_trade.id, {}).get('sl_price', latest_bar.close)
        return None

    def check_take_profit(self, current_trade: Trade, latest_bar: BarData, **kwargs) -> Optional[float]:
        trigger = self.check_bar(current_trade, latest_bar)
        if trigger == StopTriggerType.TAKE_PROFIT:
            return self.active_trade_barriers.get(current_trade.id, {}).get('tp_price', latest_bar.close)
        return None