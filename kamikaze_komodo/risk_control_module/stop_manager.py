# FILE: kamikaze_komodo/risk_control_module/stop_manager.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from kamikaze_komodo.core.models import BarData, Trade
from kamikaze_komodo.core.enums import OrderSide
from kamikaze_komodo.app_logger import get_logger
from datetime import timedelta

logger = get_logger(__name__)

class BaseStopManager(ABC):
    """
    Abstract base class for stop-loss and take-profit management.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else {}
        logger.info(f"{self.__class__.__name__} initialized with params: {self.params}")

    @abstractmethod
    def check_stop_loss(
        self,
        current_trade: Trade,
        latest_bar: BarData,
        bar_index: int # New param needed for time-based stops
    ) -> Optional[float]: # Returns stop price if triggered, else None
        """
        Checks if the stop-loss condition is met for the current trade.
        """
        pass

    @abstractmethod
    def check_take_profit(
        self,
        current_trade: Trade,
        latest_bar: BarData
    ) -> Optional[float]: # Returns take profit price if triggered, else None
        """
        Checks if the take-profit condition is met for the current trade.
        """
        pass

class PercentageStopManager(BaseStopManager):
    """
    Manages stops based on a fixed percentage from the entry price.
    """
    def __init__(self, stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.stop_loss_pct = float(self.params.get('percentagestop_losspct', stop_loss_pct if stop_loss_pct is not None else 0))
        self.take_profit_pct = float(self.params.get('percentagestop_takeprofitpct', take_profit_pct if take_profit_pct is not None else 0))

        if self.stop_loss_pct < 0 or self.stop_loss_pct >= 1.0 :
              if self.stop_loss_pct != 0:
                  raise ValueError("stop_loss_pct must be between 0 (inclusive, to disable) and 1 (exclusive).")
        if self.take_profit_pct < 0:
              if self.take_profit_pct != 0:
                  raise ValueError("take_profit_pct must be non-negative (0 to disable).")
        
        self.stop_loss_pct = None if self.stop_loss_pct == 0 else self.stop_loss_pct
        self.take_profit_pct = None if self.take_profit_pct == 0 else self.take_profit_pct
            
        logger.info(f"PercentageStopManager initialized. SL: {self.stop_loss_pct*100 if self.stop_loss_pct else 'N/A'}%, TP: {self.take_profit_pct*100 if self.take_profit_pct else 'N/A'}%")

    def check_stop_loss(self, current_trade: Trade, latest_bar: BarData, bar_index: int) -> Optional[float]:
        if not self.stop_loss_pct or not current_trade or current_trade.entry_price <= 0:
            return None

        if current_trade.side == OrderSide.BUY:
            stop_price = current_trade.entry_price * (1 - self.stop_loss_pct)
            if latest_bar.low <= stop_price:
                logger.info(f"STOP LOSS (BUY) triggered for trade {current_trade.id} ({current_trade.symbol}) at SL price {stop_price:.4f} (Bar Low: {latest_bar.low:.4f}, Entry: {current_trade.entry_price:.4f})")
                return stop_price
        elif current_trade.side == OrderSide.SELL:
            stop_price = current_trade.entry_price * (1 + self.stop_loss_pct)
            if latest_bar.high >= stop_price:
                logger.info(f"STOP LOSS (SELL) triggered for trade {current_trade.id} ({current_trade.symbol}) at SL price {stop_price:.4f} (Bar High: {latest_bar.high:.4f}, Entry: {current_trade.entry_price:.4f})")
                return stop_price
        return None

    def check_take_profit(self, current_trade: Trade, latest_bar: BarData) -> Optional[float]:
        if not self.take_profit_pct or not current_trade or current_trade.entry_price <= 0:
            return None

        if current_trade.side == OrderSide.BUY:
            profit_price = current_trade.entry_price * (1 + self.take_profit_pct)
            if latest_bar.high >= profit_price:
                logger.info(f"TAKE PROFIT (BUY) triggered for trade {current_trade.id} ({current_trade.symbol}) at TP price {profit_price:.4f} (Bar High: {latest_bar.high:.4f}, Entry: {current_trade.entry_price:.4f})")
                return profit_price
        elif current_trade.side == OrderSide.SELL:
            profit_price = current_trade.entry_price * (1 - self.take_profit_pct)
            if latest_bar.low <= profit_price:
                logger.info(f"TAKE PROFIT (SELL) triggered for trade {current_trade.id} ({current_trade.symbol}) at TP price {profit_price:.4f} (Bar Low: {latest_bar.low:.4f}, Entry: {current_trade.entry_price:.4f})")
                return profit_price
        return None

class ATRStopManager(BaseStopManager):
    """
    Manages stops based on ATR.
    """
    def __init__(self, atr_multiple: float = 2.0, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.atr_multiple = float(self.params.get('atrstop_atrmultiple', atr_multiple))
        logger.info(f"ATRStopManager initialized with ATR multiple: {self.atr_multiple}")

    def check_stop_loss(self, current_trade: Trade, latest_bar: BarData, bar_index: int) -> Optional[float]:
        atr_at_entry = current_trade.custom_fields.get("atr_at_entry") if hasattr(current_trade, 'custom_fields') and current_trade.custom_fields else None
        
        if atr_at_entry is None or atr_at_entry <= 1e-8:
            if latest_bar.atr is not None and latest_bar.atr > 1e-8:
                logger.debug(f"ATR at entry not available for trade {current_trade.id}. Using latest_bar.atr ({latest_bar.atr:.6f}) for ATR stop check.")
                atr_at_entry = latest_bar.atr
            else:
                logger.debug(f"ATR value not available or invalid for trade {current_trade.id}. Cannot apply ATR stop. ATR at entry: {atr_at_entry}, Latest bar ATR: {latest_bar.atr}")
                return None
        
        stop_distance = self.atr_multiple * atr_at_entry
        if current_trade.side == OrderSide.BUY:
            stop_price = current_trade.entry_price - stop_distance
            if latest_bar.low <= stop_price:
                logger.info(f"ATR STOP LOSS (BUY) for {current_trade.symbol} at {stop_price:.4f} (Entry: {current_trade.entry_price:.4f}, ATR Used: {atr_at_entry:.6f}, BarLow: {latest_bar.low:.4f})")
                return stop_price
        elif current_trade.side == OrderSide.SELL:
            stop_price = current_trade.entry_price + stop_distance
            if latest_bar.high >= stop_price:
                logger.info(f"ATR STOP LOSS (SELL) for {current_trade.symbol} at {stop_price:.4f} (Entry: {current_trade.entry_price:.4f}, ATR Used: {atr_at_entry:.6f}, BarHigh: {latest_bar.high:.4f})")
                return stop_price
        return None

    def check_take_profit(self, current_trade: Trade, latest_bar: BarData) -> Optional[float]:
        return None

class TripleBarrierStopManager(BaseStopManager):
    """
    Implements De Prado's Triple-Barrier Method.
    Sets a stop-loss, a take-profit, and a time-based exit.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.tp_multiple = float(self.params.get('triplebarrier_tp_multiple', 1.5))  # Take profit as a multiple of risk (ATR)
        self.sl_multiple = float(self.params.get('triplebarrier_sl_multiple', 1.0))  # Stop loss as a multiple of risk (ATR)
        self.time_limit_bars = int(self.params.get('triplebarrier_time_limit_bars', 10)) # Time limit in number of bars
        logger.info(f"TripleBarrierStopManager initialized. TP Multiple: {self.tp_multiple}, SL Multiple: {self.sl_multiple}, Time Limit: {self.time_limit_bars} bars.")

    def _get_risk_and_barriers(self, trade: Trade, bar: BarData) -> Optional[Dict[str, float]]:
        """Calculates the risk (e.g., ATR at entry) and derives the barriers."""
        risk_per_unit = trade.custom_fields.get("atr_at_entry")
        if risk_per_unit is None or risk_per_unit <= 1e-8:
            logger.warning(f"ATR at entry not found for trade {trade.id}. Cannot apply Triple Barrier.")
            return None

        entry_price = trade.entry_price
        if trade.side == OrderSide.BUY:
            sl_price = entry_price - (self.sl_multiple * risk_per_unit)
            tp_price = entry_price + (self.tp_multiple * risk_per_unit)
        else: # OrderSide.SELL
            sl_price = entry_price + (self.sl_multiple * risk_per_unit)
            tp_price = entry_price - (self.tp_multiple * risk_per_unit)

        return {"sl_price": sl_price, "tp_price": tp_price}

    def check_stop_loss(self, current_trade: Trade, latest_bar: BarData, bar_index: int) -> Optional[float]:
        barriers = self._get_risk_and_barriers(current_trade, latest_bar)
        if barriers is None:
            return None

        sl_price = barriers['sl_price']
        
        # 1. Price-based stop loss
        if current_trade.side == OrderSide.BUY and latest_bar.low <= sl_price:
            logger.info(f"Triple-Barrier SL (BUY) triggered for trade {current_trade.id} at {sl_price:.4f}")
            return sl_price
        if current_trade.side == OrderSide.SELL and latest_bar.high >= sl_price:
            logger.info(f"Triple-Barrier SL (SELL) triggered for trade {current_trade.id} at {sl_price:.4f}")
            return sl_price
        
        # 2. Time-based stop (vertical barrier)
        entry_bar_index = current_trade.custom_fields.get("entry_bar_index")
        if entry_bar_index is not None:
            bars_held = bar_index - entry_bar_index
            if bars_held >= self.time_limit_bars:
                logger.info(f"Triple-Barrier TIME LIMIT reached for trade {current_trade.id} after {bars_held} bars.")
                return latest_bar.close # Exit at the current closing price
        
        return None

    def check_take_profit(self, current_trade: Trade, latest_bar: BarData) -> Optional[float]:
        barriers = self._get_risk_and_barriers(current_trade, latest_bar)
        if barriers is None:
            return None

        tp_price = barriers['tp_price']
        
        if current_trade.side == OrderSide.BUY and latest_bar.high >= tp_price:
            logger.info(f"Triple-Barrier TP (BUY) triggered for trade {current_trade.id} at {tp_price:.4f}")
            return tp_price
        if current_trade.side == OrderSide.SELL and latest_bar.low <= tp_price:
            logger.info(f"Triple-Barrier TP (SELL) triggered for trade {current_trade.id} at {tp_price:.4f}")
            return tp_price
        
        return None