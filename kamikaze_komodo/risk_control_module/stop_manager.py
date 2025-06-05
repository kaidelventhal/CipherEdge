# kamikaze_komodo/risk_control_module/stop_manager.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from kamikaze_komodo.core.models import BarData, Trade
from kamikaze_komodo.core.enums import OrderSide
from kamikaze_komodo.app_logger import get_logger
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
        latest_bar: BarData
    ) -> Optional[float]: # Returns stop price if triggered, else None
        """
        Checks if the stop-loss condition is met for the current trade.
        Args:
            current_trade (Trade): The active trade object.
            latest_bar (BarData): The latest market data bar.
        Returns:
            Optional[float]: The price at which the stop-loss was triggered, or None if not triggered.
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
        Args:
            current_trade (Trade): The active trade object.
            latest_bar (BarData): The latest market data bar.
        Returns:
            Optional[float]: The price at which the take-profit was triggered, or None if not triggered.
        """
        pass
class PercentageStopManager(BaseStopManager):
    """
    Manages stops based on a fixed percentage from the entry price.
    """
    def __init__(self, stop_loss_pct: Optional[float] = None, take_profit_pct: Optional[float] = None, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        if self.stop_loss_pct is not None and not (0 < self.stop_loss_pct < 1.0):
            raise ValueError("stop_loss_pct must be between 0 and 1 (exclusive).")
        if self.take_profit_pct is not None and not (0 < self.take_profit_pct): # TP can be > 100%
            raise ValueError("take_profit_pct must be greater than 0.")
            
        logger.info(f"PercentageStopManager initialized. SL: {self.stop_loss_pct*100 if self.stop_loss_pct else 'N/A'}%, TP: {self.take_profit_pct*100 if self.take_profit_pct else 'N/A'}%")
    def check_stop_loss(
        self,
        current_trade: Trade,
        latest_bar: BarData
    ) -> Optional[float]:
        if not self.stop_loss_pct or not current_trade or current_trade.entry_price <= 0:
            return None
        if current_trade.side == OrderSide.BUY:
            stop_price = current_trade.entry_price * (1 - self.stop_loss_pct)
            if latest_bar.low <= stop_price:
                logger.info(f"STOP LOSS (BUY) triggered for trade {current_trade.id} ({current_trade.symbol}) at SL price {stop_price:.2f} (Bar Low: {latest_bar.low:.2f}, Entry: {current_trade.entry_price:.2f})")
                return stop_price # Exit at the calculated stop price
        elif current_trade.side == OrderSide.SELL: # For short positions
            stop_price = current_trade.entry_price * (1 + self.stop_loss_pct)
            if latest_bar.high >= stop_price:
                logger.info(f"STOP LOSS (SELL) triggered for trade {current_trade.id} ({current_trade.symbol}) at SL price {stop_price:.2f} (Bar High: {latest_bar.high:.2f}, Entry: {current_trade.entry_price:.2f})")
                return stop_price
        return None
    def check_take_profit(
        self,
        current_trade: Trade,
        latest_bar: BarData
    ) -> Optional[float]:
        if not self.take_profit_pct or not current_trade or current_trade.entry_price <= 0:
            return None
        if current_trade.side == OrderSide.BUY:
            profit_price = current_trade.entry_price * (1 + self.take_profit_pct)
            if latest_bar.high >= profit_price:
                logger.info(f"TAKE PROFIT (BUY) triggered for trade {current_trade.id} ({current_trade.symbol}) at TP price {profit_price:.2f} (Bar High: {latest_bar.high:.2f}, Entry: {current_trade.entry_price:.2f})")
                return profit_price # Exit at the calculated profit price
        elif current_trade.side == OrderSide.SELL: # For short positions
            profit_price = current_trade.entry_price * (1 - self.take_profit_pct)
            if latest_bar.low <= profit_price:
                logger.info(f"TAKE PROFIT (SELL) triggered for trade {current_trade.id} ({current_trade.symbol}) at TP price {profit_price:.2f} (Bar Low: {latest_bar.low:.2f}, Entry: {current_trade.entry_price:.2f})")
                return profit_price
        return None
class ATRStopManager(BaseStopManager):
    """
    Manages stops based on ATR.
    """
    def __init__(self, atr_multiple: float = 2.0, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.atr_multiple = atr_multiple
        # This manager expects ATR to be provided, typically calculated elsewhere (e.g. by strategy or engine)
        # and stored/passed with BarData or Trade object.
        logger.info(f"ATRStopManager initialized with ATR multiple: {self.atr_multiple}")
    def check_stop_loss(
        self,
        current_trade: Trade, # current_trade needs a field like `atr_at_entry`
        latest_bar: BarData  # latest_bar could have a field like `current_atr`
    ) -> Optional[float]:
        # This is a more advanced stop. For Phase 3, PercentageStopManager is primary.
        # This requires `atr_at_entry` to be stored with the trade.
        atr_at_entry = current_trade.custom_fields.get("atr_at_entry") if hasattr(current_trade, 'custom_fields') and current_trade.custom_fields else None
        if atr_at_entry is None or atr_at_entry <= 0:
            logger.debug(f"ATR at entry not available or invalid for trade {current_trade.id}. Cannot apply ATR stop.")
            return None
        
        stop_distance = self.atr_multiple * atr_at_entry
        if current_trade.side == OrderSide.BUY:
            stop_price = current_trade.entry_price - stop_distance
            if latest_bar.low <= stop_price:
                logger.info(f"ATR STOP LOSS (BUY) for {current_trade.symbol} at {stop_price:.2f} (Entry: {current_trade.entry_price:.2f}, ATR@Entry: {atr_at_entry:.4f}, BarLow: {latest_bar.low:.2f})")
                return stop_price
        elif current_trade.side == OrderSide.SELL:
            stop_price = current_trade.entry_price + stop_distance
            if latest_bar.high >= stop_price:
                logger.info(f"ATR STOP LOSS (SELL) for {current_trade.symbol} at {stop_price:.2f} (Entry: {current_trade.entry_price:.2f}, ATR@Entry: {atr_at_entry:.4f}, BarHigh: {latest_bar.high:.2f})")
                return stop_price
        return None
    def check_take_profit(
        self,
        current_trade: Trade,
        latest_bar: BarData
    ) -> Optional[float]:
        # ATR stops are typically not used for take profit in the same way.
        # Take profit might be a multiple of risk (e.g. 2R, where R is ATR stop distance) or other logic.
        # For simplicity, this basic ATRStopManager doesn't implement TP.
        return None