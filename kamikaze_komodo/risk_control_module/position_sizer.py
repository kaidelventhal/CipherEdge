# kamikaze_komodo/risk_control_module/position_sizer.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from kamikaze_komodo.core.models import BarData # For ATR based sizers potentially
from kamikaze_komodo.app_logger import get_logger
logger = get_logger(__name__)
class BasePositionSizer(ABC):
    """
    Abstract base class for position sizing strategies.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else {}
        logger.info(f"{self.__class__.__name__} initialized with params: {self.params}")
    @abstractmethod
    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float,
        current_portfolio_value: float, # Total equity
        strategy_signal_strength: Optional[float] = None, # e.g. ML confidence
        latest_bar: Optional[BarData] = None, # For ATR or volatility based
        atr_value: Optional[float] = None # Explicit ATR if available
    ) -> Optional[float]: # Returns position size in asset units, or None if no trade
        """
        Calculates the size of the position to take.
        Args:
            symbol (str): The asset symbol.
            current_price (float): The current price of the asset.
            available_capital (float): The cash available for trading. (May not be used by all sizers)
            current_portfolio_value (float): The total current value of the portfolio (equity).
            strategy_signal_strength (Optional[float]): Confidence or strength of the signal.
            latest_bar (Optional[BarData]): Latest bar data for volatility calculation.
            atr_value (Optional[float]): Pre-calculated ATR value.
        Returns:
            Optional[float]: The quantity of the asset to trade. None if cannot size or no trade.
        """
        pass
class FixedFractionalPositionSizer(BasePositionSizer):
    """
    Sizes positions based on a fixed fraction of the total portfolio equity.
    """
    def __init__(self, fraction: float = 0.01, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        if not 0 < fraction <= 1.0:
            logger.error(f"Fraction must be between 0 (exclusive) and 1 (inclusive). Got {fraction}")
            raise ValueError("Fraction must be > 0 and <= 1.")
        self.fraction_to_risk = fraction # This is the fraction of *equity* to risk
        logger.info(f"FixedFractionalPositionSizer initialized with fraction: {self.fraction_to_risk}")
    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float, # Cash
        current_portfolio_value: float, # Equity
        strategy_signal_strength: Optional[float] = None,
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        if current_price <= 0:
            logger.warning(f"Current price for {symbol} is non-positive ({current_price}). Cannot calculate position size.")
            return None
        if current_portfolio_value <= 0:
            logger.warning(f"Current portfolio value is non-positive ({current_portfolio_value}). Cannot calculate position size.")
            return None
        # This sizer determines how much capital to *allocate* to this trade based on fixed fraction of equity.
        # It does not inherently consider a stop-loss to determine "risk per trade".
        # A more common "fixed fractional" risks a fraction of equity *based on a stop-loss distance*.
        # The current plan says "fixed fractional", so interpreting as fraction of equity to *allocate*.
        
        capital_to_allocate = current_portfolio_value * self.fraction_to_risk
        
        # Ensure we don't allocate more than available cash if that's a constraint.
        # This check is important because `current_portfolio_value` includes value of existing positions.
        # However, for a new trade, we typically use available cash.
        # Let's assume the "fraction" applies to overall equity to decide the *value* of the new position.
        
        if capital_to_allocate > available_capital :
            logger.warning(f"Calculated capital to allocate ({capital_to_allocate:.2f}) for {symbol} exceeds available cash ({available_capital:.2f}). Using available cash.")
            capital_to_allocate = available_capital
        if capital_to_allocate <= 0:
            logger.info(f"No capital to allocate for {symbol} based on fixed fraction or available cash.")
            return None
        position_size = capital_to_allocate / current_price
        logger.info(f"FixedFractional Sizing for {symbol}: Allocating ${capital_to_allocate:.2f} (Equity: ${current_portfolio_value:.2f}, Fraction: {self.fraction_to_risk}). Position Size: {position_size:.6f} units at ${current_price:.2f}.")
        return position_size
class ATRBasedPositionSizer(BasePositionSizer):
    """
    Sizes positions based on Average True Range (ATR) to normalize risk per trade.
    This implementation assumes you risk a fixed percentage of portfolio equity,
    and the stop loss is N * ATR away.
    """
    def __init__(self, risk_per_trade_fraction: float = 0.01, atr_multiple_for_stop: float = 2.0, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.risk_per_trade_fraction = risk_per_trade_fraction # e.g., 0.01 for 1% of equity
        self.atr_multiple_for_stop = atr_multiple_for_stop # e.g., stop loss is 2 * ATR
        logger.info(f"ATRBasedPositionSizer initialized with risk_fraction: {self.risk_per_trade_fraction}, atr_multiple: {self.atr_multiple_for_stop}")
    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float,
        current_portfolio_value: float,
        strategy_signal_strength: Optional[float] = None,
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None # Allow passing pre-calculated ATR
    ) -> Optional[float]:
        if atr_value is None:
            if latest_bar and 'atr_period' in self.params:
                # Simplified: This would typically need a history of bars to calculate ATR.
                # For a single 'latest_bar', ATR calculation isn't directly possible unless it's already on the bar.
                # We'd normally use pandas_ta on a series.
                # This is a placeholder for where one would calculate ATR if not provided.
                logger.warning("ATR value not provided and history for calculation from latest_bar is not implemented in sizer directly. ATR must be passed or calculated by strategy/engine.")
                return None # Cannot calculate ATR here with just one bar without more logic
            else:
                logger.warning(f"ATR value not provided for {symbol}, and cannot calculate it. Required for ATRBasedPositionSizer.")
                return None
        
        if atr_value <= 0:
            logger.warning(f"ATR value for {symbol} is non-positive ({atr_value}). Cannot size position.")
            return None
        if current_price <= 0:
            logger.warning(f"Current price for {symbol} is non-positive ({current_price}). Cannot size position.")
            return None
        if current_portfolio_value <= 0:
            logger.warning(f"Current portfolio value is non-positive ({current_portfolio_value}). Cannot size position.")
            return None
        # Amount of capital to risk on this trade
        capital_at_risk = current_portfolio_value * self.risk_per_trade_fraction
        
        # Stop distance in price terms
        stop_distance_per_unit = self.atr_multiple_for_stop * atr_value
        if stop_distance_per_unit == 0:
            logger.warning(f"Stop distance per unit is zero for {symbol} (ATR: {atr_value}, Multiple: {self.atr_multiple_for_stop}). Cannot size position.")
            return None
        # Number of units (position size)
        position_size = capital_at_risk / stop_distance_per_unit
        # Cost of this position
        position_cost = position_size * current_price
        if position_cost > available_capital:
            logger.warning(f"Calculated position cost (${position_cost:.2f}) for {symbol} exceeds available cash (${available_capital:.2f}). Reducing size.")
            position_size = available_capital / current_price # Adjust size to available cash
            if position_size == 0: return None
        logger.info(f"ATRBased Sizing for {symbol}: Risking ${capital_at_risk:.2f} (Equity: ${current_portfolio_value:.2f}). "
                    f"ATR: {atr_value:.4f}, StopDist: ${stop_distance_per_unit:.2f}. "
                    f"Size: {position_size:.6f} units at ${current_price:.2f}.")
        return position_size