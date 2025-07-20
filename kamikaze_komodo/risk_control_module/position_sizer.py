# FILE: kamikaze_komodo/risk_control_module/position_sizer.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import numpy as np
from kamikaze_komodo.core.models import BarData # For ATR based sizers potentially
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.core.enums import SignalType # Added for OptimalF and MLConfidence sizers
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
        trade_signal: SignalType, # Added for OptimalF and MLConfidence sizers
        strategy_info: Dict[str, Any], # Added for MLConfidence sizer (e.g. ML confidence)
        latest_bar: Optional[BarData] = None, # For ATR or volatility based
        atr_value: Optional[float] = None # Explicit ATR if available
    ) -> Optional[float]: # Returns position size in asset units, or None if no trade
        """
        Calculates the size of the position to take.
        """
        pass

class FixedFractionalPositionSizer(BasePositionSizer):
    """
    Sizes positions based on a fixed fraction of the total portfolio equity.
    """
    def __init__(self, fraction: float = 0.01, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.fraction_to_allocate = float(self.params.get('fixedfractional_allocationfraction', fraction))
        if not 0 < self.fraction_to_allocate <= 1.0:
            logger.error(f"Fraction must be between 0 (exclusive) and 1 (inclusive). Got {self.fraction_to_allocate}")
            raise ValueError("Fraction must be > 0 and <= 1.")
        logger.info(f"FixedFractionalPositionSizer initialized with fraction: {self.fraction_to_allocate}")

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float, # Cash
        current_portfolio_value: float, # Equity
        trade_signal: SignalType,
        strategy_info: Dict[str, Any],
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        if current_price <= 0 or current_portfolio_value <= 0:
            return None
        
        capital_to_allocate = current_portfolio_value * self.fraction_to_allocate
        
        if trade_signal == SignalType.LONG and capital_to_allocate > available_capital :
            capital_to_allocate = available_capital
        
        if capital_to_allocate <= 1.0: # Minimum capital to allocate (e.g. $1)
            return None

        position_size = capital_to_allocate / current_price
        # FIX: Change log level to DEBUG to reduce noise
        logger.debug(f"FixedFractional Sizing for {symbol}: Allocating ${capital_to_allocate:.2f}. Position Size: {position_size:.8f} units.")
        return position_size

class ATRBasedPositionSizer(BasePositionSizer):
    """
    Sizes positions based on Average True Range (ATR) to normalize risk per trade.
    """
    def __init__(self, risk_per_trade_fraction: float = 0.01, atr_multiple_for_stop: float = 2.0, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.risk_per_trade_fraction = float(self.params.get('atrbased_riskpertradefraction', risk_per_trade_fraction))
        self.atr_multiple_for_stop = float(self.params.get('atrbased_atrmultipleforstop', atr_multiple_for_stop))
        logger.info(f"ATRBasedPositionSizer initialized with risk_fraction: {self.risk_per_trade_fraction}, atr_multiple: {self.atr_multiple_for_stop}")

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float,
        current_portfolio_value: float,
        trade_signal: SignalType,
        strategy_info: Dict[str, Any],
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        
        effective_atr = atr_value if atr_value is not None else getattr(latest_bar, 'atr', None)
        
        if effective_atr is None or not np.isfinite(effective_atr) or effective_atr <= 1e-8:
            logger.warning(f"ATR value for {symbol} is invalid ({effective_atr}). Cannot size using ATRBasedPositionSizer.")
            return None
        
        if current_price <= 0 or current_portfolio_value <= 0:
            return None

        capital_to_risk = current_portfolio_value * self.risk_per_trade_fraction
        
        stop_distance_per_unit = self.atr_multiple_for_stop * effective_atr
        if stop_distance_per_unit <= 1e-8:
            logger.warning(f"Stop distance per unit is zero or too small for {symbol}. Cannot size position.")
            return None
            
        position_size = capital_to_risk / stop_distance_per_unit
        
        position_cost = position_size * current_price
        if trade_signal == SignalType.LONG and position_cost > available_capital:
            logger.warning(f"Calculated position cost (${position_cost:.2f}) for {symbol} exceeds available cash (${available_capital:.2f}). Reducing size.")
            position_size = available_capital / current_price 
            if position_size <= 1e-8 : return None

        # FIX: Change log level to DEBUG to reduce noise
        logger.debug(f"ATRBased Sizing for {symbol}: Risking ${capital_to_risk:.2f}. "
                    f"ATR: {effective_atr:.6f}, StopDist: ${stop_distance_per_unit:.4f}. "
                    f"Calculated Size: {position_size:.8f} units.")
        return position_size

class PairTradingPositionSizer(BasePositionSizer):
    """
    Sizes positions for a pair trade, aiming for dollar neutrality.
    """
    def __init__(self, dollar_neutral: bool = True, fraction_of_equity_for_pair: float = 0.1, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.dollar_neutral = str(self.params.get('pairtradingpositionsizer_dollarneutral', dollar_neutral)).lower() == 'true'
        self.fraction_of_equity_for_pair = float(self.params.get('fraction_of_equity_for_pair', fraction_of_equity_for_pair))
        logger.info(f"PairTradingPositionSizer initialized. Dollar Neutral: {self.dollar_neutral}, Fraction for Pair: {self.fraction_of_equity_for_pair}")

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float,
        current_portfolio_value: float,
        trade_signal: SignalType,
        strategy_info: Dict[str, Any],
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None,
        other_leg_price: Optional[float] = None,
        hedge_ratio: Optional[float] = None
    ) -> Optional[float]:
        
        if current_price <= 0 or current_portfolio_value <=0: return None

        total_capital_for_pair_trade = current_portfolio_value * self.fraction_of_equity_for_pair

        if self.dollar_neutral:
            capital_for_this_leg = total_capital_for_pair_trade / 2.0
            
            if total_capital_for_pair_trade > available_capital:
                reduction_factor = available_capital / total_capital_for_pair_trade if total_capital_for_pair_trade > 0 else 0
                capital_for_this_leg *= reduction_factor

            if capital_for_this_leg <= 1.0:
                return None
            
            position_size = capital_for_this_leg / current_price
            logger.debug(f"PairTrading Sizing (Dollar Neutral) for leg {symbol}: Capital for leg ${capital_for_this_leg:.2f}. Size: {position_size:.8f} units.")
            return position_size
        else:
            if total_capital_for_pair_trade > available_capital:
                total_capital_for_pair_trade = available_capital

            if total_capital_for_pair_trade <= 1.0: return None
            
            position_size = total_capital_for_pair_trade / current_price
            logger.warning(f"PairTrading Sizing (Non-Dollar Neutral) for leg {symbol} is simplified. Allocating ${total_capital_for_pair_trade:.2f}. Size: {position_size:.8f} units.")
            return position_size