# FILE: kamikaze_komodo/risk_control_module/position_sizer.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple, Type
import numpy as np
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.core.enums import SignalType
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
        trade_signal: SignalType,
        strategy_info: Dict[str, Any],
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
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
        
        if capital_to_allocate <= 1.0:
            return None

        position_size = capital_to_allocate / current_price
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

        logger.debug(f"ATRBased Sizing for {symbol}: Risking ${capital_to_risk:.2f}. "
                     f"ATR: {effective_atr:.6f}, StopDist: ${stop_distance_per_unit:.4f}. "
                     f"Calculated Size: {position_size:.8f} units.")
        return position_size

class OptimalFPositionSizer(BasePositionSizer):
    """
    Sizes positions based on a simplified Optimal f or fractional Kelly criterion.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.default_win_rate = float(self.params.get('optimalf_win_rate_estimate', 0.51))
        self.default_payoff_ratio = float(self.params.get('optimalf_avg_win_loss_ratio_estimate', 1.1))
        self.kelly_fraction = float(self.params.get('optimalf_kelly_fraction', 0.5))

        if self.default_payoff_ratio <= 0:
            raise ValueError("Default average win/loss ratio estimate must be greater than zero.")
        
        logger.info(
            f"OptimalFPositionSizer initialized. "
            f"DefaultWinRate={self.default_win_rate}, "
            f"DefaultAvgWinLossRatio={self.default_payoff_ratio}, "
            f"KellyFraction={self.kelly_fraction}"
        )

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
        if current_price <= 0 or current_portfolio_value <= 0:
            return None

        win_rate = strategy_info.get('win_rate', self.default_win_rate)
        payoff_ratio = strategy_info.get('payoff_ratio', self.default_payoff_ratio)

        if payoff_ratio <= 0:
            logger.warning(f"Invalid payoff_ratio ({payoff_ratio}) for {symbol}. Cannot size position.")
            return None

        optimal_f = (win_rate * (payoff_ratio + 1) - 1) / payoff_ratio
        
        if optimal_f <= 0:
            logger.debug(f"Optimal f for {symbol} is not positive ({optimal_f:.4f}). No position taken.")
            return None

        allocation_fraction = optimal_f * self.kelly_fraction
        
        capital_to_allocate = current_portfolio_value * allocation_fraction
        
        if trade_signal == SignalType.LONG and capital_to_allocate > available_capital:
            capital_to_allocate = available_capital

        position_size = capital_to_allocate / current_price
        
        log_source = "Dynamic" if 'win_rate' in strategy_info else "Default"
        logger.info(
            f"OptimalF Sizing ({log_source}) for {symbol}: WR={win_rate:.2f}, PR={payoff_ratio:.2f} -> "
            f"Optimal_f={optimal_f:.4f}, Fraction={allocation_fraction:.4f}, "
            f"Allocating ${capital_to_allocate:,.2f}. Position Size: {position_size:.8f} units."
        )
        return position_size if position_size > 0 else None

class MLConfidencePositionSizer(BasePositionSizer):
    """
    Sizes positions based on the confidence level of an ML prediction.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.min_size_factor = float(self.params.get('mlconfidence_min_size_factor', 0.5))
        self.max_size_factor = float(self.params.get('mlconfidence_max_size_factor', 1.5))
        self.base_allocation_fraction = float(self.params.get('mlconfidence_base_allocation_fraction', 0.05))

        if not (0 <= self.min_size_factor <= self.max_size_factor):
            raise ValueError("min_size_factor must be >= 0 and <= max_size_factor.")
        if self.max_size_factor <= 0:
            raise ValueError("max_size_factor must be positive.")

        logger.info(
            f"MLConfidencePositionSizer initialized. "
            f"BaseAllocation={self.base_allocation_fraction*100:.2f}%, "
            f"MinFactor={self.min_size_factor}, MaxFactor={self.max_size_factor}"
        )

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
        if current_price <= 0 or current_portfolio_value <= 0:
            return None

        confidence_score = strategy_info.get('confidence_score')
        if confidence_score is None:
            logger.warning(f"ML 'confidence_score' missing for {symbol}. Cannot use MLConfidencePositionSizer. No trade.")
            return None
        
        confidence_score = max(0.0, min(1.0, float(confidence_score)))
        
        effective_scaling_factor = self.min_size_factor + (self.max_size_factor - self.min_size_factor) * confidence_score
        
        base_capital_to_allocate = current_portfolio_value * self.base_allocation_fraction
        capital_to_allocate = base_capital_to_allocate * effective_scaling_factor

        if trade_signal == SignalType.LONG and capital_to_allocate > available_capital:
            capital_to_allocate = available_capital

        position_size = capital_to_allocate / current_price
        
        logger.info(
            f"MLConfidence Sizing for {symbol}: Confidence={confidence_score:.4f}, "
            f"ScalingFactor={effective_scaling_factor:.4f}. "
            f"Allocating ${capital_to_allocate:,.2f}. Position Size: {position_size:.8f} units."
        )
        
        return position_size if position_size > 0 else None

class PairTradingPositionSizer(BasePositionSizer):
    # ... (Implementation from previous file can be placed here if it exists) ...
    pass

# --- REGISTRY DEFINITION ---
# Defined after all classes to ensure they are fully loaded.
POSITION_SIZER_REGISTRY: Dict[str, Type[BasePositionSizer]] = {
    'FixedFractionalPositionSizer': FixedFractionalPositionSizer,
    'ATRBasedPositionSizer': ATRBasedPositionSizer,
    'OptimalFPositionSizer': OptimalFPositionSizer,
    'MLConfidencePositionSizer': MLConfidencePositionSizer,
}