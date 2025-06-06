# kamikaze_komodo/risk_control_module/position_sizer.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import numpy as np
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
        # fraction is often sourced from config: FixedFractional_AllocationFraction
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
        
        capital_to_allocate = current_portfolio_value * self.fraction_to_allocate
        
        if capital_to_allocate > available_capital :
            logger.warning(f"Calculated capital to allocate ({capital_to_allocate:.2f}) for {symbol} exceeds available cash ({available_capital:.2f}). Using available cash.")
            capital_to_allocate = available_capital
        
        if capital_to_allocate <= 1.0: # Minimum capital to allocate (e.g. $1)
            logger.info(f"Not enough capital to allocate for {symbol} based on fixed fraction ({capital_to_allocate:.2f}). Min trade value not met.")
            return None

        position_size = capital_to_allocate / current_price
        logger.info(f"FixedFractional Sizing for {symbol}: Allocating ${capital_to_allocate:.2f} (Equity: ${current_portfolio_value:.2f}, Fraction: {self.fraction_to_allocate}). Position Size: {position_size:.8f} units at ${current_price:.4f}.")
        return position_size

class ATRBasedPositionSizer(BasePositionSizer):
    """
    Sizes positions based on Average True Range (ATR) to normalize risk per trade.
    This implementation assumes you risk a fixed percentage of portfolio equity,
    and the stop loss is N * ATR away.
    """
    def __init__(self, risk_per_trade_fraction: float = 0.01, atr_multiple_for_stop: float = 2.0, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        # Params often from config: ATRBased_RiskPerTradeFraction, ATRBased_ATRMultipleForStop
        self.risk_per_trade_fraction = float(self.params.get('atrbased_riskpertradefraction', risk_per_trade_fraction))
        self.atr_multiple_for_stop = float(self.params.get('atrbased_atrmultipleforstop', atr_multiple_for_stop))
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
        
        effective_atr = atr_value
        if effective_atr is None and latest_bar and latest_bar.atr is not None:
            effective_atr = latest_bar.atr
        
        if effective_atr is None or effective_atr <= 1e-8: # Check for very small or zero ATR
            logger.warning(f"ATR value for {symbol} is missing, zero or invalid ({effective_atr}). Cannot size using ATRBasedPositionSizer.")
            return None
        
        if current_price <= 0:
            logger.warning(f"Current price for {symbol} is non-positive ({current_price}). Cannot size position.")
            return None
        if current_portfolio_value <= 0:
            logger.warning(f"Current portfolio value is non-positive ({current_portfolio_value}). Cannot size position.")
            return None

        capital_to_risk = current_portfolio_value * self.risk_per_trade_fraction
        
        stop_distance_per_unit = self.atr_multiple_for_stop * effective_atr
        if stop_distance_per_unit <= 1e-8: # Avoid division by zero or tiny stop
            logger.warning(f"Stop distance per unit is zero or too small for {symbol} (ATR: {effective_atr}, Multiple: {self.atr_multiple_for_stop}). Cannot size position.")
            return None
            
        position_size = capital_to_risk / stop_distance_per_unit
        
        position_cost = position_size * current_price
        if position_cost > available_capital:
            logger.warning(f"Calculated position cost (${position_cost:.2f}) for {symbol} exceeds available cash (${available_capital:.2f}). Reducing size to available cash.")
            position_size = available_capital / current_price 
            if position_size <= 1e-8 : return None # Ensure size is not effectively zero

        logger.info(f"ATRBased Sizing for {symbol}: Risking ${capital_to_risk:.2f} (Equity: ${current_portfolio_value:.2f}). "
                    f"ATR: {effective_atr:.6f}, StopDist: ${stop_distance_per_unit:.4f}. "
                    f"Calculated Size: {position_size:.8f} units at ${current_price:.4f}.")
        return position_size

class PairTradingPositionSizer(BasePositionSizer):
    """
    Sizes positions for a pair trade, aiming for dollar neutrality if configured.
    This sizer would return a tuple or dict with sizes for both legs.
    For now, let's make it return the size for ONE leg, assuming the strategy will call it twice
    or it's used in a context where dollar neutrality is applied to a total pair capital.
    A more advanced version would calculate sizes for both legs simultaneously.
    Simplified: calculate size for one leg based on total capital allocated to the pair.
    """
    def __init__(self, dollar_neutral: bool = True, fraction_of_equity_for_pair: float = 0.1, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.dollar_neutral = str(self.params.get('pairtradingpositionsizer_dollarneutral', dollar_neutral)).lower() == 'true'
        self.fraction_of_equity_for_pair = float(self.params.get('fraction_of_equity_for_pair', fraction_of_equity_for_pair))
        logger.info(f"PairTradingPositionSizer initialized. Dollar Neutral: {self.dollar_neutral}, Fraction for Pair: {self.fraction_of_equity_for_pair}")

    def calculate_size(
        self,
        symbol: str, # Symbol of the leg being sized
        current_price: float,
        available_capital: float, # Overall available cash
        current_portfolio_value: float, # Total equity
        strategy_signal_strength: Optional[float] = None,
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None,
        # Additional params for pair trading
        other_leg_price: Optional[float] = None, # Price of the other asset in the pair
        hedge_ratio: Optional[float] = None # If sizing based on hedge ratio rather than pure dollar neutrality
    ) -> Optional[float]: # Returns size for the given 'symbol' leg
        
        if current_price <= 0: return None
        if current_portfolio_value <=0: return None

        # Capital allocated to the entire pair trade
        total_capital_for_pair_trade = current_portfolio_value * self.fraction_of_equity_for_pair

        if self.dollar_neutral:
            # Each leg gets half of the capital allocated to the pair trade.
            capital_for_this_leg = total_capital_for_pair_trade / 2.0
            
            # Ensure we don't allocate more than available cash for the whole pair (simplistic check)
            if total_capital_for_pair_trade > available_capital:
                logger.warning(f"Total capital for pair trade (${total_capital_for_pair_trade:.2f}) exceeds available cash (${available_capital:.2f}). Reducing allocation.")
                # This reduction needs to be applied carefully; for now, we'll cap this leg's capital based on a proportional reduction.
                # This is a rough adjustment. Proper handling involves checking margin and total cost of both legs.
                reduction_factor = available_capital / total_capital_for_pair_trade if total_capital_for_pair_trade > 0 else 0
                capital_for_this_leg *= reduction_factor

            if capital_for_this_leg <= 1.0: # Min capital for a leg
                logger.info(f"Not enough capital for leg {symbol} in pair trade ({capital_for_this_leg:.2f})")
                return None
            
            position_size = capital_for_this_leg / current_price
            logger.info(f"PairTrading Sizing (Dollar Neutral) for leg {symbol}: Capital for leg ${capital_for_this_leg:.2f}. Size: {position_size:.8f} units.")
            return position_size
        else:
            # Non-dollar neutral (e.g., based on hedge ratio or other logic) - complex, placeholder
            # This might involve the hedge_ratio to determine relative number of units.
            # For now, just implement a simple allocation for the one leg based on total pair capital.
            if total_capital_for_pair_trade > available_capital:
                 total_capital_for_pair_trade = available_capital # Cap at available cash

            if total_capital_for_pair_trade <= 1.0: return None # Min capital for the pair
            
            # This is naive if not dollar neutral and not using hedge ratio correctly.
            # Assume this leg takes its proportional share based on some other factor or fixed fraction.
            # For Phase 6, if not dollar neutral, it's underspecified here.
            # Let's assume it falls back to a simple fraction for this leg for now.
            capital_for_this_leg = total_capital_for_pair_trade # If only one leg is sized by this call.
            position_size = capital_for_this_leg / current_price
            logger.warning(f"PairTrading Sizing (Non-Dollar Neutral) for leg {symbol} is simplified. Allocating full pair capital portion ${capital_for_this_leg:.2f}. Size: {position_size:.8f} units.")
            return position_size


class OptimalFPositionSizer(BasePositionSizer):
    """
    Sizes positions based on Vince's Optimal f (Kelly Criterion variant).
    Requires win probability and payoff ratio, which are hard to estimate robustly.
    This is a placeholder for a more sophisticated implementation.
    """
    def __init__(self, win_probability: float = 0.55, payoff_ratio: float = 1.5, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.win_probability = float(self.params.get('optimalf_win_probability', win_probability))
        self.payoff_ratio = float(self.params.get('optimalf_payoff_ratio', payoff_ratio)) # AvgWin / AvgLoss
        if not (0 <= self.win_probability <= 1): raise ValueError("Win probability must be between 0 and 1.")
        if self.payoff_ratio <= 0: raise ValueError("Payoff ratio must be positive.")
        logger.info(f"OptimalFPositionSizer initialized. Win Prob: {self.win_probability}, Payoff Ratio: {self.payoff_ratio}")

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float,
        current_portfolio_value: float,
        strategy_signal_strength: Optional[float] = None,
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        if current_price <= 0 or current_portfolio_value <= 0: return None

        # Kelly Formula: f* = (W * P - L) / (W * L_ratio) where W=win prob, P=payoff, L=loss prob, L_ratio=payoff ratio
        # Simplified Kelly: f = W - ( (1-W) / R ) where W is win probability, R is payoff_ratio (AvgWin/AvgLoss)
        kelly_f = self.win_probability - ((1 - self.win_probability) / self.payoff_ratio)

        if kelly_f <= 0:
            logger.info(f"Optimal f ({kelly_f:.4f}) is zero or negative for {symbol}. No position taken.")
            return None
        
        # Use a fraction of Kelly (e.g., half Kelly) for risk reduction
        fractional_kelly = self.params.get('optimalf_kelly_fraction', 0.5) * kelly_f
        
        capital_to_allocate = current_portfolio_value * fractional_kelly
        
        if capital_to_allocate > available_capital:
            capital_to_allocate = available_capital
        if capital_to_allocate <= 1.0:
            logger.info(f"Not enough capital for {symbol} after Optimal F ({capital_to_allocate:.2f}).")
            return None

        position_size = capital_to_allocate / current_price
        logger.info(f"OptimalF Sizing for {symbol}: Kelly f*={kelly_f:.4f}, Using {fractional_kelly*100:.2f}% of equity. Allocating ${capital_to_allocate:.2f}. Size: {position_size:.8f} units.")
        return position_size


class MLConfidencePositionSizer(BasePositionSizer):
    """
    Sizes positions based on a base fraction of equity, modulated by ML model confidence.
    """
    def __init__(self, base_fraction: float = 0.05, min_alloc_fraction: float = 0.01, max_alloc_fraction: float = 0.2, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.base_fraction = float(self.params.get('mlconfidence_base_fraction', base_fraction))
        self.min_alloc_fraction = float(self.params.get('mlconfidence_min_alloc_fraction', min_alloc_fraction))
        self.max_alloc_fraction = float(self.params.get('mlconfidence_max_alloc_fraction', max_alloc_fraction))
        logger.info(f"MLConfidencePositionSizer: BaseFraction={self.base_fraction}, MinAlloc={self.min_alloc_fraction}, MaxAlloc={self.max_alloc_fraction}")

    def calculate_size(
        self,
        symbol: str,
        current_price: float,
        available_capital: float,
        current_portfolio_value: float,
        strategy_signal_strength: Optional[float] = None, # This is the ML confidence (0.0 to 1.0)
        latest_bar: Optional[BarData] = None,
        atr_value: Optional[float] = None
    ) -> Optional[float]:
        if current_price <= 0 or current_portfolio_value <= 0: return None
        
        if strategy_signal_strength is None:
            logger.warning("ML model confidence (strategy_signal_strength) not provided. Cannot use MLConfidencePositionSizer. Defaulting to no trade.")
            return None
        
        # Ensure confidence is within [0,1]
        confidence = max(0.0, min(1.0, strategy_signal_strength))
        
        # Modulate base fraction by confidence. Example: If confidence is high, use higher fraction.
        # Linear modulation: alloc_fraction = min_alloc + (max_alloc - min_alloc) * confidence
        # Or, scale base_fraction: alloc_fraction = base_fraction * (0.5 + confidence) # (scales from 0.5*base to 1.5*base)
        # Let's use a simpler direct scaling of base_fraction, capped by min/max.
        # Scaler: 0.5 (low conf) to 1.5 (high conf), centered at 1.0 for conf=0.5
        # confidence_scaler = 0.5 + confidence 
        # effective_fraction = self.base_fraction * confidence_scaler
        
        # More direct: if confidence is low (e.g. <0.5), use min_alloc. If high (e.g. >0.8), use max_alloc. Interpolate.
        # For simplicity, let's make it proportional to confidence, bounded by min/max overall allocation.
        # If confidence is 0.5, use base_fraction. If 1.0, use max_alloc. If 0.0 use min_alloc (or even zero).
        
        if confidence < 0.5: # Lower confidence scales down from base_fraction towards min_alloc_fraction
             # Interpolate between min_alloc_fraction and base_fraction
            # when confidence is 0, use min_alloc_fraction. when confidence is 0.5, use base_fraction.
            # slope = (base_fraction - min_alloc_fraction) / 0.5
            # effective_fraction = min_alloc_fraction + slope * confidence
            # Simpler: if confidence = 0 -> min_alloc, if confidence=0.5 -> base_fraction
            # scale_factor = confidence / 0.5 # confidence from 0 to 0.5 -> scale_factor from 0 to 1
            # effective_fraction = self.min_alloc_fraction + (self.base_fraction - self.min_alloc_fraction) * scale_factor
            # This is still a bit complex. Alternative:
            effective_fraction = self.base_fraction * (confidence * 1.5) # Scales from 0 to 0.75 * base_fraction for conf 0 to 0.5
        else: # Higher confidence scales up from base_fraction towards max_alloc_fraction
            # Interpolate between base_fraction and max_alloc_fraction
            # when confidence is 0.5, use base_fraction. when confidence is 1.0, use max_alloc_fraction
            # slope = (max_alloc_fraction - base_fraction) / 0.5
            # effective_fraction = base_fraction + slope * (confidence - 0.5)
             effective_fraction = self.base_fraction * (0.75 + (confidence-0.5)*1.5) # Scales from 0.75*base to 1.5*base for conf 0.5 to 1.0


        effective_fraction = np.clip(effective_fraction, self.min_alloc_fraction, self.max_alloc_fraction)

        capital_to_allocate = current_portfolio_value * effective_fraction
        
        if capital_to_allocate > available_capital:
            capital_to_allocate = available_capital
        if capital_to_allocate <= 1.0:
            logger.info(f"Not enough capital for {symbol} using ML confidence ({capital_to_allocate:.2f}). Confidence: {confidence:.2f}, Eff.Frac: {effective_fraction:.4f}")
            return None
            
        position_size = capital_to_allocate / current_price
        logger.info(f"MLConfidence Sizing for {symbol}: Confidence={confidence:.2f}, Eff.Alloc.Frac={effective_fraction:.4f}. Allocating ${capital_to_allocate:.2f}. Size: {position_size:.8f} units.")
        return position_size