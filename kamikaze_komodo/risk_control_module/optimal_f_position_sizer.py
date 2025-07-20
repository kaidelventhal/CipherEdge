# FILE: kamikaze_komodo/risk_control_module/optimal_f_position_sizer.py
from typing import Optional, Dict, Any

from kamikaze_komodo.risk_control_module.position_sizer import BasePositionSizer
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData

logger = get_logger(__name__)

class OptimalFPositionSizer(BasePositionSizer):
    """
    Sizes positions based on a simplified Optimal f or fractional Kelly criterion.
    This requires estimates of win rate and the average win/loss ratio.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the sizer with estimates from the parameters.
        """
        super().__init__(params)
        self.win_rate_estimate = float(self.params.get('optimalf_win_rate_estimate', 0.51))
        self.avg_win_loss_ratio_estimate = float(self.params.get('optimalf_avg_win_loss_ratio_estimate', 1.1))
        self.kelly_fraction = float(self.params.get('optimalf_kelly_fraction', 0.5))

        if self.avg_win_loss_ratio_estimate <= 0:
            raise ValueError("Average win/loss ratio estimate must be greater than zero.")
        
        logger.info(
            f"OptimalFPositionSizer initialized. "
            f"WinRateEstimate={self.win_rate_estimate}, "
            f"AvgWinLossRatio={self.avg_win_loss_ratio_estimate}, "
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
        """
        Calculates the position size based on the Optimal f formula.
        """
        if current_price <= 0 or current_portfolio_value <= 0:
            return None

        # Kelly formula: f = W - (1-W)/R  (where W is win rate, R is payoff ratio)
        # Simplified variant: f = (W * (R + 1) - 1) / R
        optimal_f = (self.win_rate_estimate * (self.avg_win_loss_ratio_estimate + 1) - 1) / self.avg_win_loss_ratio_estimate
        
        # We only take positions if the edge is positive (f > 0)
        if optimal_f <= 0:
            logger.debug(f"Optimal f for {symbol} is not positive ({optimal_f:.4f}). No position taken.")
            return None

        # Apply the Kelly fraction to be more conservative
        allocation_fraction = optimal_f * self.kelly_fraction
        
        capital_to_allocate = current_portfolio_value * allocation_fraction
        
        # Ensure we don't allocate more than available cash for a new long position
        if trade_signal == SignalType.LONG and capital_to_allocate > available_capital:
            logger.warning(
                f"Optimal F allocation (${capital_to_allocate:,.2f}) exceeds available cash (${available_capital:,.2f}). "
                f"Sizing down to available cash."
            )
            capital_to_allocate = available_capital

        position_size = capital_to_allocate / current_price
        
        logger.info(
            f"OptimalF Sizing for {symbol}: Optimal_f={optimal_f:.4f}, Fraction={allocation_fraction:.4f}, "
            f"Allocating ${capital_to_allocate:,.2f}. Position Size: {position_size:.8f} units."
        )

        return position_size if position_size > 0 else None