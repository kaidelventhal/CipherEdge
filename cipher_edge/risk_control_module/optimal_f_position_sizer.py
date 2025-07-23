from typing import Optional, Dict, Any

from cipher_edge.risk_control_module.position_sizer import BasePositionSizer
from cipher_edge.app_logger import get_logger
from cipher_edge.core.enums import SignalType
from cipher_edge.core.models import BarData

logger = get_logger(__name__)

class OptimalFPositionSizer(BasePositionSizer):
    """
    Sizes positions based on a simplified Optimal f or fractional Kelly criterion.
    **IMPROVEMENT**: It now dynamically uses `win_rate` and `payoff_ratio` from
    `strategy_info` if provided, falling back to config defaults.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the sizer with default estimates from the parameters.
        """
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
        """
        Calculates the position size based on the Optimal f formula,
        using dynamic stats if available.
        """
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
            logger.warning(
                f"Optimal F allocation (${capital_to_allocate:,.2f}) exceeds available cash (${available_capital:,.2f}). "
                f"Sizing down to available cash."
            )
            capital_to_allocate = available_capital

        position_size = capital_to_allocate / current_price
        
        log_source = "Dynamic" if 'win_rate' in strategy_info else "Default"
        logger.info(
            f"OptimalF Sizing ({log_source}) for {symbol}: WR={win_rate:.2f}, PR={payoff_ratio:.2f} -> "
            f"Optimal_f={optimal_f:.4f}, Fraction={allocation_fraction:.4f}, "
            f"Allocating ${capital_to_allocate:,.2f}. Position Size: {position_size:.8f} units."
        )

        return position_size if position_size > 0 else None