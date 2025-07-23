from typing import Dict, Optional, Any
from cipher_edge.core.models import BarData
from cipher_edge.app_logger import get_logger
from cipher_edge.risk_control_module.position_sizer import BasePositionSizer
from cipher_edge.core.enums import SignalType

logger = get_logger(__name__)

class MLConfidencePositionSizer(BasePositionSizer):
    """
    Sizes positions based on the confidence level of an ML prediction.
    Higher confidence leads to a larger position, within defined bounds.
    Assumes `strategy_info` will contain a 'confidence_score'.
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
            logger.warning(f"Calculated capital (${capital_to_allocate:,.2f}) for {symbol} exceeds cash (${available_capital:,.2f}). Sizing down.")
            capital_to_allocate = available_capital

        position_size = capital_to_allocate / current_price
        
        logger.info(
            f"MLConfidence Sizing for {symbol}: Confidence={confidence_score:.4f}, "
            f"ScalingFactor={effective_scaling_factor:.4f}. "
            f"Allocating ${capital_to_allocate:,.2f}. Position Size: {position_size:.8f} units."
        )
        
        return position_size if position_size > 0 else None