from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from cipher_edge.core.models import PortfolioSnapshot
from cipher_edge.app_logger import get_logger
from cipher_edge.risk_control_module.risk_manager import RiskManager
from cipher_edge.config.settings import settings as app_settings

logger = get_logger(__name__)

class BasePortfolioConstructor(ABC):
    """
    Abstract base class for portfolio construction and rebalancing.
    Handles target allocation calculation, volatility targeting, and rebalancing triggers.
    """
    def __init__(self, settings: Any, risk_manager: RiskManager):
        self.settings = settings
        self.risk_manager = risk_manager
        
        pc_params = self.settings.get_strategy_params('PortfolioConstructor')
        
        # Volatility Targeting Parameters
        self.volatility_targeting_enabled = pc_params.get('volatility_targeting_enable', False)
        self.target_volatility = float(pc_params.get('target_portfolio_volatility', 0.15))
        self.volatility_lookback_period = int(pc_params.get('volatility_targeting_lookback_period', 60))
        
        # Rebalancing Trigger Parameters
        self.rebalance_threshold_pct = float(pc_params.get('rebalance_threshold_pct', 0.05))

        self.equity_curve_df = pd.DataFrame(columns=['total_value_usd']).set_index(pd.to_datetime([]))

        logger.info(
            f"BasePortfolioConstructor initialized. Vol Targeting: {self.volatility_targeting_enabled} "
            f"(Target: {self.target_volatility:.2%}, Lookback: {self.volatility_lookback_period} bars). "
            f"Rebalance Threshold: {self.rebalance_threshold_pct:.2%}"
        )

    def update_equity_curve(self, timestamp: datetime, total_value: float):
        new_row = pd.DataFrame([{'total_value_usd': total_value}], index=[pd.to_datetime(timestamp, utc=True)])
        self.equity_curve_df = pd.concat([self.equity_curve_df, new_row])

    def _calculate_portfolio_volatility(self) -> float:
        if len(self.equity_curve_df) < self.volatility_lookback_period:
            return 0.0

        relevant_equity = self.equity_curve_df['total_value_usd'].iloc[-self.volatility_lookback_period:]
        returns = relevant_equity.pct_change().dropna()
        
        if returns.empty or returns.std() == 0:
            return 0.0
            
        period_volatility = returns.std()
        annualization_factor = app_settings.config.getint('BacktestingPerformance', 'AnnualizationFactor', fallback=252)
        annualized_volatility = period_volatility * np.sqrt(annualization_factor)
        
        return annualized_volatility

    def adjust_weights_for_volatility_target(self, target_weights: Dict[str, float]) -> Dict[str, float]:
        if not self.volatility_targeting_enabled:
            return target_weights

        current_vol = self._calculate_portfolio_volatility()
        if current_vol <= 1e-8:
            logger.warning("Current portfolio volatility is zero. Cannot apply volatility targeting.")
            return target_weights

        scaling_factor = self.target_volatility / current_vol
        min_scale, max_scale = 0.5, 2.0  # Should be in settings
        scaling_factor = max(min_scale, min(max_scale, scaling_factor))
        
        adjusted_weights = {asset: weight * scaling_factor for asset, weight in target_weights.items()}
        
        logger.info(f"Volatility Targeting: Current Vol={current_vol:.2%}, Target Vol={self.target_volatility:.2%}. Scaling Factor: {scaling_factor:.2f}")
        return adjusted_weights

    @abstractmethod
    def calculate_target_allocations(self, current_portfolio: PortfolioSnapshot, market_data: pd.DataFrame, trades_log: pd.DataFrame) -> Dict[str, float]:
        pass

    def rebalance_portfolio(
        self,
        current_portfolio: PortfolioSnapshot,
        market_data: pd.DataFrame,
        current_prices: Dict[str, float],
        portfolio_value: float,
        trades_log: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        
        target_allocations_pct = self.calculate_target_allocations(current_portfolio, market_data, trades_log)
        adjusted_allocations_pct = self.adjust_weights_for_volatility_target(target_allocations_pct)
        
        current_allocations_pct = {}
        if portfolio_value > 0:
            for asset, quantity in current_portfolio.positions.items():
                current_allocations_pct[asset] = (quantity * current_prices.get(asset, 0)) / portfolio_value
        
        rebalance_needed = False
        all_assets = set(current_allocations_pct.keys()) | set(adjusted_allocations_pct.keys())
        for asset in all_assets:
            current_pct = current_allocations_pct.get(asset, 0.0)
            target_pct = adjusted_allocations_pct.get(asset, 0.0)
            if abs(current_pct - target_pct) > self.rebalance_threshold_pct:
                rebalance_needed = True
                logger.info(f"Rebalancing triggered for {asset}. Current: {current_pct:.2%}, Target: {target_pct:.2%}")
                break
        
        if not rebalance_needed:
            return {}

        final_target_capital = {asset: portfolio_value * pct for asset, pct in adjusted_allocations_pct.items()}
        logger.info(f"Rebalance computed target capital allocations: {final_target_capital}")
        return final_target_capital