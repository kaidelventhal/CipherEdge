# kamikaze_komodo/portfolio_constructor/rebalancer.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from kamikaze_komodo.core.models import PortfolioSnapshot
from kamikaze_komodo.core.enums import OrderSide, OrderType
from kamikaze_komodo.app_logger import get_logger
import pandas as pd

logger = get_logger(__name__)

class BaseRebalancer(ABC):
    """
    Abstract base class for portfolio rebalancing logic.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else {}
        logger.info(f"{self.__class__.__name__} initialized with params: {self.params}")

    @abstractmethod
    def generate_rebalancing_orders(
        self,
        current_portfolio: PortfolioSnapshot,
        target_allocations_pct: Dict[str, float],
        asset_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generates orders needed to rebalance the portfolio to target allocations.
        """
        pass

class BasicRebalancer(BaseRebalancer):
    """
    Rebalances the portfolio based on a target allocation dictionary.
    If an asset with a current position is NOT in the target dictionary, it is held.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.min_order_value_usd = float(self.params.get('rebalancer_min_order_value_usd', 10.0))
        logger.info(f"BasicRebalancer initialized with Min Order Value: ${self.min_order_value_usd}")

    def generate_rebalancing_orders(
        self,
        current_portfolio: PortfolioSnapshot,
        target_allocations_pct: Dict[str, float],
        asset_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        if current_portfolio.total_value_usd <= 0:
            logger.warning("Portfolio total value is zero or negative. Cannot generate rebalancing orders.")
            return orders

        # Determine the target value in USD for each asset based on the total portfolio equity
        target_asset_values: Dict[str, float] = {
            asset: current_portfolio.total_value_usd * target_pct
            for asset, target_pct in target_allocations_pct.items()
        }

        all_assets_in_scope = set(current_portfolio.positions.keys()).union(set(target_allocations_pct.keys()))

        for asset in all_assets_in_scope:
            price = asset_prices.get(asset)
            if price is None or price <= 0:
                logger.warning(f"Cannot generate order for {asset}: price is missing or invalid ({price}).")
                continue

            current_quantity = current_portfolio.positions.get(asset, 0.0)
            
            # If an asset is not in the target dictionary, its target quantity is its current quantity.
            # This correctly implements the "HOLD" signal.
            target_value = target_asset_values.get(asset, current_quantity * price)
            target_quantity = target_value / price

            quantity_diff = target_quantity - current_quantity
            value_of_trade = abs(quantity_diff * price)

            if value_of_trade < self.min_order_value_usd:
                continue

            if abs(quantity_diff) > 1e-9: # Use a small epsilon to avoid floating point noise
                order_side = OrderSide.BUY if quantity_diff > 0 else OrderSide.SELL
                orders.append({
                    'symbol': asset,
                    'type': OrderType.MARKET,
                    'side': order_side,
                    'amount': abs(quantity_diff)
                })
                logger.info(f"Generated rebalancing order for {asset}: {order_side.value} {abs(quantity_diff):.6f} units. Target Value: ${target_value:.2f}, Current Value: ${current_quantity * price:.2f}")

        return orders