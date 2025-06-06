# kamikaze_komodo/portfolio_constructor/rebalancer.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from kamikaze_komodo.core.models import PortfolioSnapshot # For current holdings
from kamikaze_komodo.core.enums import OrderSide, OrderType # For generating orders
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
    def needs_rebalancing(
        self,
        current_portfolio: PortfolioSnapshot,
        target_allocations_pct: Dict[str, float], # e.g. {'BTC/USD': 0.6, 'ETH/USD': 0.4} as fractions
        asset_prices: Dict[str, float] # Current market prices for assets {'BTC/USD': 50000, ...}
    ) -> bool:
        """
        Determines if the portfolio needs rebalancing based on current state and targets.
        Args:
            current_portfolio (PortfolioSnapshot): The current state of the portfolio (contains positions quantity).
            target_allocations_pct (Dict[str, float]): The desired target allocations as percentages.
            asset_prices (Dict[str, float]): Current market prices of the assets.
        Returns:
            bool: True if rebalancing is needed, False otherwise.
        """
        pass

    @abstractmethod
    def generate_rebalancing_orders(
        self,
        current_portfolio: PortfolioSnapshot,
        target_allocations_pct: Dict[str, float],
        asset_prices: Dict[str, float] # Current market prices for assets
    ) -> List[Dict[str, Any]]: # List of order parameters
        """
        Generates orders needed to rebalance the portfolio to target allocations.
        Args:
            current_portfolio (PortfolioSnapshot): The current state of the portfolio.
            target_allocations_pct (Dict[str, float]): The desired target allocations as percentages.
            asset_prices (Dict[str, float]): Current market prices of the assets.
        Returns:
            List[Dict[str, Any]]: A list of order parameters (e.g., for exchange_api.create_order).
        """
        pass

class BasicRebalancer(BaseRebalancer):
    """
    Rebalances the portfolio if the current weight of any asset deviates
    from its target weight by more than a specified threshold.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.deviation_threshold = float(self.params.get('rebalancer_deviationthreshold', 0.05)) # Default 5% deviation
        self.min_order_value_usd = float(self.params.get('rebalancer_min_order_value_usd', 10.0)) # Min order value to execute
        logger.info(f"BasicRebalancer initialized with deviation threshold: {self.deviation_threshold*100}%, Min Order Value: ${self.min_order_value_usd}")

    def _get_current_weights(self, current_portfolio: PortfolioSnapshot, asset_prices: Dict[str, float]) -> Dict[str, float]:
        current_weights: Dict[str, float] = {}
        total_value_from_positions = 0.0
        asset_values : Dict[str, float] = {}

        for asset, quantity in current_portfolio.positions.items():
            if asset in asset_prices and asset_prices[asset] > 0:
                value = quantity * asset_prices[asset]
                asset_values[asset] = value
                total_value_from_positions += value
            else:
                logger.warning(f"Price for asset {asset} not available or zero. Cannot calculate its value for rebalancing.")
                asset_values[asset] = 0.0

        # Effective portfolio value for weight calculation is cash + value of positions
        effective_total_value = current_portfolio.cash_balance_usd + total_value_from_positions
        if effective_total_value <= 0:
            return {asset: 0.0 for asset in current_portfolio.positions.keys()}

        for asset, value in asset_values.items():
            current_weights[asset] = value / effective_total_value

        # Add assets that are in target but not in current holdings (weight 0)
        for asset in asset_prices.keys():
            if asset not in current_weights:
                current_weights[asset] = 0.0
        return current_weights


    def needs_rebalancing(
        self,
        current_portfolio: PortfolioSnapshot,
        target_allocations_pct: Dict[str, float],
        asset_prices: Dict[str, float]
    ) -> bool:
        if not target_allocations_pct:
            logger.debug("No target allocations provided, rebalancing not needed by BasicRebalancer.")
            return False

        current_weights = self._get_current_weights(current_portfolio, asset_prices)
        if not current_weights and any(v > 0 for v in target_allocations_pct.values()): # No current holdings but target has allocations
            logger.info("Rebalancing needed: No current holdings, but target allocations exist.")
            return True

        all_assets = set(current_weights.keys()).union(set(target_allocations_pct.keys()))

        for asset in all_assets:
            current_w = current_weights.get(asset, 0.0)
            target_w = target_allocations_pct.get(asset, 0.0)
            if abs(current_w - target_w) > self.deviation_threshold:
                logger.info(f"Rebalancing needed for {asset}. Current weight: {current_w:.4f}, Target: {target_w:.4f}, Deviation: {abs(current_w - target_w):.4f} > {self.deviation_threshold:.4f}")
                return True
        logger.debug("BasicRebalancer: No rebalancing needed based on deviation threshold.")
        return False

    def generate_rebalancing_orders(
        self,
        current_portfolio: PortfolioSnapshot,
        target_allocations_pct: Dict[str, float],
        asset_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        orders: List[Dict[str, Any]] = []
        if current_portfolio.total_value_usd <=0 :
            logger.warning("Portfolio total value is zero or negative. Cannot generate rebalancing orders.")
            return orders

        # Calculate current values of each asset
        current_asset_values: Dict[str, float] = {}
        for asset, quantity in current_portfolio.positions.items():
            if asset in asset_prices and asset_prices[asset] > 0:
                current_asset_values[asset] = quantity * asset_prices[asset]
            else:
                current_asset_values[asset] = 0.0

        # Calculate target values based on total portfolio value
        target_asset_values: Dict[str, float] = {}
        for asset, target_pct in target_allocations_pct.items():
            target_asset_values[asset] = current_portfolio.total_value_usd * target_pct

        all_assets_in_scope = set(current_asset_values.keys()).union(set(target_asset_values.keys()))

        for asset in all_assets_in_scope:
            current_value = current_asset_values.get(asset, 0.0)
            target_value = target_asset_values.get(asset, 0.0)
            price = asset_prices.get(asset)

            if price is None or price <= 0:
                logger.warning(f"Cannot generate order for {asset}: price is missing or invalid ({price}).")
                continue

            value_diff = target_value - current_value
            amount_to_trade = value_diff / price

            if abs(value_diff) < self.min_order_value_usd: # Skip if trade value is too small
                logger.debug(f"Skipping rebalance for {asset}: change in value ({value_diff:.2f}) is less than min_order_value_usd (${self.min_order_value_usd:.2f}).")
                continue

            if abs(amount_to_trade) > 1e-8: # Ensure there's a non-negligible amount to trade
                order_side = OrderSide.BUY if amount_to_trade > 0 else OrderSide.SELL
                orders.append({
                    'symbol': asset,
                    'type': OrderType.MARKET, # Or allow configurable order type
                    'side': order_side,
                    'amount': abs(amount_to_trade)
                })
                logger.info(f"Generated rebalancing order for {asset}: {order_side.value} {abs(amount_to_trade):.6f} units. Target Value: ${target_value:.2f}, Current Value: ${current_value:.2f}")

        # Orders should ideally be prioritized (e.g., sells before buys if cash is needed)
        # This basic implementation doesn't handle that.
        return orders