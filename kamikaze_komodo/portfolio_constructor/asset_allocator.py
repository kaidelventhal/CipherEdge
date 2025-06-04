# kamikaze_komodo/portfolio_constructor/asset_allocator.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from kamikaze_komodo.core.models import BarData # Or other relevant models
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class BaseAssetAllocator(ABC):
    """
    Abstract base class for asset allocation strategies.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else {}
        logger.info(f"{self.__class__.__name__} initialized with params: {self.params}")

    @abstractmethod
    def allocate(
        self,
        assets: List[str], # List of available asset symbols
        current_holdings: Dict[str, float], # Current quantity of each asset held
        market_data: Dict[str, BarData], # Current market data for available assets
        available_capital: float
    ) -> Dict[str, float]: # Target allocation in terms of capital or percentage
        """
        Determines the target allocation for assets.

        Args:
            assets (List[str]): List of asset symbols to consider for allocation.
            current_holdings (Dict[str, float]): Current holdings (e.g. {'BTC/USD': 0.5}).
            market_data (Dict[str, BarData]): Latest market data for each asset.
            available_capital (float): Total capital available for allocation.

        Returns:
            Dict[str, float]: Dictionary mapping asset symbols to target capital allocation.
                              (e.g., {'BTC/USD': 5000.0, 'ETH/USD': 5000.0})
                              or target percentage (e.g. {'BTC/USD': 0.5, 'ETH/USD': 0.5})
                              The interpretation depends on the portfolio manager.
        """
        pass

class FixedWeightAssetAllocator(BaseAssetAllocator):
    """
    Allocates assets based on predefined fixed weights.
    For a single asset strategy, this will allocate 100% to that asset if a signal is present.
    """
    def __init__(self, target_weights: Dict[str, float], params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.target_weights = target_weights
        if not self.target_weights:
            logger.warning("FixedWeightAssetAllocator initialized with no target weights.")
        elif abs(sum(self.target_weights.values()) - 1.0) > 1e-6 and sum(self.target_weights.values()) != 0 : # Allow 0 for no allocation
             logger.warning(f"Sum of target weights ({sum(self.target_weights.values())}) is not 1.0. Allocations will be normalized or may behave unexpectedly.")


    def allocate(
        self,
        assets: List[str],
        current_holdings: Dict[str, float],
        market_data: Dict[str, BarData],
        available_capital: float
    ) -> Dict[str, float]:
        """
        Returns the predefined target weights for allocation.
        In a single-asset backtest, if the asset is in target_weights, it implies full allocation for trades.
        The actual amount to buy/sell will be determined by the PositionSizer.
        This method primarily provides the desired *proportion* of the portfolio for each asset.
        """
        allocation_targets_capital: Dict[str, float] = {}
        total_weight = sum(self.target_weights.get(asset, 0.0) for asset in assets if asset in self.target_weights)

        if total_weight == 0: # No weights for available assets or all weights are zero
            logger.debug("No target weights specified for the given assets or total weight is zero. No allocation.")
            return {asset: 0.0 for asset in assets}

        for asset in assets:
            if asset in self.target_weights:
                normalized_weight = self.target_weights[asset] / total_weight # Normalize if sum is not 1
                allocation_targets_capital[asset] = available_capital * normalized_weight
            else:
                # Assets not in target_weights get zero allocation from this allocator's perspective
                allocation_targets_capital[asset] = 0.0

        logger.debug(f"FixedWeightAssetAllocator target capital allocation: {allocation_targets_capital}")
        return allocation_targets_capital

# Example for future more complex allocators:
# class OptimalFAllocator(BaseAssetAllocator): ...
# class HRPAllocator(BaseAssetAllocator): ...