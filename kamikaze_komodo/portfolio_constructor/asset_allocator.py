# kamikaze_komodo/portfolio_constructor/asset_allocator.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from kamikaze_komodo.core.models import BarData # Or other relevant models
from kamikaze_komodo.app_logger import get_logger

# For HRP
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import scipy.stats

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
        portfolio_value: float, # Total current portfolio value
        historical_data: Optional[Dict[str, pd.DataFrame]] = None, # Dict of DataFrames {symbol: df_ohlcv}
        trade_history: Optional[pd.DataFrame] = None # For OptimalF
    ) -> Dict[str, float]: # Target allocation in terms of percentage
        """
        Determines the target allocation for assets.
        Args:
            assets (List[str]): List of asset symbols to consider for allocation.
            portfolio_value (float): Total capital available for allocation.
            historical_data (Optional[Dict[str, pd.DataFrame]]): Historical OHLCV data for assets.
            trade_history (Optional[pd.DataFrame]): For OptimalF, needs past trade performance.
        Returns:
            Dict[str, float]: Dictionary mapping asset symbols to target allocation percentage (0.0 to 1.0).
        """
        pass

class FixedWeightAssetAllocator(BaseAssetAllocator):
    """
    Allocates assets based on predefined fixed weights.
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
        portfolio_value: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
        trade_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        allocation_targets_pct: Dict[str, float] = {}
        relevant_target_weights = {asset: self.target_weights.get(asset, 0.0) for asset in assets if asset in self.target_weights}
        total_weight_for_relevant_assets = sum(relevant_target_weights.values())

        if total_weight_for_relevant_assets == 0:
            logger.debug("No target weights specified for the given assets or total weight is zero. No allocation.")
            return {asset: 0.0 for asset in assets}

        for asset in assets:
            if asset in relevant_target_weights:
                normalized_weight = relevant_target_weights[asset] / total_weight_for_relevant_assets
                allocation_targets_pct[asset] = normalized_weight
            else:
                allocation_targets_pct[asset] = 0.0
        logger.debug(f"FixedWeightAssetAllocator target percentage allocation: {allocation_targets_pct}")
        return allocation_targets_pct

class OptimalFAllocator(BaseAssetAllocator):
    """
    Allocates capital based on Vince's Optimal f (Kelly Criterion variant).
    This is a placeholder and should be used with caution as estimating inputs is difficult.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.default_win_probability = float(self.params.get('optimalf_default_win_probability', 0.51))
        self.default_payoff_ratio = float(self.params.get('optimalf_default_payoff_ratio', 1.1))
        self.kelly_fraction = float(self.params.get('optimalf_kelly_fraction', 0.25))
        self.min_trades_for_stats = int(self.params.get('optimalf_min_trades_for_stats', 20))
        logger.info(f"OptimalFAllocator initialized. Default WinProb: {self.default_win_probability}, Default Payoff: {self.default_payoff_ratio}, Kelly Fraction: {self.kelly_fraction}, Min Trades: {self.min_trades_for_stats}")

    def allocate(
        self,
        assets: List[str],
        portfolio_value: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
        trade_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        # This implementation is complex and context-specific.
        # For now, it will act as a simple fallback.
        logger.warning("OptimalFAllocator not fully implemented, returning equal weights.")
        num_assets = len(assets)
        if num_assets == 0:
            return {}
        equal_weight = 1.0 / num_assets
        return {asset: equal_weight for asset in assets}


class HRPAllocator(BaseAssetAllocator):
    """
    Allocates assets using Hierarchical Risk Parity (HRP) by De Prado.
    Requires historical returns data for assets.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.linkage_method = self.params.get('hrp_linkage_method', 'ward')
        logger.info(f"HRPAllocator initialized. Linkage method: {self.linkage_method}")

    def _get_cluster_var(self, cov_matrix: pd.DataFrame, cluster_items: List[int]) -> float:
        cluster_cov_matrix = cov_matrix.iloc[cluster_items, cluster_items]
        parity_w = 1.0 / np.diag(cluster_cov_matrix)
        parity_w = parity_w / parity_w.sum()
        cluster_var = np.dot(parity_w, np.dot(cluster_cov_matrix, parity_w))
        return cluster_var

    def _get_recursive_bisection(self, sort_ix: List[int], current_weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        if len(sort_ix) == 1:
            return current_weights

        mid_point = len(sort_ix) // 2
        cluster1_items = sort_ix[:mid_point]
        cluster2_items = sort_ix[mid_point:]

        cluster1_var = self._get_cluster_var(cov_matrix, cluster1_items)
        cluster2_var = self._get_cluster_var(cov_matrix, cluster2_items)
        
        # Handle potential division by zero if a cluster has zero variance
        total_cluster_var = cluster1_var + cluster2_var
        alpha = cluster2_var / total_cluster_var if total_cluster_var != 0 else 0.5

        for i in cluster1_items:
            current_weights[i] *= alpha
        for i in cluster2_items:
            current_weights[i] *= (1 - alpha)

        current_weights = self._get_recursive_bisection(cluster1_items, current_weights, cov_matrix)
        current_weights = self._get_recursive_bisection(cluster2_items, current_weights, cov_matrix)

        return current_weights

    def allocate(
        self,
        assets: List[str],
        portfolio_value: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
        trade_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        
        num_assets = len(assets)
        if not historical_data or num_assets == 0:
            return {}

        # --- FIX for HRPAllocator Fallback ---
        if num_assets < 2:
            logger.warning("HRPAllocator requires at least 2 assets for allocation. Falling back to equal weight for the single asset.")
            return {assets[0]: 1.0} if num_assets == 1 else {}

        returns_data = {}
        for asset in assets:
            if asset in historical_data and not historical_data[asset].empty:
                returns_data[asset] = historical_data[asset]['close'].pct_change().dropna()
            else:
                logger.warning(f"No historical 'close' data for asset {asset} in HRPAllocator. Cannot perform allocation.")
                return {ast: 1.0 / num_assets for ast in assets}

        returns_df = pd.DataFrame(returns_data).dropna()
        if returns_df.shape[0] < 2 or returns_df.shape[1] < 2:
            logger.warning(f"Not enough processed return data for HRP. Shape: {returns_df.shape}. Falling back to equal weight.")
            return {asset: 1.0 / num_assets for asset in assets}

        try:
            cov_matrix = returns_df.cov()
            corr_matrix = returns_df.corr()
            
            dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            condensed_dist_matrix = squareform(dist_matrix)
            link = linkage(condensed_dist_matrix, method=self.linkage_method)

            sort_ix = dendrogram(link, no_plot=True)['leaves']

            initial_weights = np.ones(num_assets)
            hrp_weights_array = self._get_recursive_bisection(sort_ix, initial_weights, cov_matrix)
        
            hrp_weights = pd.Series(hrp_weights_array, index=[assets[i] for i in sort_ix])
            hrp_weights = hrp_weights / hrp_weights.sum()
            hrp_weights = hrp_weights.reindex(assets).fillna(0.0)

            allocations = hrp_weights.to_dict()
            logger.info(f"HRP Allocator target percentage allocation: { {k: f'{v*100:.2f}%' for k, v in allocations.items()} }")
            return allocations
        except Exception as e:
            logger.error(f"Error during HRP calculation: {e}. Falling back to equal weight.", exc_info=True)
            return {asset: 1.0 / num_assets for asset in assets}