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
    ) -> Dict[str, float]: # Target allocation in terms of capital or percentage
        """
        Determines the target allocation for assets.
        Args:
            assets (List[str]): List of asset symbols to consider for allocation.
            portfolio_value (float): Total capital available for allocation.
            historical_data (Optional[Dict[str, pd.DataFrame]]): Historical OHLCV data for assets.
            trade_history (Optional[pd.DataFrame]): For OptimalF, needs past trade performance.
        Returns:
            Dict[str, float]: Dictionary mapping asset symbols to target capital allocation.
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
        allocation_targets_capital: Dict[str, float] = {}
        relevant_target_weights = {asset: self.target_weights.get(asset, 0.0) for asset in assets if asset in self.target_weights}
        total_weight_for_relevant_assets = sum(relevant_target_weights.values())

        if total_weight_for_relevant_assets == 0:
            logger.debug("No target weights specified for the given assets or total weight is zero. No allocation.")
            return {asset: 0.0 for asset in assets}

        for asset in assets:
            if asset in relevant_target_weights:
                normalized_weight = relevant_target_weights[asset] / total_weight_for_relevant_assets
                allocation_targets_capital[asset] = portfolio_value * normalized_weight
            else:
                allocation_targets_capital[asset] = 0.0
        logger.debug(f"FixedWeightAssetAllocator target capital allocation: {allocation_targets_capital}")
        return allocation_targets_capital

class OptimalFAllocator(BaseAssetAllocator):
    """
    Allocates capital based on Vince's Optimal f (Kelly Criterion variant).
    Optimal f calculation typically needs a series of past trade returns (HPRs - Holding Period Returns).
    f = ( (R+1) * P - 1 ) / R  where P is win rate, R is avg win / avg loss (payoff ratio).
    This implementation can use defaults or calculate from provided trade_history.
    The allocation is per asset/strategy; this allocator itself doesn't combine multiple Optimal f values.
    It calculates Optimal f for a *single series of trades* that it assumes represents the strategy for the given asset.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.default_win_probability = float(self.params.get('optimalf_default_win_probability', 0.51)) # Slight edge
        self.default_payoff_ratio = float(self.params.get('optimalf_default_payoff_ratio', 1.1)) # AvgWin / AvgLoss
        self.kelly_fraction = float(self.params.get('optimalf_kelly_fraction', 0.25)) # Fraction of optimal f to use (e.g., quarter Kelly)
        self.min_trades_for_stats = int(self.params.get('optimalf_min_trades_for_stats', 20))
        logger.info(f"OptimalFAllocator initialized. Default WinProb: {self.default_win_probability}, Default Payoff: {self.default_payoff_ratio}, Kelly Fraction: {self.kelly_fraction}, Min Trades: {self.min_trades_for_stats}")

    def _calculate_stats_from_history(self, asset_symbol: str, trade_history: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
        if trade_history is None or trade_history.empty:
            return None
    
        asset_trades = trade_history[trade_history['symbol'] == asset_symbol]
        if len(asset_trades) < self.min_trades_for_stats:
            logger.debug(f"Not enough trades ({len(asset_trades)}) for {asset_symbol} to calculate Optimal F stats. Using defaults.")
            return None

        wins = asset_trades[asset_trades['pnl'] > 0]['pnl']
        losses = asset_trades[asset_trades['pnl'] < 0]['pnl'].abs() # Losses are positive for payoff ratio calculation

        if len(wins) == 0 or len(losses) == 0: # Avoid division by zero if no wins or no losses
            logger.debug(f"Not enough diversity in trades (wins: {len(wins)}, losses: {len(losses)}) for {asset_symbol}. Using defaults.")
            return None

        win_probability = len(wins) / len(asset_trades)
        average_win = wins.mean()
        average_loss = losses.mean()

        if average_loss == 0: # Avoid division by zero
            logger.debug(f"Average loss is zero for {asset_symbol}. Cannot calculate payoff ratio. Using defaults.")
            return None
        payoff_ratio = average_win / average_loss
        return {"win_probability": win_probability, "payoff_ratio": payoff_ratio}

    def allocate(
        self,
        assets: List[str],
        portfolio_value: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None,
        trade_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        allocations: Dict[str, float] = {}
        for asset in assets:
            stats = self._calculate_stats_from_history(asset, trade_history)
            win_prob = self.default_win_probability
            payoff_ratio = self.default_payoff_ratio

            if stats:
                win_prob = stats['win_probability']
                payoff_ratio = stats['payoff_ratio']
                logger.info(f"Optimal F for {asset}: Using calculated WinProb={win_prob:.3f}, PayoffRatio={payoff_ratio:.3f}")
            else:
                logger.info(f"Optimal F for {asset}: Using default WinProb={win_prob:.3f}, PayoffRatio={payoff_ratio:.3f}")

            if payoff_ratio <= 0: # Ensure payoff ratio is positive
                optimal_f = -1.0 # Indicates no bet
            else:
                # Kelly formula: f = W - (1-W)/R
                optimal_f = win_prob - ((1 - win_prob) / payoff_ratio)
        
            allocated_capital = 0.0
            if optimal_f > 0:
                fraction_to_invest = optimal_f * self.kelly_fraction
                allocated_capital = portfolio_value * fraction_to_invest
                logger.info(f"Optimal F for {asset}: f*={optimal_f:.4f}, KellyFraction={self.kelly_fraction:.2f}. Target capital: ${allocated_capital:.2f}")
            else:
                logger.info(f"Optimal f for {asset} is not positive ({optimal_f:.4f}). No allocation.")
            allocations[asset] = allocated_capital
    
        # This allocator returns capital per asset based on its own Optimal F.
        # The sum of these allocations could exceed portfolio_value if not careful,
        # or if the user intends this for individual strategy sizing rather than portfolio allocation.
        # For now, it's direct capital per asset. Normalization might be needed by the caller.
        return allocations


class HRPAllocator(BaseAssetAllocator):
    """
    Allocates assets using Hierarchical Risk Parity (HRP) by De Prado.
    Requires historical returns data for assets.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self.linkage_method = self.params.get('hrp_linkage_method', 'ward') # common: ward, single, complete
        logger.info(f"HRPAllocator initialized. Linkage method: {self.linkage_method}")

    def _get_cluster_var(self, cov_matrix: pd.DataFrame, cluster_items: List[int]) -> float:
        """Calculates variance of a cluster."""
        cluster_cov_matrix = cov_matrix.iloc[cluster_items, cluster_items]
        parity_w = 1.0 / np.diag(cluster_cov_matrix)
        parity_w = parity_w / parity_w.sum()
        cluster_var = np.dot(parity_w, np.dot(cluster_cov_matrix, parity_w))
        return cluster_var

    def _get_recursive_bisection(self, sort_ix: List[int], current_weights: np.ndarray, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Performs recursive bisection for HRP weights."""
        if len(sort_ix) == 1:
            return current_weights

        # Bisection
        mid_point = len(sort_ix) // 2
        cluster1_items = sort_ix[:mid_point]
        cluster2_items = sort_ix[mid_point:]

        cluster1_var = self._get_cluster_var(cov_matrix, cluster1_items)
        cluster2_var = self._get_cluster_var(cov_matrix, cluster2_items)

        alpha = cluster2_var / (cluster1_var + cluster2_var) # Allocation factor

        # Allocate weights to clusters
        for i in cluster1_items:
            current_weights[i] *= alpha
        for i in cluster2_items:
            current_weights[i] *= (1 - alpha)

        # Recursively bisect sub-clusters
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
        if not historical_data or len(assets) < 2:
            logger.warning("HRPAllocator requires historical data for at least 2 assets.")
            # Fallback to equal weight if only one asset or no data
            if len(assets) == 1: return {assets[0]: portfolio_value}
            return {asset: portfolio_value / len(assets) if assets else 0.0 for asset in assets}

        returns_data = {}
        for asset in assets:
            if asset in historical_data and not historical_data[asset].empty:
                returns_data[asset] = historical_data[asset]['close'].pct_change().dropna()
            else:
                logger.warning(f"No historical 'close' data for asset {asset} in HRPAllocator.")
                # Cannot proceed without data for all assets
                return {ast: portfolio_value / len(assets) if assets else 0.0 for ast in assets} # Fallback

        returns_df = pd.DataFrame(returns_data).dropna()
        if returns_df.shape[0] < 2 or returns_df.shape[1] < 2 : # Need enough observations and assets
            logger.warning(f"Not enough processed return data for HRP. Shape: {returns_df.shape}")
            return {asset: portfolio_value / len(assets) if assets else 0.0 for asset in assets}

        cov_matrix = returns_df.cov()
        corr_matrix = returns_df.corr()
    
        # Hierarchical Clustering
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix)) # Distance matrix
        condensed_dist_matrix = squareform(dist_matrix)
        link = linkage(condensed_dist_matrix, method=self.linkage_method)

        # Quasi-Diagonalization (sorting items by cluster leaves)
        sort_ix = dendrogram(link, no_plot=True)['leaves']

        # Recursive Bisection
        initial_weights = np.ones(len(assets))
        hrp_weights_array = self._get_recursive_bisection(sort_ix, initial_weights, cov_matrix)
    
        # Normalize weights
        hrp_weights = pd.Series(hrp_weights_array, index=[assets[i] for i in sort_ix])
        hrp_weights = hrp_weights / hrp_weights.sum() # Ensure they sum to 1
        hrp_weights = hrp_weights.reindex(assets).fillna(0.0) # Reorder to original asset list and fill NaNs for any missing

        allocations = {asset: portfolio_value * hrp_weights[asset] for asset in assets}
        logger.info(f"HRP Allocator target capital allocation: {allocations}")
        return allocations