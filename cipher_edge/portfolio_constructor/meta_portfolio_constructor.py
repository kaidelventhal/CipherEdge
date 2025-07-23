import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from cipher_edge.portfolio_constructor.base_portfolio_constructor import BasePortfolioConstructor
from cipher_edge.risk_control_module.risk_manager import RiskManager
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class MultiStrategyPortfolioConstructor(BasePortfolioConstructor):
    """
    Constructs a portfolio from a list of backtested strategy combinations.
    Selects the top N performers and computes their capital allocation weights.
    """

    def __init__(self, settings: Any, risk_manager: RiskManager):
        """
        Initializes the constructor. For Phase 3, it inherits from BasePortfolioConstructor
        as planned, but its primary methods operate independently of the live rebalancing logic.
        """
        super().__init__(settings, risk_manager)
        logger.info("MultiStrategyPortfolioConstructor initialized.")

    def select_top_n(
        self,
        trials_df: pd.DataFrame,
        equity_curves: Dict[int, pd.Series],
        n: int,
        rank_by: str = 'deflated_sharpe_ratio',
        correlation_threshold: float = 0.8,
        sharpe_ratio_threshold: float = 0.5
    ) -> List[int]:
        """
        Selects the top N uncorrelated and profitable strategy combinations.

        Args:
            trials_df (pd.DataFrame): DataFrame of trial results, indexed by trial_id.
            equity_curves (Dict[int, pd.Series]): Dict mapping trial_id to its equity curve.
            n (int): The desired number of combinations in the final portfolio.
            rank_by (str): The metric to use for ranking (e.g., 'deflated_sharpe_ratio').
            correlation_threshold (float): The maximum allowed pairwise correlation between equity curves.
            sharpe_ratio_threshold (float): The minimum Sharpe ratio required for a strategy to be considered.

        Returns:
            List[int]: A list of the selected trial_ids.
        """
        if rank_by not in trials_df.columns:
            logger.error(f"Ranking metric '{rank_by}' not found in trials DataFrame. Cannot select top N.")
            return []

        profitable_trials = trials_df[trials_df['sharpe_ratio'] > sharpe_ratio_threshold]
        if profitable_trials.empty:
            logger.warning(f"No trials met the minimum Sharpe Ratio threshold of {sharpe_ratio_threshold}. No portfolio will be constructed.")
            return []
            
        logger.info(f"{len(profitable_trials)} trials passed the Sharpe ratio threshold of > {sharpe_ratio_threshold}.")

        ranked_trials = profitable_trials.sort_values(by=rank_by, ascending=False)
         
        selected_ids: List[int] = []
        selected_equity_curves: List[pd.Series] = []

        logger.info(f"Starting selection of top {n} combos from {len(ranked_trials)} trials, using correlation threshold < {correlation_threshold}.")

        for trial_id, row in ranked_trials.iterrows():
            if len(selected_ids) >= n:
                break

            current_equity_curve = equity_curves.get(trial_id)
            if current_equity_curve is None:
                logger.warning(f"Equity curve for trial_id {trial_id} not found. Skipping.")
                continue

            is_diverse = True
            if selected_equity_curves:
                combined_df = pd.concat(selected_equity_curves + [current_equity_curve], axis=1).ffill()
                correlation_matrix = combined_df.corr()
                correlations_with_new = correlation_matrix.iloc[:-1, -1]
                 
                if (correlations_with_new > correlation_threshold).any():
                    is_diverse = False
                    logger.debug(f"Trial {trial_id} is highly correlated with an already selected combo. Skipping.")
             
            if is_diverse:
                logger.debug(f"Trial {trial_id} selected. Adding to portfolio.")
                selected_ids.append(trial_id)
                selected_equity_curves.append(current_equity_curve.rename(trial_id))

        if len(selected_ids) < n:
            logger.warning(f"Could only select {len(selected_ids)} diverse combos, less than the target of {n}.")

        return selected_ids

    def compute_weights(
        self,
        selected_ids: List[int],
        equity_curves: Dict[int, pd.Series],
        method: str = 'risk_parity'
    ) -> Dict[int, float]:
        """
        Computes the capital allocation weights for the selected combinations.

        Args:
            selected_ids (List[int]): List of the trial_ids for the final portfolio.
            equity_curves (Dict[int, pd.Series]): Dict mapping trial_id to its equity curve.
            method (str): The weighting method ('equal', 'risk_parity', 'performance').

        Returns:
            Dict[int, float]: A dictionary mapping trial_id to its portfolio weight (0.0 to 1.0).
        """
        if not selected_ids:
            return {}
         
        clean_method = method.strip().replace('"', '')

        weights = {}
        if clean_method == 'equal':
            weight_per_combo = 1.0 / len(selected_ids)
            weights = {combo_id: weight_per_combo for combo_id in selected_ids}

        elif clean_method == 'risk_parity':
            volatilities = {}
            for combo_id in selected_ids:
                returns = equity_curves[combo_id].pct_change().dropna()
                volatility = returns.std()
                volatilities[combo_id] = volatility if volatility > 1e-9 else 1e-9

            inverse_volatilities = {combo_id: 1.0 / vol for combo_id, vol in volatilities.items()}
            total_inverse_vol = sum(inverse_volatilities.values())

            if total_inverse_vol > 0:
                weights = {combo_id: inv_vol / total_inverse_vol for combo_id, inv_vol in inverse_volatilities.items()}
            else:
                logger.warning("Could not compute risk parity weights, falling back to equal weight.")
                return self.compute_weights(selected_ids, equity_curves, 'equal')
         
        elif clean_method == 'performance':
            logger.warning(f"Weighting method '{clean_method}' is not fully implemented. Falling back to 'equal'.")
            return self.compute_weights(selected_ids, equity_curves, 'equal')

        else:
            logger.error(f"Unknown weighting method '{clean_method}'. Falling back to 'equal'.")
            return self.compute_weights(selected_ids, equity_curves, 'equal')

        return weights

    def calculate_target_allocations(self, *args, **kwargs) -> Dict[str, float]:
        """This method is part of the base class for live rebalancing, not used in Phase 3 discovery."""
        logger.warning("calculate_target_allocations is not used in Phase 3 meta-portfolio construction.")
        return {}