import pandas as pd
import numpy as np
from typing import List, Dict, Any
from cipher_edge.core.models import Trade
from cipher_edge.app_logger import get_logger

logger = get_logger(__name__)

class MonteCarloSimulator:
    """
    Performs a Monte Carlo simulation on a series of trades to assess the
    robustness of an equity curve and determine if the result was skill or luck.
    """
    def __init__(
        self,
        trades_log: List[Trade],
        initial_capital: float = 10000.0,
        n_simulations: int = 1000
    ):
        """
        Initializes the simulator.

        Args:
            trades_log (List[Trade]): A list of completed trades from a backtest.
            initial_capital (float): The starting capital for the backtest.
            n_simulations (int): The number of random trade shuffles to perform.
        """
        self.trades_log = trades_log
        self.initial_capital = initial_capital
        self.n_simulations = n_simulations
        self.trade_pnls = [trade.pnl for trade in self.trades_log if trade.pnl is not None]

    def _generate_single_equity_curve(self) -> Dict[str, Any]:
        """
        Generates a single simulated equity curve by shuffling the trade PnLs.
        """
        if not self.trade_pnls:
            return {'final_equity': self.initial_capital, 'max_drawdown': 0.0}

        shuffled_pnls = np.random.permutation(self.trade_pnls)
        equity_curve = np.cumsum(shuffled_pnls) + self.initial_capital

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        return {
            'final_equity': equity_curve[-1],
            'max_drawdown': max_drawdown
        }

    def run_simulation(self) -> Dict[str, Any]:
        """
        Runs the full Monte Carlo simulation.

        Returns:
            Dict[str, Any]: A dictionary containing key simulation statistics.
        """
        if not self.trade_pnls:
            logger.warning("No trades with PnL found. Cannot run Monte Carlo simulation.")
            return {
                'mc_median_final_equity': self.initial_capital,
                'mc_5th_percentile_equity': self.initial_capital,
                'mc_prob_of_loss': 0.0,
                'mc_avg_max_drawdown_pct': 0.0
            }

        final_equities = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            sim_result = self._generate_single_equity_curve()
            final_equities.append(sim_result['final_equity'])
            max_drawdowns.append(sim_result['max_drawdown'])

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        median_final_equity = np.median(final_equities)
        percentile_5th_equity = np.percentile(final_equities, 5)
        prob_of_loss = np.mean(final_equities < self.initial_capital) * 100
        avg_max_drawdown_pct = np.mean(max_drawdowns) * 100

        results = {
            'mc_median_final_equity': median_final_equity,
            'mc_5th_percentile_equity': percentile_5th_equity,
            'mc_prob_of_loss_pct': prob_of_loss,
            'mc_avg_max_drawdown_pct': avg_max_drawdown_pct,
        }
        logger.info(f"Monte Carlo Simulation Results: {results}")
        return results