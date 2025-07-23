# FILE: tests/test_optimizer_wfo.py
import unittest
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta, timezone

from cipher_edge.backtesting_engine.optimizer import StrategyOptimizer
from cipher_edge.config.settings import settings

class TestWFOptimizer(unittest.TestCase):

    def setUp(self):
        """Set up a mock data feed and optimizer instance for testing."""
        if not settings:
            self.fail("Settings could not be loaded. Check config files.")

        # Create a mock data feed with a clear trend
        date_range = pd.to_datetime(pd.date_range(end=datetime.now(timezone.utc), periods=1000, freq='4h'))
        price = 100 + np.cumsum(np.random.randn(1000) * 0.5) + np.sin(np.linspace(0, 20, 1000)) * 5
        self.mock_data = pd.DataFrame({
            'open': price - 0.5,
            'high': price + 1,
            'low': price - 1,
            'close': price,
            'volume': np.random.randint(100, 1000, size=1000)
        }, index=date_range)
        self.mock_data.index.name = 'timestamp'

        self.data_feeds = {'PF_ETHUSD': self.mock_data}

        # Override config for a quick test run
        settings.PHASE3_SYMBOLS = ["PF_ETHUSD"]
        settings.PHASE3_STRATEGIES = ["EhlersInstantaneousTrendlineStrategy"] # Test one simple strategy
        settings.PHASE3_RISK_MODULES = ["ATRBasedPositionSizer"]
        settings.PHASE3_STOP_MANAGERS = ["ParabolicSARStop"]
        
        # WFO settings for test
        settings.phase3_params['wfo_enabled'] = True
        settings.phase3_params['wfo_num_windows'] = 4 # Fewer windows for faster test
        settings.phase3_params['wfo_optuna_trials'] = 5 # Fewer optuna trials

        self.optimizer = StrategyOptimizer(
            data_feeds=self.data_feeds,
            initial_capital=10000.0,
            commission_bps=10.0,
            slippage_bps=2.0
        )

    def test_wfo_discovery_run(self):
        """
        🧪 Test that the Walk-Forward Optimization runs and produces a result.
        """
        print("\n--- Running Walk-Forward Optimization Test ---")
        
        results_df, equity_curves = self.optimizer.run_phase3_discovery()

        print(f"WFO discovery finished. Found {len(results_df)} results.")
        
        # --- Assertions ---
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertFalse(results_df.empty, "The WFO run should produce at least one result row.")
        
        self.assertIsInstance(equity_curves, dict)
        self.assertFalse(not equity_curves, "Equity curves dictionary should not be empty.")

        # Check for key metrics from the WFO run and Monte Carlo simulation
        self.assertIn('sharpe_ratio', results_df.columns)
        self.assertIn('mc_median_final_equity', results_df.columns)
        self.assertIn('mc_prob_of_loss_pct', results_df.columns)
        
        # Check that the equity curve is a pandas Series
        first_key = next(iter(equity_curves))
        self.assertIsInstance(equity_curves[first_key], pd.Series)

        print("--- WFO Test Passed Successfully ---")


if __name__ == '__main__':
    # To run this test from your project root:
    # python -m tests.test_optimizer_wfo
    unittest.main()