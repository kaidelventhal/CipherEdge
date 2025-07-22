# FILE: tests/test_portfolio_manager.py
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from kamikaze_komodo.portfolio_constructor.portfolio_manager import PortfolioManager
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.config.settings import settings

# --- Mock Components for predictable testing ---

class MockAlwaysLongStrategy:
    """A simple strategy that always signals LONG."""
    def __init__(self, symbol, timeframe, params):
        self.symbol = symbol
        self.name = "AlwaysLong"
    def on_bar_data(self, bar: BarData):
        return SignalType.LONG
    def prepare_data(self, df): return df

class MockNeverTradeStrategy:
    """A simple strategy that always signals HOLD."""
    def __init__(self, symbol, timeframe, params):
        self.symbol = symbol
        self.name = "NeverTrade"
    def on_bar_data(self, bar: BarData):
        return SignalType.HOLD
    def prepare_data(self, df): return df

class TestPortfolioManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Monkey-patch the real StrategyManager to use our mock strategies for this test."""
        from kamikaze_komodo.strategy_framework import strategy_manager
        cls.original_create_strategy = strategy_manager.StrategyManager.create_strategy
        
        def mock_create_strategy(strategy_name, symbol, timeframe, params, **kwargs):
            if strategy_name == "AlwaysLong":
                return MockAlwaysLongStrategy(symbol, timeframe, params)
            if strategy_name == "NeverTrade":
                return MockNeverTradeStrategy(symbol, timeframe, params)
            return None # Should not happen in this test
        
        strategy_manager.StrategyManager.create_strategy = mock_create_strategy

    @classmethod
    def tearDownClass(cls):
        """Restore the original StrategyManager after the test."""
        from kamikaze_komodo.strategy_framework import strategy_manager
        strategy_manager.StrategyManager.create_strategy = cls.original_create_strategy

    def test_portfolio_backtest_execution(self):
        """
        🧪 Test the full portfolio backtest loop with simple, predictable strategies.
        """
        print("\n--- Running Portfolio Manager and Engine Test ---")

        # 1. Setup mock data for two assets
        dates = pd.to_datetime(pd.date_range(start="2025-01-01", periods=10, freq='4h'))
        
        price1 = np.linspace(100, 110, 10)
        mock_data1 = pd.DataFrame({
            'open': price1 - 0.5, 'high': price1 + 1,
            'low': price1 - 1, 'close': price1,
            'volume': np.random.randint(100, 1000, size=10)
        }, index=dates)

        price2 = np.linspace(200, 220, 10)
        mock_data2 = pd.DataFrame({
            'open': price2 - 0.5, 'high': price2 + 1,
            'low': price2 - 1, 'close': price2,
            'volume': np.random.randint(100, 1000, size=10)
        }, index=dates)
        
        portfolio_feeds = {'MOCK1/USD': mock_data1, 'MOCK2/USD': mock_data2}

        # 2. Define strategy configurations
        strategy_configs = [
            {
                'strategy_name': 'AlwaysLong', 'symbol': 'MOCK1/USD', 'timeframe': '4h',
                'strategy_params': {}, 'portfolio_weight': 0.60,
                'position_sizer_name': 'FixedFractionalPositionSizer'
            },
            {
                'strategy_name': 'NeverTrade', 'symbol': 'MOCK2/USD', 'timeframe': '4h',
                'strategy_params': {}, 'portfolio_weight': 0.40,
                'position_sizer_name': 'FixedFractionalPositionSizer'
            }
        ]
        
        initial_capital = 10000.0

        # 3. Initialize components
        portfolio_manager = PortfolioManager(strategy_configs, initial_capital)
        engine = BacktestingEngine(
            initial_capital=initial_capital,
            portfolio_manager=portfolio_manager,
            portfolio_data_feeds=portfolio_feeds
        )

        # 4. Run the backtest
        _, final_portfolio, equity_curve = engine.run()
        
        # --- Assertions ---
        self.assertIsNotNone(final_portfolio)
        self.assertIsNotNone(equity_curve)
        
        final_positions = portfolio_manager.positions
        self.assertIn('MOCK1/USD', final_positions)
        self.assertNotIn('MOCK2/USD', final_positions)

        # ** FIX: The assertion was too simple. Let's check the final DOLLAR VALUE. **
        # The target value for the MOCK1 position should always be close to:
        # (Current Portfolio Value) * (Strategy Weight 60%) * (Sizer Fraction 10%)
        # Let's check the final bar.
        final_portfolio_value = final_portfolio['final_portfolio_value']
        target_dollar_value = final_portfolio_value * 0.60 * 0.10
        
        final_price_mock1 = portfolio_feeds['MOCK1/USD']['close'].iloc[-1]
        actual_dollar_value = final_positions['MOCK1/USD'] * final_price_mock1
        
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Final Positions: {final_positions}")
        print(f"Final Target Value for MOCK1: ${target_dollar_value:.2f}")
        print(f"Final Actual Value for MOCK1: ${actual_dollar_value:.2f}")
        
        # Assert that the actual final dollar value is very close to the target
        self.assertAlmostEqual(actual_dollar_value, target_dollar_value, places=0)
        
        print("--- Portfolio Manager Test Passed Successfully ---")

if __name__ == '__main__':
    unittest.main()