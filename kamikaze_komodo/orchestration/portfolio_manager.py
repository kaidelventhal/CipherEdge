# FILE: kamikaze_komodo/orchestration/portfolio_manager.py
import pandas as pd
from typing import Dict, List, Optional, Any

from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.strategy_framework.strategy_manager import StrategyManager
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.portfolio_constructor.asset_allocator import HRPAllocator, BaseAssetAllocator
from kamikaze_komodo.portfolio_constructor.rebalancer import BasicRebalancer
from kamikaze_komodo.exchange_interaction.exchange_api import ExchangeAPI
from kamikaze_komodo.core.models import BarData, PortfolioSnapshot, Order
from kamikaze_komodo.core.enums import SignalType

logger = get_logger(__name__)

class PortfolioManager:
    """
    The central orchestrator for the trading bot.
    Manages data, strategies, and execution for a portfolio of assets.
    This class is designed to be used by both the live trading scheduler and the backtesting engine.
    """

    def __init__(self, exchange_api: Optional[Any] = None):
        if not settings:
            raise ValueError("Settings not loaded.")
        
        # Determine operating mode and exchange connection
        if exchange_api:
            self.exchange_api = exchange_api
            self.is_backtest = True
            logger.info("PortfolioManager initialized in backtesting mode.")
        else:
            self.exchange_api = ExchangeAPI()
            self.is_backtest = False
            logger.info("PortfolioManager initialized in live trading mode.")
        
        # Load portfolio configuration
        portfolio_config = settings.get_strategy_params('Portfolio')
        self.trading_universe_str = portfolio_config.get('tradinguniverse', settings.default_symbol)
        self.trading_universe: List[str] = [s.strip() for s in self.trading_universe_str.split(',')]
        self.timeframe = settings.default_timeframe
        
        logger.info(f"Managing portfolio for universe: {self.trading_universe}")

        # Initialize core components
        self.db_manager = DatabaseManager()
        self.strategy_manager = StrategyManager()
        self._load_strategies()

        # Initialize portfolio constructor components
        # TODO: Dynamic loading of allocator and rebalancer
        self.asset_allocator: BaseAssetAllocator = HRPAllocator()
        self.rebalancer = BasicRebalancer(params=settings.get_strategy_params('Rebalancer'))

        # Portfolio state
        self.portfolio_snapshot = PortfolioSnapshot(total_value_usd=0, cash_balance_usd=0) # Will be updated

    def _load_strategies(self):
        """
        Loads and initializes strategies for assets in the trading universe.
        This is a placeholder for a more dynamic implementation based on a config file.
        """
        logger.info("Loading strategies for the trading universe...")
        for asset in self.trading_universe:
            # For this phase, we assume a single strategy type (e.g., EWMAC) applies to all assets.
            strategy_params = settings.get_strategy_params('EWMAC_Strategy')
            strategy = EWMACStrategy(
                symbol=asset, 
                timeframe=self.timeframe, 
                params=strategy_params
            )
            self.strategy_manager.add_strategy(strategy)

    async def _update_portfolio_state(self):
        """Fetches and updates the current portfolio state from the exchange."""
        logger.info("Updating portfolio state from exchange...")
        balance = await self.exchange_api.fetch_balance()
        # This is a simplified view. A real implementation needs to handle different quote currencies,
        # margin, futures PnL, etc. For now, we assume a USD-based portfolio.
        # It also needs to fetch current positions. CCXT's fetchBalance can often do this.
        
        # Placeholder logic
        cash = balance.get('USD', {}).get('free', 0.0)
        total_value = balance.get('USD', {}).get('total', 0.0)
        positions = {} # Needs to be populated from balance info
        
        self.portfolio_snapshot = PortfolioSnapshot(
            total_value_usd=total_value,
            cash_balance_usd=cash,
            positions=positions
        )
        logger.info(f"Portfolio state updated: Total Value=${total_value:.2f}, Cash=${cash:.2f}")

    async def run_cycle(self, historical_data_for_cycle: Optional[Dict[str, pd.DataFrame]] = None):
        """
        The main trading loop. Can be triggered by a scheduler for live trading
        or by the backtesting engine for simulation.

        Args:
            historical_data_for_cycle: In backtesting, this provides the complete historical data up to the
                                       current simulation time. In live trading, this is None.
        """
        logger.info("Starting new portfolio management cycle.")

        # 1. Update Portfolio State
        if not self.is_backtest:
            await self._update_portfolio_state()

        # 2. Fetch/Update Data
        # In a live run, fetch latest data. In a backtest, data is provided.
        current_data: Dict[str, BarData] = {}
        all_asset_data: Dict[str, pd.DataFrame] = {}

        if self.is_backtest and historical_data_for_cycle:
             all_asset_data = historical_data_for_cycle
             for asset, df in all_asset_data.items():
                 if not df.empty:
                     latest_row = df.iloc[-1]
                     # The row from the dataframe contains all necessary fields from the original
                     # BarData object (except timestamp, which is now the index).
                     current_data[asset] = BarData(timestamp=df.index[-1], **latest_row.to_dict())
        else: # Live trading mode
            # TODO: Implement live data fetching logic
            logger.warning("Live data fetching not yet implemented in PortfolioManager.run_cycle.")
            return

        # 3. Generate Signals
        signals: Dict[str, SignalType] = {}
        for strategy in self.strategy_manager.get_all_strategies():
            if strategy.symbol in all_asset_data:
                strategy.data_history = all_asset_data[strategy.symbol] # Ensure strategy has latest history
                signal = strategy.on_bar_data(current_data[strategy.symbol])
                if isinstance(signal, list): # Handle complex signals if necessary
                    logger.warning(f"PortfolioManager received a list of SignalCommands from {strategy.name}, but does not yet support them. Taking first signal.")
                    signal = signal[0].signal_type if signal else SignalType.HOLD
                signals[strategy.symbol] = signal
        
        logger.info(f"Generated signals: { {k: v.value for k, v in signals.items()} }")

        # 4. Determine Target Allocation (simplified for now)
        # This step should use the Asset Allocator.
        # For Priority 1, we'll use a simplified logic: allocate to assets with LONG signals.
        target_allocations_pct: Dict[str, float] = {asset: 0.0 for asset in self.trading_universe}
        long_signals = [asset for asset, sig in signals.items() if sig == SignalType.LONG]
        if long_signals:
            equal_weight = 1.0 / len(long_signals)
            for asset in long_signals:
                target_allocations_pct[asset] = equal_weight
        
        logger.info(f"Target allocations: {target_allocations_pct}")

        # 5. Generate Rebalancing Orders
        asset_prices = {asset: bar.close for asset, bar in current_data.items() if asset in current_data}
        rebalancing_orders = self.rebalancer.generate_rebalancing_orders(
            current_portfolio=self.portfolio_snapshot,
            target_allocations_pct=target_allocations_pct,
            asset_prices=asset_prices,
        )

        # 6. Execute Orders
        if not rebalancing_orders:
            logger.info("No rebalancing orders to execute.")
        else:
            logger.info(f"Executing {len(rebalancing_orders)} rebalancing orders...")
            for order_params in rebalancing_orders:
                logger.info(f"Placing order: {order_params}")
                # In both live and backtest, this calls the (real or simulated) exchange API
                await self.exchange_api.create_order(**order_params)

        logger.info("Portfolio management cycle finished.")