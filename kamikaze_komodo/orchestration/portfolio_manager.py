# FILE: kamikaze_komodo/orchestration/portfolio_manager.py
import pandas as pd
from typing import Dict, List, Optional, Any, Type

from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.strategy_framework.strategy_manager import StrategyManager
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.strategy_framework.strategies.bollinger_band_breakout_strategy import BollingerBandBreakoutStrategy
from kamikaze_komodo.strategy_framework.strategies.funding_rate_arb_strategy import FundingRateArbStrategy
from kamikaze_komodo.strategy_framework.strategies.ml_forecaster_strategy import MLForecasterStrategy
from kamikaze_komodo.portfolio_constructor.asset_allocator import HRPAllocator, BaseAssetAllocator, FixedWeightAssetAllocator
from kamikaze_komodo.portfolio_constructor.rebalancer import BasicRebalancer
from kamikaze_komodo.exchange_interaction.exchange_api import ExchangeAPI
from kamikaze_komodo.core.models import BarData, PortfolioSnapshot
from kamikaze_komodo.core.enums import SignalType

logger = get_logger(__name__)

# Mapping of strategy names in config to their respective classes
STRATEGY_CLASS_MAP: Dict[str, Type[BaseStrategy]] = {
    "EWMAC_Strategy": EWMACStrategy,
    "BollingerBandBreakout_Strategy": BollingerBandBreakoutStrategy,
    "FundingRateArb_Strategy": FundingRateArbStrategy,
    "MLForecaster_Strategy": MLForecasterStrategy,
}


class PortfolioManager:
    """
    The central orchestrator for the trading bot.
    Manages data, strategies, and execution for a portfolio of assets.
    """

    def __init__(self, exchange_api: Optional[Any] = None):
        if not settings:
            raise ValueError("Settings not loaded.")
        
        self.exchange_api = exchange_api if exchange_api else ExchangeAPI()
        self.is_backtest = exchange_api is not None
        
        self.portfolio_config = settings.get_strategy_params('Portfolio')
        self.trading_universe: List[str] = [s.strip() for s in self.portfolio_config.get('tradinguniverse', '').split(',')]
        self.timeframe = settings.default_timeframe
        
        logger.info(f"Managing portfolio for universe: {self.trading_universe}")

        self.strategy_manager = StrategyManager()
        self._load_strategies()

        self._initialize_constructor_components(self.portfolio_config)

        self.portfolio_snapshot = PortfolioSnapshot(
            total_value_usd=10000.0, # Initialized from config now
            cash_balance_usd=10000.0,
            positions={asset: 0.0 for asset in self.trading_universe}
        )

    def _load_strategies(self):
        logger.info("Dynamically loading strategies from config...")
        active_strategies_str = self.portfolio_config.get('activestrategies', '')
        if not active_strategies_str:
            logger.warning("No 'ActiveStrategies' defined in [Portfolio] section of config.")
            return

        active_strategy_names = [s.strip() for s in active_strategies_str.split(',')]
        
        for strategy_name in active_strategy_names:
            strategy_class = STRATEGY_CLASS_MAP.get(strategy_name)
            if strategy_class:
                strategy_params = settings.get_strategy_params(strategy_name)
                for asset in self.trading_universe:
                    strategy_instance = strategy_class(symbol=asset, timeframe=self.timeframe, params=strategy_params)
                    self.strategy_manager.add_strategy(strategy_instance)
            else:
                logger.error(f"Strategy '{strategy_name}' is active but not found in STRATEGY_CLASS_MAP.")

    def _initialize_constructor_components(self, config: Dict[str, Any]):
        allocator_name = str(config.get('assetallocator', 'HRPAllocator')).lower()
        if allocator_name == 'hrpallocator':
            self.asset_allocator: BaseAssetAllocator = HRPAllocator(params=settings.get_strategy_params('HRPAllocator'))
        else:
            fixed_weights = {asset: 1.0/len(self.trading_universe) for asset in self.trading_universe}
            self.asset_allocator = FixedWeightAssetAllocator(target_weights=fixed_weights)

        self.rebalancer = BasicRebalancer(params=settings.get_strategy_params('Rebalancer'))
        logger.info(f"Initialized Asset Allocator: {self.asset_allocator.__class__.__name__}")
        logger.info(f"Initialized Rebalancer: {self.rebalancer.__class__.__name__}")

    async def run_cycle(self, historical_data_for_cycle: Optional[Dict[str, pd.DataFrame]] = None):
        if not historical_data_for_cycle:
             if not self.is_backtest:
                logger.warning("Live data fetching not yet implemented in PortfolioManager.run_cycle.")
             return

        # 1. Generate Signals from all active strategies
        signals: Dict[str, SignalType] = {}
        for strategy in self.strategy_manager.get_all_strategies():
            if strategy.symbol in historical_data_for_cycle:
                strategy.data_history = historical_data_for_cycle[strategy.symbol]
                latest_bar_data = BarData(timestamp=strategy.data_history.index[-1], **strategy.data_history.iloc[-1].to_dict())
                signal = strategy.on_bar_data(latest_bar_data)
                if isinstance(signal, list): signal = signal[0].signal_type if signal else SignalType.HOLD
                if signal != SignalType.HOLD: signals[strategy.symbol] = signal
        
        logger.info(f"Consolidated signals: { {k: v.value for k, v in signals.items()} }")

        # 2. Determine Target Allocation
        assets_with_signal = [asset for asset, sig in signals.items() if sig in [SignalType.LONG, SignalType.SHORT]]
        target_allocations_pct: Dict[str, float] = {asset: 0.0 for asset in self.trading_universe}

        if assets_with_signal:
            allocation_data = {asset: data for asset, data in historical_data_for_cycle.items() if asset in assets_with_signal}
            
            # Ensure HRP allocator gets a returns series for each asset it needs to weigh
            weights = self.asset_allocator.allocate(
                assets=assets_with_signal,
                portfolio_value=self.portfolio_snapshot.total_value_usd,
                historical_data=allocation_data
            )
            
            for asset, weight in weights.items():
                if signals.get(asset) == SignalType.SHORT:
                    target_allocations_pct[asset] = -abs(weight)
                elif signals.get(asset) == SignalType.LONG:
                    target_allocations_pct[asset] = abs(weight)

        for asset, sig in signals.items():
            if sig in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                target_allocations_pct[asset] = 0.0

        logger.info(f"Target allocations (%): { {k: f'{v*100:.2f}%' for k,v in target_allocations_pct.items()} }")

        # 3. Generate Rebalancing Orders
        asset_prices = {asset: data.iloc[-1]['close'] for asset, data in historical_data_for_cycle.items() if not data.empty}
        rebalancing_orders = self.rebalancer.generate_rebalancing_orders(
            current_portfolio=self.portfolio_snapshot,
            target_allocations_pct=target_allocations_pct,
            asset_prices=asset_prices,
        )

        # 4. Execute Orders
        if not rebalancing_orders:
            logger.info("No rebalancing orders to execute.")
        else:
            logger.info(f"Executing {len(rebalancing_orders)} rebalancing orders...")
            for order_params in rebalancing_orders:
                logger.info(f"Placing order: {order_params}")
                await self.exchange_api.create_order(**order_params)