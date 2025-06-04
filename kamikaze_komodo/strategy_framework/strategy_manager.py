# kamikaze_komodo/strategy_framework/strategy_manager.py
from typing import List, Dict, Any
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.core.models import BarData
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class StrategyManager:
    """
    Manages the loading, initialization, and execution of trading strategies.
    """
    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        logger.info("StrategyManager initialized.")

    def add_strategy(self, strategy: BaseStrategy):
        """Adds a strategy instance to the manager."""
        if not isinstance(strategy, BaseStrategy):
            logger.error("Attempted to add an invalid strategy object.")
            raise ValueError("Strategy must be an instance of BaseStrategy.")
        
        self.strategies.append(strategy)
        logger.info(f"Strategy '{strategy.name}' for {strategy.symbol} ({strategy.timeframe}) added to StrategyManager.")

    def remove_strategy(self, strategy_name: str, symbol: str, timeframe: str):
        """Removes a strategy by its name, symbol, and timeframe."""
        initial_count = len(self.strategies)
        self.strategies = [
            s for s in self.strategies 
            if not (s.name == strategy_name and s.symbol == symbol and s.timeframe == timeframe)
        ]
        if len(self.strategies) < initial_count:
            logger.info(f"Strategy '{strategy_name}' for {symbol} ({timeframe}) removed.")
        else:
            logger.warning(f"Strategy '{strategy_name}' for {symbol} ({timeframe}) not found for removal.")


    def load_strategies_from_config(self, config: Dict[str, Any]):
        """
        Loads strategies based on a configuration dictionary.
        This is a placeholder for a more dynamic loading mechanism.
        For now, strategies are added manually or via specific calls.
        """
        # Example:
        # for strategy_config in config.get('strategies', []):
        #     strategy_class = resolve_strategy_class(strategy_config['name']) # Utility to get class from name
        #     params = strategy_config.get('params', {})
        #     symbol = strategy_config.get('symbol')
        #     timeframe = strategy_config.get('timeframe')
        #     if strategy_class and symbol and timeframe:
        #         self.add_strategy(strategy_class(symbol, timeframe, params))
        logger.warning("load_strategies_from_config is a placeholder and not fully implemented.")
        pass

    def on_bar_data_all(self, bar_data: BarData) -> Dict[str, SignalType]:
        """
        Distributes new bar data to all relevant strategies and collects signals.
        A strategy is relevant if the bar_data.symbol and bar_data.timeframe match.

        Returns:
            Dict[str, SignalType]: A dictionary where keys are strategy identifiers
                                   (e.g., "EWMACStrategy_BTC/USD_1h") and values are signals.
        """
        signals_from_strategies: Dict[str, SignalType] = {}
        for strategy in self.strategies:
            if strategy.symbol == bar_data.symbol and strategy.timeframe == bar_data.timeframe:
                signal = strategy.on_bar_data(bar_data)
                if signal: # Only record actual signals, not None or HOLD if not meaningful here
                    strategy_id = f"{strategy.name}_{strategy.symbol.replace('/', '')}_{strategy.timeframe}"
                    signals_from_strategies[strategy_id] = signal
                    logger.debug(f"Signal from {strategy_id}: {signal.name}")
        return signals_from_strategies

    def get_all_strategies(self) -> List[BaseStrategy]:
        return self.strategies

# Example Usage (Conceptual)
if __name__ == '__main__':
    from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
    from kamikaze_komodo.config.settings import settings # Assuming settings are loaded

    if settings:
        manager = StrategyManager()
        
        # Create and add a strategy instance
        ewmac_params = {
            'short_window': settings.ewmac_short_window,
            'long_window': settings.ewmac_long_window
        }
        ewmac_btc_1h = EWMACStrategy(symbol="BTC/USD", timeframe="1h", params=ewmac_params)
        manager.add_strategy(ewmac_btc_1h)

        # Simulate receiving bar data
        # In a real system, this BarData would come from DataFetcher
        from datetime import datetime, timezone
        example_bar = BarData(
            timestamp=datetime.now(timezone.utc),
            open=40000, high=40500, low=39800, close=40200, volume=100,
            symbol="BTC/USD", timeframe="1h"
        )

        # To actually get a signal, the strategy needs historical data first.
        # This is a simplified call. `ewmac_btc_1h.update_data_history(bar)` would need to be called many times first.
        # For a single bar without history, it will likely return HOLD or an error if not enough data.
        # signals = manager.on_bar_data_all(example_bar)
        # logger.info(f"Signals received: {signals}")
        logger.info("StrategyManager example completed. For meaningful signals, strategies need historical data.")
    else:
        logger.error("Settings not loaded, cannot run StrategyManager example.")