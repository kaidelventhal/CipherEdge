from typing import List, Dict, Any, Optional
import pandas as pd
from cipher_edge.core.models import BarData, Trade
from cipher_edge.core.enums import SignalType
from cipher_edge.strategy_framework.strategy_manager import StrategyManager
from cipher_edge.app_logger import get_logger
from cipher_edge.risk_control_module.position_sizer import BasePositionSizer, POSITION_SIZER_REGISTRY
from cipher_edge.config.settings import settings

logger = get_logger(__name__)

class PortfolioManager:
    """
    Manages a portfolio of multiple strategies, allocating capital, aggregating
    signals, and generating net orders for execution.
    """
    def __init__(self, strategy_configs: List[Dict[str, Any]], initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {} 
        self.portfolio_value = initial_capital
        self.strategies = []
        
        logger.info(f"Initializing PortfolioManager with {len(strategy_configs)} strategies.")
        
        for config in strategy_configs:
            strategy_instance = StrategyManager.create_strategy(
                strategy_name=config['strategy_name'],
                symbol=config['symbol'],
                timeframe=config['timeframe'],
                params=config['strategy_params']
            )
            if strategy_instance:
                strategy_instance.portfolio_weight = config.get('portfolio_weight', 0.0)
                strategy_instance.position_sizer_name = config.get('position_sizer_name')
                
                self.strategies.append(strategy_instance)
                logger.info(
                    f"  - Loaded strategy '{strategy_instance.name}' for {strategy_instance.symbol} "
                    f"with weight {strategy_instance.portfolio_weight:.2%}"
                )

    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Updates the total value of the portfolio based on current market prices."""
        asset_value = 0.0
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                asset_value += quantity * current_prices[symbol]
        self.portfolio_value = self.cash + asset_value

    def on_bar(self, market_data: Dict[str, BarData]) -> List[Dict[str, Any]]:
        """
        Processes a new bar of data for all relevant symbols.
        
        Args:
            market_data (Dict[str, BarData]): A dictionary mapping symbol to its latest BarData.

        Returns:
            List[Dict[str, Any]]: A list of order parameters to be executed.
        """
        current_prices = {symbol: bar.close for symbol, bar in market_data.items()}
        self.update_portfolio_value(current_prices)
        
        target_positions: Dict[str, float] = {}

        for strategy in self.strategies:
            if strategy.symbol in market_data:
                signal = strategy.on_bar_data(market_data[strategy.symbol])
                
                if signal in [SignalType.LONG, SignalType.SHORT]:
                    sizer_class = POSITION_SIZER_REGISTRY.get(strategy.position_sizer_name)
                    if sizer_class:
                        capital_slice = self.portfolio_value * strategy.portfolio_weight
                        
                        risk_params = settings.get_strategy_params('RiskManagement')
                        sizer = sizer_class(params=risk_params)

                        size_in_units = sizer.calculate_size(
                            symbol=strategy.symbol,
                            current_price=current_prices[strategy.symbol],
                            available_capital=capital_slice,
                            current_portfolio_value=capital_slice, 
                            trade_signal=signal,
                            strategy_info={},
                            latest_bar=market_data[strategy.symbol]
                        )
                        
                        if size_in_units and size_in_units > 0:
                            direction = 1 if signal == SignalType.LONG else -1
                            target_positions[strategy.symbol] = target_positions.get(strategy.symbol, 0) + (size_in_units * direction)

                elif signal in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                    pass

        orders_to_execute = []
        all_symbols = set(target_positions.keys()) | set(self.positions.keys())

        for symbol in all_symbols:
            current_position = self.positions.get(symbol, 0.0)
            target_position = target_positions.get(symbol, 0.0)
            
            trade_amount = target_position - current_position
            
            if abs(trade_amount) > 1e-9: # If there's a change needed
                order_side = "buy" if trade_amount > 0 else "sell"
                order = {
                    'symbol': symbol,
                    'side': order_side,
                    'amount': abs(trade_amount)
                }
                orders_to_execute.append(order)

        return orders_to_execute

    def update_fill(self, trade_result: Dict[str, Any]):
        """Updates portfolio state after a trade is executed."""
        symbol = trade_result['symbol']
        side = trade_result['side']
        amount = trade_result['amount']
        price = trade_result['price']
        commission = trade_result['commission']

        current_position = self.positions.get(symbol, 0.0)
        
        self.cash -= commission
        
        if side == 'buy':
            self.positions[symbol] = current_position + amount
            self.cash -= amount * price
        else: # sell
            self.positions[symbol] = current_position - amount
            self.cash += amount * price
            
        if abs(self.positions.get(symbol, 0.0)) < 1e-9:
            del self.positions[symbol]