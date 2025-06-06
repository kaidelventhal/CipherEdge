# FILE: kamikaze_komodo/backtesting_engine/engine.py
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from kamikaze_komodo.core.models import BarData, Trade, Order, PortfolioSnapshot
from kamikaze_komodo.core.enums import SignalType, OrderSide, TradeResult, OrderType
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone
from kamikaze_komodo.orchestration.portfolio_manager import PortfolioManager

logger = get_logger(__name__)

class SimulatedExchangeAPI:
    """
    A simulated exchange API for the backtesting engine.
    It mimics the real ExchangeAPI but operates on the backtester's internal state.
    """
    def __init__(self, engine: 'BacktestingEngine'):
        self._engine = engine
        logger.info("SimulatedExchangeAPI initialized for backtesting.")

    async def fetch_balance(self) -> Dict[str, Any]:
        # Return the current state of the backtesting portfolio
        return {
            'USD': {
                'free': self._engine.current_cash,
                'used': 0.0,
                'total': self._engine.current_portfolio_value
            }
        }

    async def create_order(self, symbol: str, type: OrderType, side: OrderSide, amount: float, price: Optional[float] = None, params: Optional[Dict] = None) -> Order:
        # This is where the backtester processes a trade
        return self._engine._execute_order(symbol, side, amount, type, price)

    async def close(self):
        pass # Nothing to close in simulation

class BacktestingEngine:
    """
    A backtesting engine designed to simulate a PortfolioManager over a set of assets.
    """
    def __init__(
        self,
        data_feeds: Dict[str, pd.DataFrame],
        portfolio_manager_class: type,
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ):
        if not data_feeds:
            raise ValueError("data_feeds dictionary cannot be empty.")
        
        self.data_feeds = {symbol: df.sort_index() for symbol, df in data_feeds.items()}
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0

        # Portfolio State
        self.current_cash = initial_capital
        self.current_portfolio_value = initial_capital
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in self.data_feeds.keys()} # symbol -> quantity
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_log: List[Trade] = []
        self.trade_id_counter = 0

        # Create a simulated exchange and pass it to the PortfolioManager
        self.simulated_exchange = SimulatedExchangeAPI(self)
        self.portfolio_manager = portfolio_manager_class(exchange_api=self.simulated_exchange)
        
        logger.info(
            f"BacktestingEngine initialized for PortfolioManager '{portfolio_manager_class.__name__}'. "
            f"Assets: {list(self.data_feeds.keys())}. Initial Capital: ${initial_capital:,.2f}"
        )

    def _get_next_trade_id(self) -> str:
        self.trade_id_counter += 1
        return f"trade_{self.trade_id_counter:05d}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if self.slippage_rate == 0.0: return price
        return price * (1 + self.slippage_rate) if side == OrderSide.BUY else price * (1 - self.slippage_rate)

    def _execute_order(self, symbol: str, side: OrderSide, amount: float, order_type: OrderType, price: Optional[float] = None) -> Order:
        """Processes an order from the PortfolioManager and updates portfolio state."""
        current_bar = self.current_bars[symbol]
        execution_price = self._apply_slippage(current_bar.close, side) # Assume execution at close for simplicity
        
        cost_or_proceeds = amount * execution_price
        commission_cost = cost_or_proceeds * self.commission_rate
        
        # Update cash
        self.current_cash -= commission_cost
        if side == OrderSide.BUY:
            self.current_cash -= cost_or_proceeds
            self.positions[symbol] += amount
        else: # SELL
            self.current_cash += cost_or_proceeds
            self.positions[symbol] -= amount
            
        logger.info(
            f"{current_bar.timestamp} - SIM EXECUTE: {side.name} {amount:.6f} {symbol} @ ${execution_price:.4f}. "
            f"Commission: ${commission_cost:.2f}. New Cash: ${self.current_cash:.2f}"
        )
        
        # For simplicity, we create a filled Order and a Trade log entry simultaneously
        trade = Trade(
            id=self._get_next_trade_id(),
            symbol=symbol,
            entry_order_id=f"{side.value}_{self.trade_id_counter}",
            side=side,
            entry_price=execution_price,
            amount=amount,
            entry_timestamp=current_bar.timestamp,
            commission=commission_cost,
        )
        self.trades_log.append(trade) # Note: This simple model logs each transaction as a separate 'trade'
        
        return Order(
            id=f"order_{self.trade_id_counter}",
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=execution_price,
            timestamp=current_bar.timestamp,
            status='filled',
        )

    def _update_portfolio_history(self, timestamp: datetime):
        """Calculates and records the current portfolio value."""
        asset_value = 0.0
        for symbol, quantity in self.positions.items():
            if quantity != 0 and symbol in self.current_bars:
                asset_value += quantity * self.current_bars[symbol].close
        
        self.current_portfolio_value = self.current_cash + asset_value
        self.portfolio_history.append({
            "timestamp": timestamp,
            "cash": self.current_cash,
            "asset_value": asset_value,
            "total_value": self.current_portfolio_value
        })
        # Update the portfolio manager's snapshot for the next cycle
        self.portfolio_manager.portfolio_snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value_usd=self.current_portfolio_value,
            cash_balance_usd=self.current_cash,
            positions=self.positions.copy()
        )

    async def run(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
        logger.info("Starting portfolio backtest run...")

        # Create a unified timeline from all data feeds
        all_timestamps = sorted(list(set.union(*[set(df.index) for df in self.data_feeds.values()])))
        
        if not all_timestamps:
            logger.error("No data available in data_feeds to run backtest.")
            return [], {"initial_capital": self.initial_capital, "final_portfolio_value": self.initial_capital}, pd.DataFrame()

        self.portfolio_history.append({
            "timestamp": all_timestamps[0] - pd.Timedelta(seconds=1),
            "cash": self.initial_capital, "asset_value": 0.0, "total_value": self.initial_capital
        })

        for i, timestamp in enumerate(all_timestamps):
            self.current_bars: Dict[str, BarData] = {}
            historical_data_slice: Dict[str, pd.DataFrame] = {}
            
            # Prepare data for this timestamp
            has_new_data = False
            for asset, df in self.data_feeds.items():
                if timestamp in df.index:
                    has_new_data = True
                    historical_data_slice[asset] = df.loc[:timestamp]
                    
                    # FIX: Populate the engine's current_bars dictionary
                    latest_row = df.loc[timestamp]
                    # The to_dict() contains all necessary columns except the timestamp itself
                    self.current_bars[asset] = BarData(timestamp=timestamp, **latest_row.to_dict())
            
            if not has_new_data:
                continue

            # Run the portfolio manager's logic for this time step
            await self.portfolio_manager.run_cycle(historical_data_for_cycle=historical_data_slice)

            # Update portfolio value at the end of the bar
            self._update_portfolio_history(timestamp)

        logger.info("Portfolio backtest run completed.")
        final_portfolio_state = {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": self.current_portfolio_value,
        }
        equity_curve_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        
        # Note: The trades_log is simplified and may need post-processing to pair buys and sells
        # into complete trades for accurate performance analysis.
        return self.trades_log, final_portfolio_state, equity_curve_df