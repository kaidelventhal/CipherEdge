# FILE: kamikaze_komodo/backtesting_engine/engine.py
import pandas as pd
import pandas_ta as ta
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
import asyncio
import os

from kamikaze_komodo.core.models import BarData, Trade, Order, PortfolioSnapshot
from kamikaze_komodo.core.enums import SignalType, OrderSide, TradeResult, OrderType
from kamikaze_komodo.app_logger import get_logger, logger as root_logger
from datetime import datetime, timezone, timedelta
from kamikaze_komodo.orchestration.portfolio_manager import PortfolioManager
from kamikaze_komodo.config.settings import settings
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager, ATRStopManager, PercentageStopManager, TripleBarrierStopManager
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher

logger = get_logger(__name__)

STOP_MANAGER_CLASS_MAP = {
    "ATRStopManager": ATRStopManager,
    "PercentageStopManager": PercentageStopManager,
    "TripleBarrierStopManager": TripleBarrierStopManager,
    "TripleBarrier": TripleBarrierStopManager,
}

class SimulatedExchangeAPI:
    def __init__(self, engine: 'BacktestingEngine'):
        self._engine = engine

    async def create_order(self, symbol: str, type: OrderType, side: OrderSide, amount: float, price: Optional[float] = None, params: Optional[Dict] = None) -> Order:
        return self._engine._execute_order(symbol, side, amount, type, price)

    async def close(self): pass

class BacktestingEngine:
    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        data_feeds: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        stop_manager: Optional[BaseStopManager] = None,
    ):
        if not data_feeds: raise ValueError("data_feeds dictionary cannot be empty.")
        
        self.data_feeds = {symbol: df.sort_index() for symbol, df in data_feeds.items()}
        for asset, df in self.data_feeds.items():
            if all(col in df.columns for col in ['high', 'low', 'close']):
                df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        
        self.simulated_exchange = SimulatedExchangeAPI(self)
        self.portfolio_manager = portfolio_manager
        self.portfolio_manager.exchange_api = self.simulated_exchange

        if stop_manager:
            self.stop_manager = stop_manager
        else:
            self.stop_manager = self._initialize_stop_manager()

        self.current_cash = initial_capital
        self.current_portfolio_value = initial_capital
        self.open_positions: Dict[str, Trade] = {}
        self.completed_trades_log: List[Trade] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.order_id_counter = 0
        
        logger.info(f"BacktestingEngine initialized. Stop Manager: {self.stop_manager.__class__.__name__ if self.stop_manager else 'None'}")

    def _initialize_stop_manager(self) -> BaseStopManager:
        stop_manager_name = settings.config.get('RiskManagement', 'StopManager_Default', fallback='PercentageStopManager')
        stop_manager_class = STOP_MANAGER_CLASS_MAP.get(stop_manager_name)
        if not stop_manager_class:
            logger.error(f"StopManager '{stop_manager_name}' not found. Defaulting to PercentageStopManager.")
            stop_manager_class = PercentageStopManager
        
        params = settings.get_strategy_params(stop_manager_name) or settings.get_strategy_params('RiskManagement')
        return stop_manager_class(params=params)

    @classmethod
    async def create(cls, portfolio_manager: PortfolioManager, initial_capital: float, commission_bps: float, slippage_bps: float) -> 'BacktestingEngine':
        root_logger.info("--- BacktestingEngine: Loading Data Feeds via create() ---")
        if not settings: raise ValueError("Settings cannot be loaded.")
        data_feeds = await cls._load_data_feeds_async()
        return cls(portfolio_manager, data_feeds, initial_capital, commission_bps, slippage_bps)

    @staticmethod
    async def _load_data_feeds_async() -> Dict[str, pd.DataFrame]:
        data_feeds = {}
        portfolio_config = settings.get_strategy_params('Portfolio')
        trading_universe: List[str] = [s.strip() for s in portfolio_config.get('tradinguniverse', '').split(',')]
        timeframe = settings.default_timeframe
        
        if not trading_universe or trading_universe == ['']: return {}

        db_manager = DatabaseManager()
        data_fetcher = DataFetcher()
        start_date = datetime.now(timezone.utc) - timedelta(days=settings.historical_data_days)
        
        for asset in trading_universe:
            logger.info(f"Loading data for {asset}...")
            bars = db_manager.retrieve_bar_data(asset, timeframe, start_date=start_date)
            if not bars or len(bars) < 200:
                logger.info(f"Fetching fresh data for {asset}...")
                bars = await data_fetcher.fetch_historical_data_for_period(asset, timeframe, start_date)
                if bars: db_manager.store_bar_data(bars)
            
            if bars:
                df = pd.DataFrame([bar.model_dump() for bar in bars])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                data_feeds[asset] = df
                logger.info(f"Loaded {len(df)} bars for {asset}.")
            else:
                logger.error(f"Could not load data for asset {asset}. It will be excluded from the backtest.")
        
        await data_fetcher.close()
        db_manager.close()
        
        if not data_feeds:
            raise ValueError("No data could be loaded for any asset in the universe. Aborting backtest.")
        return data_feeds

    def _get_next_order_id(self) -> str:
        self.order_id_counter += 1
        return f"order_{self.order_id_counter:05d}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if self.slippage_rate == 0.0: return price
        return price * (1 + self.slippage_rate) if side == OrderSide.BUY else price * (1 - self.slippage_rate)

    def _execute_order(self, symbol: str, side: OrderSide, amount: float, order_type: OrderType, price: Optional[float] = None, from_stop: bool = False) -> Order:
        if amount <= 1e-9: return
        
        current_bar = self.current_bars[symbol]
        execution_price = self._apply_slippage(price if price is not None else current_bar.close, side)
        order_id = self._get_next_order_id()
        commission = (amount * execution_price) * self.commission_rate

        position = self.open_positions.get(symbol)
        
        if position and position.side != side:
            # Closing or reversing position
            close_amount = min(amount, position.amount)
            pnl = (execution_price - position.entry_price) * close_amount if position.side == OrderSide.BUY else (position.entry_price - execution_price) * close_amount
            self.current_cash += pnl - commission

            position.amount -= close_amount
            if position.amount < 1e-9:
                closed_trade = self.open_positions.pop(symbol)
                closed_trade.exit_price = execution_price
                closed_trade.exit_timestamp = current_bar.timestamp
                closed_trade.exit_order_id = order_id
                closed_trade.pnl = pnl - closed_trade.commission
                self.completed_trades_log.append(closed_trade)

            if amount > close_amount: # Reversal
                new_amount = amount - close_amount
                trade = Trade(
                    id=order_id, entry_order_id=order_id, symbol=symbol, side=side, amount=new_amount,
                    entry_price=execution_price, entry_timestamp=current_bar.timestamp, commission=0,
                    custom_fields={"atr_at_entry": self.data_feeds[symbol].loc[current_bar.timestamp].get('atr')}
                )
                self.open_positions[symbol] = trade
        else:
            # Opening or increasing position
            new_amount = amount + (position.amount if position else 0)
            new_entry_price = ((position.entry_price * position.amount) + (execution_price * amount)) / new_amount if position else execution_price
            
            if position:
                position.entry_price = new_entry_price
                position.amount = new_amount
                position.commission += commission
            else:
                trade = Trade(
                    id=order_id, entry_order_id=order_id, symbol=symbol, side=side, amount=amount,
                    entry_price=execution_price, entry_timestamp=current_bar.timestamp, commission=commission,
                    custom_fields={"atr_at_entry": self.data_feeds[symbol].loc[current_bar.timestamp].get('atr')}
                )
                self.open_positions[symbol] = trade
        
        return Order(
            id=order_id, symbol=symbol, type=order_type, side=side, amount=amount, price=execution_price,
            timestamp=current_bar.timestamp, status='filled'
        )
        
    def _update_portfolio_history(self, timestamp: datetime):
        open_pnl = 0
        positions_value = 0
        for symbol, position in self.open_positions.items():
            current_price = self.current_bars.get(symbol, position).close
            positions_value += position.amount * current_price
            pnl = (current_price - position.entry_price) * position.amount if position.side == OrderSide.BUY else (position.entry_price - current_price) * position.amount
            open_pnl += pnl
        
        self.current_portfolio_value = self.current_cash + positions_value
        self.portfolio_history.append({"timestamp": timestamp, "cash": self.current_cash, "unrealized_pnl": open_pnl, "total_value": self.current_portfolio_value})
        positions_summary = {s: p.amount if p.side == OrderSide.BUY else -p.amount for s, p in self.open_positions.items()}
        self.portfolio_manager.portfolio_snapshot = PortfolioSnapshot(timestamp=timestamp, total_value_usd=self.current_portfolio_value, cash_balance_usd=self.current_cash, positions=positions_summary)
    
    def _check_stops(self, timestamp: datetime):
        if not self.stop_manager: return
        
        trades_to_close = []
        for symbol, trade in list(self.open_positions.items()):
            current_bar = self.current_bars.get(symbol)
            if not current_bar: continue
            
            stop_price = self.stop_manager.check_stop_loss(trade, current_bar, self.current_bar_index)
            if stop_price:
                trades_to_close.append({'trade': trade, 'price': stop_price})
                continue
            
            tp_price = self.stop_manager.check_take_profit(trade, current_bar)
            if tp_price:
                trades_to_close.append({'trade': trade, 'price': tp_price})

        for item in trades_to_close:
            trade = item['trade']
            price = item['price']
            logger.info(f"STOP TRIGGERED: Closing trade {trade.id} on {trade.symbol}.")
            closing_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
            self._execute_order(symbol=trade.symbol, side=closing_side, amount=trade.amount, order_type=OrderType.MARKET, price=price, from_stop=True)

    async def run(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
        logger.info("Starting portfolio backtest run...")
        all_timestamps = sorted(list(set.union(*[set(df.index) for df in self.data_feeds.values()])))
        
        if not all_timestamps:
            return [], {"initial_capital": self.initial_capital, "final_portfolio_value": self.initial_capital}, pd.DataFrame()

        self._update_portfolio_history(all_timestamps[0] - pd.Timedelta(seconds=1))

        for i, timestamp in enumerate(all_timestamps):
            self.current_bar_index = i
            self.current_bars: Dict[str, BarData] = {}
            historical_data_slice: Dict[str, pd.DataFrame] = {}
            
            has_new_data_for_cycle = False
            for asset, df in self.data_feeds.items():
                if timestamp in df.index:
                    has_new_data_for_cycle = True
                    historical_data_slice[asset] = df.loc[:timestamp]
                    
                    bar_data_dict = df.loc[timestamp].to_dict()
                    bar_data_dict.pop('symbol', None)
                    bar_data_dict.pop('timeframe', None)
                    
                    for key, value in bar_data_dict.items():
                        if isinstance(value, float) and np.isnan(value):
                            bar_data_dict[key] = None
                            
                    self.current_bars[asset] = BarData(timestamp=timestamp, symbol=asset, timeframe=self.portfolio_manager.timeframe, **bar_data_dict)
            
            self._update_portfolio_history(timestamp)
            self._check_stops(timestamp)

            if has_new_data_for_cycle:
                await self.portfolio_manager.run_cycle(historical_data_for_cycle=historical_data_slice)

        logger.info("Portfolio backtest run completed.")
        final_portfolio_state = {"initial_capital": self.initial_capital, "final_portfolio_value": self.current_portfolio_value}
        equity_curve_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        
        return self.completed_trades_log, final_portfolio_state, equity_curve_df