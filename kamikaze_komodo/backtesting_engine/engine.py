# FILE: kamikaze_komodo/backtesting_engine/engine.py
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from kamikaze_komodo.core.models import BarData, Trade, Order, PortfolioSnapshot
from kamikaze_komodo.core.enums import SignalType, OrderSide, TradeResult, OrderType
from kamikaze_komodo.app_logger import get_logger, logger as root_logger
from datetime import datetime, timezone, timedelta
from kamikaze_komodo.orchestration.portfolio_manager import PortfolioManager
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
import os
import asyncio
from collections import deque
import numpy as np

# Import risk control modules
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager, ATRStopManager, PercentageStopManager, TripleBarrierStopManager
from kamikaze_komodo.risk_control_module.volatility_band_stop_manager import VolatilityBandStopManager


logger = get_logger(__name__)

# Mapping of Stop Manager names in config to their classes
STOP_MANAGER_CLASS_MAP = {
    "ATRBased": ATRStopManager,
    "PercentageBased": PercentageStopManager,
    "TripleBarrier": TripleBarrierStopManager,
    "VolatilityBand": VolatilityBandStopManager,
}


class SimulatedExchangeAPI:
    def __init__(self, engine: 'BacktestingEngine'):
        self._engine = engine
        logger.info("SimulatedExchangeAPI initialized for backtesting.")

    async def create_order(self, symbol: str, type: OrderType, side: OrderSide, amount: float, price: Optional[float] = None, params: Optional[Dict] = None) -> Order:
        return self._engine._execute_order(symbol, side, amount, type, price)

    async def close(self): pass

class BacktestingEngine:
    def __init__(
        self,
        portfolio_manager_class: type,
        data_feeds: Dict[str, pd.DataFrame],
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ):
        if not data_feeds: raise ValueError("data_feeds dictionary cannot be empty.")
        
        self.data_feeds = {symbol: df.sort_index() for symbol, df in data_feeds.items()}
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0

        self.current_cash = initial_capital
        self.current_portfolio_value = initial_capital
        self.open_positions: Dict[str, deque[Trade]] = {symbol: deque() for symbol in self.data_feeds.keys()}
        self.completed_trades_log: List[Trade] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.order_id_counter = 0

        self.simulated_exchange = SimulatedExchangeAPI(self)
        self.portfolio_manager = portfolio_manager_class(exchange_api=self.simulated_exchange)
        
        self.stop_manager = self._initialize_stop_manager()

        logger.info(f"BacktestingEngine initialized. Stop Manager: {self.stop_manager.__class__.__name__}")

    def _initialize_stop_manager(self) -> BaseStopManager:
        """Dynamically initializes the stop manager from config settings."""
        stop_manager_name = settings.config.get('RiskManagement', 'StopManager_Default', fallback='PercentageStopManager')
        stop_manager_class = STOP_MANAGER_CLASS_MAP.get(stop_manager_name)
        if not stop_manager_class:
            logger.error(f"StopManager '{stop_manager_name}' not found in map. Defaulting to PercentageStopManager.")
            stop_manager_class = PercentageStopManager

        params = settings.get_strategy_params('RiskManagement')
        return stop_manager_class(params=params)


    @classmethod
    async def create(cls, portfolio_manager_class: type, initial_capital: float, commission_bps: float, slippage_bps: float) -> 'BacktestingEngine':
        root_logger.info("--- BacktestingEngine: Loading Data Feeds ---")
        if not settings: raise ValueError("Settings cannot be loaded.")
        data_feeds = await cls._load_data_feeds_async()
        return cls(portfolio_manager_class, data_feeds, initial_capital, commission_bps, slippage_bps)

    @staticmethod
    async def _load_data_feeds_async() -> Dict[str, pd.DataFrame]:
        data_feeds = {}
        # 1. Load Sentiment Data (if configured)
        sentiment_df: Optional[pd.DataFrame] = None
        if settings.simulated_sentiment_data_path and settings.enable_sentiment_analysis:
            sentiment_csv_path = settings.simulated_sentiment_data_path
            logger.info(f"Attempting to load simulated sentiment data from: {sentiment_csv_path}")
            if os.path.exists(sentiment_csv_path):
                try:
                    sentiment_df = pd.read_csv(sentiment_csv_path, parse_dates=['timestamp'], index_col='timestamp')
                    if sentiment_df.index.tz is None:
                        sentiment_df.index = sentiment_df.index.tz_localize('UTC')
                    else:
                        sentiment_df.index = sentiment_df.index.tz_convert('UTC')
                    if 'sentiment_score' not in sentiment_df.columns:
                        sentiment_df = None
                    else:
                        logger.info(f"Successfully loaded {len(sentiment_df)} simulated sentiment entries.")
                except Exception as e:
                    sentiment_df = None
                    logger.error(f"Error loading simulated sentiment data from {sentiment_csv_path}: {e}", exc_info=True)
            else:
                logger.warning(f"Simulated sentiment data file NOT FOUND at: {sentiment_csv_path}. Proceeding without sentiment.")

        # 2. Load Market Data for all assets in the universe
        portfolio_config = settings.get_strategy_params('Portfolio')
        trading_universe: List[str] = [s.strip() for s in portfolio_config.get('tradinguniverse', '').split(',')]
        timeframe = settings.default_timeframe
        
        if not trading_universe or trading_universe == ['']:
            return {}

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
                
                # BUG FIX: More robust sentiment alignment. This ensures that even if sentiment data
                # is out of range, the column is not filled with 0.0, allowing strategies to bypass the filter.
                if sentiment_df is not None and not sentiment_df.empty:
                    sentiment_df_unique = sentiment_df[~sentiment_df.index.duplicated(keep='last')] if not sentiment_df.index.is_unique else sentiment_df
                    aligned_sentiment = sentiment_df_unique.reindex(df.index, method='ffill').bfill()
                    
                    # Only add the column if alignment resulted in some valid scores
                    if not aligned_sentiment['sentiment_score'].isnull().all():
                        df['sentiment_score'] = aligned_sentiment['sentiment_score']

                # --- Phase 7: Add Market Regime ---
                if settings.config.getboolean('Portfolio', 'EnableRegimeFilter', fallback=False):
                    logger.info(f"Regime filter enabled. Calculating regimes for {asset}...")
                    from kamikaze_komodo.ml_models.regime_detection.kmeans_regime_model import KMeansRegimeModel
                    
                    model_params = settings.get_strategy_params("KMeans_Regime_Model")
                    _model_base_path = model_params.get('modelsavepath', 'ml_models/trained_models')
                    _model_filename = model_params.get('modelfilename', f"kmeans_regime_{asset.replace('/', '_').lower()}_{timeframe}.joblib")
                    
                    if not os.path.isabs(_model_base_path):
                        model_save_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
                    else:
                        model_save_path_dir = _model_base_path
                    model_full_path = os.path.join(model_save_path_dir, _model_filename)

                    if os.path.exists(model_full_path):
                        regime_model = KMeansRegimeModel(model_path=model_full_path, params=model_params)
                        regime_series = regime_model.predict_series(df)
                        if regime_series is not None:
                            df['market_regime'] = regime_series
                            logger.info(f"Market regime column added for {asset}. Found regimes: {df['market_regime'].dropna().unique().tolist()}")
                    
                if 'market_regime' in df.columns:
                    df['market_regime'] = df['market_regime'].astype(object).where(df['market_regime'].notna(), None)
                
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

        is_closing_trade = self.open_positions[symbol] and side != self.open_positions[symbol][0].side

        if is_closing_trade:
            closing_amount_left = amount
            while closing_amount_left > 1e-9 and self.open_positions[symbol]:
                open_trade = self.open_positions[symbol][0]
                close_qty = min(closing_amount_left, open_trade.amount)

                if open_trade.side == OrderSide.BUY:
                    pnl = (execution_price - open_trade.entry_price) * close_qty
                else:
                    pnl = (open_trade.entry_price - execution_price) * close_qty

                prorated_commission = commission * (close_qty / amount) if amount > 0 else commission
                self.current_cash += pnl - prorated_commission

                completed_trade = open_trade.copy(deep=True)
                completed_trade.amount = close_qty
                completed_trade.exit_price = execution_price
                completed_trade.exit_timestamp = current_bar.timestamp
                completed_trade.exit_order_id = order_id
                completed_trade.pnl = pnl - completed_trade.commission - prorated_commission
                completed_trade.commission += prorated_commission
                completed_trade.result = TradeResult.WIN if completed_trade.pnl > 0 else (TradeResult.LOSS if completed_trade.pnl < 0 else TradeResult.BREAKEVEN)
                completed_trade.notes = "Closed by Stop-Loss" if from_stop else "Closed by Strategy Signal"
                self.completed_trades_log.append(completed_trade)

                if close_qty < open_trade.amount:
                    open_trade.amount -= close_qty
                else:
                    self.open_positions[symbol].popleft()
                closing_amount_left -= close_qty
        else:
            self.current_cash -= commission
            new_trade = Trade(id=order_id, symbol=symbol, side=side, amount=amount, entry_price=execution_price,
                              entry_timestamp=current_bar.timestamp, entry_order_id=order_id, commission=commission,
                              custom_fields={"entry_bar_index": self.current_bar_index, "atr_at_entry": current_bar.atr})
            self.open_positions[symbol].append(new_trade)

        return Order(id=order_id, symbol=symbol, type=order_type, side=side, amount=amount, price=execution_price, timestamp=current_bar.timestamp, status='filled')

    def _update_portfolio_history(self, timestamp: datetime):
        unrealized_pnl = 0.0
        for symbol, position_queue in self.open_positions.items():
            if not position_queue: continue
            if symbol in self.current_bars:
                current_price = self.current_bars[symbol].close
                for trade in position_queue:
                    if trade.side == OrderSide.BUY:
                        unrealized_pnl += (current_price - trade.entry_price) * trade.amount
                    else:
                        unrealized_pnl += (trade.entry_price - current_price) * trade.amount
        
        self.current_portfolio_value = self.current_cash + unrealized_pnl

        self.portfolio_history.append({"timestamp": timestamp, "cash": self.current_cash, "unrealized_pnl": unrealized_pnl, "total_value": self.current_portfolio_value})

        positions_summary = {s: (sum(t.amount for t in q) if q[0].side == OrderSide.BUY else -sum(t.amount for t in q)) if q else 0.0 for s, q in self.open_positions.items()}
        self.portfolio_manager.portfolio_snapshot = PortfolioSnapshot(timestamp=timestamp, total_value_usd=self.current_portfolio_value, cash_balance_usd=self.current_cash, positions=positions_summary)
    
    def _check_and_execute_stops(self, timestamp: datetime):
        if not self.stop_manager: return
        
        positions_to_close = []
        for symbol, position_queue in self.open_positions.items():
            if not position_queue or symbol not in self.current_bars:
                continue
                
            latest_bar = self.current_bars[symbol]
            for trade in list(position_queue):
                stop_price = self.stop_manager.check_stop_loss(trade, latest_bar, self.current_bar_index)
                if stop_price:
                    positions_to_close.append({'symbol': symbol, 'side': OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY, 'amount': trade.amount, 'price': stop_price})
                    continue 
                    
                tp_price = self.stop_manager.check_take_profit(trade, latest_bar)
                if tp_price:
                    positions_to_close.append({'symbol': symbol, 'side': OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY, 'amount': trade.amount, 'price': tp_price})
        
        for closing_order in positions_to_close:
            logger.info(f"STOP TRIGGERED: Executing closing order for {closing_order['symbol']}.")
            self._execute_order(symbol=closing_order['symbol'], side=closing_order['side'], amount=closing_order['amount'], order_type=OrderType.MARKET, price=closing_order['price'], from_stop=True)


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
                    if not has_new_data_for_cycle: has_new_data_for_cycle = True
                    historical_data_slice[asset] = df.loc[:timestamp]
                    
                    bar_data_dict = df.loc[timestamp].to_dict()
                    
                    for key, value in bar_data_dict.items():
                        if isinstance(value, float) and np.isnan(value):
                            bar_data_dict[key] = None
                            
                    self.current_bars[asset] = BarData(timestamp=timestamp, **bar_data_dict)
            
            self._update_portfolio_history(timestamp)
            self._check_and_execute_stops(timestamp)

            if has_new_data_for_cycle:
                await self.portfolio_manager.run_cycle(historical_data_for_cycle=historical_data_slice)

        logger.info("Portfolio backtest run completed.")
        final_portfolio_state = {"initial_capital": self.initial_capital, "final_portfolio_value": self.current_portfolio_value}
        equity_curve_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        
        return self.completed_trades_log, final_portfolio_state, equity_curve_df