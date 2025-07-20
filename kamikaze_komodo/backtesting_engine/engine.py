# FILE: kamikaze_komodo/backtesting_engine/engine.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from kamikaze_komodo.core.models import BarData, Trade, PortfolioSnapshot
from kamikaze_komodo.core.enums import SignalType, OrderType, OrderSide, TradeResult
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone

# Explicitly import all position sizers and stop managers
from kamikaze_komodo.risk_control_module.position_sizer import BasePositionSizer, FixedFractionalPositionSizer, ATRBasedPositionSizer, PairTradingPositionSizer
from kamikaze_komodo.risk_control_module.optimal_f_position_sizer import OptimalFPositionSizer
from kamikaze_komodo.risk_control_module.ml_confidence_position_sizer import MLConfidencePositionSizer

from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager, PercentageStopManager, ATRStopManager
from kamikaze_komodo.risk_control_module.parabolic_sar_stop import ParabolicSARStop
from kamikaze_komodo.risk_control_module.triple_barrier_stop import TripleBarrierStop, StopTriggerType

from kamikaze_komodo.risk_control_module.risk_manager import RiskManager
from kamikaze_komodo.portfolio_constructor.base_portfolio_constructor import BasePortfolioConstructor
from kamikaze_komodo.portfolio_constructor.asset_allocator import FixedWeightAssetAllocator, OptimalFAllocator, HRPAllocator, BaseAssetAllocator
from kamikaze_komodo.config.settings import settings

logger = get_logger(__name__)

class BacktestingEngine:
    def __init__(
        self,
        data_feed_df: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        position_sizer_type: Optional[str] = None,
        stop_manager_type: Optional[str] = None,
        portfolio_constructor_type: Optional[str] = None,
        asset_allocator_type: Optional[str] = None,
        slippage_bps: float = 0.0,
        slippage_model_type: str = 'fixed'
    ):
        if data_feed_df.empty:
            raise ValueError("Data feed DataFrame cannot be empty.")
        if not isinstance(data_feed_df.index, pd.DatetimeIndex):
            raise ValueError("Data feed DataFrame must be indexed by pd.DatetimeIndex.")

        self.strategy = strategy
        
        logger.info(f"Preparing data for strategy '{self.strategy.name}'...")
        self.prepared_data_feed = self.strategy.prepare_data(data_feed_df.copy())
        
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_bps = slippage_bps
        self.slippage_model_type = slippage_model_type

        self.risk_manager = RiskManager(settings)
        self.position_sizer = self._initialize_position_sizer(position_sizer_type)
        self.stop_manager = self._initialize_stop_manager(stop_manager_type)
        self.portfolio_constructor = self._initialize_portfolio_constructor(portfolio_constructor_type, asset_allocator_type)
        
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_log: List[Trade] = []
        self.current_cash = initial_capital
        self.active_trades: Dict[str, Trade] = {}
        self.trade_id_counter = 0

        logger.info(
            f"BacktestingEngine initialized for strategy '{self.strategy.name}'. "
            f"Slippage model: {self.slippage_model_type}. "
            f"Position Sizer: {self.position_sizer.__class__.__name__}. "
            f"Stop Manager: {self.stop_manager.__class__.__name__}."
        )

    def _initialize_position_sizer(self, sizer_type: Optional[str]) -> BasePositionSizer:
        params = settings.get_strategy_params('RiskManagement')
        if sizer_type is None:
            sizer_type = settings.position_sizer_type

        if sizer_type == 'FixedFractional':
            return FixedFractionalPositionSizer(params=params)
        elif sizer_type == 'ATRBased':
            return ATRBasedPositionSizer(params=params)
        elif sizer_type == 'OptimalF':
            return OptimalFPositionSizer(params=params)
        elif sizer_type == 'MLConfidence':
            return MLConfidencePositionSizer(params=params)
        else:
            logger.error(f"Unknown position sizer type: {sizer_type}. Defaulting to FixedFractional.")
            return FixedFractionalPositionSizer(params=params)

    def _initialize_stop_manager(self, stop_type: Optional[str]) -> BaseStopManager:
        params = settings.get_strategy_params('RiskManagement')
        if stop_type is None:
            stop_type = settings.stop_manager_type

        if stop_type == 'PercentageBased':
            return PercentageStopManager(params=params)
        elif stop_type == 'ATRBased':
            return ATRStopManager(params=params)
        elif stop_type == 'ParabolicSAR':
            return ParabolicSARStop(params=params)
        elif stop_type == 'TripleBarrier':
            return TripleBarrierStop(params=params)
        else:
            logger.error(f"Unknown stop manager type: {stop_type}. Defaulting to PercentageBased.")
            return PercentageStopManager(params=params)

    def _initialize_portfolio_constructor(self, constructor_type: Optional[str], allocator_type: Optional[str]) -> BasePortfolioConstructor:
        if allocator_type is None:
            allocator_type = settings.asset_allocator_type
        
        asset_allocator_instance: Optional[BaseAssetAllocator] = None
        portfolio_constructor_params = settings.get_strategy_params('PortfolioConstructor')

        if allocator_type == 'FixedWeight':
            default_symbol = settings.default_symbol
            clean_symbol_key = f'defaultallocation_{default_symbol.replace("/", "").replace(":", "").lower()}'
            default_allocation = portfolio_constructor_params.get(clean_symbol_key, 1.0)
            target_weights = {default_symbol: float(default_allocation)}
            asset_allocator_instance = FixedWeightAssetAllocator(target_weights=target_weights, params=portfolio_constructor_params)
        elif allocator_type == 'OptimalF':
            asset_allocator_instance = OptimalFAllocator(params=portfolio_constructor_params)
        elif allocator_type == 'HRP':
            asset_allocator_instance = HRPAllocator(params=portfolio_constructor_params)
        else:
            logger.error(f"Unknown asset allocator type: {allocator_type}. Defaulting to FixedWeight.")
            target_weights = {settings.default_symbol: 1.0}
            asset_allocator_instance = FixedWeightAssetAllocator(target_weights=target_weights, params=portfolio_constructor_params)
        
        class ConcretePortfolioConstructor(BasePortfolioConstructor):
            def __init__(self, settings: Any, risk_manager: RiskManager, asset_allocator: BaseAssetAllocator):
                super().__init__(settings, risk_manager)
                self.asset_allocator = asset_allocator

            def calculate_target_allocations(self, current_portfolio: PortfolioSnapshot, market_data: pd.DataFrame, trades_log: pd.DataFrame) -> Dict[str, float]:
                assets_to_allocate = [settings.default_symbol]
                historical_data_for_allocator = {settings.default_symbol: market_data}
                
                capital_allocations = self.asset_allocator.allocate(
                    assets=assets_to_allocate,
                    portfolio_value=current_portfolio.total_value_usd,
                    historical_data=historical_data_for_allocator,
                    trade_history=trades_log
                )
                
                if current_portfolio.total_value_usd <= 0: return {s: 0.0 for s in assets_to_allocate}
                
                return {asset: capital / current_portfolio.total_value_usd for asset, capital in capital_allocations.items()}

        return ConcretePortfolioConstructor(settings, self.risk_manager, asset_allocator_instance)

    def _get_next_trade_id(self) -> str:
        self.trade_id_counter += 1
        return f"trade_{self.trade_id_counter:05d}"

    def _apply_slippage(self, price: float, side: OrderSide, amount: float, latest_bar: BarData) -> float:
        if self.slippage_model_type == 'fixed':
            slippage_rate = self.slippage_bps / 10000.0
            filled_price = price * (1 + slippage_rate) if side == OrderSide.BUY else price * (1 - slippage_rate)
        elif self.slippage_model_type == 'volume_volatility_based':
            atr = getattr(latest_bar, 'atr', 0.0)
            volume = getattr(latest_bar, 'volume', 1.0)
            if price <= 0 or volume <= 0 or atr is None or atr <= 0:
                slippage_rate = self.slippage_bps / 10000.0
                return price * (1 + slippage_rate) if side == OrderSide.BUY else price * (1 - slippage_rate)

            base_slippage_pct = settings.BASE_SLIPPAGE_BPS / 10000.0
            avg_daily_volume_proxy = volume * settings.AVERAGE_DAILY_VOLUME_FACTOR
            impact_component = (amount / (avg_daily_volume_proxy + 1e-9)) 
            volatility_component = (atr / price)
            dynamic_slippage_pct = impact_component * volatility_component * settings.VOLATILITY_SLIPPAGE_FACTOR
            
            total_slippage_pct = min(base_slippage_pct + dynamic_slippage_pct, 0.1)

            filled_price = price * (1 + total_slippage_pct) if side == OrderSide.BUY else price * (1 - total_slippage_pct)
        else:
            raise NotImplementedError(f"Slippage model type '{self.slippage_model_type}' not implemented.")
        
        return filled_price if filled_price > 0 else price * 0.0001

    def _execute_trade_command(self, command: SignalCommand, current_bar: BarData, bar_index: int):
        trade_symbol = command.symbol
        execution_price_ideal = command.price if command.price else current_bar.close
        active_trade = self.active_trades.get(trade_symbol)
        
        strategy_info = {}
        if hasattr(self.strategy, 'get_latest_prediction_info') and callable(self.strategy.get_latest_prediction_info):
            strategy_info = self.strategy.get_latest_prediction_info() or {}

        if command.signal_type in [SignalType.LONG, SignalType.SHORT] and active_trade is None:
            side = OrderSide.BUY if command.signal_type == SignalType.LONG else OrderSide.SELL
            position_size = self.position_sizer.calculate_size(
                symbol=trade_symbol,
                current_price=execution_price_ideal,
                available_capital=self.current_cash,
                current_portfolio_value=self.portfolio_history[-1]['total_value_usd'],
                trade_signal=command.signal_type,
                strategy_info=strategy_info,
                latest_bar=current_bar,
                atr_value=getattr(current_bar, 'atr', None)
            )
            # FIX: Robustly check for invalid size (None, NaN, or zero)
            if position_size is None or not np.isfinite(position_size) or position_size <= 1e-8:
                return

            execution_price = self._apply_slippage(execution_price_ideal, side, position_size, current_bar)
            cost = position_size * execution_price
            commission = cost * self.commission_rate
            
            if side == OrderSide.BUY and cost + commission > self.current_cash:
                position_size = self.current_cash / (execution_price * (1 + self.commission_rate))
                cost = position_size * execution_price
                commission = cost * self.commission_rate

            self.current_cash -= commission
            if side == OrderSide.BUY: self.current_cash -= cost
            else: self.current_cash += cost

            new_trade = Trade(
                id=self._get_next_trade_id(), symbol=trade_symbol, entry_order_id=f"entry_{self.trade_id_counter}",
                side=side, entry_price=execution_price, amount=position_size, entry_timestamp=current_bar.timestamp,
                commission=commission, custom_fields={"atr_at_entry": getattr(current_bar, 'atr', None)}
            )
            self.active_trades[trade_symbol] = new_trade
            
            if isinstance(self.stop_manager, TripleBarrierStop):
                self.stop_manager.calculate_barriers(new_trade, current_bar)

        elif command.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT] and active_trade:
            if (active_trade.side == OrderSide.BUY and command.signal_type == SignalType.CLOSE_LONG) or \
               (active_trade.side == OrderSide.SELL and command.signal_type == SignalType.CLOSE_SHORT):
                
                exit_side = OrderSide.SELL if active_trade.side == OrderSide.BUY else OrderSide.BUY
                execution_price = self._apply_slippage(execution_price_ideal, exit_side, active_trade.amount, current_bar)
                exit_value = active_trade.amount * execution_price
                commission = exit_value * self.commission_rate
                
                if active_trade.side == OrderSide.BUY:
                    pnl = (execution_price - active_trade.entry_price) * active_trade.amount
                    self.current_cash += exit_value
                else:
                    pnl = (active_trade.entry_price - execution_price) * active_trade.amount
                    self.current_cash -= exit_value
                
                pnl -= (active_trade.commission + commission)
                self.current_cash -= commission
                
                self._log_and_clear_active_trade(trade_symbol, current_bar.timestamp, execution_price, pnl, commission, "SignalClose")

    def _log_and_clear_active_trade(self, symbol: str, timestamp: datetime, exit_price: float, pnl: float, exit_commission: float, reason: str):
        trade = self.active_trades.pop(symbol, None)
        if not trade: return
        trade.exit_price = exit_price
        trade.exit_timestamp = timestamp
        trade.pnl = pnl
        trade.commission += exit_commission
        trade.result = TradeResult.WIN if pnl > 0 else (TradeResult.LOSS if pnl < 0 else TradeResult.BREAKEVEN)
        trade.notes = reason
        self.trades_log.append(trade)
        if isinstance(self.stop_manager, TripleBarrierStop):
            self.stop_manager.reset_for_trade(trade.id)

    def _handle_stop_take_profit(self, current_bar: BarData, bar_index: int):
        for symbol, trade in list(self.active_trades.items()):
            stop_loss_price = self.stop_manager.check_stop_loss(trade, current_bar, bar_index, data_history_for_sar=self.prepared_data_feed.iloc[:bar_index+1])
            take_profit_price = self.stop_manager.check_take_profit(trade, current_bar)
            
            triggered_price = None
            trigger_reason = ""
            if stop_loss_price is not None:
                triggered_price = stop_loss_price
                trigger_reason = "StopLoss"
            elif take_profit_price is not None:
                triggered_price = take_profit_price
                trigger_reason = "TakeProfit"

            if triggered_price is not None:
                exit_value = trade.amount * triggered_price
                commission = exit_value * self.commission_rate
                if trade.side == OrderSide.BUY:
                    pnl = (triggered_price - trade.entry_price) * trade.amount
                    self.current_cash += exit_value
                else:
                    pnl = (trade.entry_price - triggered_price) * trade.amount
                    self.current_cash -= exit_value
                pnl -= (trade.commission + commission)
                self.current_cash -= commission
                self._log_and_clear_active_trade(symbol, current_bar.timestamp, triggered_price, pnl, commission, trigger_reason)

    def run(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
        logger.info(f"Starting backtest run for strategy '{self.strategy.name}'...")
        self.portfolio_history = [{"timestamp": self.prepared_data_feed.index[0] - pd.Timedelta(seconds=1), "total_value_usd": self.initial_capital}]

        for bar_index, (timestamp, row) in enumerate(self.prepared_data_feed.iterrows()):
            row_dict = row.to_dict()
            if 'market_regime' in row_dict and pd.isna(row_dict['market_regime']):
                row_dict['market_regime'] = None
            current_bar_data = BarData(timestamp=timestamp.to_pydatetime().replace(tzinfo=timezone.utc), **row_dict)
            
            mv_longs = sum(trade.amount * current_bar_data.close for trade in self.active_trades.values() if trade.side == OrderSide.BUY)
            mv_shorts = sum(trade.amount * current_bar_data.close for trade in self.active_trades.values() if trade.side == OrderSide.SELL)
            mtm_value = self.current_cash + mv_longs - mv_shorts

            self.portfolio_history.append({"timestamp": timestamp, "total_value_usd": mtm_value})
            equity_curve_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
            
            self.risk_manager.update_portfolio_metrics(equity_curve_df, timestamp)
            if self.risk_manager.check_portfolio_drawdown():
                for sym_to_close, trade_to_close in list(self.active_trades.items()):
                    close_signal = SignalType.CLOSE_LONG if trade_to_close.side == OrderSide.BUY else SignalType.CLOSE_SHORT
                    self._execute_trade_command(SignalCommand(signal_type=close_signal, symbol=sym_to_close), current_bar_data, bar_index)
                continue

            self._handle_stop_take_profit(current_bar_data, bar_index)
            
            if self.risk_manager.is_trading_halted():
                continue

            strategy_output = self.strategy.on_bar_data(current_bar_data)
            if strategy_output:
                commands = strategy_output if isinstance(strategy_output, list) else [SignalCommand(signal_type=strategy_output, symbol=self.strategy.symbol)]
                for command in commands:
                    if command.signal_type != SignalType.HOLD:
                        self._execute_trade_command(command, current_bar_data, bar_index)
                        
        final_portfolio_state = {"initial_capital": self.initial_capital, "final_portfolio_value": self.portfolio_history[-1]['total_value_usd']}
        equity_curve_df_final = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        return self.trades_log, final_portfolio_state, equity_curve_df_final