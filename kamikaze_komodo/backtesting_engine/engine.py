# FILE: kamikaze_komodo/backtesting_engine/engine.py
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from kamikaze_komodo.core.models import BarData, Trade, PortfolioSnapshot
from kamikaze_komodo.core.enums import SignalType, OrderType, OrderSide, TradeResult
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone

from kamikaze_komodo.risk_control_module.position_sizer import BasePositionSizer
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager
from kamikaze_komodo.config.settings import settings

# Import all component classes for instantiation by name
from kamikaze_komodo.risk_control_module.position_sizer import FixedFractionalPositionSizer, ATRBasedPositionSizer
from kamikaze_komodo.risk_control_module.optimal_f_position_sizer import OptimalFPositionSizer
from kamikaze_komodo.risk_control_module.ml_confidence_position_sizer import MLConfidencePositionSizer
from kamikaze_komodo.risk_control_module.stop_manager import PercentageStopManager, ATRStopManager
from kamikaze_komodo.risk_control_module.parabolic_sar_stop import ParabolicSARStop
from kamikaze_komodo.risk_control_module.triple_barrier_stop import TripleBarrierStop
from kamikaze_komodo.strategy_framework.strategy_manager import StrategyManager


logger = get_logger(__name__)

# Registry for component classes
POSITION_SIZER_REGISTRY: Dict[str, Type[BasePositionSizer]] = {
    'FixedFractionalPositionSizer': FixedFractionalPositionSizer,
    'ATRBasedPositionSizer': ATRBasedPositionSizer,
    'OptimalFPositionSizer': OptimalFPositionSizer,
    'MLConfidencePositionSizer': MLConfidencePositionSizer,
}

STOP_MANAGER_REGISTRY: Dict[str, Type[BaseStopManager]] = {
    'PercentageStopManager': PercentageStopManager,
    'ATRStopManager': ATRStopManager,
    'ParabolicSARStop': ParabolicSARStop,
    'TripleBarrierStopManager': TripleBarrierStop,
}


class BacktestingEngine:
    """
    **REFACTOR**: Simplified to run a single backtest trial. The multi-trial logic
    has been moved to the StrategyOptimizer for parallel execution.
    """
    def __init__(
        self,
        data_feed_df: pd.DataFrame,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        position_sizer_name: str,
        stop_manager_name: str,
        symbol: str,
        timeframe: str,
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
    ):
        if data_feed_df.empty:
            raise ValueError("Data feed DataFrame cannot be empty.")

        self.strategy = StrategyManager.create_strategy(strategy_name, symbol, timeframe, strategy_params)
        self.position_sizer = POSITION_SIZER_REGISTRY[position_sizer_name](params=settings.get_strategy_params('RiskManagement'))
        self.stop_manager = STOP_MANAGER_REGISTRY[stop_manager_name](params=settings.get_strategy_params('RiskManagement'))
         
        if not all([self.strategy, self.position_sizer, self.stop_manager]):
            raise ValueError("Failed to instantiate one or more components for the backtest.")

        self.prepared_data_feed = self.strategy.prepare_data(data_feed_df.copy())
        self.bar_data_list = self._pre_convert_to_bardata(self.prepared_data_feed)
         
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_bps = slippage_bps

        # Initialize state
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_log: List[Trade] = []
        self.current_cash = initial_capital
        self.active_trades: Dict[str, Trade] = {}
        self.trade_id_counter = 0

    def _pre_convert_to_bardata(self, df: pd.DataFrame) -> List[BarData]:
        """Converts the prepared DataFrame into a list of BarData objects upfront."""
        records = df.to_dict('records')
        timestamps = df.index
        bar_list = []
        for i, record in enumerate(records):
            if 'market_regime' in record and pd.isna(record.get('market_regime')):
                record['market_regime'] = None
            bar_list.append(BarData(timestamp=timestamps[i].to_pydatetime().replace(tzinfo=timezone.utc), **record))
        return bar_list

    def _get_next_trade_id(self) -> str:
        self.trade_id_counter += 1
        return f"trade_{self.trade_id_counter:04d}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        slippage_rate = self.slippage_bps / 10000.0
        return price * (1 + slippage_rate) if side == OrderSide.BUY else price * (1 - slippage_rate)

    def _execute_trade_command(self, command: SignalCommand, current_bar: BarData):
        trade_symbol = command.symbol
        execution_price_ideal = command.price if command.price else current_bar.close
        active_trade = self.active_trades.get(trade_symbol)

        if command.signal_type in [SignalType.LONG, SignalType.SHORT] and active_trade is None:
            side = OrderSide.BUY if command.signal_type == SignalType.LONG else OrderSide.SELL
            position_size = self.position_sizer.calculate_size(
                symbol=trade_symbol,
                current_price=execution_price_ideal,
                available_capital=self.current_cash,
                current_portfolio_value=self.portfolio_history[-1]['total_value_usd'],
                trade_signal=command.signal_type,
                strategy_info={},
                latest_bar=current_bar,
                atr_value=getattr(current_bar, 'atr', None)
            )
            if position_size is None or not np.isfinite(position_size) or position_size <= 1e-8: return

            execution_price = self._apply_slippage(execution_price_ideal, side)
            cost = position_size * execution_price
            commission = cost * self.commission_rate
             
            if side == OrderSide.BUY and cost + commission > self.current_cash: return

            self.current_cash -= commission
            if side == OrderSide.BUY: self.current_cash -= cost
            else: self.current_cash += cost

            new_trade = Trade(id=self._get_next_trade_id(), symbol=trade_symbol, entry_order_id=f"entry_{self.trade_id_counter}",
                              side=side, entry_price=execution_price, amount=position_size, entry_timestamp=current_bar.timestamp,
                              commission=commission, custom_fields={"atr_at_entry": getattr(current_bar, 'atr', None)})
            self.active_trades[trade_symbol] = new_trade
            if isinstance(self.stop_manager, TripleBarrierStop):
                self.stop_manager.calculate_barriers(new_trade, current_bar)

        elif command.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT] and active_trade:
            if (active_trade.side == OrderSide.BUY and command.signal_type == SignalType.CLOSE_LONG) or \
               (active_trade.side == OrderSide.SELL and command.signal_type == SignalType.CLOSE_SHORT):
                exit_side = OrderSide.SELL if active_trade.side == OrderSide.BUY else OrderSide.BUY
                execution_price = self._apply_slippage(execution_price_ideal, exit_side)
                self._close_active_trade(trade_symbol, current_bar.timestamp, execution_price, "SignalClose")

    def _close_active_trade(self, symbol: str, timestamp: datetime, exit_price: float, reason: str):
        trade = self.active_trades.pop(symbol, None)
        if not trade: return

        exit_value = trade.amount * exit_price
        exit_commission = exit_value * self.commission_rate
         
        entry_cost = trade.amount * trade.entry_price
        
        if trade.side == OrderSide.BUY:
            pnl = exit_value - entry_cost
            self.current_cash += exit_value
        else: # SELL
            pnl = entry_cost - exit_value
            self.current_cash -= exit_value
         
        pnl -= (trade.commission + exit_commission)
        self.current_cash -= exit_commission

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
             
            if stop_loss_price is not None:
                self._close_active_trade(symbol, current_bar.timestamp, stop_loss_price, "StopLoss")
            elif take_profit_price is not None:
                self._close_active_trade(symbol, current_bar.timestamp, take_profit_price, "TakeProfit")
     
    def run(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
        if not self.bar_data_list:
            return [], {"initial_capital": self.initial_capital, "final_portfolio_value": self.initial_capital}, pd.DataFrame([{"timestamp": pd.Timestamp.now(tz='UTC'), "total_value_usd": self.initial_capital}]).set_index('timestamp')

        initial_timestamp = self.bar_data_list[0].timestamp - pd.Timedelta(seconds=1)
        self.portfolio_history.append({"timestamp": initial_timestamp, "total_value_usd": self.initial_capital})

        for bar_index, current_bar_data in enumerate(self.bar_data_list):
            # **FIX**: Correct Mark-to-Market calculation for portfolio equity
            mtm_value = self.current_cash
            for trade in self.active_trades.values():
                if trade.side == OrderSide.BUY:
                    mtm_value += trade.amount * current_bar_data.close
                else:  # SHORT
                    unrealized_pnl = (trade.entry_price - current_bar_data.close) * trade.amount
                    # For shorts, equity is the cash received minus the current liability to buy back
                    # The cash from the sale is already in self.current_cash, so we add the unrealized PnL
                    mtm_value += unrealized_pnl
            
            self.portfolio_history.append({"timestamp": current_bar_data.timestamp, "total_value_usd": mtm_value})

            self._handle_stop_take_profit(current_bar_data, bar_index)
            strategy_output = self.strategy.on_bar_data(current_bar_data)
             
            if strategy_output and strategy_output != SignalType.HOLD:
                commands = strategy_output if isinstance(strategy_output, list) else [SignalCommand(signal_type=strategy_output, symbol=self.strategy.symbol)]
                for command in commands:
                    self._execute_trade_command(command, current_bar_data)

        final_portfolio_state = {"initial_capital": self.initial_capital, "final_portfolio_value": self.portfolio_history[-1]['total_value_usd']}
        equity_curve_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        return self.trades_log, final_portfolio_state, equity_curve_df