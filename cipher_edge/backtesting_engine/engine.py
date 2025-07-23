import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Type, TYPE_CHECKING

from tqdm import tqdm
from cipher_edge.core.models import BarData, Trade, PortfolioSnapshot
from cipher_edge.core.enums import SignalType, OrderType, OrderSide, TradeResult
from cipher_edge.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from cipher_edge.app_logger import get_logger
from datetime import datetime, timezone

from cipher_edge.risk_control_module.position_sizer import BasePositionSizer, POSITION_SIZER_REGISTRY
from cipher_edge.risk_control_module.stop_manager import BaseStopManager
from cipher_edge.config.settings import settings

if TYPE_CHECKING:
    from cipher_edge.portfolio_constructor.portfolio_manager import PortfolioManager

from cipher_edge.risk_control_module.position_sizer import FixedFractionalPositionSizer, ATRBasedPositionSizer
from cipher_edge.risk_control_module.optimal_f_position_sizer import OptimalFPositionSizer
from cipher_edge.risk_control_module.ml_confidence_position_sizer import MLConfidencePositionSizer
from cipher_edge.risk_control_module.stop_manager import PercentageStopManager, ATRStopManager
from cipher_edge.risk_control_module.parabolic_sar_stop import ParabolicSARStop
from cipher_edge.risk_control_module.triple_barrier_stop import TripleBarrierStop
from cipher_edge.strategy_framework.strategy_manager import StrategyManager


logger = get_logger(__name__)

STOP_MANAGER_REGISTRY: Dict[str, Type[BaseStopManager]] = {
    'PercentageStopManager': PercentageStopManager,
    'ATRStopManager': ATRStopManager,
    'ParabolicSARStop': ParabolicSARStop,
    'TripleBarrierStopManager': TripleBarrierStop,
}


class BacktestingEngine:
    """
    **REFACTOR**: Can now run in single-strategy mode or portfolio mode.
    """
    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        data_feed_df: Optional[pd.DataFrame] = None,
        strategy_name: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        position_sizer_name: Optional[str] = None,
        stop_manager_name: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        portfolio_manager: Optional['PortfolioManager'] = None,
        portfolio_data_feeds: Optional[Dict[str, pd.DataFrame]] = None
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_bps = slippage_bps
        self.portfolio_manager = portfolio_manager

        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_log: List[Trade] = []
        self.trade_id_counter = 0

        if self.portfolio_manager:
            self.mode = "portfolio"
            if not portfolio_data_feeds:
                raise ValueError("portfolio_data_feeds must be provided for portfolio mode.")
            self.portfolio_data_feeds = {sym: df.to_dict('index') for sym, df in portfolio_data_feeds.items()}
            self.all_timestamps = sorted(list(set(ts for feed in self.portfolio_data_feeds.values() for ts in feed.keys())))
            self.current_cash = self.portfolio_manager.cash
            self.active_trades: Dict[str, Trade] = {}
        else:
            self.mode = "single"
            if data_feed_df is None or strategy_name is None:
                raise ValueError("data_feed_df and strategy_name are required for single strategy mode.")
            
            self.strategy = StrategyManager.create_strategy(strategy_name, symbol, timeframe, strategy_params)
            self.position_sizer = POSITION_SIZER_REGISTRY[position_sizer_name](params=settings.get_strategy_params('RiskManagement'))
            self.stop_manager = STOP_MANAGER_REGISTRY[stop_manager_name](params=settings.get_strategy_params('RiskManagement'))
            
            self.prepared_data_feed = self.strategy.prepare_data(data_feed_df.copy())
            self.bar_data_list = self._pre_convert_to_bardata(self.prepared_data_feed)
            self.current_cash = initial_capital
            self.active_trades: Dict[str, Trade] = {}


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

    def _execute_trade_command(self, command: Union[SignalCommand, Dict], current_bar_or_prices: Union[BarData, Dict]):
        """Executes a trade command for either single or portfolio mode."""
        if self.mode == 'single':
            self._execute_single_trade(command, current_bar_or_prices)
        else: 
            self._execute_portfolio_trade(command, current_bar_or_prices)

    def _execute_portfolio_trade(self, order_params: Dict, current_prices: Dict[str, float]):
        """Executes a net order generated by the PortfolioManager."""
        symbol = order_params['symbol']
        side_str = order_params['side']
        amount = order_params['amount']
        side = OrderSide.BUY if side_str == 'buy' else OrderSide.SELL
        
        execution_price = self._apply_slippage(current_prices[symbol], side)
        trade_value = amount * execution_price
        commission = trade_value * self.commission_rate
        
        fill_info = {
            "symbol": symbol, "side": side_str, "amount": amount,
            "price": execution_price, "commission": commission
        }
        self.portfolio_manager.update_fill(fill_info)
        self.current_cash = self.portfolio_manager.cash # Sync cash state
        
        logger.info(f"PORTFOLIO EXECUTION: {side_str.upper()} {amount:.6f} {symbol} @ {execution_price:.4f}")

    def _execute_single_trade(self, command: SignalCommand, current_bar: BarData):
        pass

    def run(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
        if self.mode == "portfolio":
            return self._run_portfolio_backtest()
        else:
            return self._run_single_strategy_backtest()

    def _run_portfolio_backtest(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
        """Runs the backtest for the entire portfolio of strategies."""
        if not self.all_timestamps:
            return [], {"initial_capital": self.initial_capital, "final_portfolio_value": self.initial_capital}, pd.DataFrame()

        initial_timestamp = self.all_timestamps[0] - pd.Timedelta(seconds=1)
        self.portfolio_history.append({"timestamp": initial_timestamp, "total_value_usd": self.initial_capital})

        for ts in tqdm(self.all_timestamps, desc="Running Portfolio Backtest"):
            # 1. Get current market data for this timestamp
            current_market_data: Dict[str, BarData] = {}
            for symbol, feed in self.portfolio_data_feeds.items():
                if ts in feed:
                    bar_dict = feed[ts]
                    current_market_data[symbol] = BarData(timestamp=ts.to_pydatetime().replace(tzinfo=timezone.utc), **bar_dict)

            if not current_market_data:
                continue

            # 2. Update portfolio value and pass data to manager
            current_prices = {symbol: bar.close for symbol, bar in current_market_data.items()}
            self.portfolio_manager.update_portfolio_value(current_prices)
            self.portfolio_history.append({"timestamp": ts, "total_value_usd": self.portfolio_manager.portfolio_value})
            
            orders = self.portfolio_manager.on_bar(current_market_data)

            # 3. Execute generated orders
            for order in orders:
                self._execute_portfolio_trade(order, current_prices)

        final_portfolio_state = {"initial_capital": self.initial_capital, "final_portfolio_value": self.portfolio_history[-1]['total_value_usd']}
        equity_curve_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        
        # In portfolio mode, trades_log isn't populated in the same way, as we track net positions.
        # Returning an empty list for now. This can be expanded to log rebalancing trades if needed.
        return [], final_portfolio_state, equity_curve_df

    def _run_single_strategy_backtest(self):
        if not self.bar_data_list:
            return [], {"initial_capital": self.initial_capital, "final_portfolio_value": self.initial_capital}, pd.DataFrame([{"timestamp": pd.Timestamp.now(tz='UTC'), "total_value_usd": self.initial_capital}]).set_index('timestamp')

        initial_timestamp = self.bar_data_list[0].timestamp - pd.Timedelta(seconds=1)
        self.portfolio_history.append({"timestamp": initial_timestamp, "total_value_usd": self.initial_capital})

        for bar_index, current_bar_data in enumerate(self.bar_data_list):
            mtm_value = self.current_cash
            for trade in self.active_trades.values():
                if trade.side == OrderSide.BUY:
                    mtm_value += trade.amount * current_bar_data.close
                else:  # SHORT
                    unrealized_pnl = (trade.entry_price - current_bar_data.close) * trade.amount
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
    def _handle_stop_take_profit(self, current_bar: BarData, bar_index: int):
        """
        Checks and executes stop-loss or take-profit orders for all active trades.
        """
        for symbol, trade in list(self.active_trades.items()):
            stop_loss_price = self.stop_manager.check_stop_loss(
                trade, 
                current_bar, 
                bar_index, 
                data_history_for_sar=self.prepared_data_feed.iloc[:bar_index+1]
            )
            if stop_loss_price is not None:
                self._close_active_trade(symbol, current_bar.timestamp, stop_loss_price, "StopLoss")
                continue 

            take_profit_price = self.stop_manager.check_take_profit(
                trade, 
                current_bar
            )
            if take_profit_price is not None:
                self._close_active_trade(symbol, current_bar.timestamp, take_profit_price, "TakeProfit")