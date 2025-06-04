# kamikaze_komodo/backtesting_engine/engine.py
import pandas as pd
from typing import List, Dict, Any, Optional
from kamikaze_komodo.core.models import BarData, Trade, Order # Order might not be fully used in basic backtest
from kamikaze_komodo.core.enums import SignalType, OrderSide, TradeResult
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)

class BacktestingEngine:
    """
    A basic backtesting engine.
    Iterates through historical data, applies strategy logic, and simulates trades.
    """
    def __init__(
        self,
        data_feed_df: pd.DataFrame, # Expects DataFrame with OHLCV columns, indexed by timestamp
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0, # Commission in basis points (e.g., 10 bps = 0.1% = 0.001)
        stop_loss_pct: Optional[float] = None, # e.g., 0.02 for 2%
        take_profit_pct: Optional[float] = None # e.g., 0.05 for 5%
    ):
        if data_feed_df.empty:
            raise ValueError("Data feed DataFrame cannot be empty.")
        if not isinstance(data_feed_df.index, pd.DatetimeIndex):
            raise ValueError("Data feed DataFrame must be indexed by timestamp (pd.DatetimeIndex).")
        
        self.data_feed_df = data_feed_df.sort_index() # Ensure data is chronological
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0 # Convert bps to a rate (e.g., 10bps -> 0.001)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_log: List[Trade] = []
        
        self.current_capital = initial_capital
        self.current_position_size = 0.0 # In units of the asset
        self.current_position_side: Optional[OrderSide] = None
        self.entry_price: Optional[float] = None
        self.trade_id_counter = 0

        logger.info(
            f"BacktestingEngine initialized for strategy '{strategy.name}' on symbol '{strategy.symbol}'. "
            f"Initial Capital: ${initial_capital:,.2f}, Commission: {commission_bps} bps."
        )
        if stop_loss_pct: logger.info(f"Stop Loss: {stop_loss_pct*100:.2f}%")
        if take_profit_pct: logger.info(f"Take Profit: {take_profit_pct*100:.2f}%")


    def _get_next_trade_id(self) -> str:
        self.trade_id_counter += 1
        return f"trade_{self.trade_id_counter:05d}"

    def _execute_trade(
        self, 
        signal_type: SignalType, 
        timestamp: datetime, 
        price: float # Execution price (e.g., current bar's close or next bar's open)
    ):
        """Simulates executing a trade based on the signal."""
        commission_cost = 0.0

        # --- POSITION ENTRY ---
        if signal_type == SignalType.LONG and self.current_position_side is None:
            # Simple position sizing: use all available capital (for this basic engine)
            # A more advanced engine would use a PositionSizer module.
            if self.current_capital <= 0:
                logger.warning(f"{timestamp} - Cannot enter LONG trade for {self.strategy.symbol}. No capital available.")
                return

            asset_cost_without_commission = self.current_capital 
            self.current_position_size = asset_cost_without_commission / price 

            commission_cost = asset_cost_without_commission * self.commission_rate

            # Reduce cash by the total cost (assets + commission)
            self.current_capital -= asset_cost_without_commission 
            self.current_capital -= commission_cost

            self.current_position_side = OrderSide.BUY
            self.entry_price = price
            
            # Create a new trade record (still open)
            trade = Trade(
                id=self._get_next_trade_id(),
                symbol=self.strategy.symbol,
                entry_order_id=f"entry_{self.trade_id_counter}", # Simulated order ID
                side=OrderSide.BUY,
                entry_price=self.entry_price,
                amount=self.current_position_size,
                entry_timestamp=timestamp,
                commission=commission_cost # Initial commission for entry
            )
            self.trades_log.append(trade)
            logger.info(
                f"{timestamp} - EXECUTE LONG: {self.current_position_size:.4f} {self.strategy.symbol} "
                f"@ ${price:.2f}. Capital: ${self.current_capital:.2f}. Comm: ${commission_cost:.2f}."
            )

        # --- POSITION EXIT ---
        elif signal_type == SignalType.CLOSE_LONG and self.current_position_side == OrderSide.BUY:
            if self.current_position_size == 0 or self.entry_price is None:
                logger.warning(f"{timestamp} - Received CLOSE_LONG but no open BUY position or entry price for {self.strategy.symbol}.")
                return

            exit_value = self.current_position_size * price # Gross proceeds from sale
            commission_cost = exit_value * self.commission_rate

            # PnL calculation should be based on entry_price vs exit_price for the amount traded
            # PnL = (exit_price - self.entry_price) * self.current_position_size - commission_cost_entry - commission_cost_exit
            # The self.trades_log already stores entry commission.

            # Add gross proceeds to cash, then deduct exit commission
            self.current_capital += exit_value 
            self.current_capital -= commission_cost

            # Update the last trade log
            if self.trades_log:
                last_trade = self.trades_log[-1]
                if last_trade.exit_price is None:
                    # Calculate PnL for this specific trade
                    pnl_for_this_trade = (price - last_trade.entry_price) * last_trade.amount - last_trade.commission - commission_cost
                    pnl_percentage_for_this_trade = (pnl_for_this_trade / (last_trade.entry_price * last_trade.amount)) * 100 if (last_trade.entry_price * last_trade.amount) != 0 else 0

                    last_trade.exit_price = price
                    last_trade.exit_timestamp = timestamp
                    last_trade.pnl = pnl_for_this_trade # Corrected PnL
                    last_trade.pnl_percentage = pnl_percentage_for_this_trade # Corrected PnL %
                    last_trade.commission += commission_cost # Add exit commission to total commission for the trade
                    last_trade.result = TradeResult.WIN if pnl_for_this_trade > 0 else (TradeResult.LOSS if pnl_for_this_trade < 0 else TradeResult.BREAKEVEN)
                    last_trade.exit_order_id = f"exit_{last_trade.id.split('_')[1]}"

            logger.info(
                f"{timestamp} - EXECUTE CLOSE LONG: {self.current_position_size:.4f} {self.strategy.symbol} "
                f"@ ${price:.2f}. PnL: ${pnl_for_this_trade:.2f} ({pnl_percentage_for_this_trade*100:.2f}%). " # <-- Problem is here
                f"Capital: ${self.current_capital:.2f}. Comm: ${commission_cost:.2f}."
            )

            # Reset position
            self.current_position_size = 0.0
            self.current_position_side = None
            self.entry_price = None
        
        # Note: This basic engine doesn't handle SHORT signals or CLOSE_SHORT.
        elif signal_type == SignalType.SHORT or signal_type == SignalType.CLOSE_SHORT:
            logger.debug(f"{timestamp} - SHORT/CLOSE_SHORT signals are not handled by this basic backtesting engine.")


    def _check_stop_loss_take_profit(self, current_bar: BarData):
        """Checks and triggers SL/TP if conditions are met within the current bar's H/L prices."""
        if self.current_position_side == OrderSide.BUY and self.entry_price is not None:
            # Check Stop Loss
            if self.stop_loss_pct:
                stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
                if current_bar.low <= stop_loss_price:
                    logger.info(f"{current_bar.timestamp} - STOP LOSS triggered for {self.strategy.symbol} at ${stop_loss_price:.2f} (Low: {current_bar.low:.2f})")
                    self._execute_trade(SignalType.CLOSE_LONG, current_bar.timestamp, stop_loss_price)
                    return True # SL Triggered

            # Check Take Profit (only if not SL triggered)
            if self.take_profit_pct and self.current_position_side == OrderSide.BUY: # Check side again in case SL closed it
                take_profit_price = self.entry_price * (1 + self.take_profit_pct)
                if current_bar.high >= take_profit_price:
                    logger.info(f"{current_bar.timestamp} - TAKE PROFIT triggered for {self.strategy.symbol} at ${take_profit_price:.2f} (High: {current_bar.high:.2f})")
                    self._execute_trade(SignalType.CLOSE_LONG, current_bar.timestamp, take_profit_price)
                    return True # TP Triggered
        return False # No SL/TP triggered


    def run(self) -> tuple[List[Trade], Dict[str, Any]]:
        """
        Runs the backtest simulation.

        Returns:
            A tuple containing:
            - List[Trade]: The log of all executed trades.
            - Dict[str, Any]: The final portfolio state.
        """
        logger.info(f"Starting backtest run for strategy '{self.strategy.name}'...")

        # Option 1: Generate all signals first (less realistic for on_bar_data logic, good for vectorized)
        # signals = self.strategy.generate_signals(self.data_feed_df)

        # Option 2: Iterate bar by bar, calling strategy.on_bar_data (more realistic for event-driven)
        # We will use Option 2 as it aligns better with how on_bar_data is defined.
        # The strategy needs to maintain its own state.

        # Initialize strategy's internal data history if it needs it separately
        # For EWMAC, it builds its history internally via update_data_history.
        self.strategy.data_history = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']) # Reset history

        for timestamp, row in self.data_feed_df.iterrows():
            # Ensure timestamp is timezone-aware (UTC)
            ts_aware = timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')

            current_bar = BarData(
                timestamp=ts_aware,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                symbol=self.strategy.symbol,
                timeframe=self.strategy.timeframe
            )
            
            # 1. Check for Stop-Loss / Take-Profit first based on H/L prices of current bar
            # This assumes SL/TP can be triggered intra-bar.
            sl_tp_triggered = self._check_stop_loss_take_profit(current_bar)
            
            # 2. If SL/TP not triggered, get signal from strategy for the current bar's close
            # The strategy's on_bar_data uses its internal history which is updated within the method.
            if not sl_tp_triggered:
                signal = self.strategy.on_bar_data(current_bar) # Strategy updates its state and history
                
                if signal and signal != SignalType.HOLD:
                    # Assumption: Trades are executed at the close price of the bar where signal is generated.
                    # Or, for more realism, at the open of the *next* bar.
                    # For this basic engine, we'll use current bar's close.
                    execution_price = current_bar.close
                    self._execute_trade(signal, current_bar.timestamp, execution_price)

            # 3. Log portfolio state at the end of each bar
            current_portfolio_value = self.current_capital
            if self.current_position_side == OrderSide.BUY and self.entry_price is not None:
                 # Mark-to-market value of current open position
                current_portfolio_value += self.current_position_size * current_bar.close
            
            self.portfolio_history.append({
                "timestamp": current_bar.timestamp,
                "capital": self.current_capital, # Cash available
                "position_size": self.current_position_size if self.current_position_side else 0,
                "asset_value": (self.current_position_size * current_bar.close) if self.current_position_side else 0,
                "total_value": current_portfolio_value,
                "current_price": current_bar.close
            })

        # If there's an open position at the end of the backtest, close it at the last bar's close price
        if self.current_position_side == OrderSide.BUY:
            last_bar_timestamp = self.data_feed_df.index[-1]
            last_bar_close = self.data_feed_df['close'].iloc[-1]
            logger.info(f"{last_bar_timestamp} - End of backtest. Closing open LONG position for {self.strategy.symbol} at ${last_bar_close:.2f}")
            self._execute_trade(SignalType.CLOSE_LONG, last_bar_timestamp.tz_localize('UTC') if last_bar_timestamp.tzinfo is None else last_bar_timestamp.tz_convert('UTC'), last_bar_close)
        
        final_portfolio_state = {
            "initial_capital": self.initial_capital,
            "final_capital_cash": self.current_capital, # This is the cash after all trades
            "final_position_size": self.current_position_size, # Should be 0 if all closed
            "total_value": self.portfolio_history[-1]['total_value'] if self.portfolio_history else self.initial_capital,
            "end_timestamp": self.data_feed_df.index[-1]
        }
        
        logger.info(f"Backtest run completed. Final Portfolio Value: ${final_portfolio_state['total_value']:.2f}")
        return self.trades_log, final_portfolio_state