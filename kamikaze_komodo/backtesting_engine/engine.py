# kamikaze_komodo/backtesting_engine/engine.py
# Significantly updated to integrate PositionSizer and StopManager
# And to handle sentiment data (conceptual for now)
# Phase 5: BarData now has prediction_value, prediction_confidence. Engine will pass them along if present in data_feed_df
# or if strategy populates them on the BarData object.
# Phase 6: Added short selling capabilities and PnL calculation for shorts.
# Phase 6: Added placeholder for slippage and funding rate.

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from kamikaze_komodo.core.models import BarData, Trade, Order # Order not fully used
from kamikaze_komodo.core.enums import SignalType, OrderSide, TradeResult
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone

# Phase 3 imports
from kamikaze_komodo.risk_control_module.position_sizer import BasePositionSizer, FixedFractionalPositionSizer # Default
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager, PercentageStopManager # Default
from kamikaze_komodo.portfolio_constructor.asset_allocator import BaseAssetAllocator # For future use

logger = get_logger(__name__)

class BacktestingEngine:
    def __init__(
        self,
        data_feed_df: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0, # e.g., 10 bps = 0.1%
        position_sizer: Optional[BasePositionSizer] = None,
        stop_manager: Optional[BaseStopManager] = None,
        sentiment_data_df: Optional[pd.DataFrame] = None, # Timestamp-indexed series/df with sentiment scores
        # Phase 6: Slippage and Funding Rate (placeholders for now)
        slippage_bps: float = 0.0, # Slippage in basis points per side
        funding_rate_annualized: float = 0.0 # Example: 0.01 for 1% annualized
    ):
        if data_feed_df.empty: raise ValueError("Data feed DataFrame cannot be empty.")
        if not isinstance(data_feed_df.index, pd.DatetimeIndex):
            raise ValueError("Data feed DataFrame must be indexed by pd.DatetimeIndex.")

        self.data_feed_df = data_feed_df.sort_index()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0 # Phase 6
        self.funding_rate_annualized = funding_rate_annualized # Phase 6 (needs more sophisticated application)


        self.position_sizer = position_sizer if position_sizer else FixedFractionalPositionSizer(fraction=0.1)
        self.stop_manager = stop_manager if stop_manager else PercentageStopManager(stop_loss_pct=None, take_profit_pct=None)

        self.sentiment_data_df = sentiment_data_df
        if self.sentiment_data_df is not None and not self.sentiment_data_df.empty:
            if not isinstance(self.sentiment_data_df.index, pd.DatetimeIndex):
                logger.warning("Sentiment data DataFrame must be indexed by pd.DatetimeIndex. Sentiment will not be used.")
                self.sentiment_data_df = None
            else: # Ensure timezone consistency (UTC)
                if self.sentiment_data_df.index.tz is None:
                    self.sentiment_data_df.index = self.sentiment_data_df.index.tz_localize('UTC')
                else:
                    self.sentiment_data_df.index = self.sentiment_data_df.index.tz_convert('UTC')
                logger.info(f"Sentiment data loaded with {len(self.sentiment_data_df)} entries.")

        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_log: List[Trade] = []

        self.current_cash = initial_capital
        self.current_asset_value = 0.0 # Mark-to-market value of assets held (for long positions)
        self.current_short_liability = 0.0 # Market value of shorted assets (liability)
        self.current_portfolio_value = initial_capital # Total equity

        self.active_trade: Optional[Trade] = None
        self.trade_id_counter = 0

        logger.info(
            f"BacktestingEngine initialized for strategy '{strategy.name}' on symbol '{strategy.symbol}'. "
            f"Initial Capital: ${initial_capital:,.2f}, Commission: {commission_bps} bps, Slippage: {slippage_bps} bps."
        )
        logger.info(f"Position Sizer: {self.position_sizer.__class__.__name__}")
        logger.info(f"Stop Manager: {self.stop_manager.__class__.__name__}")
        if self.sentiment_data_df is not None and not self.sentiment_data_df.empty :
            logger.info("Sentiment data will be used in this backtest if strategy supports it.")


    def _get_next_trade_id(self) -> str:
        self.trade_id_counter += 1
        return f"trade_{self.trade_id_counter:05d}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Applies slippage to the execution price."""
        if self.slippage_rate == 0.0:
            return price
        if side == OrderSide.BUY: # Buy order, price might slip up
            return price * (1 + self.slippage_rate)
        elif side == OrderSide.SELL: # Sell order, price might slip down
            return price * (1 - self.slippage_rate)
        return price

    def _update_portfolio_value(self, current_bar_close_price: Optional[float] = None):
        """Updates current_asset_value, short_liability and current_portfolio_value."""
        if self.active_trade and current_bar_close_price is not None:
            if self.active_trade.side == OrderSide.BUY:
                self.current_asset_value = self.active_trade.amount * current_bar_close_price
                self.current_short_liability = 0.0
            elif self.active_trade.side == OrderSide.SELL: # Active short position
                # For shorts, "asset value" is negative (liability)
                # The value of the shorted asset is current_price * amount
                # Equity calculation for shorts: Initial Margin + (Entry Price - Current Price) * Amount - Current Cash
                # Simpler: Cash + (Entry Value of Short - Current Value of Short)
                # current_portfolio_value was cash_at_short_entry + short_entry_value
                # change in equity = (entry_price - current_bar_close_price) * self.active_trade.amount
                # self.current_portfolio_value = self.cash_at_trade_entry + change_in_equity
                # This is tricky. Let's simplify to: Cash + Unrealized PnL from short
                # Unrealized PnL = (Entry Price - Current Price) * Amount
                self.current_asset_value = 0.0 # No long asset held
                # No explicit short_liability needed if PnL is tracked correctly in portfolio_value
                # For MTM, portfolio_value = cash + (entry_price - current_price) * amount_shorted (excluding comm.)
                # This is implicitly handled by `self.current_portfolio_value` being updated based on PnL.
                # Let's adjust:
                unrealized_short_pnl = (self.active_trade.entry_price - current_bar_close_price) * self.active_trade.amount
                self.current_portfolio_value = self.current_cash + unrealized_short_pnl # This needs initial cash at trade start
                # This needs to be fixed if we MTM the portfolio value directly for shorts.
                # For now, we'll calculate portfolio value directly based on trades.
                # The self.portfolio_history append will be the primary source for equity curve.
                pass # MTM for shorts is complex to show as "asset_value"

        else: # No active trade
            self.current_asset_value = 0.0
            self.current_short_liability = 0.0

        # Portfolio value = cash + value of long positions - value of short positions (liability)
        # For MTM before closing trade, for shorts:
        # Equity = initial_capital_for_trade + (entry_price - current_price) * quantity
        # For simplicity, the final portfolio value update after trade close is most critical.
        # self.current_portfolio_value = self.current_cash + self.current_asset_value - self.current_short_liability
        # This will be updated in self.portfolio_history log correctly.


    def _execute_trade(
        self,
        signal_type: SignalType,
        timestamp: datetime,
        price: float, # Ideal execution price
        current_bar_for_atr_calc: Optional[BarData] = None
    ):
        commission_cost = 0.0
        trade_executed = False

        # --- LONG ENTRY ---
        if signal_type == SignalType.LONG and self.active_trade is None:
            execution_price = self._apply_slippage(price, OrderSide.BUY)
            atr_value_for_sizing = current_bar_for_atr_calc.atr if current_bar_for_atr_calc and hasattr(current_bar_for_atr_calc, 'atr') and current_bar_for_atr_calc.atr is not None else None
            position_size_units = self.position_sizer.calculate_size(
                symbol=self.strategy.symbol, current_price=execution_price, available_capital=self.current_cash,
                current_portfolio_value=self.current_portfolio_value, latest_bar=current_bar_for_atr_calc,
                atr_value=atr_value_for_sizing
            )
            if position_size_units is None or position_size_units <= 1e-8:
                logger.debug(f"{timestamp} - Cannot enter LONG for {self.strategy.symbol}. Sizer returned no size ({position_size_units}).")
                return

            cost_of_assets = position_size_units * execution_price
            commission_cost = cost_of_assets * self.commission_rate
            if cost_of_assets + commission_cost > self.current_cash:
                logger.warning(f"{timestamp} - Insufficient cash for LONG on {self.strategy.symbol}. Need {cost_of_assets + commission_cost:.2f}, have {self.current_cash:.2f}. Reducing size.")
                if execution_price <= 0 or (1 + self.commission_rate) <= 0 : return
                max_cost_before_commission = self.current_cash / (1 + self.commission_rate)
                adjusted_size_units = max_cost_before_commission / execution_price
                if adjusted_size_units <= 1e-8 : logger.warning(f"{timestamp} - Still insufficient cash after adjustment. Skipping trade."); return
                position_size_units = adjusted_size_units
                cost_of_assets = position_size_units * execution_price
                commission_cost = cost_of_assets * self.commission_rate

            self.current_cash -= (cost_of_assets + commission_cost)
            atr_at_entry = current_bar_for_atr_calc.atr if current_bar_for_atr_calc and current_bar_for_atr_calc.atr else None
            self.active_trade = Trade(
                id=self._get_next_trade_id(), symbol=self.strategy.symbol, entry_order_id=f"entry_{self.trade_id_counter}",
                side=OrderSide.BUY, entry_price=execution_price, amount=position_size_units, entry_timestamp=timestamp,
                commission=commission_cost, custom_fields={"atr_at_entry": atr_at_entry} if atr_at_entry else {}
            )
            self.strategy.current_position_status = SignalType.LONG # Update strategy's internal state
            logger.info(
                f"{timestamp} - EXECUTE LONG: {position_size_units:.6f} {self.strategy.symbol} @ ${execution_price:.2f} (Price w/ slippage). "
                f"Cost: ${cost_of_assets:.2f}, Comm: ${commission_cost:.2f}. Cash Left: ${self.current_cash:.2f}."
            )
            trade_executed = True

        # --- SHORT ENTRY ---
        elif signal_type == SignalType.SHORT and self.active_trade is None:
            execution_price = self._apply_slippage(price, OrderSide.SELL) # Slippage makes sell price lower
            atr_value_for_sizing = current_bar_for_atr_calc.atr if current_bar_for_atr_calc and hasattr(current_bar_for_atr_calc, 'atr') and current_bar_for_atr_calc.atr is not None else None
            position_size_units = self.position_sizer.calculate_size( # Sizer should be direction agnostic for risk capital
                symbol=self.strategy.symbol, current_price=execution_price, available_capital=self.current_cash, # Margin considerations for shorts are complex
                current_portfolio_value=self.current_portfolio_value, latest_bar=current_bar_for_atr_calc,
                atr_value=atr_value_for_sizing
            )
            if position_size_units is None or position_size_units <= 1e-8:
                logger.debug(f"{timestamp} - Cannot enter SHORT for {self.strategy.symbol}. Sizer returned no size ({position_size_units}).")
                return

            # For shorts, cash doesn't decrease by cost of assets. It increases by proceeds, minus commission.
            # Margin is held by broker. This simplified model assumes enough margin.
            proceeds_from_short = position_size_units * execution_price
            commission_cost = proceeds_from_short * self.commission_rate
            # self.current_cash += (proceeds_from_short - commission_cost) # Cash increases

            # To keep PnL tracking simple, let's make cash effectively decrease by value for symmetry in PnL calc,
            # OR track entry_value_of_short separately.
            # For now, we'll keep track of the entry price and amount, and calculate PnL at exit.
            # The cash change for shorts is more complex with margin, so we focus on equity.
            self.current_cash -= commission_cost # Cash only decreases by commission for this simplified model

            atr_at_entry = current_bar_for_atr_calc.atr if current_bar_for_atr_calc and current_bar_for_atr_calc.atr else None
            self.active_trade = Trade(
                id=self._get_next_trade_id(), symbol=self.strategy.symbol, entry_order_id=f"entry_{self.trade_id_counter}",
                side=OrderSide.SELL, entry_price=execution_price, amount=position_size_units, entry_timestamp=timestamp,
                commission=commission_cost, custom_fields={"atr_at_entry": atr_at_entry} if atr_at_entry else {}
            )
            self.strategy.current_position_status = SignalType.SHORT # Update strategy's internal state
            logger.info(
                f"{timestamp} - EXECUTE SHORT: {position_size_units:.6f} {self.strategy.symbol} @ ${execution_price:.2f} (Price w/ slippage). "
                f"Proceeds (gross): ${proceeds_from_short:.2f}, Comm: ${commission_cost:.2f}. Cash (adj for comm): ${self.current_cash:.2f}."
            )
            trade_executed = True

        # --- POSITION EXIT (from strategy signal) ---
        elif signal_type == SignalType.CLOSE_LONG and self.active_trade and self.active_trade.side == OrderSide.BUY:
            execution_price = self._apply_slippage(price, OrderSide.SELL) # Selling to close long, price might slip down
            exit_value = self.active_trade.amount * execution_price
            commission_cost = exit_value * self.commission_rate
            self.current_cash += (exit_value - commission_cost)
            pnl_for_this_trade = (execution_price - self.active_trade.entry_price) * self.active_trade.amount - self.active_trade.commission - commission_cost
            self._log_and_clear_active_trade(timestamp, execution_price, pnl_for_this_trade, commission_cost, "SignalCloseLong")
            trade_executed = True

        elif signal_type == SignalType.CLOSE_SHORT and self.active_trade and self.active_trade.side == OrderSide.SELL:
            execution_price = self._apply_slippage(price, OrderSide.BUY) # Buying to cover short, price might slip up
            cost_to_cover = self.active_trade.amount * execution_price
            commission_cost = cost_to_cover * self.commission_rate
            # self.current_cash -= (cost_to_cover + commission_cost) # Cash decreases

            # PnL for short = (Entry Price - Exit Price) * Amount - Total Commissions
            pnl_for_this_trade = (self.active_trade.entry_price - execution_price) * self.active_trade.amount - self.active_trade.commission - commission_cost
            self.current_cash += pnl_for_this_trade # Cash changes by net PnL of the short trade (already includes commissions factored in pnl_for_this_trade)

            self._log_and_clear_active_trade(timestamp, execution_price, pnl_for_this_trade, commission_cost, "SignalCloseShort")
            trade_executed = True

        if trade_executed:
            self._update_portfolio_value(price if self.active_trade else None)


    def _log_and_clear_active_trade(self, timestamp: datetime, exit_price: float, pnl: float, current_exit_commission: float, exit_reason: str):
        if not self.active_trade: return

        self.active_trade.exit_price = exit_price
        self.active_trade.exit_timestamp = timestamp
        self.active_trade.pnl = pnl
        self.active_trade.commission += current_exit_commission # Add exit commission to total
        self.active_trade.result = TradeResult.WIN if pnl > 0 else (TradeResult.LOSS if pnl < 0 else TradeResult.BREAKEVEN)
        self.active_trade.notes = exit_reason
        self.active_trade.exit_order_id = f"{exit_reason.lower()}_{self.active_trade.id.split('_')[1]}"

        # Calculate PnL percentage
        initial_trade_value_abs = abs(self.active_trade.entry_price * self.active_trade.amount)
        if initial_trade_value_abs > 0:
            self.active_trade.pnl_percentage = (pnl / initial_trade_value_abs) * 100
        else:
            self.active_trade.pnl_percentage = 0.0

        self.trades_log.append(self.active_trade.model_copy(deep=True))
        logger.info(
            f"{timestamp} - EXECUTE CLOSE {self.active_trade.side.name} ({exit_reason}): {self.active_trade.amount:.6f} {self.strategy.symbol} @ ${exit_price:.2f}. "
            f"PnL: ${pnl:.2f} ({self.active_trade.pnl_percentage:.2f}%). Total Comm: ${self.active_trade.commission:.2f}. "
            f"Cash Now: ${self.current_cash:.2f}."
        )
        self.active_trade = None
        self.strategy.current_position_status = None # Reset strategy's internal state

    def _handle_stop_take_profit(self, current_bar: BarData):
        if self.active_trade is None or self.stop_manager is None:
            return

        stop_loss_trigger_price = self.stop_manager.check_stop_loss(self.active_trade, current_bar)
        if stop_loss_trigger_price is not None:
            logger.info(f"{current_bar.timestamp} - STOP LOSS triggered for trade {self.active_trade.id} at derived price {stop_loss_trigger_price:.2f}")
            # Slippage for SL/TP should also be considered. For SL, it makes the loss worse.
            effective_exit_price = self._apply_slippage(stop_loss_trigger_price, OrderSide.SELL if self.active_trade.side == OrderSide.BUY else OrderSide.BUY)
            self._execute_exit(current_bar.timestamp, effective_exit_price, "StopLoss")
            return

        if self.active_trade: # SL might have closed the trade
            take_profit_trigger_price = self.stop_manager.check_take_profit(self.active_trade, current_bar)
            if take_profit_trigger_price is not None:
                logger.info(f"{current_bar.timestamp} - TAKE PROFIT triggered for trade {self.active_trade.id} at derived price {take_profit_trigger_price:.2f}")
                # Slippage for TP makes profit smaller
                effective_exit_price = self._apply_slippage(take_profit_trigger_price, OrderSide.SELL if self.active_trade.side == OrderSide.BUY else OrderSide.BUY)
                self._execute_exit(current_bar.timestamp, effective_exit_price, "TakeProfit")
                return

    def _execute_exit(self, timestamp: datetime, price: float, exit_reason: str):
        if not self.active_trade: return

        current_exit_commission = 0.0
        pnl_for_this_trade = 0.0

        if self.active_trade.side == OrderSide.BUY:
            exit_value = self.active_trade.amount * price
            current_exit_commission = exit_value * self.commission_rate
            self.current_cash += (exit_value - current_exit_commission)
            pnl_for_this_trade = (price - self.active_trade.entry_price) * self.active_trade.amount - self.active_trade.commission - current_exit_commission
        elif self.active_trade.side == OrderSide.SELL: # Closing a short
            cost_to_cover = self.active_trade.amount * price
            current_exit_commission = cost_to_cover * self.commission_rate
            # PnL for short = (Entry Price - Exit Price) * Amount - Total Commissions
            pnl_for_this_trade = (self.active_trade.entry_price - price) * self.active_trade.amount - self.active_trade.commission - current_exit_commission
            # Cash adjustment for short closure:
            # Initial cash impact was -(entry_commission).
            # Cash changes by net PnL of the short trade.
            self.current_cash += pnl_for_this_trade # Add net PnL (which already has all commissions subtracted)

        self._log_and_clear_active_trade(timestamp, price, pnl_for_this_trade, current_exit_commission, exit_reason)


    def run(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]: # Added equity curve df to output
        logger.info(f"Starting backtest run for strategy '{self.strategy.name}'...")
        self.strategy.data_history = pd.DataFrame(columns=[
            'open', 'high', 'low', 'close', 'volume', 'atr',
            'sentiment_score', 'prediction_value', 'prediction_confidence'
        ])
        # Initialize portfolio history with initial capital
        self.portfolio_history.append({
                "timestamp": self.data_feed_df.index[0] - pd.Timedelta(seconds=1) if not self.data_feed_df.empty else datetime.now(timezone.utc) - pd.Timedelta(seconds=1), # Synthetic start
                "cash": self.initial_capital,
                "asset_value": 0.0,
                "total_value": self.initial_capital,
                "current_price": self.data_feed_df.iloc[0]['close'] if not self.data_feed_df.empty else 0,
                "active_trade_pnl": 0.0
            })


        for timestamp, row in self.data_feed_df.iterrows():
            ts_aware = timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')

            bar_data_args = {
                "timestamp": ts_aware, "open": row['open'], "high": row['high'],
                "low": row['low'], "close": row['close'], "volume": row['volume'],
                "symbol": self.strategy.symbol, "timeframe": self.strategy.timeframe,
                "atr": row.get('atr'),
                "prediction_value": row.get('prediction_value'),
                "prediction_confidence": row.get('prediction_confidence')
            }

            current_sentiment_score = None
            if self.sentiment_data_df is not None and not self.sentiment_data_df.empty:
                try:
                    if not self.sentiment_data_df.index.is_monotonic_increasing:
                        self.sentiment_data_df = self.sentiment_data_df.sort_index()
                    sentiment_value = self.sentiment_data_df['sentiment_score'].asof(ts_aware)
                    if pd.notna(sentiment_value):
                        current_sentiment_score = sentiment_value
                        bar_data_args["sentiment_score"] = current_sentiment_score
                except KeyError: pass
                except Exception as e_sentiment: logger.warning(f"Error accessing sentiment data for {ts_aware}: {e_sentiment}")

            current_bar = BarData(**bar_data_args)

            if self.active_trade:
                self._handle_stop_take_profit(current_bar)

            if self.active_trade is None or not (self.active_trade.exit_price is not None):
                signal = self.strategy.on_bar_data(current_bar, sentiment_score=current_sentiment_score)
                if signal and signal != SignalType.HOLD:
                    execution_price = current_bar.close # Ideal price before slippage
                    self._execute_trade(signal, current_bar.timestamp, execution_price, current_bar_for_atr_calc=current_bar)

            # Log portfolio state at the end of each bar
            # MTM calculation needs to be robust for shorts
            current_mtm_portfolio_value = self.current_cash
            if self.active_trade:
                if self.active_trade.side == OrderSide.BUY:
                    current_mtm_portfolio_value += self.active_trade.amount * current_bar.close
                elif self.active_trade.side == OrderSide.SELL: # Short position
                    # Equity for short = Cash + (Entry Value - Current Value) - Commissions so far
                    # Entry value = entry_price * amount
                    # Current value = current_bar.close * amount
                    # PnL_unrealized = (self.active_trade.entry_price - current_bar.close) * self.active_trade.amount
                    # current_mtm_portfolio_value += PnL_unrealized # self.current_cash already reflects cash change from short entry commission
                    # This is tricky, let's directly use equity: cash + sum of unrealized PnL of open positions
                    # For one short trade: cash_after_entry_comm + (entry_price - current_price)*amount
                    # The self.current_cash at this point is after entry commissions.
                    # So, add the unrealized PnL.
                    unrealized_pnl = (self.active_trade.entry_price - current_bar.close) * self.active_trade.amount
                    current_mtm_portfolio_value = self.current_cash + unrealized_pnl


            self.portfolio_history.append({
                "timestamp": current_bar.timestamp,
                "cash": self.current_cash,
                "asset_value": self.active_trade.amount * current_bar.close if self.active_trade and self.active_trade.side == OrderSide.BUY else 0.0,
                "total_value": current_mtm_portfolio_value, # Equity
                "current_price": current_bar.close,
                "active_trade_pnl": ((current_bar.close - self.active_trade.entry_price) * self.active_trade.amount) if self.active_trade and self.active_trade.side == OrderSide.BUY else (((self.active_trade.entry_price - current_bar.close) * self.active_trade.amount) if self.active_trade and self.active_trade.side == OrderSide.SELL else 0.0)
            })
            # TODO: Add funding rate application here if applicable (e.g. daily for perpetuals)

        if self.active_trade:
            last_bar_data_row = self.data_feed_df.iloc[-1]
            last_bar_timestamp = self.data_feed_df.index[-1].tz_localize('UTC') if self.data_feed_df.index[-1].tzinfo is None else self.data_feed_df.index[-1].tz_convert('UTC')
            last_bar_close = last_bar_data_row['close']
            logger.info(f"{last_bar_timestamp} - End of backtest. Closing open {self.active_trade.side.name} position for {self.strategy.symbol} at ${last_bar_close:.2f}")
            effective_exit_price = self._apply_slippage(last_bar_close, OrderSide.SELL if self.active_trade.side == OrderSide.BUY else OrderSide.BUY)
            self._execute_exit(last_bar_timestamp, effective_exit_price, "EndOfBacktest")

        # Final portfolio value should be all cash if all positions closed
        final_portfolio_state = {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": self.current_cash, # After all trades closed, portfolio value is cash
            "final_cash": self.current_cash,
            "end_timestamp": self.data_feed_df.index[-1] if not self.data_feed_df.empty else datetime.now(timezone.utc)
        }
        logger.info(f"Backtest run completed. Final Portfolio Value: ${final_portfolio_state['final_portfolio_value']:.2f}")

        equity_curve_df = pd.DataFrame(self.portfolio_history)
        if not equity_curve_df.empty:
            equity_curve_df['timestamp'] = pd.to_datetime(equity_curve_df['timestamp'])
            equity_curve_df.set_index('timestamp', inplace=True)

        return self.trades_log, final_portfolio_state, equity_curve_df