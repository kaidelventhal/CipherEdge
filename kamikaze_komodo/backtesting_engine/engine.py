# FILE: kamikaze_komodo/backtesting_engine/engine.py
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from kamikaze_komodo.core.models import BarData, Trade, Order
from kamikaze_komodo.core.enums import SignalType, OrderSide, TradeResult
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.app_logger import get_logger
from datetime import datetime, timezone

from kamikaze_komodo.risk_control_module.position_sizer import BasePositionSizer, FixedFractionalPositionSizer
from kamikaze_komodo.risk_control_module.stop_manager import BaseStopManager, PercentageStopManager

logger = get_logger(__name__)

class BacktestingEngine:
    def __init__(
        self,
        data_feed_df: pd.DataFrame,
        strategy: BaseStrategy,
        initial_capital: float = 10000.0,
        commission_bps: float = 0.0,
        position_sizer: Optional[BasePositionSizer] = None,
        stop_manager: Optional[BaseStopManager] = None,
        sentiment_data_df: Optional[pd.DataFrame] = None,
        slippage_bps: float = 0.0,
        funding_rate_annualized: float = 0.0,
        data_feed_df_pair_asset2: Optional[pd.DataFrame] = None
    ):
        if data_feed_df.empty: raise ValueError("Data feed DataFrame cannot be empty.")
        if not isinstance(data_feed_df.index, pd.DatetimeIndex):
            raise ValueError("Data feed DataFrame must be indexed by pd.DatetimeIndex.")

        self.data_feed_df = data_feed_df.sort_index()
        self.data_feed_df_pair_asset2 = data_feed_df_pair_asset2.sort_index() if data_feed_df_pair_asset2 is not None else None
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        self.funding_rate_annualized = funding_rate_annualized

        self.position_sizer = position_sizer if position_sizer else FixedFractionalPositionSizer()
        self.stop_manager = stop_manager if stop_manager else PercentageStopManager()

        self.sentiment_data_df = sentiment_data_df
        if self.sentiment_data_df is not None and not self.sentiment_data_df.empty:
            if not isinstance(self.sentiment_data_df.index, pd.DatetimeIndex):
                logger.warning("Sentiment data DataFrame must be indexed by pd.DatetimeIndex. Sentiment will not be used.")
                self.sentiment_data_df = None
            else:
                if self.sentiment_data_df.index.tz is None:
                    self.sentiment_data_df.index = self.sentiment_data_df.index.tz_localize('UTC')
                else:
                    self.sentiment_data_df.index = self.sentiment_data_df.index.tz_convert('UTC')
                logger.info(f"Sentiment data loaded with {len(self.sentiment_data_df)} entries.")

        self.portfolio_history: List[Dict[str, Any]] = []
        self.trades_log: List[Trade] = []

        self.current_cash = initial_capital
        self.current_portfolio_value = initial_capital
        self.active_trades: Dict[str, Trade] = {}
        self.trade_id_counter = 0

        logger.info(
            f"BacktestingEngine initialized for strategy '{strategy.name}'. "
            f"Initial Capital: ${initial_capital:,.2f}, Commission: {commission_bps} bps, Slippage: {slippage_bps} bps, Annual Funding: {funding_rate_annualized*100:.2f}%."
        )

    def _get_next_trade_id(self) -> str:
        self.trade_id_counter += 1
        return f"trade_{self.trade_id_counter:05d}"

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if self.slippage_rate == 0.0: return price
        if side == OrderSide.BUY: return price * (1 + self.slippage_rate)
        elif side == OrderSide.SELL: return price * (1 - self.slippage_rate)
        return price

    def _apply_funding(self, position_value: float, side: OrderSide, funding_rate_per_bar: float):
        if funding_rate_per_bar == 0.0: return 0.0
        funding_cost = position_value * funding_rate_per_bar
        if side == OrderSide.BUY:
            self.current_cash -= funding_cost
            return funding_cost
        elif side == OrderSide.SELL:
            self.current_cash += funding_cost
            return -funding_cost
        return 0.0

    def _execute_trade_command(self, command: SignalCommand, current_bar_data_for_command_symbol: BarData, bar_index: int):
        signal_type = command.signal_type
        trade_symbol = command.symbol
        execution_price_ideal = command.price if command.price else current_bar_data_for_command_symbol.close
        timestamp = current_bar_data_for_command_symbol.timestamp
        bar_for_atr_calc = command.related_bar_data if command.related_bar_data else current_bar_data_for_command_symbol

        active_trade_for_symbol = self.active_trades.get(trade_symbol)

        # --- ENTRY LOGIC ---
        if signal_type in [SignalType.LONG, SignalType.SHORT] and active_trade_for_symbol is None:
            side = OrderSide.BUY if signal_type == SignalType.LONG else OrderSide.SELL
            execution_price = self._apply_slippage(execution_price_ideal, side)
            
            position_size_units = self.position_sizer.calculate_size(
                symbol=trade_symbol, current_price=execution_price, available_capital=self.current_cash,
                current_portfolio_value=self.current_portfolio_value, latest_bar=bar_for_atr_calc,
                atr_value=bar_for_atr_calc.atr
            )
            if position_size_units is None or position_size_units <= 1e-8: return

            cost_of_assets = position_size_units * execution_price
            commission_cost = cost_of_assets * self.commission_rate

            if side == OrderSide.BUY and cost_of_assets + commission_cost > self.current_cash:
                logger.warning(f"{timestamp} - Insufficient cash for LONG on {trade_symbol}. Skipping.")
                return

            self.current_cash -= commission_cost
            if side == OrderSide.BUY:
                self.current_cash -= cost_of_assets
            else: # On short entry, cash increases by proceeds (margin not modeled)
                self.current_cash += cost_of_assets

            custom_fields = {"atr_at_entry": bar_for_atr_calc.atr, "entry_bar_index": bar_index}
            self.active_trades[trade_symbol] = Trade(
                id=self._get_next_trade_id(), symbol=trade_symbol, entry_order_id=f"entry_{self.trade_id_counter}",
                side=side, entry_price=execution_price, amount=position_size_units, entry_timestamp=timestamp,
                commission=commission_cost, custom_fields=custom_fields
            )
            logger.info(f"{timestamp} - EXECUTE CMD {side.name}: {position_size_units:.6f} {trade_symbol} @ ${execution_price:.4f}")

        # --- EXIT LOGIC ---
        elif signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT] and active_trade_for_symbol:
            expected_side = OrderSide.BUY if signal_type == SignalType.CLOSE_LONG else OrderSide.SELL
            if active_trade_for_symbol.side != expected_side: return

            exit_side = OrderSide.SELL if signal_type == SignalType.CLOSE_LONG else OrderSide.BUY
            execution_price = self._apply_slippage(execution_price_ideal, exit_side)
            exit_value = active_trade_for_symbol.amount * execution_price
            commission_cost = exit_value * self.commission_rate
            
            pnl = 0.0
            if active_trade_for_symbol.side == OrderSide.BUY:
                pnl = (execution_price - active_trade_for_symbol.entry_price) * active_trade_for_symbol.amount
                self.current_cash += exit_value # Receive cash from selling the asset
            else: # SELL
                pnl = (active_trade_for_symbol.entry_price - execution_price) * active_trade_for_symbol.amount
                self.current_cash -= exit_value # Pay cash to buy back the asset (cover short)
            
            pnl -= (active_trade_for_symbol.commission + commission_cost)
            self.current_cash -= commission_cost

            self._log_and_clear_active_trade(trade_symbol, timestamp, execution_price, pnl, commission_cost, "SignalClose")


    def _log_and_clear_active_trade(self, symbol_key: str, timestamp: datetime, exit_price: float, pnl: float, exit_commission: float, exit_reason: str):
        active_trade = self.active_trades.get(symbol_key)
        if not active_trade: return

        active_trade.exit_price = exit_price
        active_trade.exit_timestamp = timestamp
        active_trade.pnl = pnl
        active_trade.commission += exit_commission
        active_trade.result = TradeResult.WIN if pnl > 0 else (TradeResult.LOSS if pnl < 0 else TradeResult.BREAKEVEN)
        active_trade.notes = exit_reason

        self.trades_log.append(active_trade.model_copy(deep=True))
        logger.info(
            f"{timestamp} - EXECUTE CLOSE {active_trade.side.name} ({exit_reason}) for {symbol_key}: {active_trade.amount:.6f} @ ${exit_price:.4f}. "
            f"PnL: ${pnl:.2f}. Total Comm: ${active_trade.commission:.2f}. Cash Now: ${self.current_cash:.2f}."
        )
        del self.active_trades[symbol_key]

    def _handle_stop_take_profit(self, current_bar_data_for_symbol: BarData, trade_symbol: str, bar_index: int):
        active_trade = self.active_trades.get(trade_symbol)
        if not active_trade or not self.stop_manager:
            return

        stop_loss_price = self.stop_manager.check_stop_loss(active_trade, current_bar_data_for_symbol, bar_index)
        if stop_loss_price is not None:
            # FIX: The variable is named current_bar_data_for_symbol
            self._execute_trade_command(SignalCommand(
                signal_type=SignalType.CLOSE_LONG if active_trade.side == OrderSide.BUY else SignalType.CLOSE_SHORT,
                symbol=trade_symbol, price=stop_loss_price
            ), current_bar_data_for_symbol, bar_index)
            return

        active_trade = self.active_trades.get(trade_symbol)
        if active_trade:
            take_profit_price = self.stop_manager.check_take_profit(active_trade, current_bar_data_for_symbol)
            if take_profit_price is not None:
                # FIX: The variable is named current_bar_data_for_symbol
                self._execute_trade_command(SignalCommand(
                    signal_type=SignalType.CLOSE_LONG if active_trade.side == OrderSide.BUY else SignalType.CLOSE_SHORT,
                    symbol=trade_symbol, price=take_profit_price
                ), current_bar_data_for_symbol, bar_index)

    def run(self) -> tuple[List[Trade], Dict[str, Any], pd.DataFrame]:
        logger.info(f"Starting backtest run for strategy '{self.strategy.name}'...")
        self.portfolio_history.append({
            "timestamp": self.data_feed_df.index[0] - pd.Timedelta(seconds=1) if not self.data_feed_df.empty else datetime.now(timezone.utc),
            "cash": self.initial_capital, "asset_value": 0.0, "total_value": self.initial_capital
        })

        for i, (timestamp, row_asset1) in enumerate(self.data_feed_df.iterrows()):
            ts_aware = timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')

            current_bar_asset1 = BarData(
                timestamp=ts_aware, open=row_asset1['open'], high=row_asset1['high'], low=row_asset1['low'],
                close=row_asset1['close'], volume=row_asset1['volume'], symbol=self.strategy.symbol,
                timeframe=self.strategy.timeframe, atr=row_asset1.get('atr'),
                prediction_value=row_asset1.get('prediction_value'), prediction_confidence=row_asset1.get('prediction_confidence'),
                market_regime=row_asset1.get('market_regime')
            )
            self.data_feed_df.name = self.strategy.symbol

            current_bar_asset2 = None
            if self.data_feed_df_pair_asset2 is not None and ts_aware in self.data_feed_df_pair_asset2.index:
                row_asset2 = self.data_feed_df_pair_asset2.loc[ts_aware]
                current_bar_asset2 = BarData(
                    timestamp=ts_aware, open=row_asset2['open'], high=row_asset2['high'], low=row_asset2['low'],
                    close=row_asset2['close'], volume=row_asset2['volume'], symbol=self.strategy.params.get('asset2_symbol', 'PAIR_ASSET2'),
                    timeframe=self.strategy.timeframe, atr=row_asset2.get('atr')
                )
                self.data_feed_df_pair_asset2.name = self.strategy.params.get('asset2_symbol', 'PAIR_ASSET2')

            sentiment_val = self.sentiment_data_df['sentiment_score'].asof(ts_aware) if self.sentiment_data_df is not None else None
            current_bar_asset1.sentiment_score = sentiment_val if pd.notna(sentiment_val) else None

            for sym_key in list(self.active_trades.keys()):
                bar_for_stop_check = current_bar_asset1 if sym_key == self.strategy.symbol else current_bar_asset2
                if bar_for_stop_check:
                    self._handle_stop_take_profit(bar_for_stop_check, sym_key, i)

            strategy_output: Union[Optional[SignalType], List[SignalCommand]] = self.strategy.on_bar_data(current_bar_asset1)
            
            if strategy_output:
                commands = strategy_output if isinstance(strategy_output, list) else []
                if isinstance(strategy_output, SignalType) and strategy_output != SignalType.HOLD:
                     commands = [SignalCommand(signal_type=strategy_output, symbol=self.strategy.symbol, price=current_bar_asset1.close, related_bar_data=current_bar_asset1)]

                for command in commands:
                    bar_for_cmd = current_bar_asset1 if command.symbol == self.strategy.symbol else current_bar_asset2
                    if bar_for_cmd:
                        self._execute_trade_command(command, bar_for_cmd, i)

            mtm_value = self.current_cash
            for trade in self.active_trades.values():
                price = current_bar_asset1.close if trade.symbol == current_bar_asset1.symbol else (current_bar_asset2.close if current_bar_asset2 else trade.entry_price)
                if trade.side == OrderSide.BUY:
                    mtm_value += trade.amount * price
                else: # Short
                    initial_proceeds = trade.amount * trade.entry_price
                    cost_to_cover = trade.amount * price
                    mtm_value += initial_proceeds - cost_to_cover
            
            self.portfolio_history.append({"timestamp": ts_aware, "cash": self.current_cash, "total_value": mtm_value})

        if self.active_trades:
            last_bar = self.data_feed_df.iloc[-1]
            last_bar_data = BarData(timestamp=last_bar.name, open=last_bar['open'], high=last_bar['high'], low=last_bar['low'], close=last_bar['close'], volume=last_bar['volume'], symbol=self.strategy.symbol)
            for sym_key in list(self.active_trades.keys()):
                active_trade = self.active_trades[sym_key]
                self._execute_trade_command(SignalCommand(
                    signal_type=SignalType.CLOSE_LONG if active_trade.side == OrderSide.BUY else SignalType.CLOSE_SHORT,
                    symbol=sym_key, price=last_bar_data.close), last_bar_data, len(self.data_feed_df) - 1)

        final_portfolio_state = {"initial_capital": self.initial_capital, "final_portfolio_value": self.current_cash}
        logger.info(f"Backtest run completed. Final Portfolio Value: ${final_portfolio_state['final_portfolio_value']:.2f}")
        equity_curve_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        return self.trades_log, final_portfolio_state, equity_curve_df