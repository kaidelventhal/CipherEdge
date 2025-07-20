# FILE: kamikaze_komodo/backtesting_engine/performance_analyzer.py
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import norm

from kamikaze_komodo.core.models import Trade
from kamikaze_komodo.core.enums import OrderSide, TradeResult
from kamikaze_komodo.app_logger import get_logger

logger = get_logger(__name__)

class PerformanceAnalyzer:
    def __init__(
        self,
        trades: List[Trade],
        initial_capital: float,
        final_capital: float,
        equity_curve_df: Optional[pd.DataFrame] = None, # Timestamp-indexed 'total_value_usd'
        risk_free_rate_annual: float = 0.02, # Annual risk-free rate (e.g., 2%)
        annualization_factor: int = 252 # Trading days in a year for Sharpe/Sortino
    ):
        if not trades:
            logger.warning("PerformanceAnalyzer initialized with no trades. Some metrics might be zero or NaN.")
        self.trades_df = pd.DataFrame([trade.model_dump() for trade in trades])
        if not self.trades_df.empty:
            self.trades_df['entry_timestamp'] = pd.to_datetime(self.trades_df['entry_timestamp'])
            self.trades_df['exit_timestamp'] = pd.to_datetime(self.trades_df['exit_timestamp'])
    
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.equity_curve_df = equity_curve_df
        self.risk_free_rate_annual = risk_free_rate_annual
        self.annualization_factor = annualization_factor
    
        logger.info(f"PerformanceAnalyzer initialized. Trades: {len(trades)}, Initial: ${initial_capital:,.2f}, Final: ${final_capital:,.2f}")
        logger.info(f"Using Annual Risk-Free Rate: {self.risk_free_rate_annual*100:.2f}%, Annualization Factor: {self.annualization_factor}")


    def _calculate_periodic_returns(self) -> Optional[pd.Series]:
        # FIX: Use 'total_value_usd' to match the backtesting engine's output
        if self.equity_curve_df is None or self.equity_curve_df.empty or 'total_value_usd' not in self.equity_curve_df.columns:
            logger.warning("Equity curve data is missing or invalid. Cannot calculate periodic returns for Sharpe/Sortino.")
            return None
        # Resample to daily returns for annualization, handling potential non-unique index if multiple records per day
        daily_equity = self.equity_curve_df['total_value_usd'].resample('D').last().ffill()
        periodic_returns = daily_equity.pct_change().dropna()
        return periodic_returns

    def calculate_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_net_profit": 0.0,
            "total_return_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "loss_rate_pct": 0.0,
            "average_pnl_per_trade": 0.0,
            "average_win_pnl": 0.0,
            "average_loss_pnl": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "calmar_ratio": np.nan,
            "total_fees_paid": 0.0,
            "average_holding_period_hours": 0.0,
            "longest_win_streak": 0,
            "longest_loss_streak": 0,
            "time_in_market_pct": 0.0,
            "turnover_rate": np.nan,
        }

        if self.trades_df.empty:
            metrics["total_net_profit"] = self.final_capital - self.initial_capital
            if self.initial_capital > 0:
                metrics["total_return_pct"] = (metrics["total_net_profit"] / self.initial_capital) * 100
            logger.warning("No trades to analyze. Returning basic capital metrics.")
            return metrics

        pnl_series = self.trades_df['pnl'].dropna()
        if pnl_series.empty:
            metrics["total_net_profit"] = self.final_capital - self.initial_capital # If PnL couldn't be calculated for trades
            if self.initial_capital > 0: metrics["total_return_pct"] = (metrics["total_net_profit"] / self.initial_capital) * 100
            metrics["total_trades"] = len(self.trades_df)
            metrics["total_fees_paid"] = self.trades_df['commission'].sum()
            return metrics
    
        metrics["total_net_profit"] = pnl_series.sum()
        if self.initial_capital > 0: metrics["total_return_pct"] = (metrics["total_net_profit"] / self.initial_capital) * 100
        metrics["total_trades"] = len(pnl_series)

        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]
        breakevens = pnl_series[pnl_series == 0]
        metrics["winning_trades"] = len(wins)
        metrics["losing_trades"] = len(losses)
        metrics["breakeven_trades"] = len(breakevens)

        if metrics["total_trades"] > 0:
            metrics["win_rate_pct"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100
            metrics["loss_rate_pct"] = (metrics["losing_trades"] / metrics["total_trades"]) * 100
            metrics["average_pnl_per_trade"] = pnl_series.mean()
        if not wins.empty: metrics["average_win_pnl"] = wins.mean()
        if not losses.empty: metrics["average_loss_pnl"] = losses.mean()

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        if gross_loss > 0: metrics["profit_factor"] = gross_profit / gross_loss
        elif gross_profit > 0: metrics["profit_factor"] = np.inf
    
        metrics["total_fees_paid"] = self.trades_df['commission'].sum()

        # Max Drawdown (from equity curve)
        # FIX: Use 'total_value_usd'
        if self.equity_curve_df is not None and not self.equity_curve_df.empty and 'total_value_usd' in self.equity_curve_df.columns:
            equity_values = self.equity_curve_df['total_value_usd']
            if len(equity_values) > 1:
                peak = equity_values.expanding(min_periods=1).max()
                drawdown = (equity_values - peak) / peak
                metrics["max_drawdown_pct"] = abs(drawdown.min()) * 100 if not drawdown.empty else 0.0
    
        # Sharpe and Sortino Ratios
        periodic_returns = self._calculate_periodic_returns()
        if periodic_returns is not None and len(periodic_returns) > 1:
            risk_free_rate_periodic = self.risk_free_rate_annual / self.annualization_factor
            excess_returns = periodic_returns - risk_free_rate_periodic
        
            # Sharpe Ratio
            sharpe_avg_excess_return = excess_returns.mean()
            sharpe_std_excess_return = excess_returns.std()
            if sharpe_std_excess_return is not None and sharpe_std_excess_return != 0:
                metrics["sharpe_ratio"] = (sharpe_avg_excess_return / sharpe_std_excess_return) * np.sqrt(self.annualization_factor)
        
            # Sortino Ratio
            downside_returns = excess_returns[excess_returns < 0]
            if not downside_returns.empty:
                downside_deviation = downside_returns.std()
                if downside_deviation is not None and downside_deviation != 0:
                    metrics["sortino_ratio"] = (sharpe_avg_excess_return / downside_deviation) * np.sqrt(self.annualization_factor)
    
        # Calmar Ratio
        if metrics["max_drawdown_pct"] is not None and metrics["max_drawdown_pct"] > 0:
            if self.equity_curve_df is not None and not self.equity_curve_df.empty:
                start_date = self.equity_curve_df.index.min()
                end_date = self.equity_curve_df.index.max()
                duration_years = (end_date - start_date).days / 365.25 if (end_date - start_date).days > 0 else 1.0/365.25
                total_return = (self.final_capital / self.initial_capital) - 1 if self.initial_capital > 0 else 0
                annualized_return = ((1 + total_return) ** (1 / duration_years)) - 1 if duration_years > 0 else total_return
                metrics["calmar_ratio"] = (annualized_return * 100) / metrics["max_drawdown_pct"]

        # Average Holding Period
        if not self.trades_df.empty and 'exit_timestamp' in self.trades_df.columns and 'entry_timestamp' in self.trades_df.columns:
            valid_trades_for_duration = self.trades_df.dropna(subset=['entry_timestamp', 'exit_timestamp'])
            if not valid_trades_for_duration.empty:
                holding_periods = (valid_trades_for_duration['exit_timestamp'] - valid_trades_for_duration['entry_timestamp'])
                metrics["average_holding_period_hours"] = holding_periods.mean().total_seconds() / 3600 if not holding_periods.empty else 0.0

        # Win/Loss Streaks
        if not pnl_series.empty:
            win_streak, loss_streak = 0, 0
            current_win_streak, current_loss_streak = 0, 0
            for pnl_val in pnl_series:
                if pnl_val > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                elif pnl_val < 0:
                    current_loss_streak += 1
                    current_win_streak = 0
                else: # Breakeven
                    current_win_streak = 0
                    current_loss_streak = 0
                win_streak = max(win_streak, current_win_streak)
                loss_streak = max(loss_streak, current_loss_streak)
            metrics["longest_win_streak"] = win_streak
            metrics["longest_loss_streak"] = loss_streak
        
        # Time in Market
        if self.equity_curve_df is not None and not self.equity_curve_df.empty:
            total_duration = self.equity_curve_df.index.max() - self.equity_curve_df.index.min()
            if total_duration.total_seconds() > 0:
                time_in_trades = timedelta(0)
                for _, trade in self.trades_df.iterrows():
                    if pd.notna(trade['exit_timestamp']):
                        time_in_trades += trade['exit_timestamp'] - trade['entry_timestamp']
                metrics["time_in_market_pct"] = (time_in_trades / total_duration) * 100

        # Turnover Rate
        # FIX: Use 'total_value_usd'
        if self.equity_curve_df is not None and not self.equity_curve_df.empty and len(self.equity_curve_df) > 1:
            total_traded_value = self.trades_df.apply(lambda x: abs(x['amount'] * x['entry_price']), axis=1).sum()
            total_traded_value += self.trades_df.apply(lambda x: abs(x['amount'] * x['exit_price']) if pd.notna(x['exit_price']) else 0, axis=1).sum()

            time_diffs = self.equity_curve_df.index.to_series().diff().dt.total_seconds().fillna(0)
            time_weighted_avg_equity = np.average(self.equity_curve_df['total_value_usd'], weights=time_diffs)
            
            if time_weighted_avg_equity > 0:
                metrics["turnover_rate"] = total_traded_value / time_weighted_avg_equity

        return metrics

    def print_summary(self, metrics: Optional[Dict[str, Any]] = None):
        if metrics is None:
            metrics = self.calculate_metrics()

        summary = f"""
        --------------------------------------------------
        |          Backtest Performance Summary          |
        --------------------------------------------------
        | Metric                       | Value               |
        --------------------------------------------------
        | Initial Capital              | ${metrics.get("initial_capital", 0):<15,.2f} |
        | Final Capital                | ${metrics.get("final_capital", 0):<15,.2f} |
        | Total Net Profit             | ${metrics.get("total_net_profit", 0):<15,.2f} |
        | Total Return                 | {metrics.get("total_return_pct", 0):<15.2f}% |
        | Total Trades                 | {metrics.get("total_trades", 0):<16} |
        | Winning Trades               | {metrics.get("winning_trades", 0):<16} |
        | Losing Trades                | {metrics.get("losing_trades", 0):<16} |
        | Breakeven Trades             | {metrics.get("breakeven_trades", 0):<16} |
        | Win Rate                     | {metrics.get("win_rate_pct", 0):<15.2f}% |
        | Loss Rate                    | {metrics.get("loss_rate_pct", 0):<15.2f}% |
        | Average PnL per Trade        | ${metrics.get("average_pnl_per_trade", 0):<15,.2f} |
        | Average Win PnL              | ${metrics.get("average_win_pnl", 0):<15,.2f} |
        | Average Loss PnL             | ${metrics.get("average_loss_pnl", 0):<15,.2f} |
        | Profit Factor                | {metrics.get("profit_factor", float('nan')):<16.2f} |
        | Max Drawdown                 | {metrics.get("max_drawdown_pct", 0):<15.2f}% |
        | Sharpe Ratio                 | {metrics.get("sharpe_ratio", float('nan')):<16.2f} |
        | Sortino Ratio                | {metrics.get("sortino_ratio", float('nan')):<16.2f} |
        | Calmar Ratio                 | {metrics.get("calmar_ratio", float('nan')):<16.2f} |
        | Avg Holding Period (hours)   | {metrics.get("average_holding_period_hours", 0):<16.2f} |
        | Longest Win Streak           | {metrics.get("longest_win_streak", 0):<16} |
        | Longest Loss Streak          | {metrics.get("longest_loss_streak", 0):<16} |
        | Time in Market               | {metrics.get("time_in_market_pct", 0):<15.2f}% |
        | Turnover Rate                | {metrics.get("turnover_rate", float('nan')):<16.2f} |
        | Total Fees Paid              | ${metrics.get("total_fees_paid", 0):<15,.2f} |
        --------------------------------------------------
        """
        print(summary)
        logger.info("Performance summary generated." + summary.replace("\n      |", "\n"))


    @staticmethod
    def calculate_deflated_sharpe_ratio(
        sharpe_ratios_series: pd.Series,
        num_bars_in_backtest: int,
        selected_sharpe: float
    ) -> Optional[float]:
        """
        Calculates the Deflated Sharpe Ratio (DSR) based on a series of Sharpe Ratios from multiple trials.
        This indicates the probability that the selected Sharpe Ratio is a false positive.
        Based on "The Deflated Sharpe Ratio" by Lopez de Prado.

        Args:
            sharpe_ratios_series (pd.Series): A series of Sharpe Ratios from an optimization run (e.g., grid search).
            num_bars_in_backtest (int): The number of observations (e.g., days, bars) in the backtest period.
            selected_sharpe (float): The Sharpe Ratio of the strategy selected from the trials.

        Returns:
            Optional[float]: The Deflated Sharpe Ratio, or None if calculation fails.
        """
        if sharpe_ratios_series.empty or num_bars_in_backtest < 30:
            logger.warning("DSR calculation requires a series of Sharpe Ratios and a sufficient number of backtest bars.")
            return None

        n_trials = len(sharpe_ratios_series)
        var_sr = sharpe_ratios_series.var()

        if pd.isna(var_sr) or var_sr == 0:
            logger.warning("Variance of Sharpe Ratios is zero or NaN. Cannot calculate DSR.")
            return None
        
        # Expected maximum Sharpe Ratio approximation
        euler_mascheroni = 0.5772156649
        e_max_sr = ( (1 - euler_mascheroni) * norm.ppf(1 - 1/n_trials) ) + \
                   ( euler_mascheroni * norm.ppf(1 - 1/(n_trials * np.e)) )
        
        # Deflated Sharpe Ratio Calculation
        # DSR = P[SR_real <= SR_selected | N trials] = CDF of Z score
        try:
            # The Z-score compares the selected SR to the expected maximum SR from random trials
            z_score = (selected_sharpe - (e_max_sr * np.sqrt(var_sr))) * \
                      np.sqrt(num_bars_in_backtest - 1)
            
            deflated_sharpe = norm.cdf(z_score)
            logger.info(f"DSR Calculation: N_trials={n_trials}, Var(SR)={var_sr:.4f}, E[MaxSR]={e_max_sr:.4f}, SelectedSR={selected_sharpe:.4f}, Z-Score={z_score:.4f}")
            return deflated_sharpe
        except Exception as e:
            logger.error(f"Error calculating Deflated Sharpe Ratio: {e}", exc_info=True)
            return None