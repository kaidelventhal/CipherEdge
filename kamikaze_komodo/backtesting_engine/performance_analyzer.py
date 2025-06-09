# FILE: kamikaze_komodo/backtesting_engine/performance_analyzer.py
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from kamikaze_komodo.core.models import Trade
from kamikaze_komodo.core.enums import OrderSide, TradeResult
from kamikaze_komodo.app_logger import get_logger
from collections import deque


logger = get_logger(__name__)

class PerformanceAnalyzer:
    def __init__(
        self,
        trades: List[Trade], # Expects a list of COMPLETED trades
        initial_capital: float,
        final_capital: float,
        equity_curve_df: Optional[pd.DataFrame] = None, # Timestamp-indexed 'total_value'
        risk_free_rate_annual: float = 0.02, # Annual risk-free rate (e.g., 2%)
        annualization_factor: int = 252 # Trading days in a year for Sharpe/Sortino
    ):
        if not trades:
            logger.warning("PerformanceAnalyzer initialized with no completed trades. Some metrics might be zero or NaN.")
        
        self.trades_df = pd.DataFrame([trade.model_dump() for trade in trades])
        if not self.trades_df.empty:
            self.trades_df['entry_timestamp'] = pd.to_datetime(self.trades_df['entry_timestamp'])
            self.trades_df['exit_timestamp'] = pd.to_datetime(self.trades_df['exit_timestamp'])
    
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.equity_curve_df = equity_curve_df
        self.risk_free_rate_annual = risk_free_rate_annual
        self.annualization_factor = annualization_factor
        
        logger.info(f"PerformanceAnalyzer initialized. Completed Trades: {len(trades)}, Initial: ${initial_capital:,.2f}, Final: ${final_capital:,.2f}")
        logger.info(f"Using Annual Risk-Free Rate: {self.risk_free_rate_annual*100:.2f}%, Annualization Factor: {self.annualization_factor}")

    def _calculate_periodic_returns(self) -> Optional[pd.Series]:
        if self.equity_curve_df is None or self.equity_curve_df.empty or 'total_value' not in self.equity_curve_df.columns:
            logger.warning("Equity curve data is missing or invalid. Cannot calculate periodic returns for Sharpe/Sortino.")
            return None
        # Resample to daily returns for annualization, handling potential non-unique index if multiple records per day
        daily_equity = self.equity_curve_df['total_value'].resample('D').last().ffill()
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
        }

        # Calculate metrics from the equity curve first, as they don't depend on trades
        metrics["total_net_profit"] = self.final_capital - self.initial_capital
        if self.initial_capital > 0:
            metrics["total_return_pct"] = (metrics["total_net_profit"] / self.initial_capital) * 100
        
        # Max Drawdown (from equity curve)
        if self.equity_curve_df is not None and not self.equity_curve_df.empty and 'total_value' in self.equity_curve_df.columns:
            # FIX: Clip equity at 0 to prevent drawdowns > 100%
            equity_values = self.equity_curve_df['total_value'].clip(lower=0)
            if len(equity_values) > 1:
                peak = equity_values.expanding(min_periods=1).max()
                # Ensure peak is not zero to avoid division by zero if equity starts at 0
                peak_safe = peak.replace(0, np.nan)
                drawdown = (equity_values - peak_safe) / peak_safe
                metrics["max_drawdown_pct"] = abs(drawdown.min()) * 100 if pd.notna(drawdown.min()) else 0.0

        if self.trades_df.empty:
            logger.warning("No completed trades to analyze. Returning portfolio-level metrics only.")
            return metrics

        pnl_series = self.trades_df['pnl'].dropna()
        if pnl_series.empty:
            logger.warning("No PnL data in trades to analyze. Returning portfolio-level metrics only.")
            return metrics
        
        # --- Trade-based metrics ---
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

        # --- Portfolio Ratios ---
        periodic_returns = self._calculate_periodic_returns()
        if periodic_returns is not None and len(periodic_returns) > 1:
            risk_free_rate_periodic = self.risk_free_rate_annual / self.annualization_factor
            excess_returns = periodic_returns - risk_free_rate_periodic
        
            # Sharpe Ratio
            sharpe_avg_excess_return = excess_returns.mean()
            sharpe_std_excess_return = excess_returns.std()
            if sharpe_std_excess_return is not None and sharpe_std_excess_return > 1e-9:
                metrics["sharpe_ratio"] = (sharpe_avg_excess_return / sharpe_std_excess_return) * np.sqrt(self.annualization_factor)
        
            # Sortino Ratio
            downside_returns = excess_returns[excess_returns < 0]
            if not downside_returns.empty:
                downside_deviation = downside_returns.std()
                if downside_deviation is not None and downside_deviation > 1e-9:
                    metrics["sortino_ratio"] = (sharpe_avg_excess_return / downside_deviation) * np.sqrt(self.annualization_factor)
    
        # Calmar Ratio
        if metrics["max_drawdown_pct"] is not None and metrics["max_drawdown_pct"] > 0:
            if self.equity_curve_df is not None and not self.equity_curve_df.empty:
                start_date = self.equity_curve_df.index.min()
                end_date = self.equity_curve_df.index.max()
                duration_years = (end_date - start_date).days / 365.25
                if duration_years > 0:
                    total_return = (self.final_capital / self.initial_capital) - 1 if self.initial_capital > 0 else 0
                    annualized_return = ((1 + total_return) ** (1 / duration_years)) - 1
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
                    current_win_streak += 1; current_loss_streak = 0
                elif pnl_val < 0:
                    current_loss_streak += 1; current_win_streak = 0
                else:
                    current_win_streak = 0; current_loss_streak = 0
                win_streak = max(win_streak, current_win_streak)
                loss_streak = max(loss_streak, current_loss_streak)
            metrics["longest_win_streak"] = win_streak
            metrics["longest_loss_streak"] = loss_streak

        return metrics

    def print_summary(self, metrics: Optional[Dict[str, Any]] = None):
        if metrics is None:
            metrics = self.calculate_metrics()

        summary = f"""
        --------------------------------------------------
        |              Backtest Performance Summary              |
        --------------------------------------------------
        | Metric                       | Value                 |
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
        | Total Fees Paid              | ${metrics.get("total_fees_paid", 0):<15,.2f} |
        --------------------------------------------------
        """
        print(summary)
        logger.info("Performance summary generated." + summary.replace("\n        |", "\n"))