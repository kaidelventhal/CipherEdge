# kamikaze_komodo/backtesting_engine/performance_analyzer.py

from datetime import datetime
import pandas as pd

import numpy as np

from typing import List, Dict, Any, Optional

from kamikaze_komodo.core.models import Trade

from kamikaze_komodo.core.enums import OrderSide, TradeResult

from kamikaze_komodo.app_logger import get_logger


logger = get_logger(__name__)


class PerformanceAnalyzer:

    """

    Calculates and analyzes performance metrics from a list of trades.

    """

    def __init__(self, trades: List[Trade], initial_capital: float, final_capital: float):

        if not trades:

            logger.warning("PerformanceAnalyzer initialized with no trades. Some metrics might be zero or NaN.")

        self.trades = pd.DataFrame([trade.model_dump() for trade in trades])

        if not self.trades.empty:

            self.trades['entry_timestamp'] = pd.to_datetime(self.trades['entry_timestamp'])

            self.trades['exit_timestamp'] = pd.to_datetime(self.trades['exit_timestamp'])

        

        self.initial_capital = initial_capital

        self.final_capital = final_capital # This should be the final total portfolio value

        logger.info(f"PerformanceAnalyzer initialized with {len(trades)} trades. Initial: ${initial_capital:,.2f}, Final: ${final_capital:,.2f}")



    def get_pnl_series(self) -> pd.Series:

        """Returns a Series of PnL for each trade."""

        if self.trades.empty or 'pnl' not in self.trades.columns:

            return pd.Series(dtype=float)

        return self.trades['pnl'].dropna()


    def get_equity_curve(self) -> pd.Series:

        """Generates an equity curve based on trade PnLs."""

        if self.trades.empty or 'pnl' not in self.trades.columns:

            equity = pd.Series([self.initial_capital], index=[pd.Timestamp(0)]) # Placeholder

            equity.name = "Equity"

            return equity

            

        # Ensure PnL is numeric and trades are sorted by exit time for accurate curve

        pnl_series = self.trades.set_index('exit_timestamp')['pnl'].dropna().astype(float)

        if pnl_series.empty:

             equity = pd.Series([self.initial_capital], index=[pd.Timestamp(0)])

             equity.name = "Equity"

             return equity


        cumulative_pnl = pnl_series.cumsum()

        equity = self.initial_capital + cumulative_pnl

        

        # Add initial capital point

        # Find the earliest entry time, or use a synthetic start time if no trades

        start_time = self.trades['entry_timestamp'].min() if not self.trades.empty else pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)

        # Ensure start_time is before the first trade's exit_timestamp for proper plotting.

        # If pnl_series index is not empty, make sure start_time is before its first element.

        if not pnl_series.empty and start_time > pnl_series.index[0]:

            start_time = pnl_series.index[0] - pd.Timedelta(seconds=1)


        equity = pd.concat([pd.Series([self.initial_capital], index=[start_time]), equity])

        equity.name = "Equity"

        return equity



    def calculate_metrics(self) -> Dict[str, Any]:

        """

        Calculates a comprehensive suite of performance metrics.

        """

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

            "profit_factor": np.nan, # Gross Profit / Gross Loss

            "max_drawdown_pct": 0.0, # Needs equity curve

            "sharpe_ratio": np.nan, # Needs daily/periodic returns & risk-free rate

            "sortino_ratio": np.nan, # Needs daily/periodic returns & risk-free rate & downside deviation

            "total_fees_paid": 0.0,

        }


        if self.trades.empty:

            metrics["total_net_profit"] = self.final_capital - self.initial_capital

            if self.initial_capital > 0:

                 metrics["total_return_pct"] = (metrics["total_net_profit"] / self.initial_capital) * 100

            logger.warning("No trades to analyze. Returning basic capital metrics.")

            return metrics


        # PnL calculations

        pnl_series = self.get_pnl_series()

        if pnl_series.empty and not self.trades.empty: # PnL might be all NaN if trades didn't close

            logger.warning("PnL series is empty or all NaN, cannot calculate detailed metrics.")

            metrics["total_net_profit"] = self.final_capital - self.initial_capital

            if self.initial_capital > 0:

                 metrics["total_return_pct"] = (metrics["total_net_profit"] / self.initial_capital) * 100

            metrics["total_trades"] = len(self.trades)

            metrics["total_fees_paid"] = self.trades['commission'].sum() if 'commission' in self.trades.columns else 0.0

            return metrics


        metrics["total_net_profit"] = pnl_series.sum()

        if self.initial_capital > 0: # Avoid division by zero

            metrics["total_return_pct"] = (metrics["total_net_profit"] / self.initial_capital) * 100

        

        metrics["total_trades"] = len(pnl_series)

        if metrics["total_trades"] == 0: # No closed trades with PnL

             metrics["total_fees_paid"] = self.trades['commission'].sum() if 'commission' in self.trades.columns else 0.0

             return metrics



        # Win/Loss Analysis

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


        if not wins.empty:

            metrics["average_win_pnl"] = wins.mean()

        if not losses.empty:

            metrics["average_loss_pnl"] = losses.mean() # This will be negative


        # Profit Factor

        gross_profit = wins.sum()

        gross_loss = abs(losses.sum()) # Absolute sum of losses

        if gross_loss > 0:

            metrics["profit_factor"] = gross_profit / gross_loss

        elif gross_profit > 0 and gross_loss == 0 : # All wins, no losses

            metrics["profit_factor"] = np.inf



        # Max Drawdown (from equity curve)

        equity_curve = self.get_equity_curve()

        if not equity_curve.empty and len(equity_curve) > 1:

            peak = equity_curve.expanding(min_periods=1).max()

            drawdown = (equity_curve - peak) / peak

            metrics["max_drawdown_pct"] = abs(drawdown.min()) * 100

        

        # Sharpe and Sortino Ratios (Simplified: assumes daily returns if data allows)

        # These require more complex calculations of periodic returns and risk-free rate.

        # For a basic version, we can skip or use a simplified placeholder if PnL series represents portfolio value changes.

        # daily_returns = equity_curve.pct_change().dropna() # if equity curve represents daily values

        # if not daily_returns.empty and len(daily_returns) > 1:

        #     # Assuming risk-free rate of 0 for simplicity

        #     sharpe_avg_return = daily_returns.mean()

        #     sharpe_std_return = daily_returns.std()

        #     if sharpe_std_return != 0:

        #         metrics["sharpe_ratio"] = (sharpe_avg_return / sharpe_std_return) * np.sqrt(252) # Annualized (approx for crypto: 365)

            

        #     downside_returns = daily_returns[daily_returns < 0]

        #     if not downside_returns.empty:

        #         downside_std = downside_returns.std()

        #         if downside_std != 0:

        #             metrics["sortino_ratio"] = (sharpe_avg_return / downside_std) * np.sqrt(252) # Annualized


        metrics["total_fees_paid"] = self.trades['commission'].sum() if 'commission' in self.trades.columns else 0.0


        return metrics


    def print_summary(self, metrics: Optional[Dict[str, Any]] = None):

        """Prints a summary of the performance metrics."""

        if metrics is None:

            metrics = self.calculate_metrics()


        summary = f"""

        --------------------------------------------------

        |              Backtest Performance Summary      |

        --------------------------------------------------

        | Metric                      | Value            |

        --------------------------------------------------

        | Initial Capital             | ${metrics.get("initial_capital", 0):<15,.2f} |

        | Final Capital               | ${metrics.get("final_capital", 0):<15,.2f} |

        | Total Net Profit            | ${metrics.get("total_net_profit", 0):<15,.2f} |

        | Total Return                | {metrics.get("total_return_pct", 0):<15.2f}% |

        | Total Trades                | {metrics.get("total_trades", 0):<16} |

        | Winning Trades              | {metrics.get("winning_trades", 0):<16} |

        | Losing Trades               | {metrics.get("losing_trades", 0):<16} |

        | Breakeven Trades            | {metrics.get("breakeven_trades", 0):<16} |

        | Win Rate                    | {metrics.get("win_rate_pct", 0):<15.2f}% |

        | Loss Rate                   | {metrics.get("loss_rate_pct", 0):<15.2f}% |

        | Avg PnL per Trade           | ${metrics.get("average_pnl_per_trade", 0):<15,.2f} |

        | Avg Win PnL                 | ${metrics.get("average_win_pnl", 0):<15,.2f} |

        | Avg Loss PnL                | ${metrics.get("average_loss_pnl", 0):<15,.2f} |

        | Profit Factor               | {metrics.get("profit_factor", float('nan')):<16.2f} |

        | Max Drawdown                | {metrics.get("max_drawdown_pct", 0):<15.2f}% |

        | Total Fees Paid             | ${metrics.get("total_fees_paid", 0):<15,.2f} |

        | Sharpe Ratio (approx)       | {metrics.get("sharpe_ratio", float('nan')):<16.2f} |

        | Sortino Ratio (approx)      | {metrics.get("sortino_ratio", float('nan')):<16.2f} |

        --------------------------------------------------

        """

        print(summary)

        logger.info("Performance summary generated." + summary.replace("\n        |", "\n")) # Loggable format



# Example Usage:

if __name__ == '__main__':

    # Create some dummy trade data

    dummy_trades_data = [

        Trade(id="t1", symbol="BTC/USD", entry_order_id="e1", exit_order_id="ex1", side=OrderSide.BUY,

              entry_price=30000, exit_price=31000, amount=1, entry_timestamp=datetime(2023,1,1,10), exit_timestamp=datetime(2023,1,1,12),

              pnl=980, pnl_percentage=(1000/30000 - 0.002)*100, commission=20, result=TradeResult.WIN),

        Trade(id="t2", symbol="BTC/USD", entry_order_id="e2", exit_order_id="ex2", side=OrderSide.BUY,

              entry_price=31500, exit_price=31000, amount=1, entry_timestamp=datetime(2023,1,2,10), exit_timestamp=datetime(2023,1,2,15),

              pnl=-520, pnl_percentage=(-500/31500 - 0.002)*100, commission=20, result=TradeResult.LOSS),

        Trade(id="t3", symbol="BTC/USD", entry_order_id="e3", exit_order_id="ex3", side=OrderSide.BUY,

              entry_price=32000, exit_price=33000, amount=0.5, entry_timestamp=datetime(2023,1,3,10), exit_timestamp=datetime(2023,1,3,18),

              pnl=485, pnl_percentage=(1000/32000 * 0.5 - 0.002)*100, commission=15, result=TradeResult.WIN), # PnL adjusted for 0.5 amount logic

    ]

    # Correcting PnL for T3: (33000-32000)*0.5 = 500.  Net PnL = 500 - 15 (commission) = 485

    # Pct: ((33000/32000)-1)*100 = 3.125%. After commission approx.


    initial_cap = 100000

    final_cap = initial_cap + (980 - 520 + 485) # 100000 + 945 = 100945


    analyzer = PerformanceAnalyzer(trades=dummy_trades_data, initial_capital=initial_cap, final_capital=final_cap)

    metrics_calculated = analyzer.calculate_metrics()

    analyzer.print_summary(metrics_calculated)


    # equity = analyzer.get_equity_curve()

    # print("\nEquity Curve:")

    # print(equity)