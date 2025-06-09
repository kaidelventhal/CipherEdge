# FILE: kamikaze_komodo/main.py
import asyncio
import os
import pandas as pd
from typing import List, Dict, Any, Optional

from kamikaze_komodo.app_logger import get_logger, logger as root_logger
from kamikaze_komodo.config.settings import settings
from kamikaze_komodo.orchestration.scheduler import TaskScheduler
from kamikaze_komodo.orchestration.portfolio_manager import PortfolioManager
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer

logger = get_logger(__name__)

async def run_portfolio_backtest():
    """
    Initializes and runs the backtesting engine, then prints performance results.
    """
    root_logger.info("--- Initializing Portfolio Backtest ---")

    # First, create the PortfolioManager instance that will be tested.
    # By default, it loads its configuration from the settings file.
    portfolio_manager = PortfolioManager()

    # Use the asynchronous factory `create` to instantiate the engine
    backtest_engine = await BacktestingEngine.create(
        portfolio_manager=portfolio_manager,
        initial_capital=float(settings.config.get('Backtesting', 'InitialCapital', fallback=10000.0)),
        commission_bps=float(settings.config.get('Trading', 'CommissionBPS', fallback=0.0)),
        slippage_bps=float(settings.config.get('Trading', 'SlippageBPS', fallback=0.0)),
    )

    trades_log, final_portfolio, equity_curve_df = await backtest_engine.run()

    # Analyze and print performance
    logger.info("--- Portfolio Backtest Finished ---")
    if final_portfolio:
        logger.info(f"Initial Capital: ${final_portfolio.get('initial_capital', 0):,.2f}")
        logger.info(f"Final Portfolio Value: ${final_portfolio.get('final_portfolio_value', 0):,.2f}")

    if equity_curve_df is not None and not equity_curve_df.empty:
        logger.info("Equity curve generated. See logs or plotting output for details.")

        performance_analyzer = PerformanceAnalyzer(
            trades=trades_log,
            initial_capital=final_portfolio.get('initial_capital', 0),
            final_capital=final_portfolio.get('final_portfolio_value', 0),
            equity_curve_df=equity_curve_df,
            risk_free_rate_annual=float(settings.config.get('BacktestingPerformance', 'RiskFreeRateAnnual')),
            annualization_factor=int(settings.config.get('BacktestingPerformance', 'AnnualizationFactor'))
        )
        metrics = performance_analyzer.calculate_metrics()
        performance_analyzer.print_summary(metrics)

async def run_live_trading():
    """
    Initializes the PortfolioManager and TaskScheduler for live trading.
    """
    logger.info("--- Initializing for LIVE TRADING ---")

    portfolio_manager = PortfolioManager() # ExchangeAPI is initialized internally for live mode
    scheduler_manager = TaskScheduler()

    # Schedule the portfolio manager's main execution cycle
    # The interval should match the strategy's timeframe or desired frequency
    run_interval_minutes = int(settings.config.get('Scheduler', 'RunIntervalMinutes', fallback=240))
    logger.info(f"Scheduling portfolio manager to run every {run_interval_minutes} minutes.")

    scheduler_manager.add_job(
        portfolio_manager.run_cycle,
        'interval',
        minutes=run_interval_minutes,
        id='portfolio_run_cycle'
    )

    try:
        scheduler_manager.start()
        # Keep the application running
        logger.info("Live trading scheduler started. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down live trading...")
        # await portfolio_manager.close() # Implement close method if needed for cleanup
        scheduler_manager.shutdown()
    logger.info("Live trading shut down.")


async def main():
    """
    Main entry point for the Kamikaze Komodo trading bot.
    Initializes and runs the bot in the mode specified in the config.
    """
    root_logger.info(">>> Kamikaze Komodo Program Starting <<<")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot start.")
        return

    run_mode = settings.config.get('General', 'RunMode', fallback='backtest').lower()
    root_logger.info(f"Running in '{run_mode}' mode.")

    if run_mode == 'backtest':
        await run_portfolio_backtest()
    elif run_mode == 'live':
        await run_live_trading()
    else:
        logger.error(f"Unknown RunMode '{run_mode}' in config.ini. Use 'live' or 'backtest'.")

    root_logger.info(">>> Kamikaze Komodo Program Finished <<<")

if __name__ == "__main__":
    try:
        # Ensure log directory exists
        if not os.path.exists("logs"):
            os.makedirs("logs")

        asyncio.run(main())
    except KeyboardInterrupt:
        root_logger.info("Kamikaze Komodo program terminated by user.")
    except Exception as e:
        root_logger.critical(f"Critical error in main execution: {e}", exc_info=True)