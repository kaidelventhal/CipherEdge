# FILE: kamikaze_komodo/main.py
import asyncio
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

from kamikaze_komodo.app_logger import get_logger, logger as root_logger
from kamikaze_komodo.config.settings import settings
from kamikaze_komodo.orchestration.scheduler import TaskScheduler
from kamikaze_komodo.orchestration.portfolio_manager import PortfolioManager
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher

logger = get_logger(__name__)

async def run_portfolio_backtest():
    """
    Runs a backtest for the entire portfolio as defined in the configuration.
    """
    root_logger.info("--- Starting Portfolio Backtest ---")
    if not settings:
        root_logger.critical("Settings not loaded. Cannot run backtest.")
        return

    # 1. Load Sentiment Data (if configured)
    sentiment_df: Optional[pd.DataFrame] = None
    if settings.simulated_sentiment_data_path and settings.enable_sentiment_analysis:
        sentiment_csv_path = settings.simulated_sentiment_data_path
        logger.info(f"Attempting to load simulated sentiment data from: {sentiment_csv_path}")
        if os.path.exists(sentiment_csv_path):
            try:
                sentiment_df = pd.read_csv(sentiment_csv_path, parse_dates=['timestamp'], index_col='timestamp')
                if sentiment_df.index.tz is None:
                    sentiment_df.index = sentiment_df.index.tz_localize('UTC')
                else:
                    sentiment_df.index = sentiment_df.index.tz_convert('UTC')
                
                if 'sentiment_score' not in sentiment_df.columns:
                    logger.error(f"'sentiment_score' column not found in {sentiment_csv_path}. Sentiment will not be used.")
                    sentiment_df = None
                else:
                    logger.info(f"Successfully loaded {len(sentiment_df)} simulated sentiment entries.")
            except Exception as e:
                logger.error(f"Error loading simulated sentiment data from {sentiment_csv_path}: {e}", exc_info=True)
                sentiment_df = None
        else:
            logger.warning(f"Simulated sentiment data file NOT FOUND at: {sentiment_csv_path}. Proceeding without sentiment.")


    # 2. Load Market Data for all assets in the universe
    portfolio_config = settings.get_strategy_params('Portfolio')
    trading_universe: List[str] = [s.strip() for s in portfolio_config.get('tradinguniverse', '').split(',')]
    timeframe = settings.default_timeframe
    
    if not trading_universe:
        logger.error("TradingUniverse is not defined in the [Portfolio] section of config.ini.")
        return

    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()
    data_feeds: Dict[str, pd.DataFrame] = {}
    start_date = datetime.now(timezone.utc) - timedelta(days=settings.historical_data_days)

    for asset in trading_universe:
        logger.info(f"Loading data for {asset}...")
        bars = db_manager.retrieve_bar_data(asset, timeframe, start_date=start_date)
        if not bars or len(bars) < 200:
            logger.info(f"Fetching fresh data for {asset}...")
            bars = await data_fetcher.fetch_historical_data_for_period(asset, timeframe, start_date)
            if bars:
                db_manager.store_bar_data(bars)
        
        if bars:
            df = pd.DataFrame([bar.model_dump() for bar in bars])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # --- FIX: Replace concat/reindex with a more robust reindex/ffill approach ---
            if sentiment_df is not None and not sentiment_df.empty:
                # Ensure the sentiment index is unique by taking the last entry for any duplicate timestamp.
                if not sentiment_df.index.is_unique:
                    logger.warning(f"Duplicate timestamps found in sentiment data. Keeping last entry for each.")
                    sentiment_df = sentiment_df[~sentiment_df.index.duplicated(keep='last')]

                # Create a new sentiment index aligned to the market data, forward-filling values
                aligned_sentiment = sentiment_df.reindex(df.index, method='ffill')
                df['sentiment_score'] = aligned_sentiment['sentiment_score']
                logger.debug(f"Merged sentiment data into data feed for {asset}.")

            # Ensure sentiment_score column exists and fill any remaining NaNs (e.g., at the very start)
            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = 0.0
            else:
                df['sentiment_score'] = df['sentiment_score'].fillna(0.0)

            data_feeds[asset] = df
            logger.info(f"Loaded {len(df)} bars for {asset}.")
        else:
            logger.error(f"Could not load data for asset {asset}. It will be excluded from the backtest.")

    await data_fetcher.close()
    db_manager.close()

    if not data_feeds:
        logger.error("No data loaded for any asset in the universe. Aborting backtest.")
        return

    # 3. Initialize and run the backtesting engine
    backtest_engine = BacktestingEngine(
        data_feeds=data_feeds,
        portfolio_manager_class=PortfolioManager,
        initial_capital=10000.0, # Example value
        commission_bps=float(settings.config.get('Trading', 'CommissionBPS', fallback=0.0)),
        slippage_bps=float(settings.config.get('Trading', 'SlippageBPS', fallback=0.0)),
    )
    
    trades_log, final_portfolio, equity_curve_df = await backtest_engine.run()

    # 4. Analyze and print performance
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

async def main():
    """
    Main entry point for the Kamikaze Komodo trading bot.
    Initializes and runs the PortfolioManager based on the configured schedule.
    """
    root_logger.info(">>> Kamikaze Komodo Program Starting <<<")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot start.")
        return
        
    run_mode = settings.config.get('General', 'RunMode', fallback='backtest').lower()

    if run_mode == 'backtest':
        await run_portfolio_backtest()
    elif run_mode == 'live':
        logger.info("Initializing for LIVE TRADING...")
        # Placeholder for live trading logic using the scheduler
        # portfolio_manager = PortfolioManager()
        # scheduler_manager = TaskScheduler()
        #
        # # Schedule the portfolio manager's main loop
        # # The interval should match the strategy's timeframe
        # scheduler_manager.add_job(portfolio_manager.run_cycle, 'interval', minutes=240, id='portfolio_run_cycle')
        #
        # try:
        #     scheduler_manager.start()
        #     # Keep the application running
        #     while True:
        #         await asyncio.sleep(60)
        # except (KeyboardInterrupt, SystemExit):
        #     logger.info("Shutting down live trading...")
        #     await portfolio_manager.close()
        #     scheduler_manager.shutdown()
        logger.warning("Live trading mode is not fully implemented in Phase 7, Priority 1. Exiting.")
    else:
        logger.error(f"Unknown RunMode '{run_mode}' in config.ini. Use 'live' or 'backtest'.")


    root_logger.info(">>> Kamikaze Komodo Program Finished <<<")

if __name__ == "__main__":
    try:
        if settings and not os.path.exists("logs"):
            os.makedirs("logs")
        
        # Add a RunMode to the config for easy switching
        if settings and not settings.config.has_option('General', 'RunMode'):
            root_logger.warning("No 'RunMode' found in [General] section of config.ini. Defaulting to 'backtest'.")
            root_logger.warning("Add 'RunMode = live' or 'RunMode = backtest' to your config.")

        asyncio.run(main())
    except KeyboardInterrupt:
        root_logger.info("Kamikaze Komodo program terminated by user.")
    except Exception as e:
        root_logger.critical(f"Critical error in main execution: {e}", exc_info=True)