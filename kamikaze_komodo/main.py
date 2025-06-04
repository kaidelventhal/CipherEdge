# kamikaze_komodo/main.py
import asyncio
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from kamikaze_komodo.app_logger import get_logger, logger as root_logger
from kamikaze_komodo.config.settings import settings

# Phase 1 & 2 imports (already present)
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.exchange_interaction.exchange_api import ExchangeAPI
from kamikaze_komodo.strategy_framework.strategy_manager import StrategyManager
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.core.models import BarData, NewsArticle # Added NewsArticle
from kamikaze_komodo.core.enums import SignalType # For strategy checking

# Phase 3 imports
from kamikaze_komodo.risk_control_module.position_sizer import FixedFractionalPositionSizer, ATRBasedPositionSizer
from kamikaze_komodo.risk_control_module.stop_manager import PercentageStopManager, ATRStopManager
# Portfolio constructor components are more for multi-asset; for single-asset backtest, they are less directly demonstrated here.

# Phase 4 imports
from kamikaze_komodo.ai_news_analysis_agent_module.news_scraper import NewsScraper
from kamikaze_komodo.ai_news_analysis_agent_module.sentiment_analyzer import SentimentAnalyzer
from kamikaze_komodo.ai_news_analysis_agent_module.browser_agent import BrowserAgent # Optional advanced feature
from kamikaze_komodo.ai_news_analysis_agent_module.notification_listener import NotificationListener, dummy_notification_callback


logger = get_logger(__name__)

async def run_phase1_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 1: Core Infrastructure & Data")
    if not settings: root_logger.critical("Settings failed to load."); return

    db_manager = DatabaseManager()
    data_fetcher = DataFetcher() # Uses exchange_id from settings
    exchange_api = ExchangeAPI() # Uses exchange_id from settings

    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    start_period = datetime.now(timezone.utc) - timedelta(days=settings.historical_data_days if settings.historical_data_days else 30)
    end_period = datetime.now(timezone.utc)

    historical_data = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_period, end_period)
    if historical_data:
        logger.info(f"Fetched {len(historical_data)} bars for {symbol} ({timeframe}).")
        db_manager.store_bar_data(historical_data)
        retrieved_data = db_manager.retrieve_bar_data(symbol, timeframe, start_date=start_period, end_date=end_period)
        logger.info(f"Retrieved {len(retrieved_data)} bars from DB for {symbol} ({timeframe}). First bar: {retrieved_data[0].timestamp if retrieved_data else 'N/A'}")
    else:
        logger.warning(f"No historical data fetched for {symbol}.")

    balance = await exchange_api.fetch_balance()
    if balance:
        base_currency = symbol.split('/')[0] if '/' in symbol else symbol.split(':')[0] if ':' in symbol else "N/A"
        quote_currency = symbol.split('/')[1] if '/' in symbol else "USD" # Assuming USD or similar
        logger.info(f"Fetched balance. Free {base_currency}: {balance.get(base_currency, {}).get('free', 'N/A')}, Free {quote_currency}: {balance.get(quote_currency, {}).get('free', 'N/A')}")
    else:
        logger.warning("Could not fetch account balance.")

    await data_fetcher.close()
    await exchange_api.close()
    db_manager.close()
    root_logger.info("Phase 1 Demonstration completed.")


async def run_phase2_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 2: Basic Strategy & Backtesting")
    if not settings: root_logger.critical("Settings failed to load."); return

    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()

    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days if settings.historical_data_days > 0 else 365
    start_date = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date = datetime.now(timezone.utc)

    historical_bars: List[BarData] = db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)
    required_bars_for_strategy = settings.ewmac_long_window + 5 # Example buffer

    if not historical_bars or len(historical_bars) < required_bars_for_strategy:
        logger.info(f"Insufficient data in DB for {symbol}. Fetching fresh for {hist_days} days...")
        historical_bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
        if historical_bars: db_manager.store_bar_data(historical_bars)
        else: logger.error(f"Failed to fetch sufficient data for {symbol}. Backtest cannot proceed."); await data_fetcher.close(); db_manager.close(); return

    await data_fetcher.close()
    db_manager.close()

    if not historical_bars or len(historical_bars) < required_bars_for_strategy:
        logger.error(f"Still not enough historical data for {symbol} ({len(historical_bars)} bars) after fetch attempt. Backtest aborted."); return
    logger.info(f"Using {len(historical_bars)} bars for backtesting {symbol}.")

    data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)

    if data_df.empty or len(data_df) < max(settings.ewmac_short_window, settings.ewmac_long_window):
        logger.error(f"DataFrame conversion error or insufficient points for EWMAC ({len(data_df)})."); return

    ewmac_params = settings.get_strategy_params("EWMAC") # Gets params from [EWMAC_Strategy]
    ewmac_strategy = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params)

    initial_capital = 10000.00
    commission_bps = settings.commission_bps
    backtest_engine = BacktestingEngine(data_feed_df=data_df, strategy=ewmac_strategy, initial_capital=initial_capital, commission_bps=commission_bps)

    logger.info(f"Running basic backtest for EWMAC on {symbol}...")
    trades_log, final_portfolio = backtest_engine.run()

    if trades_log:
        logger.info(f"Backtest (Phase 2) completed. Generated {len(trades_log)} trades.")
        performance_analyzer = PerformanceAnalyzer(trades=trades_log, initial_capital=initial_capital, final_capital=final_portfolio['final_portfolio_value'])
        metrics = performance_analyzer.calculate_metrics()
        performance_analyzer.print_summary(metrics)
    else:
        logger.info(f"Backtest (Phase 2) completed. No trades executed. Final portfolio value: ${final_portfolio['final_portfolio_value']:.2f}")
    root_logger.info("Phase 2 Demonstration completed.")

async def run_phase3_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 3: Risk Management Integration")
    if not settings: root_logger.critical("Settings failed to load."); return

    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()
    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days if settings.historical_data_days > 0 else 365
    start_date = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date = datetime.now(timezone.utc)

    historical_bars = db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)
    # Strategy itself will define ATR period in its params
    ewmac_temp_params = settings.get_strategy_params("EWMAC")
    atr_period_for_min_bars = int(ewmac_temp_params.get("atr_period", 14))
    min_bars_needed = (settings.ewmac_long_window + atr_period_for_min_bars + 5) # For EMA + ATR + buffer

    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.info(f"Fetching fresh data for Phase 3 backtest (need ~{min_bars_needed} bars)...")
        historical_bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
        if historical_bars: db_manager.store_bar_data(historical_bars)
        else: logger.error(f"Failed to fetch data for {symbol}. Aborting."); await data_fetcher.close(); db_manager.close(); return

    await data_fetcher.close(); db_manager.close()
    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.error(f"Not enough data ({len(historical_bars)} bars) for Phase 3 backtest. Aborting."); return

    data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)
    # Note: EWMACStrategy will calculate ATR internally.

    ewmac_params = settings.get_strategy_params("EWMAC")
    # Ensure ATR period is in params for the strategy to use it
    if 'atr_period' not in ewmac_params: ewmac_params['atr_period'] = 14
    ewmac_strategy = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params)

    # Position Sizer (from config)
    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer = ATRBasedPositionSizer(
            risk_per_trade_fraction=settings.atr_based_risk_per_trade_fraction,
            atr_multiple_for_stop=settings.atr_based_atr_multiple_for_stop
        )
    else: # Default to FixedFractional
        position_sizer = FixedFractionalPositionSizer(fraction=settings.fixed_fractional_allocation_fraction)
    logger.info(f"Using Position Sizer: {position_sizer.__class__.__name__}")

    # Stop Manager (from config)
    if settings.stop_manager_type.lower() == 'atrbased':
        stop_manager = ATRStopManager(atr_multiple=settings.atr_stop_atr_multiple)
    else: # Default to PercentageBased
        stop_manager = PercentageStopManager(
            stop_loss_pct=settings.percentage_stop_loss_pct,
            take_profit_pct=settings.percentage_stop_take_profit_pct
        )
    logger.info(f"Using Stop Manager: {stop_manager.__class__.__name__}")

    initial_capital = 10000.00
    commission_bps = settings.commission_bps
    backtest_engine = BacktestingEngine(
        data_feed_df=data_df,
        strategy=ewmac_strategy,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        position_sizer=position_sizer,
        stop_manager=stop_manager
    )

    logger.info(f"Running Phase 3 backtest (EWMAC with Risk Management) on {symbol}...")
    trades_log, final_portfolio = backtest_engine.run()

    if trades_log:
        logger.info(f"Backtest (Phase 3) completed. Generated {len(trades_log)} trades.")
        performance_analyzer = PerformanceAnalyzer(trades=trades_log, initial_capital=initial_capital, final_capital=final_portfolio['final_portfolio_value'])
        metrics = performance_analyzer.calculate_metrics()
        performance_analyzer.print_summary(metrics)
    else:
        logger.info(f"Backtest (Phase 3) completed. No trades executed. Final portfolio: ${final_portfolio['final_portfolio_value']:.2f}")
    root_logger.info("Phase 3 Demonstration completed.")


async def run_phase4_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 4: AI News & Sentiment Integration")
    if not settings: root_logger.critical("Settings failed to load."); return

    # ... (News Scraping, BrowserAgent, NotificationListener sections remain the same) ...
    # Ensure those sections are complete as in the previous response if you copy this one.

    # 4. Backtesting with Simulated Sentiment Data
    logger.info("Proceeding to backtest with sentiment integration...")
    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()
    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days if settings.historical_data_days > 0 else 90 
    start_date = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date = datetime.now(timezone.utc)

    historical_bars = db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)
    ewmac_temp_params_sentiment = settings.get_strategy_params("EWMAC")
    atr_period_for_min_bars_sentiment = int(ewmac_temp_params_sentiment.get("atr_period", 14))
    min_bars_needed_sentiment = (settings.ewmac_long_window + atr_period_for_min_bars_sentiment + 5)

    if not historical_bars or len(historical_bars) < min_bars_needed_sentiment:
        logger.info(f"Fetching fresh data for Phase 4 backtest (need ~{min_bars_needed_sentiment} bars)...")
        historical_bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
        if historical_bars: db_manager.store_bar_data(historical_bars)
        else: logger.error(f"Failed to fetch data for {symbol}. Aborting Phase 4 backtest."); await data_fetcher.close(); db_manager.close(); return

    await data_fetcher.close(); db_manager.close()
    if not historical_bars or len(historical_bars) < min_bars_needed_sentiment:
        logger.error(f"Not enough data ({len(historical_bars)} bars) for Phase 4 backtest. Aborting."); return

    data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)

    # Load Simulated Sentiment Data
    sentiment_df: Optional[pd.DataFrame] = None
    if settings.simulated_sentiment_data_path and settings.enable_sentiment_analysis:
        sentiment_csv_path = settings.simulated_sentiment_data_path
        logger.info(f"Attempting to load simulated sentiment data from: {sentiment_csv_path}")
        if os.path.exists(sentiment_csv_path):
            try:
                sentiment_df = pd.read_csv(sentiment_csv_path, parse_dates=['timestamp'], index_col='timestamp')
                if sentiment_df.empty:
                    logger.warning(f"Simulated sentiment data file is empty: {sentiment_csv_path}")
                    sentiment_df = None
                else:
                    if sentiment_df.index.tz is None:
                        sentiment_df.index = sentiment_df.index.tz_localize('UTC')
                    else:
                        sentiment_df.index = sentiment_df.index.tz_convert('UTC')
                    
                    if 'sentiment_score' not in sentiment_df.columns:
                        logger.error(f"'sentiment_score' column not found in {sentiment_csv_path}. Sentiment will not be used.")
                        sentiment_df = None
                    else:
                        logger.info(f"Successfully loaded simulated sentiment data from: {sentiment_csv_path} with {len(sentiment_df)} entries.")
                        logger.info(f"First 5 sentiment scores:\n{sentiment_df['sentiment_score'].head().to_string()}")
                        logger.info(f"Last 5 sentiment scores:\n{sentiment_df['sentiment_score'].tail().to_string()}")
                        logger.info(f"Sentiment data timespan: {sentiment_df.index.min()} to {sentiment_df.index.max()}")
            except Exception as e_csv:
                logger.error(f"Error loading simulated sentiment data from {sentiment_csv_path}: {e_csv}. Proceeding without sentiment.", exc_info=True)
                sentiment_df = None
        else:
            logger.warning(f"Simulated sentiment data file NOT FOUND at: {sentiment_csv_path}. Proceeding without external sentiment.")
    elif not settings.enable_sentiment_analysis:
        logger.info("Sentiment analysis is disabled in settings. Backtest will not use external sentiment data.")
    else:
        logger.info("No simulated sentiment data path provided in settings. Proceeding without external sentiment.")

    ewmac_params_sentiment = settings.get_strategy_params("EWMAC")
    if 'sentiment_filter_long_threshold' not in ewmac_params_sentiment:
        ewmac_params_sentiment['sentiment_filter_long_threshold'] = settings.sentiment_filter_threshold_long
    if 'sentiment_filter_short_threshold' not in ewmac_params_sentiment:
        ewmac_params_sentiment['sentiment_filter_short_threshold'] = settings.sentiment_filter_threshold_short
    if 'atr_period' not in ewmac_params_sentiment: ewmac_params_sentiment['atr_period'] = 14

    ewmac_strategy_sentiment = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params_sentiment)

    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer_sent = ATRBasedPositionSizer(settings.atr_based_risk_per_trade_fraction, settings.atr_based_atr_multiple_for_stop)
    else:
        position_sizer_sent = FixedFractionalPositionSizer(settings.fixed_fractional_allocation_fraction)

    if settings.stop_manager_type.lower() == 'atrbased':
        stop_manager_sent = ATRStopManager(settings.atr_stop_atr_multiple)
    else:
        stop_manager_sent = PercentageStopManager(settings.percentage_stop_loss_pct, settings.percentage_stop_take_profit_pct)

    initial_capital = 10000.00
    commission_bps = settings.commission_bps
    backtest_engine_sentiment = BacktestingEngine(
        data_feed_df=data_df,
        strategy=ewmac_strategy_sentiment,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        position_sizer=position_sizer_sent,
        stop_manager=stop_manager_sent,
        sentiment_data_df=sentiment_df 
    )

    logger.info(f"Running Phase 4 backtest (EWMAC with Sentiment & Risk Mgmt) on {symbol}...")
    trades_log, final_portfolio = backtest_engine_sentiment.run()

    if trades_log:
        logger.info(f"Backtest (Phase 4) completed. Generated {len(trades_log)} trades.")
        performance_analyzer = PerformanceAnalyzer(trades=trades_log, initial_capital=initial_capital, final_capital=final_portfolio['final_portfolio_value'])
        metrics = performance_analyzer.calculate_metrics()
        performance_analyzer.print_summary(metrics)
    else:
        logger.info(f"Backtest (Phase 4) completed. No trades executed. Final portfolio: ${final_portfolio['final_portfolio_value']:.2f}")
    root_logger.info("Phase 4 Demonstration completed.")


async def main():
    root_logger.info("Kamikaze Komodo Program Starting...")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot start.")
        return

    # --- Select phases to run ---
    # await run_phase1_demonstration()
    # await run_phase2_demonstration()
    await run_phase3_demonstration()
    await run_phase4_demonstration()
    # For Phase 4, ensure Ollama is running if live sentiment analysis is part of it,
    # and 'data/simulated_sentiment_data.csv' exists (relative to kamikaze_komodo/) for backtesting.
    # Example simulated_sentiment_data.csv in kamikaze_komodo/data/:
    # timestamp,sentiment_score
    # 2023-01-01T00:00:00Z,0.5
    # 2023-01-01T01:00:00Z,-0.2

    root_logger.info("Kamikaze Komodo Program Finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        root_logger.info("Kamikaze Komodo program terminated by user.")
    except Exception as e:
        root_logger.critical(f"Critical error in main execution: {e}", exc_info=True)