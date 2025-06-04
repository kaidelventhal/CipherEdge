# kamikaze_komodo/main.py
import asyncio
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from kamikaze_komodo.app_logger import get_logger, logger as root_logger
from kamikaze_komodo.config.settings import settings

# Phase 1 & 2 imports
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.exchange_interaction.exchange_api import ExchangeAPI
# from kamikaze_komodo.strategy_framework.strategy_manager import StrategyManager # Not directly used in demos
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.core.models import BarData, NewsArticle
# from kamikaze_komodo.core.enums import SignalType # For strategy checking, not directly used here

# Phase 3 imports
from kamikaze_komodo.risk_control_module.position_sizer import FixedFractionalPositionSizer, ATRBasedPositionSizer
from kamikaze_komodo.risk_control_module.stop_manager import PercentageStopManager, ATRStopManager

# Portfolio constructor components are more for multi-asset; for single-asset backtest, they are less directly demonstrated here.

# Phase 4 imports
from kamikaze_komodo.ai_news_analysis_agent_module.news_scraper import NewsScraper
from kamikaze_komodo.ai_news_analysis_agent_module.sentiment_analyzer import SentimentAnalyzer
# from kamikaze_komodo.ai_news_analysis_agent_module.browser_agent import BrowserAgent # Optional advanced feature
# from kamikaze_komodo.ai_news_analysis_agent_module.notification_listener import NotificationListener, dummy_notification_callback # Placeholder

logger = get_logger(__name__)

async def run_phase1_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 1: Core Infrastructure & Data")
    if not settings: root_logger.critical("Settings failed to load."); return

    db_manager = DatabaseManager()
    data_fetcher = DataFetcher() # Uses exchange_id from settings
    exchange_api = ExchangeAPI() # Uses exchange_id from settings

    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    start_period = datetime.now(timezone.utc) - timedelta(days=settings.historical_data_days if settings.historical_data_days > 0 else 30)
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
        # Updated currency extraction logic
        base_currency = symbol.split('/')[0].split(':')[0] if '/' in symbol or ':' in symbol else "N/A"
        quote_currency = symbol.split('/')[1] if '/' in symbol else (symbol.split(':')[1] if ':' in symbol else "USD")
        
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

    ewmac_params_for_min_bars = settings.get_strategy_params("EWMAC")
    min_bars_needed = int(ewmac_params_for_min_bars.get('longwindow', settings.ewmac_long_window)) + 5 # Buffer

    historical_bars: List[BarData] = db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)
    
    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.info(f"Insufficient data in DB for {symbol} ({len(historical_bars)}/{min_bars_needed}). Fetching fresh for {hist_days} days...")
        historical_bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
        if historical_bars: db_manager.store_bar_data(historical_bars)
        else: logger.error(f"Failed to fetch sufficient data for {symbol}. Backtest cannot proceed."); await data_fetcher.close(); db_manager.close(); return
    
    await data_fetcher.close() # Close after fetching
    db_manager.close() # Close after retrieving/storing

    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.error(f"Still not enough historical data for {symbol} ({len(historical_bars)} bars vs {min_bars_needed} needed) after fetch attempt. Backtest aborted."); return
    
    logger.info(f"Using {len(historical_bars)} bars for backtesting {symbol}.")
    data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)

    if data_df.empty or len(data_df) < min_bars_needed:
        logger.error(f"DataFrame conversion error or insufficient points for EWMAC ({len(data_df)})."); return

    ewmac_params = settings.get_strategy_params("EWMAC") # Gets params from [EWMAC_Strategy] in config
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

    ewmac_params_for_min_bars_phase3 = settings.get_strategy_params("EWMAC")
    # For ATR-based risk, strategy needs enough data to calculate ATR.
    # ATR period is now part of strategy params from get_strategy_params.
    atr_period_from_params = int(ewmac_params_for_min_bars_phase3.get("atr_period", settings.ewmac_atr_period))
    long_window_from_params = int(ewmac_params_for_min_bars_phase3.get("longwindow", settings.ewmac_long_window))
    min_bars_needed = max(long_window_from_params, atr_period_from_params) + 5 # Buffer

    historical_bars = db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)
    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.info(f"Fetching fresh data for Phase 3 backtest (need ~{min_bars_needed} bars)...")
        historical_bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
        if historical_bars: db_manager.store_bar_data(historical_bars)
        else: logger.error(f"Failed to fetch data for {symbol}. Aborting."); await data_fetcher.close(); db_manager.close(); return
    
    await data_fetcher.close()
    db_manager.close()

    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.error(f"Not enough data ({len(historical_bars)} bars vs {min_bars_needed} needed) for Phase 3 backtest. Aborting."); return
    
    data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)
    
    # Strategy will calculate ATR internally if 'atr_period' is in its params
    ewmac_params = settings.get_strategy_params("EWMAC")
    ewmac_strategy = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params)

    position_sizer: Optional[Any] = None
    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer = ATRBasedPositionSizer(
            risk_per_trade_fraction=settings.atr_based_risk_per_trade_fraction,
            atr_multiple_for_stop=settings.atr_based_atr_multiple_for_stop
        )
    else: # Default to FixedFractional
        position_sizer = FixedFractionalPositionSizer(fraction=settings.fixed_fractional_allocation_fraction)
    logger.info(f"Using Position Sizer: {position_sizer.__class__.__name__}")

    stop_manager: Optional[Any] = None
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

    # 1. News Scraping (Optional live scrape for demo, or assume pre-populated DB)
    if settings.news_scraper_enable:
        logger.info("--- Running News Scraper (Phase 4 Demo) ---")
        try:
            news_scraper = NewsScraper() # Configured from settings
            # Scrape recent articles, e.g., last 48 hours for RSS, limit per source
            scraped_articles = await news_scraper.scrape_all(limit_per_source=5, since_hours_rss=48)
            if scraped_articles:
                logger.info(f"Scraped {len(scraped_articles)} unique articles.")
                
                # 2. Sentiment Analysis (Optional live analysis for demo)
                if settings.enable_sentiment_analysis and settings.sentiment_llm_provider == "VertexAI" and settings.vertex_ai_project_id:
                    logger.info("--- Running Sentiment Analyzer on Scraped Articles (Phase 4 Demo) ---")
                    try:
                        sentiment_analyzer = SentimentAnalyzer() # Configured from settings
                        analyzed_articles: List[NewsArticle] = []
                        for article_to_analyze in scraped_articles[:5]: # Analyze first 5 for demo speed
                            logger.info(f"Analyzing sentiment for: {article_to_analyze.title[:50]}...")
                            updated_article = await sentiment_analyzer.get_sentiment_for_article(article_to_analyze)
                            analyzed_articles.append(updated_article)
                            if updated_article.sentiment_label:
                                logger.info(f"  -> Sentiment: {updated_article.sentiment_label} ({updated_article.sentiment_score:.2f})")
                        
                        # Store analyzed articles (including sentiment) in DB
                        if analyzed_articles:
                            db_manager_news = DatabaseManager()
                            db_manager_news.store_news_articles(analyzed_articles)
                            logger.info(f"Stored {len(analyzed_articles)} analyzed articles in the database.")
                            db_manager_news.close()
                    except Exception as e_sa_live:
                        logger.error(f"Error during live sentiment analysis demo: {e_sa_live}", exc_info=True)
                elif not settings.enable_sentiment_analysis:
                    logger.info("Live sentiment analysis is disabled in settings.")
                elif settings.sentiment_llm_provider != "VertexAI" or not settings.vertex_ai_project_id:
                     logger.warning("Live sentiment analysis configured for non-VertexAI or VertexAI not fully configured (Project ID missing). Skipping live analysis demo.")
            else:
                logger.info("No articles scraped in this run.")
        except Exception as e_scrape_live:
            logger.error(f"Error during live news scraping demo: {e_scrape_live}", exc_info=True)
    else:
        logger.info("News Scraper is disabled in settings. Skipping live scraping/analysis for Phase 4 demo.")


    # 3. Backtesting with Simulated Sentiment Data (Primary focus of Phase 4 completion)
    logger.info("--- Proceeding to backtest with sentiment integration using simulated data ---")
    db_manager_bt = DatabaseManager()
    data_fetcher_bt = DataFetcher()
    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days if settings.historical_data_days > 0 else 90 # Shorter for faster demo if needed
    start_date_bt = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date_bt = datetime.now(timezone.utc)

    ewmac_params_phase4 = settings.get_strategy_params("EWMAC")
    atr_period_phase4 = int(ewmac_params_phase4.get("atr_period", settings.ewmac_atr_period))
    long_window_phase4 = int(ewmac_params_phase4.get("longwindow", settings.ewmac_long_window))
    min_bars_needed_phase4 = max(long_window_phase4, atr_period_phase4) + 5

    historical_bars_bt = db_manager_bt.retrieve_bar_data(symbol, timeframe, start_date_bt, end_date_bt)
    if not historical_bars_bt or len(historical_bars_bt) < min_bars_needed_phase4:
        logger.info(f"Fetching fresh data for Phase 4 backtest (need ~{min_bars_needed_phase4} bars)...")
        historical_bars_bt = await data_fetcher_bt.fetch_historical_data_for_period(symbol, timeframe, start_date_bt, end_date_bt)
        if historical_bars_bt: db_manager_bt.store_bar_data(historical_bars_bt)
        else: logger.error(f"Failed to fetch data for {symbol}. Aborting Phase 4 backtest."); await data_fetcher_bt.close(); db_manager_bt.close(); return
    
    await data_fetcher_bt.close()
    db_manager_bt.close()

    if not historical_bars_bt or len(historical_bars_bt) < min_bars_needed_phase4:
        logger.error(f"Not enough data ({len(historical_bars_bt)} bars) for Phase 4 backtest. Aborting."); return

    data_df_bt = pd.DataFrame([bar.model_dump() for bar in historical_bars_bt])
    data_df_bt['timestamp'] = pd.to_datetime(data_df_bt['timestamp'])
    data_df_bt.set_index('timestamp', inplace=True)

    # Load Simulated Sentiment Data
    sentiment_df: Optional[pd.DataFrame] = None
    if settings.simulated_sentiment_data_path and settings.enable_sentiment_analysis:
        sentiment_csv_path = settings.simulated_sentiment_data_path
        logger.info(f"Attempting to load simulated sentiment data from: {sentiment_csv_path}")
        if os.path.exists(sentiment_csv_path):
            try:
                sentiment_df = pd.read_csv(sentiment_csv_path, parse_dates=['timestamp'], index_col='timestamp')
                if sentiment_df.empty:
                    logger.warning(f"Simulated sentiment data file is empty: {sentiment_csv_path}"); sentiment_df = None
                else:
                    if sentiment_df.index.tz is None: sentiment_df.index = sentiment_df.index.tz_localize('UTC')
                    else: sentiment_df.index = sentiment_df.index.tz_convert('UTC')
                    if 'sentiment_score' not in sentiment_df.columns:
                        logger.error(f"'sentiment_score' column not found in {sentiment_csv_path}. Sentiment will not be used."); sentiment_df = None
                    else:
                        logger.info(f"Successfully loaded {len(sentiment_df)} simulated sentiment entries from: {sentiment_csv_path}")
            except Exception as e_csv:
                logger.error(f"Error loading simulated sentiment data from {sentiment_csv_path}: {e_csv}. Proceeding without sentiment.", exc_info=True); sentiment_df = None
        else:
            logger.warning(f"Simulated sentiment data file NOT FOUND at: {sentiment_csv_path}. Proceeding without external sentiment.")
    elif not settings.enable_sentiment_analysis:
        logger.info("Sentiment analysis is disabled in settings. Backtest will not use external sentiment data.")
    else: # Path not provided
        logger.info("No simulated sentiment data path provided. Proceeding without external sentiment.")

    ewmac_strategy_sentiment = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params_phase4)
    
    position_sizer_sent: Optional[Any] = None
    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer_sent = ATRBasedPositionSizer(settings.atr_based_risk_per_trade_fraction, settings.atr_based_atr_multiple_for_stop)
    else:
        position_sizer_sent = FixedFractionalPositionSizer(settings.fixed_fractional_allocation_fraction)

    stop_manager_sent: Optional[Any] = None
    if settings.stop_manager_type.lower() == 'atrbased':
        stop_manager_sent = ATRStopManager(settings.atr_stop_atr_multiple)
    else:
        stop_manager_sent = PercentageStopManager(settings.percentage_stop_loss_pct, settings.percentage_stop_take_profit_pct)

    initial_capital = 10000.00
    commission_bps = settings.commission_bps
    
    backtest_engine_sentiment = BacktestingEngine(
        data_feed_df=data_df_bt,
        strategy=ewmac_strategy_sentiment,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        position_sizer=position_sizer_sent,
        stop_manager=stop_manager_sent,
        sentiment_data_df=sentiment_df # Pass loaded sentiment data
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
    # await run_phase3_demonstration()
    await run_phase4_demonstration()
    
    root_logger.info("Kamikaze Komodo Program Finished.")

if __name__ == "__main__":
    try:
        # Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment for Vertex AI
        # e.g., export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-file.json"
        # Also, ensure playwright browsers are installed if using BrowserAgent:
        # playwright install --with-deps chromium
        if settings and settings.sentiment_llm_provider == "VertexAI" and not settings.vertex_ai_project_id:
             root_logger.warning("Vertex AI is selected, but Project ID is not set in config.ini. AI features may fail.")
             root_logger.warning("Please set your GCP Project ID in kamikaze_komodo/config/config.ini ([VertexAI] -> ProjectID)")
             root_logger.warning("And ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set.")


        asyncio.run(main())
    except KeyboardInterrupt:
        root_logger.info("Kamikaze Komodo program terminated by user.")
    except Exception as e:
        root_logger.critical(f"Critical error in main execution: {e}", exc_info=True)