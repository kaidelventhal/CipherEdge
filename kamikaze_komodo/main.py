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

# Phase 5 imports
from kamikaze_komodo.strategy_framework.strategies.ehlers_instantaneous_trendline import EhlersInstantaneousTrendlineStrategy
from kamikaze_komodo.strategy_framework.strategies.ml_forecaster_strategy import MLForecasterStrategy
from kamikaze_komodo.ml_models.training_pipelines.lightgbm_pipeline import LightGBMTrainingPipeline # For training demo
# from kamikaze_komodo.ml_models.inference_pipelines.lightgbm_inference import LightGBMInference # Used by MLForecasterStrategy internally

# Phase 6 imports (strategies and models will be demonstrated if settings allow)
# from kamikaze_komodo.strategy_framework.strategies.bollinger_band_breakout_strategy import BollingerBandBreakoutStrategy
# from kamikaze_komodo.strategy_framework.strategies.pair_trading_strategy import PairTradingStrategy
# from kamikaze_komodo.ml_models.training_pipelines.xgboost_classifier_pipeline import XGBoostClassifierTrainingPipeline


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

    ewmac_params_for_min_bars = settings.get_strategy_params("EWMAC_Strategy")
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

    ewmac_params = settings.get_strategy_params("EWMAC_Strategy")
    ewmac_strategy = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params)

    initial_capital = 10000.00
    commission_bps = settings.commission_bps
    backtest_engine = BacktestingEngine(data_feed_df=data_df, strategy=ewmac_strategy, initial_capital=initial_capital, commission_bps=commission_bps)

    logger.info(f"Running basic backtest for EWMAC on {symbol}...")
    trades_log, final_portfolio, equity_curve_df = backtest_engine.run() # Capture equity_curve

    if trades_log:
        logger.info(f"Backtest (Phase 2) completed. Generated {len(trades_log)} trades.")
        performance_analyzer = PerformanceAnalyzer(
            trades=trades_log,
            initial_capital=initial_capital,
            final_capital=final_portfolio['final_portfolio_value'],
            equity_curve=equity_curve_df # Pass equity curve here
        )
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

    ewmac_params_for_min_bars_phase3 = settings.get_strategy_params("EWMAC_Strategy")
    atr_period_from_params = int(ewmac_params_for_min_bars_phase3.get("atr_period", settings.ewmac_atr_period))
    long_window_from_params = int(ewmac_params_for_min_bars_phase3.get("longwindow", settings.ewmac_long_window))
    min_bars_needed = max(long_window_from_params, atr_period_from_params) + 5

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

    ewmac_params = settings.get_strategy_params("EWMAC_Strategy")
    ewmac_strategy = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params)

    position_sizer: Optional[Any] = None
    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer = ATRBasedPositionSizer(
            risk_per_trade_fraction=settings.atr_based_risk_per_trade_fraction,
            atr_multiple_for_stop=settings.atr_based_atr_multiple_for_stop
        )
    else:
        position_sizer = FixedFractionalPositionSizer(fraction=settings.fixed_fractional_allocation_fraction)
    logger.info(f"Using Position Sizer: {position_sizer.__class__.__name__}")

    stop_manager: Optional[Any] = None
    if settings.stop_manager_type.lower() == 'atrbased':
        stop_manager = ATRStopManager(atr_multiple=settings.atr_stop_atr_multiple)
    else:
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
    trades_log, final_portfolio, equity_curve_df = backtest_engine.run()

    if trades_log:
        logger.info(f"Backtest (Phase 3) completed. Generated {len(trades_log)} trades.")
        performance_analyzer = PerformanceAnalyzer(
            trades=trades_log,
            initial_capital=initial_capital,
            final_capital=final_portfolio['final_portfolio_value'],
            equity_curve=equity_curve_df
            )
        metrics = performance_analyzer.calculate_metrics()
        performance_analyzer.print_summary(metrics)
    else:
        logger.info(f"Backtest (Phase 3) completed. No trades executed. Final portfolio: ${final_portfolio['final_portfolio_value']:.2f}")
    root_logger.info("Phase 3 Demonstration completed.")


async def run_phase4_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 4: AI News & Sentiment Integration")
    if not settings: root_logger.critical("Settings failed to load."); return

    if settings.news_scraper_enable:
        logger.info("--- Running News Scraper (Phase 4 Demo) ---")
        try:
            news_scraper = NewsScraper()
            scraped_articles = await news_scraper.scrape_all(limit_per_source=5, since_hours_rss=48)
            if scraped_articles:
                logger.info(f"Scraped {len(scraped_articles)} unique articles.")

                if settings.enable_sentiment_analysis and settings.sentiment_llm_provider == "VertexAI" and settings.vertex_ai_project_id:
                    logger.info("--- Running Sentiment Analyzer on Scraped Articles (Phase 4 Demo) ---")
                    try:
                        sentiment_analyzer = SentimentAnalyzer()
                        analyzed_articles: List[NewsArticle] = []
                        for article_to_analyze in scraped_articles[:5]:
                            logger.info(f"Analyzing sentiment for: {article_to_analyze.title[:50]}...")
                            updated_article = await sentiment_analyzer.get_sentiment_for_article(article_to_analyze)
                            analyzed_articles.append(updated_article)
                            if updated_article.sentiment_label:
                                logger.info(f"  -> Sentiment: {updated_article.sentiment_label} ({updated_article.sentiment_score:.2f})")

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


    logger.info("--- Proceeding to backtest with sentiment integration using simulated data ---")
    db_manager_bt = DatabaseManager()
    data_fetcher_bt = DataFetcher()
    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days if settings.historical_data_days > 0 else 90
    start_date_bt = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date_bt = datetime.now(timezone.utc)

    ewmac_params_phase4 = settings.get_strategy_params("EWMAC_Strategy")
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
    else:
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
        sentiment_data_df=sentiment_df
    )
    logger.info(f"Running Phase 4 backtest (EWMAC with Sentiment & Risk Mgmt) on {symbol}...")
    trades_log, final_portfolio, equity_curve_df = backtest_engine_sentiment.run()

    if trades_log:
        logger.info(f"Backtest (Phase 4) completed. Generated {len(trades_log)} trades.")
        performance_analyzer = PerformanceAnalyzer(
            trades=trades_log,
            initial_capital=initial_capital,
            final_capital=final_portfolio['final_portfolio_value'],
            equity_curve=equity_curve_df
            )
        metrics = performance_analyzer.calculate_metrics()
        performance_analyzer.print_summary(metrics)
    else:
        logger.info(f"Backtest (Phase 4) completed. No trades executed. Final portfolio: ${final_portfolio['final_portfolio_value']:.2f}")
    root_logger.info("Phase 4 Demonstration completed.")


async def run_phase5_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 5: Advanced Strategies & ML Models")
    if not settings: root_logger.critical("Settings failed to load."); return

    # --- 1. Optional: Train ML Model (LightGBM Example) ---
    run_ml_training = True # Set to True to run training as part of demo. Ensures model is available.
    if run_ml_training:
        if not settings.config.has_section("LightGBM_Forecaster"):
            logger.error("Config section [LightGBM_Forecaster] not found. Skipping ML model training.")
        else:
            logger.info("--- Running LightGBM Training Pipeline (Phase 5 Demo) ---")
            try:
                training_pipeline = LightGBMTrainingPipeline(
                    symbol=settings.default_symbol,
                    timeframe=settings.default_timeframe
                )
                await training_pipeline.run_training()
                logger.info("LightGBM Training Pipeline completed for Phase 5 demo.")
            except Exception as e_train:
                logger.error(f"Error during LightGBM training demo: {e_train}", exc_info=True)
    else:
        logger.info("Skipping ML model training for this Phase 5 demonstration run. Ensure model exists if MLForecasterStrategy is used.")

    # --- 2. Backtest with Ehlers Instantaneous Trendline Strategy ---
    logger.info("--- Starting Backtest with Ehlers Instantaneous Trendline Strategy (Phase 5 Demo) ---")
    db_manager_ehlers = DatabaseManager()
    data_fetcher_ehlers = DataFetcher()
    symbol_ehlers = settings.default_symbol
    timeframe_ehlers = settings.default_timeframe
    hist_days_ehlers = settings.historical_data_days if settings.historical_data_days > 0 else 180 # Shorter for demo
    start_date_ehlers = datetime.now(timezone.utc) - timedelta(days=hist_days_ehlers)
    end_date_ehlers = datetime.now(timezone.utc)

    ehlers_params = settings.get_strategy_params("EhlersInstantaneousTrendline_Strategy")
    min_bars_needed_ehlers = int(ehlers_params.get("it_lag_trigger", 1)) + int(ehlers_params.get("atr_period", 14)) + 20 # Increased buffer

    historical_bars_ehlers = db_manager_ehlers.retrieve_bar_data(symbol_ehlers, timeframe_ehlers, start_date_ehlers, end_date_ehlers)
    if not historical_bars_ehlers or len(historical_bars_ehlers) < min_bars_needed_ehlers:
        logger.info(f"Fetching data for Ehlers IT backtest (need ~{min_bars_needed_ehlers} bars)...")
        historical_bars_ehlers = await data_fetcher_ehlers.fetch_historical_data_for_period(symbol_ehlers, timeframe_ehlers, start_date_ehlers, end_date_ehlers)
        if historical_bars_ehlers: db_manager_ehlers.store_bar_data(historical_bars_ehlers)

    await data_fetcher_ehlers.close()
    db_manager_ehlers.close()

    if not historical_bars_ehlers or len(historical_bars_ehlers) < min_bars_needed_ehlers:
        logger.error(f"Not enough data for Ehlers IT backtest ({len(historical_bars_ehlers)} bars). Aborting."); return

    data_df_ehlers = pd.DataFrame([bar.model_dump() for bar in historical_bars_ehlers])
    data_df_ehlers['timestamp'] = pd.to_datetime(data_df_ehlers['timestamp'])
    data_df_ehlers.set_index('timestamp', inplace=True)

    ehlers_strategy = EhlersInstantaneousTrendlineStrategy(symbol=symbol_ehlers, timeframe=timeframe_ehlers, params=ehlers_params)

    position_sizer_ehlers: Optional[Any] = None
    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer_ehlers = ATRBasedPositionSizer(settings.atr_based_risk_per_trade_fraction, settings.atr_based_atr_multiple_for_stop)
    else: position_sizer_ehlers = FixedFractionalPositionSizer(settings.fixed_fractional_allocation_fraction)

    stop_manager_ehlers: Optional[Any] = None
    if settings.stop_manager_type.lower() == 'atrbased':
        stop_manager_ehlers = ATRStopManager(settings.atr_stop_atr_multiple)
    else: stop_manager_ehlers = PercentageStopManager(settings.percentage_stop_loss_pct, settings.percentage_stop_take_profit_pct)

    initial_capital = 10000.00
    commission_bps = settings.commission_bps

    backtest_engine_ehlers = BacktestingEngine(
        data_feed_df=data_df_ehlers, strategy=ehlers_strategy, initial_capital=initial_capital,
        commission_bps=commission_bps, position_sizer=position_sizer_ehlers, stop_manager=stop_manager_ehlers
    )
    logger.info(f"Running Phase 5 backtest (Ehlers IT Strategy) on {symbol_ehlers}...")
    trades_log_ehlers, final_portfolio_ehlers, equity_curve_df_ehlers = backtest_engine_ehlers.run()

    if trades_log_ehlers:
        logger.info(f"Ehlers IT Backtest completed. Generated {len(trades_log_ehlers)} trades.")
        pa_ehlers = PerformanceAnalyzer(
            trades_log_ehlers,
            initial_capital,
            final_portfolio_ehlers['final_portfolio_value'],
            equity_curve_df_ehlers
            )
        metrics_ehlers = pa_ehlers.calculate_metrics()
        pa_ehlers.print_summary(metrics_ehlers)
    else:
        logger.info(f"Ehlers IT Backtest completed. No trades executed. Final portfolio: ${final_portfolio_ehlers['final_portfolio_value']:.2f}")


    # --- 3. Backtest with MLForecasterStrategy ---
    logger.info("--- Starting Backtest with MLForecasterStrategy (Phase 5 Demo) ---")

    db_manager_ml = DatabaseManager()
    data_fetcher_ml = DataFetcher()
    symbol_ml = settings.default_symbol
    timeframe_ml = settings.default_timeframe
    hist_days_ml = settings.historical_data_days if settings.historical_data_days > 0 else 180
    start_date_ml = datetime.now(timezone.utc) - timedelta(days=hist_days_ml)
    end_date_ml = datetime.now(timezone.utc)

    ml_strategy_params = settings.get_strategy_params("MLForecaster_Strategy")
    min_bars_for_pred = int(ml_strategy_params.get('min_bars_for_prediction', 50))
    min_bars_needed_ml = max(min_bars_for_pred, int(ml_strategy_params.get("atr_period", 14))) + 20 # Increased buffer

    historical_bars_ml = db_manager_ml.retrieve_bar_data(symbol_ml, timeframe_ml, start_date_ml, end_date_ml)
    if not historical_bars_ml or len(historical_bars_ml) < min_bars_needed_ml:
        logger.info(f"Fetching data for ML Strategy backtest (need ~{min_bars_needed_ml} bars)...")
        historical_bars_ml = await data_fetcher_ml.fetch_historical_data_for_period(symbol_ml, timeframe_ml, start_date_ml, end_date_ml)
        if historical_bars_ml: db_manager_ml.store_bar_data(historical_bars_ml)

    await data_fetcher_ml.close()
    db_manager_ml.close()

    if not historical_bars_ml or len(historical_bars_ml) < min_bars_needed_ml:
        logger.error(f"Not enough data for ML Strategy backtest ({len(historical_bars_ml)} bars). Aborting."); return

    data_df_ml = pd.DataFrame([bar.model_dump() for bar in historical_bars_ml])
    data_df_ml['timestamp'] = pd.to_datetime(data_df_ml['timestamp'])
    data_df_ml.set_index('timestamp', inplace=True)

    try:
        ml_strategy = MLForecasterStrategy(symbol=symbol_ml, timeframe=timeframe_ml, params=ml_strategy_params)
        if ml_strategy.inference_engine is None or ml_strategy.inference_engine.forecaster.model is None:
             logger.error(f"MLForecasterStrategy for {symbol_ml} ({timeframe_ml}) failed to load its model. Cannot proceed with backtest.")
             root_logger.info("Phase 5 MLForecasterStrategy Backtest SKIPPED due to model loading issue.")
             return # Skip this part of the demo if model isn't loaded
    except Exception as e_strat_init:
        logger.error(f"Failed to initialize MLForecasterStrategy: {e_strat_init}", exc_info=True)
        root_logger.info("Phase 5 MLForecasterStrategy Backtest SKIPPED due to strategy initialization error.")
        return

    position_sizer_ml: Optional[Any] = None
    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer_ml = ATRBasedPositionSizer(settings.atr_based_risk_per_trade_fraction, settings.atr_based_atr_multiple_for_stop)
    else: position_sizer_ml = FixedFractionalPositionSizer(settings.fixed_fractional_allocation_fraction)

    stop_manager_ml: Optional[Any] = None
    if settings.stop_manager_type.lower() == 'atrbased':
        stop_manager_ml = ATRStopManager(settings.atr_stop_atr_multiple)
    else: stop_manager_ml = PercentageStopManager(settings.percentage_stop_loss_pct, settings.percentage_stop_take_profit_pct)

    sentiment_df_ml: Optional[pd.DataFrame] = None
    if settings.simulated_sentiment_data_path and settings.enable_sentiment_analysis:
        sentiment_csv_path = settings.simulated_sentiment_data_path
        if os.path.exists(sentiment_csv_path):
            try:
                sentiment_df_ml = pd.read_csv(sentiment_csv_path, parse_dates=['timestamp'], index_col='timestamp')
                if not sentiment_df_ml.empty:
                    if sentiment_df_ml.index.tz is None: sentiment_df_ml.index = sentiment_df_ml.index.tz_localize('UTC')
                    else: sentiment_df_ml.index = sentiment_df_ml.index.tz_convert('UTC')
                    if 'sentiment_score' in sentiment_df_ml.columns:
                        logger.info(f"Loaded {len(sentiment_df_ml)} simulated sentiment entries for ML strategy backtest.")
                    else: sentiment_df_ml = None; logger.error("sentiment_score col missing in CSV for ML test.")
                else: sentiment_df_ml = None
            except Exception: sentiment_df_ml = None; logger.error("Error loading sentiment CSV for ML test.", exc_info=True)
        else: logger.warning("Sentiment CSV for ML test not found.")


    backtest_engine_ml = BacktestingEngine(
        data_feed_df=data_df_ml, strategy=ml_strategy, initial_capital=initial_capital,
        commission_bps=commission_bps, position_sizer=position_sizer_ml, stop_manager=stop_manager_ml,
        sentiment_data_df=sentiment_df_ml
    )
    logger.info(f"Running Phase 5 backtest (MLForecasterStrategy) on {symbol_ml}...")
    trades_log_ml, final_portfolio_ml, equity_curve_df_ml = backtest_engine_ml.run()

    if trades_log_ml:
        logger.info(f"MLForecasterStrategy Backtest completed. Generated {len(trades_log_ml)} trades.")
        pa_ml = PerformanceAnalyzer(
            trades_log_ml,
            initial_capital,
            final_portfolio_ml['final_portfolio_value'],
            equity_curve_df_ml
            )
        metrics_ml = pa_ml.calculate_metrics()
        pa_ml.print_summary(metrics_ml)
    else:
        logger.info(f"MLForecasterStrategy Backtest completed. No trades executed. Final portfolio: ${final_portfolio_ml['final_portfolio_value']:.2f}")

    root_logger.info("Phase 5 Demonstration completed.")


async def run_phase6_demonstration():
    root_logger.info("Starting Kamikaze Komodo - Phase 6: Advanced Trading & Backtesting Demo")
    if not settings: root_logger.critical("Settings failed to load."); return

    # --- Backtest EWMAC Strategy with Shorting Enabled ---
    logger.info("--- Starting Backtest with EWMAC Strategy (Shorting Enabled - Phase 6 Demo) ---")
    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()
    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days if settings.historical_data_days > 0 else 365
    start_date = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date = datetime.now(timezone.utc)

    ewmac_params = settings.get_strategy_params("EWMAC_Strategy")
    min_bars_needed = int(ewmac_params.get('longwindow', 26)) + 5 # Buffer

    historical_bars = db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)
    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.info(f"Fetching data for EWMAC (shorting) backtest (need ~{min_bars_needed} bars)...")
        historical_bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
        if historical_bars: db_manager.store_bar_data(historical_bars)

    await data_fetcher.close()
    db_manager.close()

    if not historical_bars or len(historical_bars) < min_bars_needed:
        logger.error(f"Not enough data for EWMAC (shorting) backtest ({len(historical_bars)} bars). Aborting."); return

    data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)

    ewmac_strategy_shorting = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params)
    # Make sure EWMAC can generate SHORT signals based on its logic (e.g. death cross)
    # And it handles self.current_position_status correctly for SHORT.

    position_sizer: Optional[Any] = None
    if settings.position_sizer_type.lower() == 'atrbased':
        position_sizer = ATRBasedPositionSizer(settings.atr_based_risk_per_trade_fraction, settings.atr_based_atr_multiple_for_stop)
    else:
        position_sizer = FixedFractionalPositionSizer(settings.fixed_fractional_allocation_fraction)

    stop_manager: Optional[Any] = None
    if settings.stop_manager_type.lower() == 'atrbased':
        stop_manager = ATRStopManager(settings.atr_stop_atr_multiple)
    else:
        stop_manager = PercentageStopManager(settings.percentage_stop_loss_pct, settings.percentage_stop_take_profit_pct)

    initial_capital = 10000.00
    commission_bps = settings.commission_bps

    backtest_engine_shorting = BacktestingEngine(
        data_feed_df=data_df,
        strategy=ewmac_strategy_shorting,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        position_sizer=position_sizer,
        stop_manager=stop_manager
    )
    logger.info(f"Running Phase 6 backtest (EWMAC Strategy with Shorting) on {symbol}...")
    trades_log, final_portfolio, equity_curve_df = backtest_engine_shorting.run()

    if trades_log:
        logger.info(f"EWMAC (Shorting) Backtest completed. Generated {len(trades_log)} trades.")
        pa_shorting = PerformanceAnalyzer(trades_log, initial_capital, final_portfolio['final_portfolio_value'], equity_curve_df)
        metrics = pa_shorting.calculate_metrics()
        pa_shorting.print_summary(metrics)
    else:
        logger.info(f"EWMAC (Shorting) Backtest completed. No trades executed. Final portfolio: ${final_portfolio['final_portfolio_value']:.2f}")

    root_logger.info("Phase 6 Demonstration completed.")


async def main():
    root_logger.info("Kamikaze Komodo Program Starting...")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot start.")
        return

    # --- Select phases to run ---
    # await run_phase1_demonstration()
    # await run_phase2_demonstration()
    # await run_phase3_demonstration()
    # await run_phase4_demonstration()
    # await run_phase5_demonstration()
    await run_phase6_demonstration() # New phase demo

    root_logger.info("Kamikaze Komodo Program Finished.")

if __name__ == "__main__":
    try:
        if settings and settings.sentiment_llm_provider == "VertexAI" and not settings.vertex_ai_project_id:
             root_logger.warning("Vertex AI is selected, but Project ID is not set in config.ini. AI features may fail.")
             root_logger.warning("Please set your GCP Project ID in kamikaze_komodo/config/config.ini ([VertexAI] -> ProjectID)")
             root_logger.warning("And ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set.")

        if not os.path.exists("logs"): os.makedirs("logs")

        # Check if model directory exists, create if not.
        # Training pipeline also attempts to create it.
        model_dir_from_config = "ml_models/trained_models" # Default or read from config
        if settings and settings.config.has_option("LightGBM_Forecaster", "ModelSavePath"):
            model_dir_from_config = settings.config.get("LightGBM_Forecaster", "ModelSavePath")
        # Further path construction for model_dir_from_config happens in settings or pipelines

        asyncio.run(main())
    except KeyboardInterrupt:
        root_logger.info("Kamikaze Komodo program terminated by user.")
    except Exception as e:
        root_logger.critical(f"Critical error in main execution: {e}", exc_info=True)