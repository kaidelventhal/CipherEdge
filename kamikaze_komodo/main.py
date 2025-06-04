# kamikaze_komodo/main.py
import asyncio
from kamikaze_komodo.app_logger import get_logger, logger as root_logger # Use root logger for main app messages
from kamikaze_komodo.config.settings import settings
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.exchange_interaction.exchange_api import ExchangeAPI

# --- Import Phase 2 components ---
from kamikaze_komodo.strategy_framework.strategy_manager import StrategyManager
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.backtesting_engine.engine import BacktestingEngine
from kamikaze_komodo.backtesting_engine.performance_analyzer import PerformanceAnalyzer
from kamikaze_komodo.core.models import BarData # For type hinting or example data
from datetime import datetime, timedelta, timezone
import pandas as pd


logger = get_logger(__name__) # Module-specific logger

async def run_phase1_demonstration():
    """Demonstrates Phase 1 components."""
    root_logger.info("Starting Kamikaze Komodo Quantitative Trading Program - Phase 1 Demonstration")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot continue.")
        return

    root_logger.info(f"Log Level: {settings.log_level}")
    root_logger.info(f"Default Symbol: {settings.default_symbol}")

    # Initialize Database Manager
    db_manager = DatabaseManager()

    # Initialize Data Fetcher
    data_fetcher = DataFetcher(exchange_id='kraken') # Default to Kraken as per plan
    
    # Fetch some historical data
    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    # Fetch a small amount of recent data for demonstration
    # For a proper run, fetch_historical_data_for_period would be better.
    # since_date = datetime.now(timezone.utc) - timedelta(days=10)
    # historical_data = await data_fetcher.fetch_historical_ohlcv(symbol, timeframe, since=since_date, limit=100)
    
    start_period = datetime.now(timezone.utc) - timedelta(days=settings.historical_data_days if settings.historical_data_days else 60)
    end_period = datetime.now(timezone.utc)
    
    historical_data = await data_fetcher.fetch_historical_data_for_period(
        symbol, timeframe, start_period, end_period
    )

    if historical_data:
        logger.info(f"Fetched {len(historical_data)} data points for {symbol} ({timeframe}).")
        # Store data
        db_manager.store_bar_data(historical_data)
        # Retrieve data (example)
        retrieved_data = db_manager.retrieve_bar_data(symbol, timeframe, start_date=start_period)
        logger.info(f"Retrieved {len(retrieved_data)} data points from DB for {symbol} ({timeframe}).")
    else:
        logger.warning(f"No historical data fetched for {symbol}.")

    await data_fetcher.close()

    # Initialize Exchange API
    exchange_api = ExchangeAPI(exchange_id='kraken')
    balance = await exchange_api.fetch_balance() # Will show warning if API keys are dummy
    if balance:
        # logger.info(f"Account Balance: {balance.get('total', {})}") # Structure varies
        logger.info(f"Fetched balance. Free USD: {balance.get('USD', {}).get('free', 'N/A')}")
    else:
        logger.warning("Could not fetch account balance (may be due to dummy API keys or network issues).")
    
    await exchange_api.close()
    db_manager.close()
    root_logger.info("Phase 1 Demonstration completed.")


async def run_phase2_demonstration():
    """Demonstrates Phase 2 components: Basic Strategy & Backtesting."""
    root_logger.info("Starting Kamikaze Komodo Quantitative Trading Program - Phase 2 Demonstration")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot continue.")
        return

    # 1. Get Data (either from DB or fetch fresh)
    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()

    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    start_date = datetime.now(timezone.utc) - timedelta(days=settings.historical_data_days if settings.historical_data_days > 0 else 365) # Ensure positive days
    end_date = datetime.now(timezone.utc)

    # Try to retrieve from DB first
    historical_bars: List[BarData] = db_manager.retrieve_bar_data(symbol, timeframe, start_date, end_date)

    if not historical_bars or len(historical_bars) < (settings.ewmac_long_window + 5): # Need enough data for EMA + a bit more
        logger.info(f"Not enough data in DB or data is old for {symbol} ({timeframe}). Fetching fresh data...")
        historical_bars = await data_fetcher.fetch_historical_data_for_period(symbol, timeframe, start_date, end_date)
        if historical_bars:
            db_manager.store_bar_data(historical_bars) # Store newly fetched data
        else:
            logger.error(f"Failed to fetch sufficient historical data for {symbol} ({timeframe}). Cannot proceed with backtest.")
            await data_fetcher.close()
            db_manager.close()
            return
    
    await data_fetcher.close() # Close fetcher connection after use
    db_manager.close() # Close DB connection after use

    if not historical_bars:
        logger.error(f"No historical data available for {symbol} to run backtest.")
        return
    
    logger.info(f"Using {len(historical_bars)} bars of historical data for backtesting {symbol} ({timeframe}).")

    # Convert BarData list to Pandas DataFrame for strategy processing
    # The strategy will expect a DataFrame.
    data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)
    
    if data_df.empty or len(data_df) < max(settings.ewmac_short_window, settings.ewmac_long_window):
        logger.error(f"Not enough data points ({len(data_df)}) after DataFrame conversion for EWMAC strategy. Need at least {max(settings.ewmac_short_window, settings.ewmac_long_window)}.")
        return

    # 2. Initialize Strategy
    ewmac_params = {
        'short_window': settings.ewmac_short_window,
        'long_window': settings.ewmac_long_window,
        # 'signal_window': settings.config.getint('EWMAC_Strategy', 'SignalWindow') # if using MACD
    }
    ewmac_strategy = EWMACStrategy(symbol=symbol, timeframe=timeframe, params=ewmac_params)

    # 3. Initialize Strategy Manager (Optional for single strategy backtest, but good for structure)
    strategy_manager = StrategyManager()
    strategy_manager.add_strategy(ewmac_strategy)

    # 4. Initialize Backtesting Engine
    # For this basic backtest, we'll pass data and strategy directly.
    # A more advanced engine would take data from a source and manage strategy execution.
    initial_capital = 10000.00 # Example starting capital
    commission_bps = 0.001 # Example commission 0.1% (10 bps)
    
    backtest_engine = BacktestingEngine(
        data_feed_df=data_df, # Engine expects DataFrame
        strategy=ewmac_strategy, # Pass the single strategy instance
        initial_capital=initial_capital,
        commission_bps=commission_bps
    )
    
    # 5. Run Backtest
    logger.info(f"Running backtest for EWMAC strategy on {symbol}...")
    trades_log, final_portfolio = backtest_engine.run()

    # 6. Analyze Performance
    if trades_log:
        logger.info(f"Backtest completed. Generated {len(trades_log)} trades.")
        performance_analyzer = PerformanceAnalyzer(
            trades=trades_log, 
            initial_capital=initial_capital,
            final_capital=final_portfolio['total_value'] # Access total_value from portfolio dict
            )
        
        metrics = performance_analyzer.calculate_metrics()
        performance_analyzer.print_summary(metrics)
        
        # Store PnL series for plotting or further analysis if needed
        # pnl_series = performance_analyzer.get_pnl_series()
        # equity_curve = performance_analyzer.get_equity_curve()
        # logger.debug(f"Equity Curve (first 5): \n{equity_curve.head()}")
    else:
        logger.info("Backtest completed. No trades were executed.")
        logger.info(f"Final portfolio value: ${final_portfolio['total_value']:.2f}")


    root_logger.info("Phase 2 Demonstration completed.")


async def main():
    """
    Main asynchronous function to run the application phases.
    """
    # Uncomment the phase you want to demonstrate
    # await run_phase1_demonstration()
    await run_phase2_demonstration()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        root_logger.info("Kamikaze Komodo program terminated by user.")
    except Exception as e:
        root_logger.critical(f"Critical error in main execution: {e}", exc_info=True)