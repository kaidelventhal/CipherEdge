# FILE: optimizer_main.py
import asyncio
import pandas as pd
from kamikaze_komodo.backtesting_engine.optimizer import StrategyOptimizer
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.config.settings import settings
from datetime import datetime, timedelta, timezone
from kamikaze_komodo.app_logger import get_logger

# Import all strategies to be tested
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.strategy_framework.strategies.bollinger_band_breakout_strategy import BollingerBandBreakoutStrategy
from kamikaze_komodo.strategy_framework.strategies.ehlers_instantaneous_trendline import EhlersInstantaneousTrendlineStrategy


logger = get_logger("Optimizer")

async def main():
    """
    Main function to set up and run a comparative optimization across multiple strategies.
    """
    if not settings:
        logger.critical("Settings not loaded. Cannot run optimizer.")
        return

    # --- 1. Load Data for a Single Asset ---
    asset_to_optimize = "PF_XBTUSD"
    timeframe = settings.default_timeframe

    logger.info(f"Loading data for optimization: {asset_to_optimize} ({timeframe})")
    db_manager = DatabaseManager()
    data_fetcher = DataFetcher()
    start_date = datetime.now(timezone.utc) - timedelta(days=settings.historical_data_days)

    bars = db_manager.retrieve_bar_data(asset_to_optimize, timeframe, start_date=start_date)
    if not bars or len(bars) < 200:
        logger.info(f"Fetching fresh data for {asset_to_optimize}...")
        bars = await data_fetcher.fetch_historical_data_for_period(asset_to_optimize, timeframe, start_date)
        if bars: db_manager.store_bar_data(bars)

    await data_fetcher.close()
    db_manager.close()

    if not bars:
        logger.error(f"Could not load data for {asset_to_optimize}. Aborting optimization.")
        return

    data_df = pd.DataFrame([bar.model_dump() for bar in bars])
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
    data_df.set_index('timestamp', inplace=True)
    logger.info(f"Loaded {len(data_df)} bars for {asset_to_optimize}.")

    # --- 2. Define Strategies and Parameter Grids for Optimization ---
    # Common risk parameters to be tested with each strategy
    risk_params = {
        'triplebarrier_tp_multiple': [2.0, 3.0],
        'triplebarrier_sl_multiple': [1.5, 2.0],
        'triplebarrier_time_limit_bars': [20, 30],
    }

    # Dictionary mapping strategy classes to their specific parameter grids
    optimization_tasks = {
        EWMACStrategy: {
            'shortwindow': [20, 30],
            'longwindow': [50, 60],
            **risk_params
        },
        BollingerBandBreakoutStrategy: {
            'bb_period': [20, 30],
            'bb_std_dev': [2.0, 2.5],
            **risk_params
        },
        EhlersInstantaneousTrendlineStrategy: {
            'it_lag_trigger': [1, 2],
            **risk_params
        },
    }

    overall_best_result = {
        'strategy': None,
        'params': None,
        'metric_value': -float('inf')
    }

    optimization_metric = 'sharpe_ratio'

    # --- 3. Loop Through and Optimize Each Strategy ---
    for strategy_class, param_grid in optimization_tasks.items():
        logger.info(f"\n--- Starting Optimization for {strategy_class.__name__} ---")

        optimizer = StrategyOptimizer(
            strategy_class=strategy_class,
            data_feed_df=data_df,
            param_grid=param_grid,
            optimization_metric=optimization_metric,
            initial_capital=float(settings.config.get('Backtesting', 'InitialCapital')),
            commission_bps=float(settings.config.get('Trading', 'CommissionBPS')),
            slippage_bps=float(settings.config.get('Trading', 'SlippageBPS')),
            symbol=asset_to_optimize,
            timeframe=timeframe,
            position_sizer_class_name="FixedFractionalPositionSizer",
            stop_manager_class_name="TripleBarrierStopManager"
        )

        logger.info(f"Starting Grid Search for '{strategy_class.__name__}'...")
        best_params, best_metric, results_df = await optimizer.grid_search()

        logger.info(f"--- Optimization for {strategy_class.__name__} Finished ---")
        if best_params:
            logger.info(f"Best Metric ({optimization_metric}): {best_metric:.4f}")
            logger.info(f"Best Parameters: {best_params}")

            if best_metric > overall_best_result['metric_value']:
                overall_best_result['strategy'] = strategy_class.__name__
                overall_best_result['params'] = best_params
                overall_best_result['metric_value'] = best_metric
                logger.info(f"*** New Overall Best Found: {strategy_class.__name__} with metric {best_metric:.4f} ***")
        else:
            logger.warning(f"Optimization for {strategy_class.__name__} did not yield any successful results.")

    # --- 4. Report Final Winner ---
    logger.info("\n\n---=== OVERALL OPTIMIZATION COMPLETE ===---")
    if overall_best_result['strategy']:
        logger.info(f"The best performing strategy is: {overall_best_result['strategy']}")
        logger.info(f"Best overall metric ({optimization_metric}): {overall_best_result['metric_value']:.4f}")
        logger.info(f"Best overall parameters: {overall_best_result['params']}")
        logger.info("\nUpdate your config.ini with these new 'best' parameters to improve performance.")
    else:
        logger.warning("No successful results found across all strategies.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error in optimizer execution: {e}", exc_info=True)