# FILE: kamikaze_komodo/main.py
import asyncio
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import warnings
import multiprocessing

# Suppress the specific pandas FutureWarning
warnings.filterwarnings(
    "ignore",
    message="Downcasting behavior in `replace` is deprecated and will be removed in a future version.",
    category=FutureWarning
)

from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT
from kamikaze_komodo.data_handling.data_handler import DataHandler
from kamikaze_komodo.backtesting_engine.optimizer import StrategyOptimizer
from kamikaze_komodo.portfolio_constructor.meta_portfolio_constructor import MultiStrategyPortfolioConstructor
from kamikaze_komodo.risk_control_module.risk_manager import RiskManager
from kamikaze_komodo.ml_models.training_pipelines.lightgbm_pipeline import LightGBMTrainingPipeline
from kamikaze_komodo.ml_models.training_pipelines.xgboost_classifier_pipeline import XGBoostClassifierTrainingPipeline
from kamikaze_komodo.ml_models.training_pipelines.lstm_pipeline import LSTMTrainingPipeline
from kamikaze_komodo.ml_models.training_pipelines.kmeans_regime_pipeline import KMeansRegimeTrainingPipeline


root_logger = get_logger(__name__)
logger = get_logger(__name__)


async def train_all_models_if_needed():
    """
    Checks for the existence of required ML models and trains them only if they are missing and needed.
    """
    root_logger.info("Checking for pre-trained ML models...")
    if not settings:
        root_logger.critical("Settings not loaded, cannot check or train models.")
        return

    symbols_to_check = settings.PHASE3_SYMBOLS or [settings.default_symbol]
    timeframe = settings.default_timeframe
    strategies_in_use = settings.PHASE3_STRATEGIES

    # Determine which types of models are actually needed by the configured strategies
    needs_ml_models = any(s.startswith("MLForecaster") or s.startswith("Ensemble") for s in strategies_in_use)
    # The regime model is always needed for the optimizer's regime-aware filtering feature
    needs_regime_model = True

    for symbol in symbols_to_check:
        root_logger.info(f"--- Checking models for {symbol} ---")
        
        models_to_check = {
            "LGBM": (LightGBMTrainingPipeline, f"lgbm_{symbol.replace('/', '_').lower()}_{timeframe}.joblib"),
            "XGB_Classifier": (XGBoostClassifierTrainingPipeline, f"xgb_classifier_{symbol.replace('/', '_').lower()}_{timeframe}.joblib"),
            "LSTM": (LSTMTrainingPipeline, f"lstm_{symbol.replace('/', '_').lower()}_{timeframe}.pth"),
            "KMeans_Regime": (KMeansRegimeTrainingPipeline, f"kmeans_regime_{symbol.replace('/', '_').lower()}_{timeframe}.joblib"),
        }
        
        for model_name, (pipeline_class, model_filename) in models_to_check.items():
            # Skip training if this model type is not required by the config
            if "KMeans" in model_name and not needs_regime_model:
                logger.info(f"Skipping KMeans model check for {symbol} as it's not required by the current configuration.")
                continue
            if ("LGBM" in model_name or "XGB" in model_name or "LSTM" in model_name) and not needs_ml_models:
                logger.info(f"Skipping {model_name} model check for {symbol} as it's not required by the current configuration.")
                continue

            model_path = os.path.join(PROJECT_ROOT, 'ml_models/trained_models', model_filename)
            if not os.path.exists(model_path):
                root_logger.warning(f"{model_name} model not found for {symbol} at {model_path}. Starting training...")
                try:
                    pipeline = pipeline_class(symbol=symbol, timeframe=timeframe)
                    await pipeline.run_training()
                    
                    # FIX: Verify that the model file was actually created before logging success
                    if os.path.exists(model_path):
                        root_logger.info(f"Successfully trained and saved {model_name} model for {symbol}.")
                    else:
                        root_logger.error(f"Training for {model_name} for {symbol} ran, but the model file was NOT created at {model_path}.")
                except Exception as e:
                    root_logger.error(f"Failed to train {model_name} model for {symbol}: {e}", exc_info=True)
            else:
                root_logger.info(f"Found existing {model_name} model for {symbol}.")


async def run_phase3_discovery(data_handler: DataHandler):
    """
    Executes the Phase 3 strategy discovery and portfolio construction process.
    """
    root_logger.info("🚀 Starting Kamikaze Komodo - Phase 3: Strategy Discovery & Portfolio Construction")
    if not settings:
        root_logger.critical("Settings failed to load.")
        return

    all_symbols = settings.PHASE3_SYMBOLS
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days
    start_date = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date = datetime.now(timezone.utc)

    data_feeds = {}
    for symbol in all_symbols:
        root_logger.info(f"Fetching historical data for {symbol}...")
        df = await data_handler.get_prepared_data(
            symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date,
            needs_funding_rate=True, needs_sentiment=True
        )
        if not df.empty:
            data_feeds[symbol] = df
        else:
            root_logger.error(f"Failed to fetch data for {symbol}. It will be excluded from the analysis.")

    if not data_feeds:
        root_logger.critical("No data could be fetched for any symbol. Aborting Phase 3.")
        return

    optimizer = StrategyOptimizer(
        data_feeds=data_feeds, initial_capital=10000.0,
        commission_bps=settings.commission_bps, slippage_bps=settings.slippage_bps
    )
    
    results_df, equity_curves = optimizer.run_phase3_discovery()
    
    if results_df.empty:
        root_logger.error("Strategy discovery yielded no results. Portfolio construction aborted.")
        return

    risk_manager = RiskManager(settings=settings)
    portfolio_constructor = MultiStrategyPortfolioConstructor(settings=settings, risk_manager=risk_manager)
    
    top_n = settings.PHASE3_TOP_COMBOS_COUNT
    selected_ids = portfolio_constructor.select_top_n(
        trials_df=results_df, equity_curves=equity_curves, n=top_n
    )

    if not selected_ids:
        root_logger.warning("No top combinations were selected after filtering. Portfolio construction aborted.")
        return

    weights = portfolio_constructor.compute_weights(
        selected_ids=selected_ids, equity_curves=equity_curves,
        method=settings.PHASE3_COMPUTE_WEIGHTS_METHOD
    )

    root_logger.info("--- Top Performing & Diversified Combinations ---")
    top_combos_df = results_df.loc[selected_ids].copy()
    for combo_id, weight in weights.items():
        top_combos_df.loc[combo_id, 'portfolio_weight'] = weight

    display_cols = [
        'symbol', 'strategy_name', 'position_sizer_name', 'stop_manager_name',
        'sharpe_ratio', 'deflated_sharpe_ratio', 'total_return_pct', 'max_drawdown_pct',
        'portfolio_weight', 'strategy_params'
    ]
    display_cols = [col for col in display_cols if col in top_combos_df.columns]

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    root_logger.info(f"\n{top_combos_df[display_cols].to_string()}")
    root_logger.info("--- Phase 3 Discovery Complete ---")


async def main():
    root_logger.info("Kamikaze Komodo Program Starting...")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot start.")
        return
     
    data_handler = DataHandler()
    try:
        await train_all_models_if_needed()
        await run_phase3_discovery(data_handler)
    finally:
        await data_handler.close()
        root_logger.info("Data handler connections closed.")

    root_logger.info("Kamikaze Komodo Program Finished.")

if __name__ == "__main__":
    try:
        if multiprocessing.get_start_method() != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
            root_logger.info("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        root_logger.info("Multiprocessing context already set.")

    try:
        if settings and settings.sentiment_llm_provider == "VertexAI" and not settings.vertex_ai_project_id:
            root_logger.warning("Vertex AI is selected, but Project ID is not set in config.ini. AI features may fail.")
         
        if not os.path.exists("logs"):
            os.makedirs("logs")

        asyncio.run(main())
    except KeyboardInterrupt:
        root_logger.info("Kamikaze Komodo program terminated by user.")
    except Exception as e:
        root_logger.critical(f"Critical error in main execution: {e}", exc_info=True)