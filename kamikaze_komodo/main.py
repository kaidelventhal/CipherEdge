# FILE: kamikaze_komodo/main.py
import asyncio
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Type, Optional, Union, List

from kamikaze_komodo.app_logger import get_logger, logger as root_logger
from kamikaze_komodo.config.settings import settings

# Core components
from kamikaze_komodo.data_handling.data_handler import DataHandler
from kamikaze_komodo.backtesting_engine.optimizer import StrategyOptimizer
from kamikaze_komodo.orchestration.scheduler import TaskScheduler
from kamikaze_komodo.core.enums import SignalType
from kamikaze_komodo.core.models import BarData

# Strategies for the Gauntlet
from kamikaze_komodo.strategy_framework.base_strategy import BaseStrategy, SignalCommand
from kamikaze_komodo.strategy_framework.strategies.ewmac import EWMACStrategy
from kamikaze_komodo.strategy_framework.strategies.bollinger_band_breakout_strategy import BollingerBandBreakoutStrategy
from kamikaze_komodo.strategy_framework.strategies.volatility_squeeze_breakout_strategy import VolatilitySqueezeBreakoutStrategy
from kamikaze_komodo.strategy_framework.strategies.funding_rate_strategy import FundingRateStrategy
from kamikaze_komodo.strategy_framework.strategies.ensemble_ml_strategy import EnsembleMLStrategy
from kamikaze_komodo.strategy_framework.strategies.regime_switching_strategy import RegimeSwitchingStrategy
from kamikaze_komodo.strategy_framework.strategies.bollinger_band_mean_reversion_strategy import BollingerBandMeanReversionStrategy
from kamikaze_komodo.strategy_framework.strategies.ehlers_instantaneous_trendline import EhlersInstantaneousTrendlineStrategy


# ML Models and Training Pipelines
from kamikaze_komodo.ml_models.training_pipelines.lightgbm_pipeline import LightGBMTrainingPipeline
from kamikaze_komodo.ml_models.training_pipelines.xgboost_classifier_pipeline import XGBoostClassifierTrainingPipeline
from kamikaze_komodo.ml_models.training_pipelines.lstm_pipeline import LSTMTrainingPipeline
from kamikaze_komodo.ml_models.training_pipelines.kmeans_regime_pipeline import KMeansRegimeTrainingPipeline
from kamikaze_komodo.ml_models.regime_detection.kmeans_regime_model import KMeansRegimeModel
from kamikaze_komodo.config.settings import PROJECT_ROOT


logger = get_logger(__name__)

# Define a simple concrete class for the HOLD action in the Regime Switcher
class HoldStrategy(BaseStrategy):
    """A simple strategy that always returns a HOLD signal."""
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # No indicators needed for this strategy
        return data

    def on_bar_data(self, current_bar: BarData) -> Union[Optional[SignalType], List[SignalCommand]]:
        return SignalType.HOLD


async def train_all_models_if_needed():
    """
    Checks for the existence of required ML models and trains them if they are missing.
    """
    root_logger.info("Checking for pre-trained ML models...")
    if not settings:
        root_logger.critical("Settings not loaded, cannot check or train models.")
        return

    symbol = settings.default_symbol
    timeframe = settings.default_timeframe

    models_to_train = {
        "LightGBM": {
            "pipeline": LightGBMTrainingPipeline,
            "config_section": "LightGBM_Forecaster",
            "default_filename": f"lgbm_{symbol.replace('/', '_').lower()}_{timeframe}.joblib"
        },
        "XGBoost": {
            "pipeline": XGBoostClassifierTrainingPipeline,
            "config_section": "XGBoost_Classifier_Forecaster",
            "default_filename": f"xgb_classifier_{symbol.replace('/', '_').lower()}_{timeframe}.joblib"
        },
        "LSTM": {
            "pipeline": LSTMTrainingPipeline,
            "config_section": "LSTM_Forecaster",
            "default_filename": f"lstm_{symbol.replace('/', '_').lower()}_{timeframe}.pth"
        },
        "KMeansRegime": {
            "pipeline": KMeansRegimeTrainingPipeline,
            "config_section": "KMeans_Regime_Model",
            "default_filename": f"kmeans_regime_{symbol.replace('/', '_').lower()}_{timeframe}.joblib"
        }
    }

    for model_name, config in models_to_train.items():
        model_params = settings.get_strategy_params(config["config_section"])
        model_base_path = model_params.get('modelsavepath', 'ml_models/trained_models')
        model_filename = model_params.get('modelfilename', config["default_filename"])
        
        full_model_path = os.path.join(PROJECT_ROOT, model_base_path, model_filename)

        if os.path.exists(full_model_path):
            logger.info(f"{model_name} model found at {full_model_path}. Skipping training.")
        else:
            logger.warning(f"{model_name} model not found at {full_model_path}. Starting training process...")
            try:
                pipeline = config["pipeline"](symbol=symbol, timeframe=timeframe)
                await pipeline.run_training()
                logger.info(f"Successfully trained and saved {model_name} model.")
            except Exception as e:
                logger.error(f"An error occurred during {model_name} model training: {e}", exc_info=True)


async def run_phase1_gauntlet():
    root_logger.info("Starting Kamikaze Komodo - Phase 1: Backtesting & Core Strategy Refinement Gauntlet")
    if not settings:
        root_logger.critical("Settings failed to load.")
        return

    data_handler = DataHandler()
    symbol = settings.default_symbol
    timeframe = settings.default_timeframe
    hist_days = settings.historical_data_days
    start_date = datetime.now(timezone.utc) - timedelta(days=hist_days)
    end_date = datetime.now(timezone.utc)

    # --- Gauntlet Configuration ---
    SLIPPAGE_MODEL_TO_TEST = 'volume_volatility_based' # Options: 'fixed', 'volume_volatility_based'
    
    # --- Prepare Regime Switching Sub-strategies ---
    regime_params = settings.get_strategy_params("RegimeSwitchingStrategy")
    trending_params = settings.get_strategy_params(regime_params.get('trending_strategy_section', 'EhlersInstantaneousTrendlineStrategy'))
    ranging_params = settings.get_strategy_params(regime_params.get('ranging_strategy_section', 'BollingerBandMeanReversionStrategy'))
    
    trending_strategy_instance = EhlersInstantaneousTrendlineStrategy(symbol, timeframe, trending_params)
    ranging_strategy_instance = BollingerBandMeanReversionStrategy(symbol, timeframe, ranging_params)

    strategy_map = {
        0: ranging_strategy_instance,
        1: trending_strategy_instance,
        2: HoldStrategy(symbol, timeframe)
    }

    strategies_to_test = {
        "EWMACStrategy": {
            "class": EWMACStrategy,
            "param_grid": { 'shortwindow': [12], 'longwindow': [26] }, # Reduced for speed
        },
        "BollingerBandBreakoutStrategy": {
            "class": BollingerBandBreakoutStrategy,
            "param_grid": { 'bb_period': [20], 'bb_std_dev': [2.0] }, # Reduced for speed
        },
        "VolatilitySqueezeBreakoutStrategy": {
            "class": VolatilitySqueezeBreakoutStrategy,
            "param_grid": { 'kc_atr_multiplier': [1.5], 'bb_std_dev': [2.0] }, # Reduced for speed
        },
        "FundingRateStrategy": {
            "class": FundingRateStrategy,
            "param_grid": { 'short_threshold': [0.0005], 'long_threshold': [-0.0005] }, # Reduced for speed
        },
        "EnsembleMLStrategy": {
            "class": EnsembleMLStrategy,
            "param_grid": { 'ensemble_method': ['majority_vote', 'weighted_average'] },
        },
        "RegimeSwitchingStrategy": {
            "class": RegimeSwitchingStrategy,
            "param_grid": { 'regime_confirmation_period': [2, 3] },
            "init_kwargs": {"strategy_mapping": strategy_map}
        },
    }
    
    # --- Prepare Data with Market Regimes ---
    root_logger.info("Preparing main data feed with market regimes for the gauntlet...")
    main_data_df = await data_handler.get_prepared_data(
        symbol, timeframe, start_date, end_date,
        needs_funding_rate=True, needs_sentiment=True
    )
    
    kmeans_params = settings.get_strategy_params("KMeans_Regime_Model")
    kmeans_config = settings.get_strategy_params("KMeans_Regime_Model")
    kmeans_path = os.path.join(PROJECT_ROOT, kmeans_config.get('modelsavepath', 'ml_models/trained_models'), f"kmeans_regime_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
    kmeans_model = KMeansRegimeModel(model_path=kmeans_path, params=kmeans_params)

    if kmeans_model.model:
        main_data_df['market_regime'] = kmeans_model.predict_regimes_for_dataframe(main_data_df)
        main_data_df['market_regime'] = main_data_df['market_regime'].ffill()
    else:
        logger.warning("KMeans model not loaded, 'market_regime' column will be empty.")
        main_data_df['market_regime'] = None
    
    if main_data_df.empty:
        logger.error("Failed to prepare main data feed. Aborting gauntlet.")
        await data_handler.close()
        return

    # --- WFO Settings ---
    total_days = (end_date - start_date).days
    train_days = int(total_days * 0.4)
    test_days = int(total_days * 0.1)
    tf_hours = 4
    if 'h' in timeframe:
        try: tf_hours = int(timeframe.replace('h', ''))
        except: pass
    elif 'd' in timeframe:
        try: tf_hours = int(timeframe.replace('d', '')) * 24
        except: pass
    bars_per_day = 24 / tf_hours if tf_hours > 0 else 1
    train_bars = int(train_days * bars_per_day)
    test_bars = int(test_days * bars_per_day)
    step_bars = test_bars
    
    gauntlet_summary = []

    for name, config in strategies_to_test.items():
        root_logger.info(f"\n{'='*25} GAUNTLET: {name} {'='*25}")
        
        if len(main_data_df) < (train_bars + test_bars):
            logger.error(f"Not enough data for {name} ({len(main_data_df)} bars). Required ~{train_bars + test_bars}. Skipping.")
            continue
            
        optimizer = StrategyOptimizer(
            strategy_class=config["class"],
            data_feed_df=main_data_df,
            param_grid=config["param_grid"],
            optimization_metric='sortino_ratio',
            initial_capital=10000.0,
            commission_bps=settings.commission_bps,
            slippage_bps=settings.slippage_bps,
            slippage_model_type=SLIPPAGE_MODEL_TO_TEST,
            symbol=symbol,
            timeframe=timeframe,
            warmup_period=200,
            strategy_init_kwargs=config.get("init_kwargs", {})
        )

        wfo_results = optimizer.walk_forward_optimization(
            training_period_bars=train_bars,
            testing_period_bars=test_bars,
            step_size_bars=step_bars
        )
        
        if wfo_results:
            results_df = pd.DataFrame(wfo_results)
            avg_sortino = results_df[f'test_sortino_ratio'].mean()
            avg_calmar = results_df[f'test_calmar_ratio'].mean() if f'test_calmar_ratio' in results_df.columns else float('nan')
            
            logger.info(f"--- WFO Results Summary for {name} ---")
            print(results_df[['train_end_date', 'test_end_date', 'best_params', 'test_sortino_ratio']].to_string())
            logger.info(f"Average Out-of-Sample Sortino Ratio: {avg_sortino:.4f}")
            logger.info(f"Average Out-of-Sample Calmar Ratio: {avg_calmar:.4f}")

            gauntlet_summary.append({
                "Strategy": name,
                "Avg Out-of-Sample Sortino": avg_sortino,
                "Avg Out-of-Sample Calmar": avg_calmar
            })
        else:
            logger.warning(f"WFO for {name} produced no results.")

    root_logger.info(f"\n{'='*25} GAUNTLET FINAL SUMMARY {'='*25}")
    if gauntlet_summary:
        summary_df = pd.DataFrame(gauntlet_summary).sort_values(by="Avg Out-of-Sample Sortino", ascending=False)
        print(summary_df.to_string())
    else:
        logger.info("No strategies completed the gauntlet.")
    
    await data_handler.close()
    root_logger.info("Phase 1 Gauntlet completed.")


async def main():
    root_logger.info("Kamikaze Komodo Program Starting...")
    if not settings:
        root_logger.critical("Settings failed to load. Application cannot start.")
        return

    await train_all_models_if_needed()
    await run_phase1_gauntlet()

    root_logger.info("Kamikaze Komodo Program Finished.")

if __name__ == "__main__":
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