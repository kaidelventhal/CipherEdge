# FILE: kamikaze_komodo/ml_models/training_pipelines/xgboost_classifier_pipeline.py
import pandas as pd
from datetime import datetime, timedelta, timezone
import os
from kamikaze_komodo.ml_models.price_forecasting.xgboost_classifier_forecaster import XGBoostClassifierForecaster
from kamikaze_komodo.data_handling.data_fetcher import DataFetcher
from kamikaze_komodo.data_handling.database_manager import DatabaseManager
from kamikaze_komodo.app_logger import get_logger
from kamikaze_komodo.config.settings import settings, PROJECT_ROOT

logger = get_logger(__name__)

class XGBoostClassifierTrainingPipeline:
    def __init__(self, symbol: str, timeframe: str, model_config_section: str = "XGBoost_Classifier_Forecaster"):
        if not settings:
            raise ValueError("Settings not loaded.")
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_config_section = model_config_section
        
        self.model_params = settings.get_strategy_params(model_config_section)
        
        _model_base_path = self.model_params.get('modelsavepath', 'ml_models/trained_models')
        _model_filename = self.model_params.get('modelfilename', f"xgb_classifier_{symbol.replace('/', '_').lower()}_{timeframe}.joblib")
        
        if not os.path.isabs(_model_base_path):
            self.model_save_path_dir = os.path.join(PROJECT_ROOT, _model_base_path)
        else:
            self.model_save_path_dir = _model_base_path
            
        if not os.path.exists(self.model_save_path_dir):
            os.makedirs(self.model_save_path_dir, exist_ok=True)
            logger.info(f"Created directory for trained XGBoost models: {self.model_save_path_dir}")
            
        self.model_full_save_path = os.path.join(self.model_save_path_dir, _model_filename)
        
        self.forecaster = XGBoostClassifierForecaster(params=self.model_params)
        logger.info(f"XGBoost Training Pipeline initialized for {symbol} ({timeframe}). Model will be saved to: {self.model_full_save_path}")

    async def fetch_training_data(self, days_history: int = 730) -> pd.DataFrame:
        db_manager = DatabaseManager()
        data_fetcher = DataFetcher()
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days_history)
        end_date = datetime.now(timezone.utc)
        logger.info(f"Attempting to retrieve data from DB for {self.symbol} ({self.timeframe}) from {start_date} to {end_date}")
        historical_bars = db_manager.retrieve_bar_data(self.symbol, self.timeframe, start_date, end_date)
        
        required_bars = int(self.model_params.get('minbarsfortraining', 200))
        if not historical_bars or len(historical_bars) < required_bars:
            logger.info(f"Insufficient data in DB ({len(historical_bars)} bars found). Fetching fresh data for XGBoost training...")
            historical_bars = await data_fetcher.fetch_historical_data_for_period(self.symbol, self.timeframe, start_date, end_date)
            if historical_bars:
                db_manager.store_bar_data(historical_bars)
            else:
                logger.error("Failed to fetch training data for XGBoost.")
                await data_fetcher.close()
                db_manager.close()
                return pd.DataFrame()
        
        await data_fetcher.close()
        db_manager.close()
        if not historical_bars:
            logger.error("No historical bars available for XGBoost training.")
            return pd.DataFrame()
            
        data_df = pd.DataFrame([bar.model_dump() for bar in historical_bars])
        data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
        data_df.set_index('timestamp', inplace=True)
        data_df.sort_index(inplace=True)
        
        if settings.enable_sentiment_analysis and settings.simulated_sentiment_data_path:
            sentiment_path = settings.simulated_sentiment_data_path
            if os.path.exists(sentiment_path):
                logger.info(f"Loading sentiment data from {sentiment_path} to merge for training.")
                sentiment_df = pd.read_csv(sentiment_path, parse_dates=['timestamp'], index_col='timestamp')
                if not sentiment_df.index.tz:
                    sentiment_df.index = sentiment_df.index.tz_localize('UTC')
                else:
                    sentiment_df.index = sentiment_df.index.tz_convert('UTC')
                
                # --- FIX: Drop existing sentiment_score column to prevent overlap error ---
                if 'sentiment_score' in data_df.columns:
                    data_df = data_df.drop(columns=['sentiment_score'])
                    
                data_df = data_df.join(sentiment_df['sentiment_score'], how='left')
                data_df['sentiment_score'].ffill(inplace=True)
                data_df['sentiment_score'].fillna(0.0, inplace=True)
                logger.info("Successfully merged sentiment data into the training set.")
            else:
                logger.warning(f"Sentiment data file not found at {sentiment_path}. Training without sentiment feature.")
                data_df['sentiment_score'] = 0.0
        else:
            logger.info("Sentiment analysis not enabled or no data path provided. Training without sentiment feature.")
            data_df['sentiment_score'] = 0.0

        logger.info(f"Fetched and prepared {len(data_df)} bars for XGBoost training {self.symbol} ({self.timeframe}).")
        return data_df

    async def run_training(self):
        days_history = int(self.model_params.get('trainingdayshistory', 730))
        historical_df = await self.fetch_training_data(days_history=days_history)
        if historical_df.empty:
            logger.error("Cannot run XGBoost training, no historical data.")
            return

        target_def = self.model_params.get('targetdefinition', 'next_bar_direction')
        feature_cols_str = self.model_params.get('feature_columns')
        feature_columns = [col.strip() for col in feature_cols_str.split(',')] if feature_cols_str else None
        
        logger.info(f"Starting XGBoost training with target definition: '{target_def}', features: {feature_columns if feature_columns else 'default in forecaster'}")
        self.forecaster.train(historical_df, target_definition=target_def, feature_columns=feature_columns)
        
        if self.forecaster.model:
            self.forecaster.save_model(self.model_full_save_path)
        else:
            logger.error("XGBoost training did not produce a model. Model not saved.")